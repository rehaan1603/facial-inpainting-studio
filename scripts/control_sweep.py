"""Tune simple mask/compositing controls, then assess one frozen combination."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import lpips
from benchmark_v2 import case
from pilot import morph, metrics

ROOT=Path(__file__).resolve().parents[1]


def feather(mask,width):
    if width==0: return mask.astype(np.float32)
    # Padding defines the outside even for a mask touching the image boundary.
    dist=distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
    return np.minimum(dist/width,1).astype(np.float32)


def choose(rows):
    values=defaultdict(list)
    for r in rows:
        if r['partition']=='tuning': values[(r['radius'],r['feather'])].append(r['full_face_lpips'])
    means={k:float(np.mean(v)) for k,v in values.items()}
    best=min(means,key=lambda k:(means[k],*k))
    return best,means


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['lama','resshift'],default='lama');args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/control_sweep.json').read_text())
    local=json.loads((ROOT/'configs/local.json').read_text())
    cases=json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text())
    manifest=ROOT/'data/manifests/celebahq_clean.csv'
    sources={int(r['hq_id']):r for r in csv.DictReader(manifest.open())}
    out=ROOT/'outputs'/f'control_sweep_{args.backbone}';out.mkdir(parents=True,exist_ok=True)
    signature={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['configs/control_sweep.json','scripts/control_sweep.py','data/manifests/celebahq_clean.csv','outputs/benchmark_v2/cases.json']}
    signature['backbone']=args.backbone
    sigfile=out/'run_signature.json'
    if sigfile.exists(): assert json.loads(sigfile.read_text())==signature,'Existing run differs; use a new output directory'
    else: sigfile.write_text(json.dumps(signature,indent=2))
    torch.set_num_threads(4);torch.hub.set_dir(str(Path(local['cache'])/'torch'))
    if args.backbone=='lama': model=torch.jit.load(str(Path(local['cache'])/'models/big-lama.pt'),map_location='cuda').eval()
    else:
        from resshift_adapter import ResShiftFace
        model=ResShiftFace()
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    def tensor(x): return torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0).cuda()
    @torch.inference_mode()
    def infer(obs,mask,seed):
        if args.backbone=='resshift': return model(obs,mask,seed=seed)
        raw=model(tensor(obs),torch.from_numpy(mask.astype('float32')).cuda()[None,None])[0].permute(1,2,0).cpu().numpy().clip(0,1)
        return np.where(mask[...,None],raw,obs)
    @torch.inference_mode()
    def scores(predictions,target):
        batch=torch.cat([tensor(p) for p in predictions])
        return perceptual(batch*2-1,tensor(target).expand(len(predictions),-1,-1,-1)*2-1).flatten().cpu().tolist()
    records=[];selected=None
    for partition in ['tuning','assessment']:
        combinations=[(r,w) for r in cfg['radii'] for w in cfg['feather_widths']] if partition=='tuning' else [selected]
        subset=[c for c in cases if c['partition']==partition]
        for i,c in enumerate(subset):
            chunk=out/f"{partition}_{c['hq_id']}_{c['family']}.json"
            if chunk.exists(): records.extend(json.loads(chunk.read_text()));continue
            row=sources[c['hq_id']];assert row['split']=='val'
            assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==c['source_sha256']
            with Image.open(row['image_path']) as im: target=np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
            folder=ROOT/'outputs/benchmark_v2/cases'/f"{c['hq_id']}_{c['family']}"
            true=np.array(Image.open(folder/'true_mask.png'))>0
            obs,conditions,params=case(target,true,c['seed'])
            assert params==c['parameters']
            assert np.array_equal((obs*255).round().astype('uint8'),np.array(Image.open(folder/'observed.png')))
            chunkrows=[];cache={}
            for condition,supplied in conditions.items():
                assert np.array_equal(supplied,np.array(Image.open(folder/f'{condition}.png'))>0)
                for radius in sorted({r for r,w in combinations}):
                    expanded=morph(supplied,radius);key=hashlib.sha256(expanded.tobytes()).hexdigest()
                    if key not in cache: cache[key]=infer(obs,expanded,c['seed'])
                    raw=cache[key];widths=[w for r,w in combinations if r==radius]
                    predictions=[]
                    for width in widths:
                        alpha=feather(expanded,width)[...,None]
                        predictions.append(alpha*raw+(1-alpha)*obs)
                    values=scores(predictions,target)
                    for width,pred,value in zip(widths,predictions,values):
                        assert np.isfinite(pred).all() and np.isfinite(value)
                        chunkrows.append({'hq_id':c['hq_id'],'identity':c['identity'],'partition':partition,'family':c['family'],'condition':condition,'radius':radius,'feather':width,'full_face_lpips':value,**metrics(pred,target,obs,true,supplied)})
            tmp=chunk.with_suffix('.tmp');tmp.write_text(json.dumps(chunkrows));tmp.replace(chunk)
            records.extend(chunkrows)
            print(f'{args.backbone} {partition}: {i+1}/{len(subset)} cases',flush=True)
        if partition=='tuning':
            selected,means=choose(records)
            (out/'selection.json').write_text(json.dumps({'radius':selected[0],'feather':selected[1],'tuning_means':{f'{r},{w}':v for (r,w),v in means.items()}},indent=2))
            print('Frozen tuning selection:',selected,flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    assessment=[r for r in records if r['partition']=='assessment']
    result={'backbone':args.backbone,'selected_radius':selected[0],'selected_feather':selected[1],'tuning_rows':sum(r['partition']=='tuning' for r in records),'assessment_rows':len(assessment),'assessment':{k:float(np.mean([r[k] for r in assessment])) for k in ['full_face_lpips','hole_mae','visible_mae']},'signature':signature,'limits':cfg['scope']}
    (ROOT/'research'/f'control_sweep_{args.backbone}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
