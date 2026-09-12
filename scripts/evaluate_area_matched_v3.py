"""Frozen v2 controls on v3 assessment geometry; no v3 tuning."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import lpips
from area_matched_v3 import corrupt
from control_sweep import feather
from pilot import morph,metrics
from resshift_adapter import ResShiftFace

ROOT=Path(__file__).resolve().parents[1]


def main():
    local=json.loads((ROOT/'configs/local.json').read_text())
    path=ROOT/'outputs/area_matched_v3_margin12/cases.json'
    verification=json.loads((ROOT/'research/area_matched_v3_verification.json').read_text())
    assert verification['all_checks_passed'] and hashlib.sha256(path.read_bytes()).hexdigest()==verification['cases_sha256']
    cases=[c for c in json.loads(path.read_text()) if c['partition']=='assessment']
    assert len(cases)>0 and len({c['identity'] for c in cases})==verification['identities']['assessment']
    sources={int(r['hq_id']):r for r in csv.DictReader((ROOT/'data/manifests/celebahq_clean.csv').open())}
    out=ROOT/'outputs/area_matched_v3_margin12_evaluation';out.mkdir(parents=True,exist_ok=True)
    signatures={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/evaluate_area_matched_v3.py','scripts/area_matched_v3.py','outputs/area_matched_v3_margin12/cases.json','scripts/resshift_adapter.py','scripts/control_sweep.py']}
    guard=out/'signature.json'
    if guard.exists():assert json.loads(guard.read_text())==signatures
    else:guard.write_text(json.dumps(signatures,indent=2))
    torch.set_num_threads(4);torch.hub.set_dir(str(Path(local['cache'])/'torch'))
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    def tensor(a):return torch.from_numpy(a.transpose(2,0,1).copy()).cuda()[None]
    @torch.inference_mode()
    def score(pred,target):return float(perceptual(tensor(pred)*2-1,tensor(target)*2-1).item())
    allrows=[]
    for backbone,controls in [('lama',[(8,0)]),('resshift',[(8,0),(12,4)])]:
        model=torch.jit.load(str(Path(local['cache'])/'models/big-lama.pt'),map_location='cuda').eval() if backbone=='lama' else ResShiftFace()
        @torch.inference_mode()
        def infer(obs,mask,seed):
            if backbone=='resshift':return model(obs,mask,seed=seed)
            raw=model(tensor(obs),torch.from_numpy(mask.astype('float32')).cuda()[None,None])[0].permute(1,2,0).cpu().numpy().clip(0,1)
            return np.where(mask[...,None],raw,obs)
        for i,c in enumerate(cases):
            chunk=out/f"{backbone}_{c['hq_id']}_{c['missing_pixels']}_{c['location']}.json"
            if chunk.exists():allrows.extend(json.loads(chunk.read_text()));continue
            source=sources[c['hq_id']];assert source['split']=='val'
            assert hashlib.sha256(Path(source['image_path']).read_bytes()).hexdigest()==c['source_sha256']
            with Image.open(source['image_path']) as im:target=np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
            folder=ROOT/c['mask_directory']
            def read(name):
                p=folder/f'{name}.png';assert hashlib.sha256(p.read_bytes()).hexdigest()==c['mask_sha256'][name]
                with Image.open(p) as im:return np.array(im)>0
            true=read('true');obs=corrupt(target,true,c['seed'],c['center_xy']);rows=[]
            for condition in c['conditions']:
                supplied=read(condition)
                for radius,width in controls:
                    mask=morph(supplied,radius);pred=infer(obs,mask,c['seed']);alpha=feather(mask,width)[...,None]
                    pred=alpha*pred+(1-alpha)*obs
                    rows.append({'backbone':backbone,'radius':radius,'feather':width,'hq_id':c['hq_id'],'identity':c['identity'],'location':c['location'],'missing_pixels':c['missing_pixels'],'condition':condition,'full_face_lpips':score(pred,target),**metrics(pred,target,obs,true,supplied)})
            tmp=chunk.with_suffix('.tmp');tmp.write_text(json.dumps(rows));tmp.replace(chunk);allrows.extend(rows)
            if (i+1)%12==0:print(f'{backbone}: {i+1}/{len(cases)} cases',flush=True)
        del model;torch.cuda.empty_cache()
    assert len(allrows)==len(cases)*12
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=allrows[0]);writer.writeheader();writer.writerows(allrows)
    print(f'Complete: {len(allrows)} metric rows',flush=True)


if __name__=='__main__':main()
