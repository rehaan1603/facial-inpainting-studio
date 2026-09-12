"""Development-only comparison of extended training and learned probability compositing."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch,lpips
from benchmark_v2 import case
from area_matched_v3 import corrupt
from inpaint import RefinerPredictor,Inpainter
from pilot import metrics
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probability_blend(raw,observed,probability,effective):
    alpha=np.where(effective,probability,0).astype('float32')[...,None]
    return alpha*raw+(1-alpha)*observed
def main():
    local=json.loads((ROOT/'configs/local.json').read_text());cfg=json.loads((ROOT/'configs/refiner_extension_v1.json').read_text());torch.set_num_threads(4);torch.hub.set_dir(str(Path(local['cache'])/'torch'))
    manifest=ROOT/'data/manifests/celebahq_reviewed_v2.csv';sources={int(r['hq_id']):r for r in csv.DictReader(manifest.open())}
    tuning=[c for c in json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text()) if c['partition']=='tuning'];assessment=[c for c in json.loads((ROOT/'outputs/area_matched_v3_margin12/cases.json').read_text()) if c['partition']=='assessment']
    assert len({c['identity'] for c in tuning})==len({c['identity'] for c in assessment})==48
    assert not {c['identity'] for c in tuning}&{c['identity'] for c in assessment}
    out=ROOT/'outputs/extension_evaluation_v1';out.mkdir(exist_ok=True);signed=[Path(__file__),ROOT/'scripts/inpaint.py',ROOT/'scripts/refiner.py',ROOT/'scripts/benchmark_v2.py',ROOT/'scripts/area_matched_v3.py',ROOT/'scripts/pilot.py',manifest,ROOT/'outputs/benchmark_v2/cases.json',ROOT/'outputs/area_matched_v3_margin12/cases.json']
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            folder=ROOT/'outputs/refiner_extension_v1'/f'{variant}_{seed}';assert (folder/'training.json').exists(),'Finish all six matched runs before assessing'
            signed += [folder/'best.pt',folder/'training.json']
    signature={str(p.relative_to(ROOT)):digest(p) for p in signed};sig=out/'signature.json'
    if sig.exists():assert json.loads(sig.read_text())==signature
    else:sig.write_text(json.dumps(signature,indent=2))
    def target(c):
        row=sources[int(c['hq_id'])];assert row['split']=='val' and digest(row['image_path'])==c['source_sha256']
        with Image.open(row['image_path']) as im:return np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
    models=[]
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            path=ROOT/'outputs/refiner_extension_v1'/f'{variant}_{seed}/best.pt';predictor=RefinerPredictor(path);scores={t:[] for t in [.2,.35,.5,.65,.8]}
            for c in tuning:
                x=target(c);folder=ROOT/'outputs/benchmark_v2/cases'/f"{c['hq_id']}_{c['family']}"
                with Image.open(folder/'true_mask.png') as im:true=np.asarray(im)>0
                observed,masks,_=case(x,true,c['seed'])
                for supplied in masks.values():
                    probability=predictor.probability(observed,supplied)
                    for threshold in scores:
                        selected=probability>=threshold;scores[threshold].append(float((~selected&true).sum()/true.sum()+4*(selected&~true).sum()/(~true).sum()))
            scores={t:float(np.mean(v)) for t,v in scores.items()};threshold=min(scores,key=lambda t:(scores[t],-t))
            (out/f'{variant}_{seed}_selection.json').write_text(json.dumps({'threshold':threshold,'scores':scores,'checkpoint_sha256':digest(path),'selection':'Same five-threshold FNR+4*FPR tuning rule as first stage. Soft output uses p directly; no extra parameter selection.'},indent=2))
            models.append((variant,seed,path,threshold));del predictor;print('Selected',variant,seed,threshold,flush=True)
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    @torch.inference_mode()
    def score(predictions,x):
        tensor=lambda a:torch.from_numpy(a.transpose(2,0,1).copy()).cuda()[None]*2-1
        return perceptual(torch.cat([tensor(p) for p in predictions]),tensor(x).expand(len(predictions),-1,-1,-1)).flatten().cpu().numpy()
    records=[]
    for backbone in ['lama','resshift']:
        inpainter=Inpainter(backbone)
        for variant,seed,path,threshold in models:
            if backbone=='resshift' and seed!=17:continue
            folder=out/f'{backbone}_{variant}_{seed}';folder.mkdir(exist_ok=True);predictor=RefinerPredictor(path)
            for i,c in enumerate(assessment):
                chunk=folder/f"{c['hq_id']}_{c['location']}_{c['missing_pixels']}.json"
                if chunk.exists():records.extend(json.loads(chunk.read_text()));continue
                x=target(c);mask_folder=ROOT/c['mask_directory']
                with Image.open(mask_folder/'true.png') as im:true=np.asarray(im)>0
                observed=corrupt(x,true,c['seed'],c['center_xy']);current=[]
                for condition in c['conditions']:
                    with Image.open(mask_folder/f'{condition}.png') as im:supplied=np.asarray(im)>0
                    probability=predictor.probability(observed,supplied);effective=probability>=threshold;raw=inpainter(observed,effective,c['seed']);soft=probability_blend(raw,observed,probability,effective)
                    predictions=[raw,soft];values=score(predictions,x)
                    for compositor,pred,value in zip(['hard','probability_alpha'],predictions,values):
                        assert np.isfinite(pred).all() and np.isfinite(value)
                        current.append({'backbone':backbone,'variant':variant,'training_seed':seed,'compositor':compositor,'threshold':threshold,'hq_id':c['hq_id'],'identity':c['identity'],'location':c['location'],'missing_pixels':c['missing_pixels'],'condition':condition,'full_face_lpips':float(value),'brier_score':float(np.mean((probability-true.astype('float32'))**2)),'effective_fp_fraction':float((effective&~true).mean()),'effective_fn_fraction':float((~effective&true).mean()),**metrics(pred,x,observed,true,supplied)})
                tmp=chunk.with_suffix('.tmp');tmp.write_text(json.dumps(current));tmp.replace(chunk);records.extend(current)
                if (i+1)%72==0:print(backbone,variant,seed,i+1,'/',len(assessment),flush=True)
            del predictor
        del inpainter;torch.cuda.empty_cache()
    assert len(records)==len(assessment)*4*2*8==27648
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print('Finished extension assessment',len(records),'rows; no test inference')
if __name__=='__main__':main()
