"""Choose segmentation thresholds on v2 tuning; evaluate frozen models on v3 assessment."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import lpips
from benchmark_v2 import case
from area_matched_v3 import corrupt
from inpaint import RefinerPredictor,Inpainter
from pilot import metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=json.loads((ROOT/'configs/learned_refiner.json').read_text());local=json.loads((ROOT/'configs/local.json').read_text());torch.set_num_threads(4);torch.hub.set_dir(str(Path(local['cache'])/'torch'))
    manifest=ROOT/cfg['manifest'];sources={int(r['hq_id']):r for r in csv.DictReader(manifest.open())}
    tuning=[c for c in json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text()) if c['partition']=='tuning']
    assess=[c for c in json.loads((ROOT/'outputs/area_matched_v3_margin12/cases.json').read_text()) if c['partition']=='assessment']
    assert not {c['identity'] for c in tuning}&{c['identity'] for c in assess}
    out=ROOT/'outputs/refiner_evaluation';out.mkdir(parents=True,exist_ok=True)
    signed=[Path(__file__),ROOT/'scripts/inpaint.py',ROOT/'scripts/refiner.py',manifest,ROOT/'outputs/benchmark_v2/cases.json',ROOT/'outputs/area_matched_v3_margin12/cases.json']
    signed += list((ROOT/'outputs/learned_refiner').glob('*/best.pt'))
    signature={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in signed}
    signature_path=out/'signature.json'
    if signature_path.exists():assert json.loads(signature_path.read_text())==signature,'Inputs changed: use a new output directory'
    else:signature_path.write_text(json.dumps(signature,indent=2))
    thresholds=[.2,.35,.5,.65,.8];models=[]
    def target(c):
        row=sources[int(c['hq_id'])];assert row['split']=='val';assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==c['source_sha256']
        with Image.open(row['image_path']) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            path=ROOT/'outputs/learned_refiner'/f'{variant}_{seed}'/'best.pt';completion=path.with_name('training.json');assert completion.exists(),'Training must finish before evaluation'
            predictor=RefinerPredictor(path);scores={t:[] for t in thresholds}
            for c in tuning:
                x=target(c);folder=ROOT/'outputs/benchmark_v2/cases'/f"{c['hq_id']}_{c['family']}"
                with Image.open(folder/'true_mask.png') as im:true=np.array(im)>0
                observed,masks,_=case(x,true,c['seed'])
                for supplied in masks.values():
                    prob=predictor.probability(observed,supplied)
                    for threshold in thresholds:
                        selected=prob>=threshold
                        # Same predeclared cost for both training variants; no assessment selection.
                        fn=float((~selected&true).sum()/true.sum());fp=float((selected&~true).sum()/(~true).sum())
                        scores[threshold].append(fn+4*fp)
            means={t:float(np.mean(v)) for t,v in scores.items()};chosen=min(means,key=lambda t:(means[t],-t))
            selection={'variant':variant,'seed':seed,'threshold':chosen,'tuning_costs':means,'selection_rule':'Minimize FNR + 4*FPR on v2 tuning only; ties prefer higher threshold.','checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (out/f'{variant}_{seed}_selection.json').write_text(json.dumps(selection,indent=2));models.append((variant,seed,path,chosen));del predictor
            print(f'Selected {variant} seed={seed} threshold={chosen}',flush=True)
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    @torch.inference_mode()
    def lp(pred,tgt):
        tensor=lambda a:torch.from_numpy(a.transpose(2,0,1).copy()).cuda()[None]*2-1
        return float(perceptual(tensor(pred),tensor(tgt)).item())
    records=[]
    # All three seeds on LaMa; predefined seed 17 on ResShift is exploratory transfer.
    for backbone in ['lama','resshift']:
        inpainter=Inpainter(backbone)
        for variant,seed,path,threshold in models:
            if backbone=='resshift' and seed!=17:continue
            predictor=RefinerPredictor(path);chunk=out/f'{backbone}_{variant}_{seed}.csv'
            if chunk.exists():records.extend(list(csv.DictReader(chunk.open())));continue
            current=[]
            for i,c in enumerate(assess):
                x=target(c);folder=ROOT/c['mask_directory']
                with Image.open(folder/'true.png') as im:true=np.array(im)>0
                observed=corrupt(x,true,c['seed'],c['center_xy'])
                for condition in c['conditions']:
                    with Image.open(folder/f'{condition}.png') as im:supplied=np.array(im)>0
                    probability=predictor.probability(observed,supplied);effective=probability>=threshold;pred=inpainter(observed,effective,c['seed'])
                    current.append({'backbone':backbone,'variant':variant,'training_seed':seed,'threshold':threshold,'hq_id':c['hq_id'],'identity':c['identity'],'location':c['location'],'missing_pixels':c['missing_pixels'],'condition':condition,'full_face_lpips':lp(pred,x),'effective_fp_fraction':float((effective&~true).mean()),'effective_fn_fraction':float((~effective&true).mean()),**metrics(pred,x,observed,true,supplied)})
                if (i+1)%72==0:print(f'{backbone} {variant} seed={seed}: {i+1}/{len(assess)}',flush=True)
            with chunk.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=current[0]);writer.writeheader();writer.writerows(current)
            records.extend(current);del predictor
        del inpainter;torch.cuda.empty_cache()
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    assert len(records)==len(assess)*4*8
    print('Learned control evaluation complete:',len(records),'rows')


if __name__=='__main__':main()
