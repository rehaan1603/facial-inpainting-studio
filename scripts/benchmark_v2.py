"""Irregular/semantic synthetic mask diagnostic with disjoint tuning/assessment identities."""
import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift
import torch
import lpips
from pilot import morph, metrics

ROOT=Path(__file__).resolve().parents[1]


def semantic_mask(hq,idx,family,size):
    components={'eyes':['l_eye','r_eye','l_brow','r_brow'],'mouth':['mouth','u_lip','l_lip']}[family]
    union=np.zeros((size,size),bool)
    for name in components:
        p=hq/'CelebAMask-HQ-mask-anno'/str(idx//2000)/f'{idx:05d}_{name}.png'
        if p.exists():
            with Image.open(p) as im:
                union |= np.array(im.convert('L').resize((size,size),Image.Resampling.NEAREST))>0
    if not union.any(): return None
    yy,xx=np.where(union)
    # Bounding region simulates an occluder covering a facial component, not the parser as input.
    pad=8 if family=='eyes' else 12
    mask=np.zeros_like(union)
    mask[max(0,yy.min()-pad):min(size,yy.max()+pad+1),max(0,xx.min()-pad):min(size,xx.max()+pad+1)]=True
    return mask


def case(target, true, seed):
    rng=np.random.default_rng(seed); size=true.shape[0]
    texture=rng.integers(0,256,(12,12,3),dtype=np.uint8)
    occ=np.array(Image.fromarray(texture).resize((size,size),Image.Resampling.BILINEAR)).astype(np.float32)/255
    observed=np.where(true[...,None],occ,target).astype(np.float32)
    erode=int(rng.integers(3,10)); expand=int(rng.integers(3,10))
    offsets=[int(rng.choice([-1,1])*rng.integers(4,11)) for _ in range(2)]
    signed=distance_transform_edt(true)-distance_transform_edt(~true)
    noise=gaussian_filter(rng.normal(size=(size,size)),sigma=10)
    noise=noise/max(float(noise.std()),1e-8)*5
    conditions={'accurate':true.copy(),'under':morph(true,-erode),'over':morph(true,expand),
                'shift':shift(true.astype(float),offsets,order=0,mode='constant',cval=0)>0.5,
                'mixed_boundary':signed+noise>0}
    return observed,conditions,{'erode':erode,'expand':expand,'shift':offsets}


def brush(seed,size):
    rng=np.random.default_rng(seed); im=Image.new('L',(size,size)); draw=ImageDraw.Draw(im)
    pts=[tuple(map(int,p)) for p in rng.integers(size//5,4*size//5,size=(4,2))]
    width=int(rng.integers(20,45)); draw.line(pts,fill=255,width=width,joint='curve')
    for x,y in pts: draw.ellipse((x-width//2,y-width//2,x+width//2,y+width//2),fill=255)
    return np.array(im)>0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=['lama','resshift'],default='lama')
    args=parser.parse_args()
    local=json.loads((ROOT/'configs/local.json').read_text())
    config_path=ROOT/'configs/benchmark_v2.json'; cfg=json.loads(config_path.read_text())
    run_name='benchmark_v2' if args.backbone=='lama' else 'benchmark_v2_resshift'
    out=ROOT/'outputs'/run_name; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.hub.set_dir(str(Path(local['cache'])/'torch'))
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    if args.backbone=='lama':
        model=torch.jit.load(str(Path(local['cache'])/'models/big-lama.pt'),map_location='cuda').eval()
    else:
        from resshift_adapter import ResShiftFace
        model=ResShiftFace()
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    manifest=ROOT/'data/manifests/celebahq_clean.csv'
    rows=list(csv.DictReader(manifest.open()))
    prior={r['identity'] for r in json.loads((ROOT/'outputs/pilot_clean_v1/cases.json').read_text())}
    rows=[r for r in rows if r['split']=='val' and r['identity'] not in prior]
    rows.sort(key=lambda r:hashlib.sha256(f"{cfg['seed']}:{r['hq_id']}".encode()).hexdigest())
    selected=[]; seen=set(); skipped=[]
    hq=Path(local['hq']); size=cfg['resolution']
    for r in rows:
        if r['identity'] in seen: continue
        idx=int(r['hq_id'])
        masks={'brush':brush(cfg['seed']+idx,size),'eyes':semantic_mask(hq,idx,'eyes',size),
               'mouth':semantic_mask(hq,idx,'mouth',size)}
        if any(m is None or not (0.01<float(m.mean())<0.65) for m in masks.values()):
            skipped.append(r['hq_id']); continue
        selected.append((r,masks)); seen.add(r['identity'])
        if len(selected)==cfg['images']: break
    assert len(selected)==cfg['images']
    tuning={r['identity'] for r,_ in selected[:cfg['tuning_images']]}
    assessment={r['identity'] for r,_ in selected[cfg['tuning_images']:]}
    assert not tuning & assessment and not (tuning|assessment)&prior
    if args.backbone=='resshift':
        original=json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text())
        assert [int(r['hq_id']) for r,_ in selected]==list(dict.fromkeys(c['hq_id'] for c in original)), 'Cases differ from LaMa benchmark'
    @torch.inference_mode()
    def tensor(im): return torch.from_numpy(im.transpose(2,0,1).copy()).unsqueeze(0).cuda()
    @torch.inference_mode()
    def infer(obs,mask,seed):
        if args.backbone=='resshift': return model(obs,mask,seed=seed)
        x=tensor(obs); m=torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        raw=model(x,m)[0].permute(1,2,0).cpu().numpy().clip(0,1)
        return np.where(mask[...,None],raw,obs)
    @torch.inference_mode()
    def score(pred,tgt): return float(perceptual(tensor(pred)*2-1,tensor(tgt)*2-1).item())
    records=[]; cases=[]; preview=[]
    for i,(r,masks) in enumerate(selected):
        idx=int(r['hq_id']); partition='tuning' if r['identity'] in tuning else 'assessment'
        with Image.open(r['image_path']) as im: target=np.array(im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)).astype(np.float32)/255
        assert hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()==r['source_sha256']
        for fi,(family,true) in enumerate(masks.items()):
            seed=cfg['seed']+idx*10+fi
            obs,conditions,params=case(target,true,seed)
            folder=out/'cases'/f'{idx}_{family}'; folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray((obs*255).round().astype('uint8')).save(folder/'observed.png')
            Image.fromarray(true.astype('uint8')*255).save(folder/'true_mask.png')
            cache={}
            def cached(mask):
                key=hashlib.sha256(mask.tobytes()).hexdigest()
                if key not in cache:
                    pred=infer(obs,mask,seed); cache[key]=(pred,score(pred,target))
                return cache[key]
            oracle,oracle_lpips=cached(true)
            cases.append({'hq_id':idx,'identity':r['identity'],'partition':partition,'family':family,
                          'seed':seed,'parameters':params,'source_sha256':r['source_sha256']})
            for condition,supplied in conditions.items():
                Image.fromarray(supplied.astype('uint8')*255).save(folder/f'{condition}.png')
                base={'hq_id':idx,'identity':r['identity'],'partition':partition,'family':family,'condition':condition}
                records.append({**base,'radius':'oracle','full_face_lpips':oracle_lpips,**metrics(oracle,target,obs,true,supplied)})
                for radius in cfg['dilation_radii']:
                    pred,value=cached(morph(supplied,radius))
                    records.append({**base,'radius':radius,'full_face_lpips':value,**metrics(pred,target,obs,true,supplied)})
                    if i==cfg['tuning_images'] and condition=='mixed_boundary' and radius==0:
                        dilated,_=cached(morph(supplied,4))
                        preview.append((family,[target,obs,supplied[...,None].repeat(3,2).astype(float),pred,dilated,oracle]))
        print(f'Benchmark {i+1}/{len(selected)} ({partition})',flush=True)
        # Persist completed identities so interruption does not erase diagnostic measurements.
        with (out/'metrics.partial.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    (out/'cases.json').write_text(json.dumps(cases,indent=2))
    means={r:float(np.mean([x['full_face_lpips'] for x in records if x['partition']=='tuning' and x['radius']==r])) for r in cfg['dilation_radii']}
    best=min(means,key=lambda r:(means[r],r))
    summary=[]
    for partition in ['tuning','assessment']:
        for radius in cfg['dilation_radii']+['oracle']:
            subset=[x for x in records if x['partition']==partition and x['radius']==radius]
            summary.append({'partition':partition,'radius':radius,**{k:float(np.mean([x[k] for x in subset]))
                        for k in ['full_face_lpips','hole_mae','visible_mae']}})
    # Bootstrap matched differences over assessment identities, not correlated masks.
    by_id=defaultdict(dict)
    for x in records:
        if x['partition']=='assessment': by_id[x['identity']].setdefault(x['radius'],[]).append(x['full_face_lpips'])
    delta=np.array([np.mean(v[best])-np.mean(v[0]) for v in by_id.values()])
    rng=np.random.default_rng(cfg['seed'])
    samples=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
    report={'backbone':args.backbone,'diffusion_seed_policy':'One fixed seed per identity/family, shared across mask controls; no multi-seed robustness claim.',
        'config':cfg,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'selected_radius':best,'selection_tuning_means':means,'summary':summary,
        'assessment_lpips_selected_minus_radius0':{'mean':float(delta.mean()),'bootstrap_95_interval':np.percentile(samples,[2.5,97.5]).tolist(),
            'unit':'identity','bootstrap_replicates':2000},'skipped_missing_or_extreme_semantic_masks':skipped,
        'limits':['Validation development study, not final test.','Synthetic textures and masks only.',
            'Semantic masks depend on ground truth only for corruption construction, not inference.',
            'Mask families are not area-matched.','Pretraining exposure unresolved; no novel method tested.']}
    (ROOT/'research'/f'{run_name}_results.json').write_text(json.dumps(report,indent=2))
    if preview:
        canvas=Image.new('RGB',(size*6,280*len(preview)),'white');draw=ImageDraw.Draw(canvas)
        labels=['Target','Observed','Supplied mask',f'{args.backbone} radius 0',f'{args.backbone} radius 4','Exact-mask diagnostic']
        for row,(family,images) in enumerate(preview):
            for col,im in enumerate(images):
                canvas.paste(Image.fromarray((im*255).round().astype('uint8')),(col*size,row*280+24))
                draw.text((col*size+3,row*280+3),f'{family}: {labels[col]}',fill='black')
        canvas.save(out/'preview.png')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
