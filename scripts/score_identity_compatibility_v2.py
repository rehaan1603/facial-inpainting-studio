"""Stream all generated rows into frozen metrics, then paired seed/identity tables."""
import json,time
import numpy as np
import torch
from extended_evaluation_metrics import ROOT,Metrics,read_rgb,sha
from score_target_compatibility_v1 import components
from region_metrics_v1 import score_regions
from train_identity_compatibility_v2 import OUT

KEYS=['psnr','ssim','lpips','mae','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips','niqe','brisque']
HIGH={'psnr','ssim','facenet_cosine','arcface_conditioning_cosine'}

def paired(rows,arm,condition,seed,key):
    units=[]
    for identity in sorted({r['identity'] for r in rows}):
        deltas=[]
        for s in ([seed] if seed is not None else [17,29,43]):
            match={r['arm']:r for r in rows if r['identity']==identity and r['seed']==s and r['condition']==condition}
            a=match.get(arm,{}).get('metrics',{}).get(key);b=match.get('baseline',{}).get('metrics',{}).get(key)
            if a is not None and b is not None:deltas.append(a-b)
        if deltas:units.append(dict(identity=identity,paired_seeds=len(deltas),delta=float(np.mean(deltas))))
    d=np.array([r['delta'] for r in units]);n=len(d)
    result=dict(units=units,n_identities=n,mean_delta=float(d.mean()) if n else None,
        median_delta=float(np.median(d)) if n else None,sd_delta=float(d.std(ddof=1)) if n>1 else None,
        improved=int(np.sum(d>1e-6 if key in HIGH else d< -1e-6)),worsened=int(np.sum(d< -1e-6 if key in HIGH else d>1e-6)))
    if n:
        rng=np.random.default_rng(20261002);boot=d[rng.integers(0,n,(2000,n))].mean(1)
        result['identity_bootstrap_95_ci']=np.quantile(boot,[.025,.975]).tolist()
    return result

def main():
    dest=OUT/'scores.json';assert not dest.exists()
    cases={r['identity']:r for r in json.loads((ROOT/'outputs/reference_intervention_pilot_v1/manifest.json').read_text())['cases']}
    m=Metrics();cache={};rows=[];seen=set();deadline=time.monotonic()+2700
    while len(rows)<96 and time.monotonic()<deadline:
        path=OUT/'evaluation/rows.json'
        try:generated=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(3);continue
        for original in generated:
            key=(original['identity'],original['arm'],original['seed'],original['condition'])
            if key in seen:continue
            r=dict(original);identity=r['identity']
            if r['status']=='complete':
                try:
                    if identity not in cache:
                        base=ROOT/'outputs/reference_intervention_pilot_v1/images'/identity
                        target=read_rgb(base/'target.png');observed=read_rgb(base/'input.png');tv,td=m.features(target)
                        parts,receipts=components(cases[identity]['target']['hq_id'],target.shape[:2])
                        cache[identity]=(target,observed,tv,td,parts,receipts)
                    target,observed,tv,td,parts,receipts=cache[identity]
                    rgb=read_rgb(OUT/'evaluation'/r['file'],r['sha256']);vec,det=m.features(rgb)
                    region=score_regions(m,rgb,target,observed,np.ones(target.shape[:2],bool),parts)
                    values=m.quality(rgb)
                    values.update({k:region['whole'][k] for k in ['psnr','ssim','mae']})
                    for k,v in vec.items():values[k+'_cosine']=float(np.dot(v,tv[k])) if v is not None and tv[k] is not None else None
                    with torch.inference_mode():values['lpips']=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(target)*2-1))
                    values['eye_bbox_lpips']=region['eyes']['bbox_lpips']
                    r.update(metrics=values,regions=region,output_detections=det,target_detections=td,component_sources=receipts)
                except Exception as error:r['evaluation_error']=repr(error)
            rows.append(r);seen.add(key)
            (OUT/'scores.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            print('SCORED',len(rows),*key,'ok' if 'metrics' in r else 'failure',flush=True)
        if len(rows)<96:time.sleep(3)
    if len(rows)!=96:raise RuntimeError(f'Incomplete experiment: {len(rows)}/96')
    contrasts={};summary={};gates={}
    for arm in ['baseline','regional_v1','diffusion','identity']:
        for condition in ['clean','wrong_patch']:
            for seed in [None,17,29,43]:
                name=f'{arm}/{condition}/{seed if seed is not None else "all"}'
                subset=[r for r in rows if r['arm']==arm and r['condition']==condition and (seed is None or r['seed']==seed)]
                summary[name]={k:dict(n=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,sd=float(np.std(v,ddof=1)) if len(v)>1 else None) for k in KEYS for v in [[r.get('metrics',{}).get(k) for r in subset if r.get('metrics',{}).get(k) is not None]]}
                if arm!='baseline':contrasts[name]={k:paired(rows,arm,condition,seed,k) for k in KEYS}
        if arm=='baseline':continue
        violations=[];missing=[]
        for condition in ['clean','wrong_patch']:
            for seed in [17,29,43]:
                cs=contrasts[f'{arm}/{condition}/{seed}']
                for k in ['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']:
                    x=cs[k]
                    if x['n_identities']!=4:missing.append(f'{condition}/{seed}/{k}')
                    elif (x['mean_delta']< -1e-6 if k in HIGH else x['mean_delta']>1e-6):violations.append(f'{condition}/{seed}/{k}')
        gains=all(contrasts[f'{arm}/{c}/all']['lpips']['mean_delta']< -1e-6 for c in ['clean','wrong_patch'])
        gates[arm]=dict(promote=not violations and not missing and gains,violations=violations,incomplete_support=missing,lpips_improves_both_conditions=gains)
    result=dict(scope='Four observed identities, three fixed inference seeds, two conditions; small screen only',rows=rows,
        summary=summary,contrasts=contrasts,gates=gates,
        limitations=['FaceNet supervises identity arm and is therefore not an independent metric','ArcFace remains separate from training loss',
            'Aligned full-image recognition training crop differs from evaluation MTCNN crop','No final test or new Human Faces data',
            'One training seed; three inference seeds do not establish training-seed robustness','Other requested corruptions and ablations remain future work'],
        config_sha256=sha(OUT/'config.json'),script_sha256=sha(__file__),metrics_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'))
    dest.write_text(json.dumps(result,indent=2,allow_nan=False))
    (ROOT/'research/identity_compatibility_v2_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('SCORING COMPLETE',json.dumps(gates),flush=True)

if __name__=='__main__':main()
