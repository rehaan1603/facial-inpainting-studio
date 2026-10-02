"""Read-only reproduction of v1 clean scores and every missing identity measurement."""
import json
import numpy as np
import torch
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha
from score_target_compatibility_v1 import components
from region_metrics_v1 import score_regions

def main():
    source=ROOT/'outputs/target_compatibility_v1'
    base=ROOT/'outputs/reference_intervention_pilot_v1'
    dest=ROOT/'outputs/compatibility_audit_v2';dest.mkdir(exist_ok=False)
    saved=json.loads((source/'scores.json').read_text())
    training=json.loads((source/'training/receipt.json').read_text())
    for r in saved['rows']:assert sha(source/'evaluation'/r['file'])==r['sha256']
    for r in training['arms']:assert sha(source/'training'/(r['arm']+'.pt'))==r['sha256']
    cases={r['identity']:r for r in json.loads((base/'manifest.json').read_text())['cases']}
    selected=[r for r in saved['rows'] if r['condition']=='clean' or r['metrics']['facenet_cosine'] is None]
    m=Metrics();cache={};rows=[]
    for r in selected:
        ident=r['identity']
        if ident not in cache:
            target=read_rgb(base/'images'/ident/'target.png');observed=read_rgb(base/'images'/ident/'input.png')
            vec,det=m.features(target);parts,_=components(cases[ident]['target']['hq_id'],target.shape[:2])
            cache[ident]=(target,observed,vec,parts)
        target,observed,vec,parts=cache[ident]
        rgb=read_rgb(source/'evaluation'/r['file']);pv,pd=m.features(rgb)
        region=score_regions(m,rgb,target,observed,np.ones(target.shape[:2],bool),parts)
        values={k:region['whole'][k] for k in ['psnr','ssim','mae']}
        with torch.inference_mode():values['lpips']=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(target)*2-1))
        for k,v in pv.items():values[k+'_cosine']=float(np.dot(v,vec[k])) if v is not None and vec[k] is not None else None
        differences={k:abs(v-r['metrics'][k]) if v is not None and r['metrics'][k] is not None else None for k,v in values.items()}
        same_missing=all((v is None)==(r['metrics'][k] is None) for k,v in values.items())
        eye_delta=abs(region['eyes']['bbox_lpips']-r['regions']['eyes']['bbox_lpips'])
        rows.append(dict(identity=ident,arm=r['arm'],condition=r['condition'],values=values,
            absolute_differences=differences,eye_lpips_difference=eye_delta,same_missing=same_missing,
            detections=pd,pass_tolerance=same_missing and max([eye_delta,*[v for v in differences.values() if v is not None]])<=1e-6))
        (dest/'receipt.json').write_text(json.dumps(dict(rows=rows,checked_image_hashes=96,checked_adapter_hashes=2,
            tolerance=1e-6,scope='12 clean rows plus all 3 missing FaceNet rows; no regeneration or training'),indent=2))
        print(ident,r['arm'],r['condition'],rows[-1]['pass_tolerance'],pd['facenet']['status'],flush=True)
    assert len(rows)==15 and all(r['pass_tolerance'] for r in rows)
    print('AUDIT REPRODUCTION PASS',flush=True)

if __name__=='__main__':main()
