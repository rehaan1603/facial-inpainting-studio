"""Check studio mask-union correction without altering historical refiner studies."""
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from verify_studio_fixes_v2 import request,png
from unfamiliar_studio_audit_v2 import cases,BASE


def main():
    output=ROOT/'outputs/studio_refiner_union_v2';output.mkdir(exist_ok=False)
    case=next(c for c in cases() if c['case_id']=='1037_removal')
    assert sha(case['observed'])==case['observed_sha256'] and sha(case['mask'])==case['mask_sha256']
    rows=[]
    for model in ['lama','resshift']:
        deadline=time.monotonic()+600
        while request('/api/session')['busy']:
            if time.monotonic()>deadline:raise TimeoutError('GPU remained occupied; no job interrupted.')
            time.sleep(2)
        session=request('/api/session')
        data=dict(image=png(Image.open(case['observed'])),mask=png(Image.open(case['mask'])),backbone=model,mode='learned',detail='standard')
        row=dict(case_id=case['case_id'],backbone=model,status='failed')
        try:
            job=request('/api/inpaint',data,session['token']);row['job_id']=job['id']
            while True:
                state=request('/api/jobs/'+job['id'])
                if state['status'] in ['complete','error']:break
                if time.monotonic()>deadline:raise TimeoutError(job['id'])
                time.sleep(1)
            if state['status']!='complete':raise ValueError(state['message'])
            folder=ROOT/'outputs/webapp_runs'/job['id']
            supplied=np.asarray(Image.open(folder/'mask.png'))>=128
            effective=np.asarray(Image.open(folder/'effective_mask.png'))>=128
            a=np.asarray(Image.open(folder/'result.png').convert('RGB'));b=np.asarray(Image.open(folder/'input.png').convert('RGB'))
            row.update(status='complete',output_sha256=sha(folder/'result.png'),requested_pixels=int(supplied.sum()),dropped_requested_pixels=int((supplied&~effective).sum()),known_pixels_unchanged=bool(np.array_equal(a[~effective],b[~effective])))
            assert row['dropped_requested_pixels']==0 and row['known_pixels_unchanged']
        except Exception as error:row.update(status='failed',error=str(error))
        rows.append(row);write_new(output/(model+'.json'),row);print(model,row['status'],flush=True)
    from extended_evaluation_metrics import Metrics,read_rgb
    metrics=Metrics();e=next(c for c in json.loads((BASE/'evaluation_manifest.json').read_text())['cases'] if c['case_id']==case['case_id'])
    target=read_rgb(e['target']['path'],e['target']['sha256']);tv,td=metrics.features(target)
    original=np.asarray(Image.open(case['mask']))>=128
    before=json.loads((ROOT/'research/studio_fixes_quality_v2_mask_corrected.json').read_text())['rows']
    for row in rows:
        if row['status']!='complete':continue
        folder=ROOT/'outputs/webapp_runs'/row['job_id'];rgb=read_rgb(folder/'result.png',row['output_sha256']);observed=read_rgb(case['observed'],case['observed_sha256']);effective=np.asarray(Image.open(folder/'effective_mask.png'))>=128
        values=metrics.score(rgb,target,observed,effective,tv)
        values['original_damage_mae']=float(np.abs(rgb.astype(float)-target.astype(float))[original].mean()/255)
        prior=next(r for r in before if r['check']==row['backbone']+'_learned')['evaluation']['metrics']
        row.update(metrics=values,delta_vs_prior_learned={k:values[k]-prior[k] for k in ['facenet_cosine','original_damage_mae','lpips']})
    write_new(ROOT/'research/studio_refiner_union_verification_v2.json',dict(scope='Two functional runs on one already-observed development identity. Union ensures painted damage cannot be dropped; not a novel method or general quality guarantee.',server_sha256=sha(ROOT/'webapp/server.py'),rows=rows,final_test_used=False))


if __name__=='__main__':main()
