"""Separate runtime retries and restored-default verification; never replace audit rows."""
import base64,json,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from studio_accuracy_audit import cases,BASE


def main():
    out=ROOT/'outputs/studio_accuracy_recovery_v1';out.mkdir(exist_ok=False)
    def request(path,data=None,token=None):
        req=Request('http://127.0.0.1:8765'+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json',**({'X-Local-Token':token} if token else {})})
        with urlopen(req,timeout=30) as r:return json.load(r)
    def png(p,h):
        assert sha(p)==h
        return 'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
    rows=[]
    for case,model,detail,neutralize in [('2790_removal','reference','standard',True),('2790_removal','reference_select','detailed',True),('1306_removal','reference','standard',False)]:
        c=next(c for c in cases() if c['case_id']==case);session=request('/api/session');assert not session['busy']
        payload=dict(image=png(c['observed'],c['observed_sha256']),mask=png(c['mask'],c['mask_sha256']),references=[png(r['path'],r['sha256']) for r in c['references']],backbone=model,detail=detail,mode='painted',blend='poisson',neutralize=neutralize)
        row=dict(case_id=case,model=model,detail=detail,neutralize=neutralize,status='failed')
        try:
            job=request('/api/inpaint',payload,session['token']);row['job_id']=job['id'];start=time.monotonic()
            while True:
                result=request('/api/jobs/'+job['id'])
                if result['status'] in ['complete','error']:break
                if time.monotonic()-start>1900:raise TimeoutError(job['id'])
                time.sleep(1)
            if result['status']!='complete':raise RuntimeError(result['message'])
            path=ROOT/'outputs/webapp_runs'/job['id']/'result.png';row.update(status='complete',output=str(path),output_sha256=sha(path))
        except Exception as e:row['error']=str(e)
        rows.append(row);write_new(out/(str(len(rows))+'.json'),row);print(row,flush=True)
    from extended_evaluation_metrics import Metrics,read_rgb
    metrics=Metrics();evals={e['case_id']:e for e in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    for row in rows:
        if row['status']!='complete':continue
        c=next(c for c in cases() if c['case_id']==row['case_id']);e=evals[row['case_id']]
        target=read_rgb(e['target']['path'],e['target']['sha256']);tv,td=metrics.features(target)
        rgb=read_rgb(row['output'],row['output_sha256']);observed=read_rgb(c['observed'],c['observed_sha256']);mask=read_rgb(c['mask'],c['mask_sha256'])[:,:,0]>=128
        values=metrics.score(rgb,target,observed,mask,tv);ov,od=metrics.features(rgb)
        galleries=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in e['gallery']]
        for name in ['facenet','arcface_conditioning']:
            sims=[float(np.dot(ov[name],v[name])) for v,s in galleries if ov[name] is not None and v[name] is not None]
            values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None
        row.update(metrics=values,known_pixels_unchanged=bool(np.array_equal(rgb[~mask],observed[~mask])))
        if not row['neutralize']:
            old=next(r for r in json.loads((BASE/'evaluation.json').read_text())['rows'] if r['key']==row['case_id']+'_s0.99_a0.8_seed17')
            row['matches_historical_png']=row['output_sha256']==old['output_sha256']
        row.pop('output')
    write_new(ROOT/'research/studio_accuracy_recovery_v1.json',{'scope':'Separate unchanged-setting retries after main audit completed and CPU evaluator stopped; original failures remain counted. Third row verifies restored legacy default.','rows':rows})


if __name__=='__main__':main()
