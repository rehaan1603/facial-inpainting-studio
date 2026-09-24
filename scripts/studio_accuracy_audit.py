"""Development-only quality audit of every studio generation mode.

Generation never opens target/gallery images. Evaluation uses frozen metrics.
Run generation and evaluation separately; evaluation can follow saved job rows.
"""
import argparse,base64,io,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from distortion_aware_study import BASE,RESERVED
OUT=ROOT/'outputs/studio_accuracy_v1'
PROTOCOL=ROOT/'research/protocols/studio_accuracy_v1.json'
MODES=[('lama','lama','standard',None),('resshift','resshift','standard',None),
       ('reference_512','reference','standard',None),('reference_1024','reference','detailed',None),
       ('selection_512','reference_select','standard',None),('selection_1024','reference_select','detailed',None)]
PARTIAL=[('evidence_low','reference','standard',.5),('evidence_high','reference','standard',.99),('refldm_partial','refldm','standard',.99)]


def cases():
    selected=[c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases'] if c['kind'] in ['removal','mixed']]
    assert len(selected)==8 and {c['identity'] for c in selected}=={'1306','2790','1043','787'}
    assert not {c['identity'] for c in selected}&RESERVED
    return selected


def generate():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads(PROTOCOL.read_text())
    assert sha(__file__)==protocol['runner_sha256']
    def request(path,data=None,token=None):
        req=Request('http://127.0.0.1:8765'+path,data=json.dumps(data).encode() if data is not None else None,
                    headers={'Content-Type':'application/json',**({'X-Local-Token':token} if token else {})})
        with urlopen(req,timeout=30) as r:return json.load(r)
    def png(path,expected):
        assert sha(path)==expected
        return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
    for c in cases():
        controls=dict(case_id=c['case_id'],identity=c['identity'],kind=c['kind'],arm='observed',mode='composited',status='complete',output=c['observed'],output_sha256=c['observed_sha256'])
        controlpath=OUT/'rows'/(c['case_id']+'__observed.json')
        if not controlpath.exists():write_new(controlpath,dict(controls,key=controlpath.stem))
        for arm,backbone,detail,strength in MODES if c['kind']=='removal' else PARTIAL:
            key=c['case_id']+'__'+arm;dest=OUT/'rows'/(key+'.json')
            if dest.exists():continue
            row=dict(key=key,case_id=c['case_id'],identity=c['identity'],kind=c['kind'],arm=arm,mode='composited',status='failed')
            session=request('/api/session')
            if session['busy']:raise RuntimeError('GPU occupied; rerun after it is free. No job interrupted.')
            try:
                payload=dict(image=png(c['observed'],c['observed_sha256']),mask=png(c['mask'],c['mask_sha256']),backbone=backbone,detail=detail,mode='painted',blend='poisson')
                if backbone not in ['lama','resshift']:payload['references']=[png(r['path'],r['sha256']) for r in c['references']]
                if strength is not None:
                    mask=np.array(Image.open(c['mask']).convert('L'))>=128
                    weights=np.where(mask,128,255).astype('uint8');stream=io.BytesIO();Image.fromarray(weights).save(stream,format='PNG')
                    payload.update(confidence='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode(),strength=strength)
                job=request('/api/inpaint',payload,session['token']);row['job_id']=job['id'];start=time.monotonic()
                while True:
                    result=request('/api/jobs/'+job['id'])
                    if result['status'] in ['complete','error']:break
                    if time.monotonic()-start>1900:raise TimeoutError(job['id'])
                    time.sleep(1)
                if result['status']!='complete':raise RuntimeError(result['message'])
                output=ROOT/'outputs/webapp_runs'/job['id']/Path(result['result']).name
                row.update(status='complete',output=str(output),output_sha256=sha(output),seconds=result['seconds'])
            except Exception as error:row['error']=str(error)
            write_new(dest,row);print(key,row['status'],flush=True)
    write_new(OUT/'generation_complete.json',{'protocol_sha256':sha(PROTOCOL),'expected_rows':44})


def evaluate():
    from extended_evaluation_metrics import Metrics,read_rgb
    metrics=Metrics();lookup={c['case_id']:c for c in cases()}
    targets={c['case_id']:c for c in json.loads((BASE/'evaluation_manifest.json').read_text())['cases'] if c['case_id'] in lookup}
    feature_cache={};deadline=time.monotonic()+10800
    while True:
        for file in sorted((OUT/'rows').glob('*.json')):
            dest=OUT/'scores'/file.name
            if dest.exists():continue
            row=json.loads(file.read_text());assert row['identity'] not in RESERVED
            score={'status':'failed'}
            try:
                if row['status']!='complete':raise ValueError(row.get('error'))
                c=lookup[row['case_id']];e=targets[row['case_id']]
                if row['identity'] not in feature_cache:
                    target=read_rgb(e['target']['path'],e['target']['sha256']);tv,td=metrics.features(target)
                    galleries=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in e['gallery']]
                    feature_cache[row['identity']]=(target,tv,td,galleries)
                target,tv,td,galleries=feature_cache[row['identity']]
                rgb=read_rgb(row['output'],row['output_sha256']);observed=read_rgb(c['observed'],c['observed_sha256']);mask=read_rgb(c['mask'],c['mask_sha256'])[:,:,0]>=128
                values=metrics.score(rgb,target,observed,mask,tv);ov,od=metrics.features(rgb)
                values['known_pixels_unchanged']=bool(np.array_equal(rgb[~mask],observed[~mask]))
                for name in ['facenet','arcface_conditioning']:
                    sims=[float(np.dot(ov[name],v[name])) for v,s in galleries if ov[name] is not None and v[name] is not None]
                    values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None
                    values[name+'_gallery_valid_count']=len(sims)
                score=dict(status='complete',metrics=values,output_detections=od,target_detections=td)
            except Exception as error:score['error']=str(error)
            write_new(dest,dict(row,evaluation=score));print('SCORED',file.stem,score['status'],flush=True)
        if (OUT/'generation_complete.json').exists():break
        if time.monotonic()>deadline:raise TimeoutError('Generation not complete; saved scores remain resumable.')
        time.sleep(3)
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'scores').glob('*.json'))]
    write_new(OUT/'evaluation.json',dict(rows=rows,protocol_sha256=sha(PROTOCOL),metric_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),final_test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','evaluate']);a=p.parse_args()
    (generate if a.stage=='generate' else evaluate)()
