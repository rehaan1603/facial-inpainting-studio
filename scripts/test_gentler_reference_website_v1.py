"""Four real local API GPU jobs: preserved default, then the three CFG3 replays."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
from test_focused_reference_website_v1 import ROOT,request as old_request,sha
import base64
from urllib.request import Request,urlopen

OUT=ROOT/'outputs/gentler_reference_website_v1'

def request(path,data=None,token=None):
    req=Request('http://127.0.0.1:8766'+path,data=json.dumps(data).encode() if data is not None else None,
                headers={'Content-Type':'application/json',**({'X-Local-Token':token} if token else {})})
    with urlopen(req,timeout=30) as response:return json.load(response)

def main():
    OUT.mkdir(exist_ok=False)
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    cfg=json.loads((ROOT/'outputs/focused_eye_guidance_v1/config.json').read_text());rows=[]
    for ident,gentler in [('263',False),('263',True),('6681',True),('10149',True)]:
        assert not old_request('/api/session')['busy'],'Original studio user job active; stop submissions'
        session=request('/api/session');assert not session['busy'],'User GPU job active; stop without interrupting'
        assert session['gentler_details_available']
        c=cases[ident]
        def data(p):return 'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        refs=cfg['references'][ident+'/clean']
        for r in refs:assert sha(r['path'])==r['sha256']
        payload=dict(image=data(c['input']),mask=data(c['mask']),references=[data(r['path']) for r in refs],
                     backbone='reference',mode='painted',detail='detailed',blend='poisson',neutralize=True,gentler_details=gentler)
        job=request('/api/inpaint',payload,session['token']);start=time.monotonic()
        while True:
            result=request('/api/jobs/'+job['id'])
            if result['status'] in ['complete','error']:break
            if time.monotonic()-start>1200:raise TimeoutError('GPU job active; stop submissions')
            time.sleep(2)
        row=dict(identity=ident,gentler_details=gentler,job_id=job['id'],status=result['status'],seconds=time.monotonic()-start)
        if result['status']=='complete':
            folder=ROOT/'outputs/webapp_runs'/job['id'];p=folder/'result.png';saved=OUT/f'{ident}_{"gentler" if gentler else "control"}.png';saved.write_bytes(p.read_bytes())
            rgb=np.asarray(Image.open(saved).convert('RGB'));observed=np.asarray(Image.open(c['input']).convert('RGB'));mask=np.asarray(Image.open(c['mask']))>127
            assert np.array_equal(rgb[~mask],observed[~mask])
            metadata=json.loads((folder/'metadata.json').read_text());assert metadata['guidance_scale']==(3. if gentler else 5.) and metadata['gentler_details']==gentler
            expected=ROOT/'outputs'/('focused_eye_guidance_v1' if gentler else 'focused_reference_finish_v1')/f'{ident}_eyes_clean_17.png'
            raw_expected=ROOT/'outputs'/('focused_eye_guidance_v1' if gentler else 'masked_sdxl_resolution_v1')/f'{ident}_eyes_clean_17_raw.png'
            row.update(output=str(saved),sha256=sha(saved),visible_exact=True,finished_exact=sha(saved)==sha(expected),
                       raw_exact=sha(folder/'result_raw.png')==sha(raw_expected),metadata_sha256=sha(folder/'metadata.json'))
        else:row['error']=result['message']
        rows.append(row);(OUT/'receipt.json').write_text(json.dumps(rows,indent=2));print(ident,gentler,row['status'],row.get('finished_exact'),row.get('raw_exact'),flush=True)
        assert row['status']=='complete' and row['finished_exact'] and row['raw_exact'],'Website differs from matched research result'

if __name__=='__main__':main()
