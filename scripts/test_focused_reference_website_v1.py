"""Exercise the actual local studio for the three ready-to-test eye cases."""
import base64,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/focused_reference_website_v1'


def request(path,data=None,token=None):
    req=Request('http://127.0.0.1:8765'+path,data=json.dumps(data).encode() if data is not None else None,
                headers={'Content-Type':'application/json',**({'X-Local-Token':token} if token else {})})
    with urlopen(req,timeout=30) as response:return json.load(response)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    refs=json.loads((ROOT/'outputs/masked_sdxl_v1/config.json').read_text())['references']
    OUT.mkdir(exist_ok=False);rows=[]
    for ident in ['263','6681','10149']:
        session=request('/api/session');assert not session['busy'],'Studio is busy; do not interrupt user job'
        c=cases[ident]
        def data(path):return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        rr=refs[ident+'/clean']
        for r in rr:assert sha(r['path'])==r['sha256']
        payload=dict(image=data(c['input']),mask=data(c['mask']),references=[data(r['path']) for r in rr],
                     backbone='reference',mode='painted',detail='detailed',blend='poisson',neutralize=True)
        job=request('/api/inpaint',payload,session['token']);start=time.monotonic()
        while True:
            result=request('/api/jobs/'+job['id'])
            if result['status'] in ['complete','error']:break
            if time.monotonic()-start>1200:raise TimeoutError('Website job remains active; no further submissions')
            time.sleep(2)
        row=dict(identity=ident,job_id=job['id'],status=result['status'],seconds=time.monotonic()-start)
        if result['status']=='complete':
            source=ROOT/'outputs/webapp_runs'/job['id']/Path(result['result']).name
            saved=OUT/(ident+'_eyes.png');saved.write_bytes(source.read_bytes())
            rgb=np.asarray(Image.open(saved).convert('RGB'));observed=np.asarray(Image.open(c['input']).convert('RGB'))
            mask=np.asarray(Image.open(c['mask']).convert('L'))>127
            assert np.array_equal(rgb[~mask],observed[~mask])
            expected=ROOT/'outputs/focused_reference_finish_v1'/f'{ident}_eyes_clean_17.png'
            row.update(output=str(saved),sha256=sha(saved),visible_pixels_exact=True,experiment_replay_exact=sha(saved)==sha(expected),result_url=result['result'])
        else:row['error']=result['message']
        rows.append(row);(OUT/'receipt.json').write_text(json.dumps(rows,indent=2));print(ident,row['status'],row.get('experiment_replay_exact'),flush=True)
        if result['status']!='complete':raise RuntimeError(result['message'])


if __name__=='__main__':main()
