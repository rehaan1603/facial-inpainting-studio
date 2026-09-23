"""Real local HTTP confidence reconstruction check on an observed development case."""
import base64,json,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/confidence_web_check_v2';out.mkdir(exist_ok=False)
    url='http://127.0.0.1:8771'
    def get(path):
        with urllib.request.urlopen(url+path,timeout=10) as response:return json.load(response)
    with (out/'server.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'webapp/server.py'),'--port','8771'],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(30):
                try:session=get('/api/session');break
                except OSError:time.sleep(1)
            else:raise RuntimeError('Test server did not start')
            assert not session['busy']
            c=next(c for c in json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases'] if c['case_id']=='1306_mixed')
            enc=lambda p:'data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
            data={'image':enc(c['observed']),'mask':enc(c['mask']),'confidence':enc(ROOT/'outputs/preservation_cli_check_v1/confidence.png'),'references':[enc(r['path']) for r in c['references']],'backbone':'reference','mode':'painted','detail':'standard','blend':'poisson','strength':.5}
            def post(payload):
                req=urllib.request.Request(url+'/api/inpaint',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Local-Token':session['token'],'Origin':url})
                with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
            for changes in [{'strength':True},{'strength':.2},{'mode':'expand'},{'backbone':'lama'}]:
                try:post(dict(data,**changes))
                except urllib.error.HTTPError as e:assert e.code==400
                else:raise AssertionError('Invalid confidence request accepted')
            job=post(data);print('JOB',job['id'],flush=True)
            for _ in range(600):
                result=get('/api/jobs/'+job['id'])
                if result['status'] in ['complete','error']:break
                time.sleep(1)
            (out/'job.json').write_text(json.dumps(result,indent=2))
            assert result['status']=='complete',result
            with urllib.request.urlopen(url+result['result'],timeout=10) as r:assert r.status==200
            import numpy as np
            from PIL import Image
            sys.path.insert(0,str(ROOT))
            from src.preservation.confidence import preserve_observation
            from src.research_integrity import sha
            folder=ROOT/'outputs/webapp_runs'/job['id'];read=lambda n:np.asarray(Image.open(folder/n).convert('RGB'))
            a=read('input.png');g=read('result.png');r=read('preserved.png');mask=read('effective_mask.png')[:,:,0]>=128;confidence=read('confidence.png')[:,:,0]/255
            assert np.array_equal(r,preserve_observation(a,g,mask,confidence)) and np.array_equal(r[~mask],a[~mask])
            meta=json.loads((folder/'metadata.json').read_text());assert meta['result_sha256']==sha(folder/'preserved.png')
            receipt={'date':'2026-09-21','passed':True,'job_id':job['id'],'real_gpu_generation':True,'image_http_200':True,'pixel_formula_exact':True,'known_pixels_unchanged':True,'invalid_requests_rejected':4,'result_sha256':meta['result_sha256'],'final_test_used':False,'scope':'Real HTTP integration; browser-click workflow requires separate verification'}
            (ROOT/'research/confidence_web_integration_v2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
        finally:process.terminate();process.wait(timeout=20)
if __name__=='__main__':main()
