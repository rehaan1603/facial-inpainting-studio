"""Real HTTP -> GPU -> saved PNG integration checks using the research sample only."""
import argparse,base64,io,json,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8765');args=p.parse_args()
    def fetch(path,data=None,token=None):
        request=urllib.request.Request(args.url+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json','X-Local-Token':token or ''})
        with urllib.request.urlopen(request,timeout=30) as response:return response.read()
    token=json.loads(fetch('/api/session'))['token'];demo=json.loads(fetch('/api/demo'));results=[]
    for backbone,mode in [('lama','painted'),('lama','expand'),('lama','learned'),('resshift','painted'),('resshift','expand')]:
        job=json.loads(fetch('/api/inpaint',dict(demo,backbone=backbone,mode=mode),token));deadline=time.monotonic()+180
        while True:
            status=json.loads(fetch('/api/jobs/'+job['id']))
            if status['status']=='error':raise RuntimeError(status)
            if status['status']=='complete':break
            if time.monotonic()>deadline:raise TimeoutError(status)
            time.sleep(.5)
        read=lambda url:np.array(Image.open(io.BytesIO(fetch(url))))
        source=read(status['input']);result=read(status['result']);mask=read(status['mask'])>=128
        assert result.shape==(256,256,3) and mask.any()
        assert np.array_equal(result[~mask],source[~mask]),'Visible pixels changed outside effective mask'
        assert np.any(result[mask]!=source[mask]),'Masked pixels were not reconstructed'
        record={'backbone':backbone,'mode':mode,'job_id':job['id'],'changed_pixels':int((result!=source).any(2).sum()),'outside_effective_mask_exact':True,'inference_seconds':status['seconds']}
        results.append(record);print(record,flush=True)
    (ROOT/'research/webapp_inference_checks.json').write_text(json.dumps({'scope':'Five real local HTTP/GPU checks; functional evidence, not a quality benchmark. Sample images excluded from release archive.','results':results},indent=2))
if __name__=='__main__':main()
