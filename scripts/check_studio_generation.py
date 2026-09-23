"""Repeatable local HTTP checks using explicitly supplied previous upload jobs.

Photos remain in ignored outputs. This is a functional check, not a benchmark.
"""
import argparse
import base64
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8765'


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def get(path):
        with urlopen(BASE+path,timeout=20) as r:return json.load(r)
    def png(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    records=[];tiles=[]
    for source_id in a.jobs:
        folder=ROOT/'outputs/webapp_runs'/source_id
        for model,detail in [('lama','standard'),('resshift','standard'),('reference','standard'),('reference','detailed'),('reference_select','standard')]:
            session=get('/api/session')
            if session['busy']:raise RuntimeError('GPU already in use; do not interrupt it.')
            payload=dict(image=png(folder/'input.png'),mask=png(folder/'mask.png'),backbone=model,detail=detail,mode='expand',blend='poisson')
            if model.startswith('reference'):payload['references']=[png(f) for f in sorted(folder.glob('reference_*.png'))]
            req=Request(BASE+'/api/inpaint',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Local-Token':session['token']})
            with urlopen(req,timeout=20) as r:job=json.load(r)
            start=time.monotonic()
            while True:
                result=get('/api/jobs/'+job['id'])
                if result['status'] in ['complete','error']:break
                if time.monotonic()-start>1800:raise TimeoutError(job['id'])
                time.sleep(1)
            record=dict(source_job=source_id,model=model,detail=detail,job_id=job['id'],status=result['status'])
            if result['status']=='complete':
                out=ROOT/'outputs/webapp_runs'/job['id']
                rgb=np.array(Image.open(out/'result.png'));observed=np.array(Image.open(out/'input.png'));mask=np.array(Image.open(out/'effective_mask.png'))>=128
                record.update(outside_exact=bool(np.array_equal(rgb[~mask],observed[~mask])),changed_inside=int(np.any(rgb!=observed,axis=2)[mask].sum()),size=list(rgb.shape[:2]))
                tile=Image.new('RGB',(768,280),'white');draw=ImageDraw.Draw(tile);draw.text((5,5),model+' '+detail+' '+source_id[:8],fill='black')
                for j,name in enumerate(['input.png','effective_mask.png','result.png']):tile.paste(Image.open(out/name).convert('RGB').resize((256,256)),(j*256,24))
                tiles.append(tile)
            else:record['error']=result['message']
            records.append(record);(a.output/'checks.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
    sheet=Image.new('RGB',(768,280*len(tiles)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,(0,280*i))
    sheet.save(a.output/'review.png')


if __name__=='__main__':main()
