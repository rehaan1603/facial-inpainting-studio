"""Fetch official code and checkpoints with bounded network timeouts."""
import hashlib
import json
import subprocess
import shutil
import base64
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/local.json').read_text())
cache=Path(cfg['cache']); repo=cache/'ResShift'; folder=cache/'models';folder.mkdir(parents=True,exist_ok=True)
commit='bb03b7d21614cace01787e097c8a6ab6b945227d'
if not repo.exists():
    subprocess.run(['git','clone','--depth','1','--branch','journal','https://github.com/zsyOAOA/ResShift.git',str(repo)],check=True)
actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if actual!=commit:
    raise RuntimeError(f'Source revision differs ({actual}); review before updating the pinned baseline.')
records=[]
for name in ['resshift_inpainting_face_s4.pth','celeba256_vq_f4_dim3_face.pth']:
    path=folder/name; url='https://github.com/zsyOAOA/ResShift/releases/download/v2.0/'+name
    if not path.exists():
        temp=path.with_suffix('.curl-part')
        previous=path.with_suffix('.download')
        if previous.exists() and (not temp.exists() or previous.stat().st_size>temp.stat().st_size):
            shutil.copyfile(previous,temp)
        print('Downloading',name,flush=True)
        subprocess.run(['curl.exe','--http1.1','--fail','--location','--silent','--show-error','--connect-timeout','20',
            '--max-time','600','--speed-time','60','--speed-limit','1024','--retry','2',
            '--continue-at','-','--output',str(temp),url],check=True)
        temp.replace(path)
    raw=path.read_bytes()
    if name=='resshift_inpainting_face_s4.pth':
        assert base64.b64encode(hashlib.md5(raw).digest()).decode()=='pAGMjmr798m04mKyyguZng==', 'Release asset Content-MD5 mismatch'
    records.append({'filename':name,'url':url,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    print(name,len(raw),flush=True)
(ROOT/'research/resshift_downloads.json').write_text(json.dumps({'commit':commit,'files':records},indent=2))
