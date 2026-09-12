"""Fetch the pinned author OSOR phase-II and SDXL fp16 weights; verify HF LFS hashes."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);tasks=[]
    for tag,repo in [('osor','QinmingZhou/OSOR'),('sdxl','diffusers/stable-diffusion-xl-1.0-inpainting-0.1')]:
        metadata=json.loads((ROOT/f'research/{tag}_remote_files.json').read_text())
        for row in metadata['siblings']:
            name=row['rfilename']
            wanted=(name=='osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth') if tag=='osor' else (name.endswith(('.json','.txt')) or name.endswith('fp16.safetensors'))
            if wanted:tasks.append((tag,repo,metadata['sha'],row))
    def fetch(task):
        tag,repo,revision,row=task;name=row['rfilename'];path=cache/'osor_models'/tag/name;path.parent.mkdir(parents=True,exist_ok=True);expected=row.get('lfs',{}).get('sha256')
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}';tmp=path.with_suffix(path.suffix+'.download')
        if path.exists():
            assert path.stat().st_size==row['size'] and (not expected or digest(path)==expected)
        else:
            for attempt in range(4):
                try:
                    start=tmp.stat().st_size if tmp.exists() else 0
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
                    with urllib.request.urlopen(req,timeout=120) as response:
                        mode='ab' if start and response.status==206 else 'wb'
                        with tmp.open(mode) as f:
                            while blob:=response.read(4*1024*1024):f.write(blob)
                    assert tmp.stat().st_size==row['size'],(name,tmp.stat().st_size,row['size'])
                    assert not expected or digest(tmp)==expected,name
                    tmp.replace(path);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2)
        print('Verified',tag,name,flush=True)
        return {'repository':repo,'revision':revision,'filename':name,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'published_lfs_sha256':expected}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,tasks))
    (ROOT/'research/osor_downloads.json').write_text(json.dumps(records,indent=2));print('All OSOR inference assets verified')
if __name__=='__main__':main()
