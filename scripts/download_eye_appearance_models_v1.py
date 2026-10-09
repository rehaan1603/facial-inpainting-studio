"""Pinned official appearance assets; stream and verify, preserve existing files."""
import hashlib, json, urllib.request, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
REV='018e402774aeeddd60609b4ecdb7e298259dc729'
FILES=[('models/image_encoder/config.json',560,None),
       ('models/image_encoder/model.safetensors',2528373448,'6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030'),
       ('sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors',847517512,'677ad8860204f7d0bfba12d29e6c31ded9beefdf3e4bbd102518357d31a292c1')]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def segmented(url,path,size):
    """Verified byte ranges, bounded memory; preserve a prior contiguous partial."""
    previous=path.with_suffix(path.suffix+'.download')
    offset=previous.stat().st_size if previous.exists() else 0
    assert offset<=size
    folder=path.parent/(path.name+'.ranges');folder.mkdir(exist_ok=True)
    # Shorter ranges avoid losing long requests to a slow or stalled connection.
    # Reuse every fully received larger range from an earlier attempt.
    step=16*1024*1024
    cached={}
    for part in folder.glob('*.part'):
        start,end=map(int,part.stem.split('-'))
        if offset<=start<=end<size and part.stat().st_size==end-start+1:cached[start]=end
    ranges=[];cursor=offset
    while cursor<size:
        if cursor in cached:end=cached[cursor]
        else:
            next_cached=min((start for start in cached if start>cursor),default=size)
            end=min(cursor+step,next_cached,size)-1
        ranges.append((cursor,end));cursor=end+1
    def fetch(pair):
        start,end=pair; chunk=folder/f'{start}-{end}.part'; expected=end-start+1
        if chunk.exists() and chunk.stat().st_size==expected:return chunk
        for attempt in range(3):
            try:
                request=urllib.request.Request(url+f'?download=true&range_start={start}',headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}', 'Server did not honor byte range'
                    with chunk.open('wb') as f:shutil.copyfileobj(response,f,4*1024*1024)
                assert chunk.stat().st_size==expected
                print('RANGE',path.name,start,end,flush=True);return chunk
            except Exception as exc:
                print('RETRY',path.name,start,attempt+1,type(exc).__name__,flush=True)
                if attempt==2:raise
                time.sleep(2*(attempt+1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,pair) for pair in ranges]
        for future in as_completed(futures):future.result()
    assembled=path.with_suffix(path.suffix+'.parallel.download')
    with assembled.open('wb') as f:
        if offset:
            with previous.open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
        for start,end in ranges:
            with (folder/f'{start}-{end}.part').open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
    assert assembled.stat().st_size==size
    return assembled

def main():
    dest=CACHE/'eye_appearance_models_v1'; dest.mkdir(exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/h94/IP-Adapter/revision/{REV}?blobs=true',timeout=30) as r:meta=json.load(r)
    assert meta['sha']==REV
    siblings={r['rfilename']:r for r in meta['siblings']}; receipts=[]
    for name,size,digest in FILES:
        row=siblings[name]; assert row['size']==size
        if digest:assert row['lfs']['sha256']==digest
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            url=f'https://huggingface.co/h94/IP-Adapter/resolve/{REV}/{name}'
            if size>64*1024*1024:
                temp=segmented(url,path,size)
                assert sha(temp)==digest
                temp.rename(path)
                print('ASSEMBLED',name,flush=True)
                actual=sha(path)
                receipts.append(dict(file=name,path=str(path),bytes=size,sha256=actual,git_blob=row['blobId']))
                continue
            temp=path.with_suffix(path.suffix+'.download'); offset=temp.stat().st_size if temp.exists() else 0
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=60) as response:
                resumed=offset>0 and response.status==206
                with temp.open('ab' if resumed else 'wb') as f:
                    downloaded=offset if resumed else 0; checkpoint=downloaded//(256*1024*1024)
                    for chunk in iter(lambda:response.read(4*1024*1024),b''):
                        f.write(chunk);downloaded+=len(chunk)
                        if downloaded//(256*1024*1024)>checkpoint:
                            checkpoint=downloaded//(256*1024*1024);print(name,downloaded,'/',size,flush=True)
            assert temp.stat().st_size==size
            actual=sha(temp)
            assert digest is None or actual==digest
            temp.rename(path)
        assert path.stat().st_size==size
        actual=sha(path);assert digest is None or actual==digest
        if digest is None:
            # Verify small config against Git's pinned blob identifier as well.
            content=path.read_bytes(); blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            assert blob==row['blobId']
        receipts.append(dict(file=name,path=str(path),bytes=size,sha256=actual,git_blob=row['blobId']))
        print('VERIFIED',name,flush=True)
    (dest/'receipt.json').write_text(json.dumps(dict(repository='h94/IP-Adapter',revision=REV,files=receipts),indent=2))

if __name__=='__main__':main()
