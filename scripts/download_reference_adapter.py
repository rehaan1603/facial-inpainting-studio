"""Download a pinned author multi-reference FaceID Portrait adapter and verify SHA256."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    metadata=json.loads((ROOT/'research/faceid_remote_files.json').read_text());name='ip-adapter-faceid-portrait_sdxl.bin'
    row=next(r for r in metadata['siblings'] if r['rfilename']==name);cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);dest=cache/'reference_models'/name;dest.parent.mkdir(exist_ok=True)
    expected=row['lfs']['sha256'];temp=dest.with_suffix('.download');url=f"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/{metadata['sha']}/{name}"
    if not dest.exists():
        start=temp.stat().st_size if temp.exists() else 0
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {}),timeout=120) as response:
            with temp.open('ab' if start and response.status==206 else 'wb') as f:
                while chunk:=response.read(4*1024*1024):f.write(chunk)
        assert temp.stat().st_size==row['size'] and hashlib.sha256(temp.read_bytes()).hexdigest()==expected
        temp.replace(dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected
    (ROOT/'research/reference_adapter_provenance.json').write_text(json.dumps({'repository':'h94/IP-Adapter-FaceID','revision':metadata['sha'],'file':name,'bytes':dest.stat().st_size,'sha256':expected,'local_path':str(dest),'terms':'Author restricts FaceID models to noncommercial research; do not redistribute weights in the source archive.','method':'Existing multi-reference FaceID Portrait adapter, not a novel project contribution.'},indent=2));print('Reference adapter downloaded and verified',flush=True)
if __name__=='__main__':main()
