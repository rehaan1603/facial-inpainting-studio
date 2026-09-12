"""Verify and extract the official InsightFace research model package locally."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    row=json.loads((ROOT/'research/insightface_remote_asset.json').read_text());cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    folder=cache/'reference_models';folder.mkdir(exist_ok=True);path=folder/'buffalo_l.zip';tmp=path.with_suffix('.download');expected=row['digest'].split(':',1)[1]
    if not path.exists():
        start=tmp.stat().st_size if tmp.exists() else 0
        with urllib.request.urlopen(urllib.request.Request(row['browser_download_url'],headers={'Range':f'bytes={start}-'} if start else {}),timeout=120) as response:
            with tmp.open('ab' if start and response.status==206 else 'wb') as f:
                while blob:=response.read(4*1024*1024):f.write(blob)
        assert tmp.stat().st_size==row['size'] and hashlib.sha256(tmp.read_bytes()).hexdigest()==expected;tmp.replace(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    dest=(folder/'insightface/models/buffalo_l').resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for info in z.infolist():assert (dest/info.filename).resolve().is_relative_to(dest),'Unsafe archive path'
        z.extractall(dest)
    files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    (ROOT/'research/reference_faces_provenance.json').write_text(json.dumps({'source':row['browser_download_url'],'zip_sha256':expected,'files':files,'purpose':'Local reference-face alignment and conditioning embeddings; no identification against an external gallery.','terms':'Noncommercial research model use; source image uploads stay on this laptop.'},indent=2));print('InsightFace model package verified and extracted',flush=True)
if __name__=='__main__':main()
