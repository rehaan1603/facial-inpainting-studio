"""Fetch the official public COCO annotation archive and extract only val instances."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);cache.mkdir(parents=True,exist_ok=True)
    archive=cache/'coco_annotations.zip';url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    if not archive.exists():
        tmp=archive.with_suffix('.download')
        with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as f:
            while chunk:=response.read(1024*1024):f.write(chunk)
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:data=json.loads(z.read('annotations/instances_val2017.json'))
    (cache/'coco_val_instances.json').write_text(json.dumps(data))
    report={'url':url,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'extracted_member':'annotations/instances_val2017.json','terms':'https://cocodataset.org/#termsofuse','transport':'Official HTTP download link; local hashes record received bytes, not independently authenticated publisher checksums.'}
    (ROOT/'research/coco_download.json').write_text(json.dumps(report,indent=2));print('COCO validation annotation metadata ready')
if __name__=='__main__':main()
