"""Download only three pinned public DINOv2 files; never upload images."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'
FILES=['config.json','preprocessor_config.json','model.safetensors']
EXPECTED='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'

def main():
    folder=ROOT/'outputs/dense_geometry_assets_v4/dinov2-small';folder.mkdir(parents=True,exist_ok=True)
    receipt=dict(repository='facebook/dinov2-small',revision=REVISION,files=[])
    for name in FILES:
        target=folder/name
        if not target.exists():
            url=f'https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/{name}'
            temporary=target.with_suffix(target.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
            temporary.replace(target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if name=='model.safetensors':assert digest==EXPECTED,'Published model hash mismatch'
        receipt['files'].append(dict(file=name,sha256=digest,size=target.stat().st_size))
        print(name,target.stat().st_size,flush=True)
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
