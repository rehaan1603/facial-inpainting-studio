import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=json.loads((ROOT/'configs/learned_refiner.json').read_text());local=json.loads((ROOT/'configs/local.json').read_text())
    manifest=ROOT/cfg['manifest'];rows=list(csv.DictReader(manifest.open()));train=[r for r in rows if r['split']=='train']
    selected={str(c['hq_id']) for c in json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text()) if c['partition']=='tuning'}
    val=[r for r in rows if r['hq_id'] in selected];assert len(val)==48
    assert not {r['identity'] for r in train}&{r['identity'] for r in val}
    out=Path(local['cache'])/'refiner_data';out.mkdir(parents=True,exist_ok=True)
    signature={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'resolution':cfg['resolution'],'tuning_ids':sorted(selected)}
    meta=out/'metadata.json'
    if meta.exists():assert json.loads(meta.read_text())['signature']==signature;print('Prepared data already verified');return
    def load(r):
        blob=Path(r['image_path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==r['source_sha256']
        with Image.open(r['image_path']) as im:return np.array(im.convert('RGB').resize((cfg['resolution'],cfg['resolution']),Image.Resampling.LANCZOS))
    for name,subset in [('train',train),('tuning',val)]:
        target=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype='uint8',shape=(len(subset),cfg['resolution'],cfg['resolution'],3))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,arr in enumerate(pool.map(load,subset)):
                target[i]=arr
                if (i+1)%4000==0:print(f'{name} prepared {i+1}/{len(subset)}',flush=True)
        target.flush();del target
    meta.write_text(json.dumps({'signature':signature,'train':train,'tuning':val},indent=2));print('Prepared',len(train),'training images and',len(val),'tuning identities')


if __name__=='__main__':main()
