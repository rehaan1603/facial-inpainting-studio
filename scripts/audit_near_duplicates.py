"""Read-only cross-partition duplicate candidate screening; never identity recognition."""
import csv
import hashlib
import io
import json
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/near_duplicate_audit'


def fingerprint(row):
    blob=Path(row['image_path']).read_bytes()
    assert hashlib.sha256(blob).hexdigest()==row['source_sha256'],row['image_path']
    with Image.open(io.BytesIO(blob)) as im:
        rgb=ImageOps.exif_transpose(im).convert('RGB')
        pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        small=np.array(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype='float32')
        coeff=dctn(small,type=2,norm='ortho')[:8,:8].flatten()
        bits=coeff>np.median(coeff[1:]);bits[0]=False
        phash=int.from_bytes(np.packbits(bits).tobytes(),'big')
        return {**row,'decoded_rgb_sha256':pixels,'phash':f'{phash:016x}','width':rgb.width,'height':rgb.height}


def candidates(hashes,radius=6):
    # Seven disjoint blocks guarantee a shared exact block for Hamming distance <= 6.
    buckets=defaultdict(list);widths=[10,9,9,9,9,9,9];offsets=np.cumsum([0]+widths[:-1]).tolist()
    for i,value in enumerate(hashes):
        keys=[(j,(value>>offset)&((1<<width)-1)) for j,(offset,width) in enumerate(zip(offsets,widths))]
        possible=set()
        for key in keys:possible.update(buckets[key])
        for other in sorted(possible):
            distance=(value^hashes[other]).bit_count()
            if distance<=radius:yield other,i,distance
        for key in keys:buckets[key].append(i)


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];digests={}
    for dataset in ['celebahq','lapa']:
        path=ROOT/'data/manifests'/f'{dataset}_clean.csv';digests[dataset]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in csv.DictReader(path.open()):rows.append({k:v for k,v in {'dataset':dataset,**r}.items() if k in ['dataset','split','identity','hq_id','image_path','source_sha256']})
    guard=OUT/'signature.json'
    sig={'manifests':digests,'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if guard.exists():assert json.loads(guard.read_text())==sig,'Inputs changed; version the audit directory'
    else:guard.write_text(json.dumps(sig,indent=2))
    cache=OUT/'fingerprints.jsonl';saved={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r=json.loads(line);saved[r['image_path']]=r
    pending=[r for r in rows if r['image_path'] not in saved]
    with cache.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(fingerprint,pending)):
            f.write(json.dumps(r)+'\n');saved[r['image_path']]=r
            if (i+1)%2000==0:f.flush();print(f'Fingerprinted {len(saved)}/{len(rows)}',flush=True)
    records=[saved[r['image_path']] for r in rows]
    hashes=[int(r['phash'],16) for r in records];pairs=[]
    for a,b,distance in candidates(hashes):
        x,y=records[a],records[b]
        if (x['dataset'],x['split'])==(y['dataset'],y['split']):continue
        pairs.append({'a':a,'b':b,'hamming_distance':distance,'decoded_rgb_equal':x['decoded_rgb_sha256']==y['decoded_rgb_sha256']})
    pairs.sort(key=lambda p:(not p['decoded_rgb_equal'],p['hamming_distance'],p['a'],p['b']))
    (OUT/'candidate_pairs.json').write_text(json.dumps(pairs,indent=2))
    (OUT/'indexed_images.json').write_text(json.dumps(records))
    categories=Counter()
    for p in pairs:
        a,b=records[p['a']],records[p['b']]
        categories[' | '.join(sorted([a['dataset']+':'+a['split'],b['dataset']+':'+b['split']]))]+=1
    # Preview ranked candidates only. Similar appearance is not proof of duplication.
    for page in range(min(4,(len(pairs)+7)//8)):
        canvas=Image.new('RGB',(600,224*8),'white');draw=ImageDraw.Draw(canvas)
        for slot,p in enumerate(pairs[page*8:(page+1)*8]):
            for col,key in enumerate(['a','b']):
                row=records[p[key]]
                with Image.open(row['image_path']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(220,190))
                canvas.paste(thumb,(col*300,slot*224+30));draw.text((col*300+3,slot*224+3),f"#{page*8+slot} {row['dataset']} {row['split']} d={p['hamming_distance']}",fill='black')
        canvas.save(OUT/f'review_{page+1}.png')
    result={'images_screened':len(records),'hamming_radius':6,'candidate_pairs':len(pairs),'decoded_rgb_equal_pairs':sum(p['decoded_rgb_equal'] for p in pairs),'candidate_categories':dict(categories),'manifest_hashes':digests,'limits':['Perceptual hash candidates require verification; similar faces can collide.','No identity recognition or same-person matching performed.','Screening may miss crops, flips, color changes and other transformed duplicates.','Within-dataset within-split near duplicates are not included in candidate output.','No source files or working manifests were changed.','Reading test images for integrity screening is not model evaluation.']}
    (ROOT/'research/near_duplicate_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
