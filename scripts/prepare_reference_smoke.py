"""Freeze one dataset-label group for a functional multi-reference example."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def main():
    groups=defaultdict(list)
    for row in csv.DictReader((ROOT/'data/manifests/celebahq_reviewed_v2.csv').open()):
        if row['split']=='val':groups[row['identity']].append(row)
    ids=sorted([i for i,rows in groups.items() if len(rows)>=5],key=lambda i:hashlib.sha256(('reference-smoke-17:'+i).encode()).hexdigest())
    rows=sorted(groups[ids[0]],key=lambda r:int(r['hq_id']))[:5]
    out=ROOT/'outputs/reference_smoke';out.mkdir(exist_ok=True)
    manifest=[]
    for index,row in enumerate(rows):
        path=Path(row['image_path']);sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==row['source_sha256']
        image=Image.open(path).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
        name='target.png' if index==0 else f'reference_{index}.png';image.save(out/name);manifest.append({'role':'target' if index==0 else 'reference','hq_id':row['hq_id'],'identity_label':row['identity'],'source_sha256':sha,'file':name})
    mask=Image.new('L',(512,512));draw=ImageDraw.Draw(mask);draw.rounded_rectangle((115,180,395,290),radius=25,fill=255);mask.save(out/'mask.png')
    target=Image.open(out/'target.png');Image.composite(Image.new('RGB',(512,512),(90,100,110)),target,mask).save(out/'observed.png')
    (out/'manifest.json').write_text(json.dumps({'selection':'First SHA256-ranked validation identity label with at least five reviewed images; first five numeric HQ IDs. Functional example, not a random test estimate.','images':manifest},indent=2));print(out)
if __name__=='__main__':main()
