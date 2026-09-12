"""Build a deterministic, source-recorded COCO object-cutout pool for assessment only."""
import hashlib,json,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    source=cache/'coco_val_instances.json';data=json.loads(source.read_text())
    images={i['id']:i for i in data['images']};licenses={l['id']:l for l in data['licenses']}
    categories={c['id']:c['name'] for c in data['categories']}
    out=cache/'object_assets';out.mkdir(exist_ok=True);records=[];used=set()
    for category in [28,44,47,77,84,88]:
        candidates=[]
        for a in data['annotations']:
            im=images[a['image_id']];x,y,w,h=a['bbox']
            if a['category_id']==category and im['license']==4 and not a['iscrowd'] and isinstance(a['segmentation'],list) and a['area']>=3000 and min(w,h)>=45 and x>2 and y>2 and x+w<im['width']-2 and y+h<im['height']-2:
                candidates.append(a)
        candidates.sort(key=lambda a:hashlib.sha256(f"objects-v1:{a['id']}".encode()).hexdigest())
        count=0
        for a in candidates:
            im=images[a['image_id']]
            if im['id'] in used:continue
            raw=out/im['file_name']
            if not raw.exists():
                with urllib.request.urlopen(im['coco_url'],timeout=60) as response:raw.write_bytes(response.read())
            with Image.open(raw) as image:
                image=image.convert('RGBA');assert image.size==(im['width'],im['height'])
                mask=Image.new('L',image.size);draw=ImageDraw.Draw(mask)
                for poly in a['segmentation']:draw.polygon(list(zip(poly[::2],poly[1::2])),fill=255)
                image.putalpha(mask);image=image.crop(mask.getbbox());path=out/f"object_{a['id']}.png";image.save(path)
            records.append({'asset_id':a['id'],'category':categories[category],'image_id':im['id'],'path':str(path),'image_url':im['coco_url'],'flickr_source':im['flickr_url'],'license':licenses[im['license']],'original_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'asset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'modification':'COCO polygon rasterized with Pillow; cropped RGBA, no generated pixels.'})
            used.add(im['id']);count+=1;print(categories[category],a['id'],flush=True)
            if count==2:break
        assert count==2,category
    report={'annotation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection':'Hash order objects-v1; two non-crowd polygon assets per predeclared category; CC BY 2.0 per COCO metadata; no training use.','limits':'Synthetic composites of photographed objects, not real occluded face captures. Source photographer attribution must be completed before redistributing example images. Backbone exposure unknown.','assets':records}
    (ROOT/'research/object_assets.json').write_text(json.dumps(report,indent=2))
    sheet=Image.new('RGB',(720,480),'#dddddd');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        with Image.open(r['path']) as im:
            tile=ImageOps.contain(im,(220,190));x=(i%3)*240;y=(i//3)*120
            tile=ImageOps.contain(tile,(210,95));sheet.paste(tile,(x,y),tile);draw.text((x,y+100),f"{r['category']} {r['asset_id']}",fill='black')
    sheet.save(out/'contact_sheet.png')
if __name__=='__main__':main()
