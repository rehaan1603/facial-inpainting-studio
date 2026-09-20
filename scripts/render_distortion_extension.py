"""Render all saved development extension cases, retaining failed arms."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'outputs/distortion_extension_v1'
    rows=json.loads((base/'evaluation.json').read_text())['rows']
    cases={c['case_id']:c for c in json.loads((base/'evaluation_manifest.json').read_text())['cases']}
    out=base/'contact_sheets';out.mkdir(exist_ok=True)
    for cid in sorted(cases):
        group=[r for r in rows if r['case_id']==cid]
        assert group and all(r['identity'] not in {'7789','8726','5774','6632','7170','6225','5170','8561'} for r in group)
        photos=[('Clean scoring target',cases[cid]['target']['path'])]
        for mode,strength in [('observed',None),('standard',.5),('standard',.99),('preserve',.5),('preserve',.99)]:
            r=next(r for r in group if r['mode']==mode and (mode=='observed' or r['strength']==strength))
            photos.append((f'{mode} {strength or ""}',r.get('output') if r['status']=='complete' else None))
        sheet=Image.new('RGB',(768,560),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate(photos):
            x,y=i%3*256,i//3*280;draw.text((x+4,y+4),label,fill='black')
            if path:
                with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
            else:draw.text((x+8,y+90),'REFERENCE DETECTION FAILED',fill='red')
        sheet.save(out/(cid+'.png'))
    print(f'{len(cases)} local sheets rendered; restricted photographs are not publication assets.')

if __name__=='__main__':main()
