import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/near_duplicate_audit'


def main():
    records=json.loads((OUT/'indexed_images.json').read_text());pairs=json.loads((OUT/'candidate_pairs.json').read_text())
    cache={}
    def pixels(i):
        if i not in cache:
            with Image.open(records[i]['image_path']) as im:cache[i]=np.array(ImageOps.exif_transpose(im).convert('RGB').resize((128,128),Image.Resampling.LANCZOS)).astype('float32')/255
        return cache[i]
    for index,p in enumerate(pairs):
        a,b=pixels(p['a']),pixels(p['b']);p['candidate_id']=index
        p['rgb_mae_128']=float(np.abs(a-b).mean());p['rgb_correlation_128']=float(np.corrcoef(a.flatten(),b.flatten())[0,1])
    pairs.sort(key=lambda p:p['rgb_mae_128'])
    (OUT/'ranked_candidates.json').write_text(json.dumps(pairs,indent=2))
    for page in range(4):
        canvas=Image.new('RGB',(600,224*8),'white');draw=ImageDraw.Draw(canvas)
        for slot,p in enumerate(pairs[page*8:(page+1)*8]):
            for col,key in enumerate(['a','b']):
                row=records[p[key]]
                with Image.open(row['image_path']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(220,190))
                canvas.paste(thumb,(col*300,slot*224+30));draw.text((col*300+3,slot*224+3),f"#{p['candidate_id']} {row['dataset']} {row['split']} MAE={p['rgb_mae_128']:.3f}",fill='black')
        canvas.save(OUT/f'ranked_review_{page+1}.png')
    selected={str(c['hq_id']) for c in json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text())}
    involved=[p['candidate_id'] for p in pairs if any(records[p[key]].get('hq_id') in selected for key in ['a','b'])]
    print(json.dumps({'lowest_mae_candidates':[{k:p[k] for k in ['candidate_id','hamming_distance','rgb_mae_128','rgb_correlation_128']} for p in pairs[:20]],'candidates_touching_v2':involved},indent=2))


if __name__=='__main__':main()
