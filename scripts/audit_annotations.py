"""Fully decode segmentation files and validate landmark syntax."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def check_mask(task):
    path,kind=task
    try:
        with Image.open(path) as im:
            im.load()
            colors=im.getcolors(256)
            values={c for _,c in colors} if colors else set()
            if kind=='hq':
                if im.size!=(512,512): raise ValueError(f'Unexpected HQ mask size {im.size}')
                allowed={0,255} if im.mode=='L' else {(0,0,0),(255,255,255)} if im.mode=='RGB' else set()
                if not values or not values.issubset(allowed): raise ValueError(f'Nonbinary HQ mask {im.mode}: {values}')
            else:
                if im.mode not in ['L','P']: raise ValueError(f'Unexpected label mode {im.mode}')
                if not values.issubset(set(range(11))): raise ValueError(f'Invalid LaPa labels {values}')
        return None
    except Exception as exc: return {'path':str(path),'error':str(exc)}


def main():
    cfg=json.loads((ROOT/'configs/local.json').read_text())
    hq=list((Path(cfg['hq'])/'CelebAMask-HQ-mask-anno').rglob('*.png'))
    lapa=list(csv.DictReader((ROOT/'data/manifests/lapa.csv').open()))
    tasks=[(p,'hq') for p in hq]+[(Path(r['label_path']),'lapa') for r in lapa]
    errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,result in enumerate(pool.map(check_mask,tasks),1):
            if result: errors.append(result)
            if i%50000==0: print(f'Decoded {i}/{len(tasks)} masks',flush=True)
    landmark_errors=[]
    for r in lapa:
        try:
            lines=Path(r['landmark_path']).read_text().splitlines()
            if int(lines[0])!=106 or len(lines)!=107: raise ValueError('Expected 106 points')
            for line in lines[1:]:
                xy=[float(v) for v in line.split()]
                if len(xy)!=2 or not all(abs(v)<1e10 for v in xy): raise ValueError('Invalid coordinates')
        except Exception as exc: landmark_errors.append({'path':r['landmark_path'],'error':str(exc)})
    report={'decoded_hq_masks':len(hq),'decoded_lapa_labels':len(lapa),
        'mask_errors':errors,'landmark_syntax_errors':landmark_errors,
        'limits':'File decoding, mask values and landmark syntax checked; annotation accuracy and landmark bounds not evaluated.'}
    (ROOT/'research/annotation_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'decoded_hq_masks':len(hq),'decoded_lapa_labels':len(lapa),
        'mask_error_count':len(errors),'landmark_error_count':len(landmark_errors),
        'error_examples':errors[:3]+landmark_errors[:3]},indent=2),flush=True)
    if errors or landmark_errors: raise RuntimeError('Annotation audit failed')


if __name__=='__main__': main()
