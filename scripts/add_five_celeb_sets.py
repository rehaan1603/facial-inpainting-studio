"""Add five annotated training-split practice identities, with single-face screening."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import shutil
import cv2
from insightface.app import FaceAnalysis

ROOT=Path(__file__).resolve().parents[1]
DEST=Path.home()/'Downloads/Celeb TEST data'
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])
    groups=defaultdict(list)
    with (ROOT/'data/manifests/celebahq_reviewed_v2.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split']=='train' and not (DEST/f"Person {row['identity']}").exists():
                groups[row['identity']].append(row)
    analyzer=FaceAnalysis(name=str(cache/'reference_models/insightface/models/buffalo_l'),allowed_modules=['detection'],providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1,det_size=(640,640))
    selected=[]
    for identity,rows in sorted(groups.items(),key=lambda x:int(x[0])):
        if len(rows)<5:
            continue
        accepted=[]
        seen=set()
        for row in sorted(rows,key=lambda x:int(x['hq_id'])):
            path=Path(row['image_path'])
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=row['source_sha256']:
                raise ValueError('Dataset source changed')
            if digest in seen:
                continue
            image=cv2.imread(str(path))
            if image is None or len(analyzer.get(image))!=1:
                continue
            seen.add(digest)
            accepted.append(row)
            if len(accepted)==5:
                break
        if len(accepted)<5:
            continue
        selected.append({'identity':identity,'split':'train','photos':accepted})
        print(f'Selected person {identity}: five distinct files with one detected face each',flush=True)
        if len(selected)==5:
            break
    if len(selected)!=5:
        raise ValueError('Could not find five suitable groups')
    for group in selected:
        folder=DEST/f"Person {group['identity']}"
        folder.mkdir(parents=True,exist_ok=False)
        for i,row in enumerate(group['photos']):
            name=('Input_photo' if i==0 else f'reference_{i}')+f"_{row['hq_id']}.jpg"
            destination=folder/name
            shutil.copy2(row['image_path'],destination)
            assert hashlib.sha256(destination.read_bytes()).hexdigest()==row['source_sha256']
        (folder/'START HERE.txt').write_text('Upload Input_photo to the studio, then paint the region you want to reconstruct. Add the four reference_ photos from this same folder. These are original clean photographs; no damaged input or premade mask is included. Same-person grouping follows CelebA annotations. These training-split practice photos are not a fresh held-out test and do not mean the current generator was trained locally on them. Keep photos local.\n',encoding='utf-8')
    report=DEST/'ADDITIONAL FIVE SETS.json'
    if report.exists():
        report=DEST/('ADDITIONAL SETS '+ '-'.join(g['identity'] for g in selected)+'.json')
    report.write_text(json.dumps({'sets':selected,'verified_photos':25,'selection':'First eligible training identity labels with five distinct-hash, single-detector-face images, excluding existing folders.'},indent=2),encoding='utf-8')
    print('Added five sets / 25 verified original photos:',', '.join(g['identity'] for g in selected),flush=True)

if __name__=='__main__':
    main()
