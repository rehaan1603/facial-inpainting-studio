"""CPU-only geometry prerequisite on the existing eight TRAIN identities."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial_reference_geometry_v1 import estimate_geometry
from run_structure_transport import LandmarkExtractor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_image(record):
    p=Path(record['image_path'])
    if sha(p)!=record['source_sha256']:raise ValueError('Source image changed')
    with Image.open(p) as im:return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)).copy()

def main():
    out=ROOT/'outputs/spatial_reference_correspondence_v1/geometry'
    if out.exists():raise ValueError('Preserve previous attempt')
    protocol=ROOT/'research/SPATIAL_REFERENCE_CORRESPONDENCE_PREFLIGHT_V1.md'
    manifest=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/manifest.json'
    cache_path=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache/receipt.json'
    cache=json.loads(cache_path.read_text())
    ids=['6792','2250','2083','4164','674','476','1281','8748']
    if not protocol.exists() or sha(manifest)!='b90a1ccddad5cd7c6efafe39209567765e918aabbf0a68022955e3baea2a875f':raise ValueError('Protocol/manifest mismatch')
    if cache['train_identities']!=ids or set(ids)&set(cache['excluded_identities']):raise ValueError('Split mismatch')
    if not {'263','6681','10149','7400'}<=set(cache['excluded_identities']):raise ValueError('Missing development exclusions')
    for c in cache['records']:
        if c['identity'] not in ids or c['role']!='train' or c['target']['identity']!=c['identity']:raise ValueError('Unexpected identity')
        if len(c['references'])!=4 or any(r['identity']!=c['identity'] for r in c['references']):raise ValueError('Unexpected reference identity')
    local=json.loads((ROOT/'configs/local.json').read_text());model_dir=Path(local['cache'])/'reference_models/insightface/models/buffalo_l'
    out.mkdir(parents=True)
    receipt=dict(status='running',device='CPU',cuda_training=False,protocol_sha256=sha(protocol),manifest_sha256=sha(manifest),cache_receipt_sha256=sha(cache_path),
                 sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/spatial_reference_geometry_v1.py',ROOT/'scripts/run_structure_transport.py',ROOT/'src/preservation/structure_transport.py']},
                 training_ids=ids,excluded_development=['263','6681','10149','7400'],rows=[])
    def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf8')
    save();start=time.monotonic()
    try:
        extractor=LandmarkExtractor(model_dir)
        for c in cache['records']:
            target=read_image(c['target']);mask=np.zeros((512,512),bool)
            x0,y0,x1,y1=c['mask']['bbox_xyxy_exclusive'];mask[y0:y1,x0:x1]=True
            if hashlib.sha256(mask.tobytes()).hexdigest()!=c['mask']['binary_sha256']:raise ValueError('Training mask changed')
            observed=np.where(mask[...,None],np.uint8(128),target)
            del target
            refs=[read_image(r) for r in c['references']]
            result=estimate_geometry(observed,mask,refs,extractor)
            receipt['rows'].append(dict(identity=c['identity'],geometry=result))
            save();print(c['identity'],result['status'],result.get('reason'),flush=True)
        receipt['usable_cases']=sum(r['geometry']['status']=='usable' for r in receipt['rows'])
        receipt['gate_passed']=receipt['usable_cases']==8
        receipt['status']='complete'
        receipt['decision']='eligible_for_mechanism_and_GPU_probe' if receipt['gate_passed'] else 'stop_before_GPU_training_and_diagnose_geometry'
    except Exception as exc:
        receipt.update(status='failed',gate_passed=False,error=repr(exc));raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start;save()
    print(json.dumps({k:receipt[k] for k in ['status','usable_cases','gate_passed','decision','elapsed_seconds']},indent=2))

if __name__=='__main__':main()
