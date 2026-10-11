"""Run the preregistered CPU diagnostic; preserve all existing outputs."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_spatial_reference_geometry_v1 import read_image,sha
from visible_feature_geometry_v2 import match_geometry

def main():
    out=ROOT/'outputs/spatial_reference_correspondence_v2/geometry'
    if out.exists(): raise ValueError('Preserve existing attempt')
    cache_path=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache/receipt.json'
    old=json.loads((ROOT/'outputs/spatial_reference_correspondence_v1/geometry/receipt.json').read_text())
    assert sha(cache_path)==old['cache_receipt_sha256']
    cache=json.loads(cache_path.read_text())
    assert cache['train_identities']==old['training_ids']
    assert not set(cache['train_identities'])&set(cache['excluded_identities'])
    files=['research/VISIBLE_FEATURE_GEOMETRY_PREFLIGHT_V2.md','scripts/visible_feature_geometry_v2.py','scripts/audit_visible_feature_geometry_v2.py','scripts/spatial_reference_geometry_v1.py','scripts/audit_spatial_reference_geometry_v1.py']
    receipt=dict(status='running',device='CPU',training=False,source_hashes={f:sha(ROOT/f) for f in files},cache_sha256=sha(cache_path),rows=[])
    out.mkdir(parents=True)
    def save(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    save()
    try:
        for c in cache['records']:
            assert c['role']=='train' and c['identity'] in old['training_ids']
            assert len(c['references'])==4 and all(r['identity']==c['identity'] for r in c['references'])
            mask=np.zeros((512,512),bool);x0,y0,x1,y1=c['mask']['bbox_xyxy_exclusive'];mask[y0:y1,x0:x1]=True
            assert hashlib.sha256(mask.tobytes()).hexdigest()==c['mask']['binary_sha256']
            observed=np.where(mask[...,None],np.uint8(128),read_image(c['target']))
            rows=[dict(index=i,**match_geometry(observed,mask,read_image(r))) for i,r in enumerate(c['references'])]
            accepted=sum(r['eligible'] for r in rows)
            receipt['rows'].append(dict(identity=c['identity'],references=rows,accepted_references=accepted,usable=accepted>=2))
            save();print(c['identity'],accepted,'of 4 references pass',flush=True)
        receipt.update(status='complete',usable_cases=sum(r['usable'] for r in receipt['rows']))
        receipt['gate_passed']=len(receipt['rows'])==8 and receipt['usable_cases']==8
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally: save()

if __name__=='__main__': main()
