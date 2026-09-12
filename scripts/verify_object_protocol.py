"""Verify frozen object-composite construction independently of model predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from object_test import make_case
ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/'outputs/object_test/protocol.json';protocol=json.loads(path.read_text());rows=[]
    for c in protocol['cases']:
        target,observed,true,masks=make_case(c)
        assert target.shape==observed.shape==(256,256,3)
        assert np.isfinite(observed).all() and np.array_equal(observed[~true],target[~true])
        assert np.array_equal(masks['accurate'],true)
        assert not (masks['under']&~true).any() and not (true&~masks['over']).any()
        assert not true[0].any() and not true[-1].any() and not true[:,0].any() and not true[:,-1].any()
        with Image.open(c['image_path']) as im:w,h=im.size
        b=c['crop'];padded=bool(b and (b[0]<0 or b[1]<0 or b[2]>w or b[3]>h))
        rows.append({'case_id':c['case_id'],'true_fraction':float(true.mean()),'empty_under_mask':not bool(masks['under'].any()),'crop_requires_padding':padded})
    report={'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':len(rows),'checks':'Finite geometry, nonempty true masks, visible pixels unchanged by corruption, accurate/under/over mask polarity, no true-mask boundary clipping. Not a model quality test.','minimum_missing_fraction':min(r['true_fraction'] for r in rows),'maximum_missing_fraction':max(r['true_fraction'] for r in rows),'empty_under_masks':sum(r['empty_under_mask'] for r in rows),'padded_crops':sum(r['crop_requires_padding'] for r in rows),'cases_detail':rows}
    (ROOT/'research/object_protocol_verification.json').write_text(json.dumps(report,indent=2));print('Verified object construction:',len(rows),'cases')
if __name__=='__main__':main()
