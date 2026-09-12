import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from area_matched_v3 import translate,corrupt

ROOT=Path(__file__).resolve().parents[1]


def main():
    cases_path=ROOT/'outputs/area_matched_v3_margin12/cases.json'
    cases=json.loads(cases_path.read_text());groups=defaultdict(list)
    files=0
    for c in cases:groups[(c['hq_id'],c['missing_pixels'])].append(c)
    ids={p:{c['identity'] for c in cases if c['partition']==p} for p in ['tuning','assessment']}
    assert not ids['tuning']&ids['assessment']
    for group in groups.values():
        assert {c['location'] for c in group}=={'eye_center','mouth_center','upper_image'} and len(group)==3
        reference=None;rgb_reference=None
        for c in group:
            masks={}
            for name,digest in c['mask_sha256'].items():
                path=ROOT/c['mask_directory']/f'{name}.png';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
                with Image.open(path) as im:arr=np.array(im)
                assert arr.shape==(256,256) and set(np.unique(arr))<={0,255}
                masks[name]=arr>0;files+=1
                yy,xx=np.where(masks[name]);assert min(yy.min(),xx.min(),255-yy.max(),255-xx.max())>=12
            true=masks['true'];assert int(true.sum())==c['missing_pixels']
            for name,expected in c['conditions'].items():
                assert int((masks[name]&~true).sum())==expected['fp_pixels']
                assert int((~masks[name]&true).sum())==expected['fn_pixels']
            cx,cy=c['center_xy'];unshifted={k:translate(v,128-cx,128-cy) for k,v in masks.items()}
            assert all(v is not None for v in unshifted.values())
            if reference is None:reference=unshifted
            else:
                for name in masks:np.testing.assert_array_equal(unshifted[name],reference[name])
            rgb=corrupt(np.zeros((256,256,3),np.float32),true,c['seed'],c['center_xy'])[true]
            if rgb_reference is None:rgb_reference=rgb
            else:np.testing.assert_array_equal(rgb,rgb_reference)
    result={'all_checks_passed':True,'matched_groups':len(groups),'cases':len(cases),'mask_files_verified':files,'identities':{p:len(v) for p,v in ids.items()},'checks':['Exact true/FP/FN counts','Masks equal under inverse translation','Occluder RGB equal across locations','File hashes, shapes and binary values','Disjoint development partitions','At least 12 pixels of margin for all supplied masks; correction dilation cannot clip'],'cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest()}
    (ROOT/'research/area_matched_v3_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
