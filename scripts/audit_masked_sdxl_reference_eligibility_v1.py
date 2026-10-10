"""Audit all original training references at the unchanged detector settings.

No targets/test images are opened. No faces or photographs are substituted.
Preserve every failure; cache only count/score/status/hash metadata, no embeddings.
"""
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/reference_eligibility.json'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(document):
    OUT.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    if OUT.exists(): raise ValueError('Refusing eligibility audit overwrite')
    source=ROOT/'outputs/reference_intervention_pilot_v1/manifest.json'
    expected='f01f52bae44001bc9c23d36c0b4c5b48cd8e60e8f19cb1ba717be085143f117e'
    assert sha(source)==expected
    manifest=json.loads(source.read_text());cases=[c for c in manifest['cases'] if c['role']=='train']
    assert len(cases)==16 and not {c['identity'] for c in cases}&{'263','6681','7400','10149'}
    local=json.loads((ROOT/'configs/local.json').read_text());cache=Path(local['cache'])
    provenance=json.loads((ROOT/'research/reference_faces_provenance.json').read_text())
    face_dir=cache/'reference_models/insightface/models/buffalo_l'
    for name,h in provenance['files'].items(): assert sha(face_dir/name)==h
    for c in cases:
        assert len(c['references'])==4
        for r in c['references']: assert sha(r['image_path'])==r['source_sha256']
    import cv2
    import numpy as np
    from PIL import Image,ImageOps
    from insightface.app import FaceAnalysis
    cv2.setNumThreads(1)
    app=FaceAnalysis(name=str(face_dir),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1,det_size=(640,640))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',manifest_sha256=expected,script_sha256=sha(__file__),
        detector='buffalo_l CPU det_size640; unchanged default threshold',
        eligibility='All four original references must have exactly one finite face embedding and landmarks',
        target_photos_decoded=0,evaluation_photos_decoded=0,records=[],eligible_ids=[],excluded_ids=[])
    save(report);start=time.monotonic()
    for c in cases:
        refs=[]
        for i,r in enumerate(c['references']):
            assert sha(r['image_path'])==r['source_sha256']
            with Image.open(r['image_path']) as image: rgb=np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
            faces=app.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            finite=len(faces)==1 and faces[0].normed_embedding is not None and np.isfinite(faces[0].normed_embedding).all() and np.isfinite(faces[0].kps).all()
            refs.append(dict(position=i,hq_id=r['hq_id'],source_sha256=r['source_sha256'],face_count=len(faces),
                finite_features=bool(finite),scores=[float(f.det_score) for f in faces],eligible=bool(finite)))
        eligible=all(r['eligible'] for r in refs)
        report['records'].append(dict(identity=c['identity'],references=refs,eligible=eligible))
        report['eligible_ids' if eligible else 'excluded_ids'].append(c['identity'])
        report['elapsed_seconds']=time.monotonic()-start;save(report)
        print('ELIGIBILITY',c['identity'],eligible,[r['face_count'] for r in refs],flush=True)
    report['status']='complete';report['reference_photos_decoded']=64;save(report)
    print(json.dumps(dict(eligible_ids=report['eligible_ids'],excluded_ids=report['excluded_ids'])),flush=True)

if __name__=='__main__': main()
