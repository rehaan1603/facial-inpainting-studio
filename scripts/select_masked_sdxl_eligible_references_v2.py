"""Deterministic training-only reference eligibility recovery; preserve old data."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2'
PROTOCOL=ROOT/'research/MASKED_SDXL_REFERENCE_ELIGIBILITY_RECOVERY_V2.md'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    audit_path=OUT/'reference_selection.json';manifest_path=OUT/'manifest.json'
    if audit_path.exists() or manifest_path.exists(): raise ValueError('Refusing selection overwrite')
    assert PROTOCOL.is_file()
    old_path=ROOT/'outputs/reference_intervention_pilot_v1/manifest.json'
    assert sha(old_path)=='f01f52bae44001bc9c23d36c0b4c5b48cd8e60e8f19cb1ba717be085143f117e'
    old=json.loads(old_path.read_text());original=[c for c in old['cases'] if c['role']=='train']
    csv_path=ROOT/'data/manifests/celebahq_reviewed_v2.csv';fp_path=ROOT/'outputs/near_duplicate_audit/fingerprints.jsonl'
    assert sha(csv_path)==old['source_sha256'] and sha(fp_path)==old['fingerprints_sha256']
    rows=list(csv.DictReader(csv_path.open()))
    fp={str(r['hq_id']):r for r in map(json.loads,fp_path.read_text().splitlines()) if r['dataset']=='celebahq'}
    excluded=set(old['excluded_identities'])|{c['identity'] for c in old['cases'] if c['role']!='train'}
    protected=[r for r in fp.values() if r['identity'] in excluded]
    protected_sha={r['decoded_rgb_sha256'] for r in protected}
    protected_ph=[int(r['phash'],16) for r in protected]
    targets=[fp[c['target']['hq_id']] for c in original]
    target_ids={c['target']['hq_id'] for c in original}
    target_sha={r['decoded_rgb_sha256'] for r in targets}
    target_ph=[int(r['phash'],16) for r in targets]
    local=json.loads((ROOT/'configs/local.json').read_text());cache=Path(local['cache'])
    face_dir=cache/'reference_models/insightface/models/buffalo_l'
    for name,h in json.loads((ROOT/'research/reference_faces_provenance.json').read_text())['files'].items(): assert sha(face_dir/name)==h
    import cv2,numpy as np
    from PIL import Image,ImageOps
    from insightface.app import FaceAnalysis
    cv2.setNumThreads(1)
    app=FaceAnalysis(name=str(face_dir),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1,det_size=(640,640))
    report=dict(status='running',source_manifest_sha256=sha(old_path),protocol_sha256=sha(PROTOCOL),script_sha256=sha(__file__),
        source_csv_sha256=sha(csv_path),fingerprints_sha256=sha(fp_path),candidates=[],eligible_ids=[],excluded_ids=[],target_decodes=0,evaluation_decodes=0)
    retained_ph=[];retained_sha=set();selected_cases=[];start=time.monotonic();save(audit_path,report)
    for c in original:
        identity=c['identity'];selected=[];local_ph=[];local_sha=set()
        candidates=sorted((r for r in rows if r['identity']==identity and r['split']=='train'),key=lambda r:int(r['hq_id']))
        for row in candidates:
            if len(selected)==4: break
            idx=row['hq_id'];finger=fp[idx]
            rec=dict(identity=identity,hq_id=idx,source_sha256=row['source_sha256'])
            if idx in target_ids:
                rec['reason']='training_target';report['candidates'].append(rec);continue
            decoded=finger['decoded_rgb_sha256'];ph=int(finger['phash'],16)
            if decoded in protected_sha|target_sha|retained_sha|local_sha or any((ph^p).bit_count()<=6 for p in protected_ph+target_ph+retained_ph+local_ph):
                rec['reason']='duplicate_or_protected_near_duplicate';report['candidates'].append(rec);continue
            assert sha(row['image_path'])==row['source_sha256']
            with Image.open(row['image_path']) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
            faces=app.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            valid=len(faces)==1 and faces[0].normed_embedding is not None and np.isfinite(faces[0].normed_embedding).all() and np.isfinite(faces[0].kps).all()
            rec.update(face_count=len(faces),scores=[float(f.det_score) for f in faces],reason='selected' if valid else 'not_exactly_one_finite_face')
            report['candidates'].append(rec)
            if valid: selected.append(row);local_ph.append(ph);local_sha.add(decoded)
        if len(selected)==4:
            selected_cases.append(dict(c, references=selected));retained_ph+=local_ph;retained_sha|=local_sha;report['eligible_ids'].append(identity)
        else: report['excluded_ids'].append(identity)
        report['elapsed_seconds']=time.monotonic()-start;save(audit_path,report)
        print('REFERENCE SELECTION',identity,len(selected),flush=True)
    report['status']='complete';save(audit_path,report)
    assert len(selected_cases)>=4,'Insufficient same-person encodable reference support; no training permitted'
    new=dict(old,cases=selected_cases,excluded_identities=sorted(excluded|set(report['excluded_ids']),key=int),
        stage='Training-only encoder eligibility revision before any SDXL optimization',
        original_manifest_sha256=sha(old_path),eligibility_selection_sha256=sha(audit_path),
        selection_protocol_sha256=sha(PROTOCOL),limitations=old['limitations']+['FaceID eligibility and near-duplicate filtering determine revised training-only reference selection; no evaluation-quality selection'])
    save(manifest_path,new)
    print(json.dumps(dict(eligible_ids=report['eligible_ids'],manifest_sha256=sha(manifest_path))),flush=True)

if __name__=='__main__': main()
