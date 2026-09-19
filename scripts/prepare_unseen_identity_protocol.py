"""Freeze fresh LOCAL identity roles. Run in reference_env_v2; no generation."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, rank, write_new

def exclusions(rows):
    ledger={'reviewed_training_partition':{'identity_labels':sorted({r['identity'] for r in rows if r['split']=='train'},key=int)}}
    for rel in ['outputs/pilot/cases.json','outputs/pilot_clean_v1/cases.json',
                'outputs/benchmark_v2/cases.json','outputs/benchmark_v2_resshift/cases.json',
                'outputs/area_matched_v3/cases.json','outputs/area_matched_v3_margin12/cases.json']:
        p=ROOT/rel; data=json.loads(p.read_text());ledger[rel]={'sha256':sha(p),'identity_labels':sorted({str(c['identity']) for c in data},key=int)}
    p=ROOT/'outputs/object_test/protocol.json'
    ledger[str(p.relative_to(ROOT))]={'sha256':sha(p),'identity_labels':sorted({str(c['unit']) for c in json.loads(p.read_text())['cases'] if c['dataset']=='celebahq'},key=int)}
    for p in [ROOT/'outputs/reference_smoke/manifest.json', ROOT/'outputs/reference_diagnostics_v1/selection_audit.json']:
        data=json.loads(p.read_text())
        records=data['attempts'] if 'attempts' in data else data['images']
        ledger[str(p.relative_to(ROOT))]={'sha256':sha(p),'identity_labels':sorted({str(c['identity_label']) for c in records},key=int)}
    ledger['earlier_engineering_examples']={'identity_labels':['916','3037']}
    return ledger,set().union(*(set(v['identity_labels']) for v in ledger.values()))

def main():
    import numpy as np
    import cv2
    from PIL import Image,ImageOps
    from insightface.app import FaceAnalysis
    out=ROOT/'outputs/generalization_v1';out.mkdir(exist_ok=False)
    source=ROOT/'data/manifests/celebahq_reviewed_v2.csv';rows=list(csv.DictReader(source.open()))
    ledger,excluded=exclusions(rows);groups=defaultdict(list)
    for r in rows:
        if r['identity'] not in excluded:groups[(r['split'],r['identity'])].append(r)
    fp_path=ROOT/'outputs/near_duplicate_audit/fingerprints.jsonl'
    fp={str(r['hq_id']):r for r in map(json.loads,fp_path.read_text().splitlines()) if r['dataset']=='celebahq'}
    protected=[r for r in fp.values() if str(r['identity']) in excluded]
    protected_sha={r['source_sha256'] for r in protected};protected_rgb={r['decoded_rgb_sha256'] for r in protected}
    protected_phash=[int(r['phash'],16) for r in protected]
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    detector=FaceAnalysis(name=str(cache/'reference_models/insightface/models/buffalo_l'),allowed_modules=['detection'],providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,det_size=(640,640));cv2.setNumThreads(2)
    selected=[];retained=[];attempts=[]
    for split,count in [('val',4),('test',8)]:
        eligible=[identity for s,identity in groups if s==split and len({r['source_sha256'] for r in groups[s,identity]})>=8]
        for identity in sorted(eligible,key=lambda v:rank(v,split)):
            kept=[];attempt={'split':split,'identity':identity,'photos':[]}
            for row in sorted(groups[split,identity],key=lambda r:rank(r['hq_id'],'photo')):
                f=fp[row['hq_id']];record=dict(row,decoded_rgb_sha256=f['decoded_rgb_sha256'],phash=f['phash']);reasons=[]
                if sha(row['image_path'])!=row['source_sha256']:raise ValueError('Source changed')
                with Image.open(row['image_path']) as original:
                    image=ImageOps.exif_transpose(original).convert('RGB')
                    import hashlib
                    if hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()!=f['decoded_rgb_sha256']:raise ValueError('Decoded image changed')
                    image=image.resize((512,512),Image.Resampling.LANCZOS)
                if f['source_sha256'] in protected_sha or f['decoded_rgb_sha256'] in protected_rgb:reasons.append('protected_exact_overlap')
                ph=int(f['phash'],16)
                if any((ph^p).bit_count()<=6 for p in protected_phash):reasons.append('protected_phash_distance_le_6')
                if any(f['source_sha256']==p['source_sha256'] or f['decoded_rgb_sha256']==p['decoded_rgb_sha256'] or (ph^int(p['phash'],16)).bit_count()<=6 for p in retained+kept):reasons.append('role_or_split_duplicate_screen')
                if not reasons and split=='val':
                    faces=detector.get(np.asarray(image)[:,:,::-1].copy())
                    record['face_count']=len(faces)
                    if len(faces)!=1:reasons.append('face_count_not_one')
                    else:record['construction_detection']={'bbox':faces[0].bbox.tolist(),'landmarks':faces[0].kps.tolist()}
                attempt['photos'].append({'hq_id':row['hq_id'],'reasons':reasons})
                if not reasons:kept.append(record)
                if len(kept)==8:break
            attempt['selected']=len(kept)==8;attempts.append(attempt)
            if len(kept)==8:
                for i,r in enumerate(kept):r['role']='target' if i==0 else f'reference_{i}' if i<5 else f'gallery_{i-4}'
                selected.append({'identity':identity,'split':'validation' if split=='val' else 'reserved_final','images':kept});retained.extend(kept)
            print(split,identity,len(kept),'selected',sum(c['split']==('validation' if split=='val' else 'reserved_final') for c in selected),flush=True)
            if sum(c['split']==('validation' if split=='val' else 'reserved_final') for c in selected)==count:break
        if sum(c['split']==('validation' if split=='val' else 'reserved_final') for c in selected)!=count:
            write_new(out/'selection_failure.json',{'attempts':attempts,'selected':selected});raise RuntimeError('Insufficient eligible identities; do not silently weaken protocol')
    protocol={'version':1,'date':'2026-09-19','purpose':'Fresh local validation and reserved final identities; pretraining exposure unknown',
              'source_sha256':sha(source),'fingerprints_sha256':sha(fp_path),'exclusion_ledger':ledger,
              'split_rule':'Exclude every reviewed training identity and previously inspected experiment identity. Hash-rank eight-photo groups. All role/split pHash distances >6 and exact hashes distinct. Validation detector-only screening; final metadata/duplicate screening only.',
              'limitations':['Dataset identity labels are not independently verified','pHash is conservative, not exhaustive transformed-duplicate detection','Unknown pretrained overlap','Detector-screened validation is not population representative'],
              'cases':selected,'attempts':attempts,'experiment':{'validation_only':True,'conditions':['eyes_removal_medium','mouth_removal_medium','central_face_mixed_medium'],'seeds':[17,29],
              'policies':['first','random','identity_only','quality_only','all','mask_aware','zero_scale'],'steps':30,'strength':.99,'scale':.8,'blend':'poisson','resolution':512,
              'primary_metric':'facenet_cosine','statistical_unit':'identity','primary_comparisons':['mask_aware - random','mask_aware - identity_only','mask_aware - quality_only','mask_aware - all'],
              'multiplicity':'Holm across four primary comparisons; other metrics exploratory','generator':'Frozen scripts/reference_inpaint.py','generator_sha256':sha(ROOT/'scripts/reference_inpaint.py')}}
    write_new(ROOT/'research/protocols/unseen_identity_protocol_v1.json',protocol)
    (ROOT/'research/protocols/unseen_identity_protocol_v1.md').write_text('# Unseen identity protocol v1\n\nFour locally fresh validation groups and eight reserved final groups, eight distinct photos each. No final-test generations in this cycle.\n\nThe JSON contains source hashes, identity exclusions, target/reference/gallery roles, screening failures, seeds and fixed comparisons. All reviewed training identities are excluded. Pretraining overlap is unknown. Final data must not be exported as user practice sets.\n',encoding='utf-8')
    print('FROZEN',[(c['split'],c['identity']) for c in selected],flush=True)

if __name__=='__main__':main()
