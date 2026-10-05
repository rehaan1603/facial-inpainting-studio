"""Conservative history index: explicit identity/HQ fields in project JSON records."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];CACHE=Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_official_benchmark_v1')
cfg=json.loads((ROOT/'configs/local.json').read_text());identity=dict(x.split() for x in (Path(cfg['annotations'])/'identity_CelebA.txt').read_text().splitlines())
hq={str(int(p[0])):identity[p[2]] for line in (Path(cfg['hq'])/'CelebA-HQ-to-CelebA-mapping.txt').read_text().splitlines()[1:] if (p:=line.split())}
valid=set(identity.values());hits=defaultdict(set);measured=defaultdict(set);errors=[];files=[]
keys={'identity','identity_label','identity_labels','identities','identity_ids','selected_identity_labels','excluded_identities','train_identities','validation_identities','test_identities','reserved_identities'}
def walk(v,src,key=''):
 if isinstance(v,dict):
  ident=str(v.get('identity',''))
  if ident in valid and (v.get('status')=='complete' or isinstance(v.get('metrics'),dict) or 'diffusion_raw' in v):measured[ident].add(src)
  for k,x in v.items():walk(x,src,k)
 elif isinstance(v,list):
  for x in v:walk(x,src,key)
 elif isinstance(v,(str,int)) and str(v).isdigit():
  n=str(int(v))
  if key in keys and n in valid:hits[n].add(src)
  if key=='hq_id' and n in hq:hits[hq[n]].add(src)
for folder in ['outputs','research','configs']:
 for p in sorted((ROOT/folder).rglob('*.json')):
  rel=p.relative_to(ROOT).as_posix()
  if 'refldm_official_benchmark' in rel or 'presentation_build' in rel:continue
  try:
   data=p.read_bytes();walk(json.loads(data.decode('utf-8-sig')),rel)
   files.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest()})
  except Exception as e:errors.append({'path':rel,'error':str(e)[:200]})
audit=json.loads((CACHE/'overlap_audit.json').read_text());rows=[]
for r in audit['records']:
 r=dict(r);known={r['identity'],*r['reference_identities']} & set(hits)
 r['history_referenced_identities']=sorted(known);r['completed_record_identities']=sorted({r['identity'],*r['reference_identities']} & set(measured));rows.append(r)
summary={'scope':'Conservative JSON metadata-reference audit, not proof of image exposure. Includes candidate/exclusion lists. Does not scan arbitrary code, logs, CSVs or external histories.', 'json_files_scanned':len(files),'parse_errors':errors,'identity_labels_referenced':len(hits),'benchmark_rows':len(rows),'rows_with_history_reference':sum(bool(r['history_referenced_identities']) for r in rows),'rows_without_history_reference':sum(not r['history_referenced_identities'] for r in rows),'completed_record_identity_labels':len(measured),'rows_with_completed_record_identity':sum(bool(r['completed_record_identities']) for r in rows),'images_decoded':0}
(CACHE/'history_audit.json').write_text(json.dumps({'summary':summary,'files':files,'identity_sources':{k:sorted(v) for k,v in hits.items()},'completed_record_sources':{k:sorted(v) for k,v in measured.items()},'records':rows},indent=2))
(ROOT/'research/refldm_official_history_audit_v1.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
