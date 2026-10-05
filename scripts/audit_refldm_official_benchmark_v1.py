"""Audit official benchmark membership without opening any image pixels."""
import ast,csv,json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_official_benchmark_v1')
config=json.loads((ROOT/'configs/local.json').read_text())
identity_path=Path(config['annotations'])/'identity_CelebA.txt'
ids=dict(line.split() for line in identity_path.read_text().splitlines())
map_path=Path(config['hq'])/'CelebA-HQ-to-CelebA-mapping.txt'
hq={str(int(parts[0])):ids[parts[2]] for line in map_path.read_text().splitlines()[1:] if (parts:=line.split())}
pilot=json.loads((ROOT/'outputs/reference_intervention_pilot_v1/manifest.json').read_text())
train={x['identity'] for x in pilot['cases'] if x['role']=='train'}
dev={x['identity'] for x in pilot['cases'] if x['role']!='train'}
excluded=set(pilot['excluded_identities'])
source=CACHE/'metadata/CelebA-Test-Ref/test_references.csv'
counts=Counter();records=[]
for row in csv.DictReader(source.open()):
 target=str(int(Path(row['gt_image']).stem));refs=ast.literal_eval(row['ref_image'])
 assert isinstance(refs,list) and all(isinstance(x,str) for x in refs)
 refids=[hq[str(int(Path(x).stem))] for x in refs];identity=hq[target];allids={identity,*refids}
 flags=[]
 for label,group in [('pilot_training_overlap',train),('pilot_development_overlap',dev),('pilot_exclusion_overlap',excluded)]:
  if allids & group:flags.append(label)
 mismatch=sum(x!=identity for x in refids)
 if mismatch:flags.append('official_reference_identity_disagreement')
 missing=[x for x in refs if not (Path(config['hq'])/'CelebA-HQ-img'/f'{int(Path(x).stem)}.jpg').exists()]
 if missing:flags.append('missing_local_reference')
 counts.update(flags)
 records.append(dict(target=row['gt_image'],lq=row['lq_image'],identity=identity,references=refs,reference_identities=refids,flags=flags,reference_identity_disagreements=mismatch))
receipt=dict(scope='Metadata-only audit against current pilot train/development/exclusion identities; not a complete historical exposure audit',cases=len(records),target_identities=len({r['identity'] for r in records}),flag_counts=dict(counts),without_flags=sum(not r['flags'] for r in records),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),excluded_identity_count=len(excluded),images_decoded=0,records=records)
(CACHE/'overlap_audit.json').write_text(json.dumps(receipt,indent=2))
public={k:v for k,v in receipt.items() if k!='records'}
(ROOT/'research/refldm_official_benchmark_overlap_v1.json').write_text(json.dumps(public,indent=2))
print(json.dumps(public,indent=2))
