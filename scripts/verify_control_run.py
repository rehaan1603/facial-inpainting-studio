"""Audit completed control outputs and unchanged controls against the v2 run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['lama','resshift'],required=True);args=p.parse_args()
    model=args.backbone;out=ROOT/'outputs'/f'control_sweep_{model}'
    signature=json.loads((out/'run_signature.json').read_text())
    for name,digest in signature.items():
        if name!='backbone':assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    tuning=[r for r in rows if r['partition']=='tuning'];assessment=[r for r in rows if r['partition']=='assessment']
    assert len(tuning)==20160 and len(assessment)==720
    ids={part:{r['identity'] for r in rows if r['partition']==part} for part in ['tuning','assessment']}
    assert len(ids['tuning'])==len(ids['assessment'])==48 and not ids['tuning']&ids['assessment']
    key=lambda r:(r['hq_id'],r['family'],r['condition'],r['radius'])
    assert len({(*key(r),r['feather']) for r in rows})==len(rows)
    chosen=json.loads((out/'selection.json').read_text())
    assert all(int(r['radius'])==chosen['radius'] and int(r['feather'])==chosen['feather'] for r in assessment)
    oldrun='benchmark_v2' if model=='lama' else 'benchmark_v2_resshift'
    original={key(r):r for r in csv.DictReader((ROOT/'outputs'/oldrun/'metrics.csv').open()) if r['partition']=='tuning' and r['radius']!='oracle'}
    unchanged=[r for r in tuning if r['feather']=='0' and r['radius'] in ['0','2','4','8']]
    assert len(unchanged)==2880
    errors={m:max(abs(float(r[m])-float(original[key(r)][m])) for r in unchanged) for m in ['full_face_lpips','hole_mae','visible_mae']}
    assert errors['full_face_lpips']<1e-5 and errors['hole_mae']<1e-6 and errors['visible_mae']<1e-6,errors
    assert all(np.isfinite(float(r[m])) for r in rows for m in ['full_face_lpips','hole_mae','visible_mae'])
    report={'backbone':model,'tuning_identities':48,'assessment_identities':48,'tuning_rows':len(tuning),'assessment_rows':len(assessment),'unchanged_controls_checked':len(unchanged),'max_absolute_reproduction_errors':errors,'all_checks_passed':True,'metrics_sha256':hashlib.sha256((out/'metrics.csv').read_bytes()).hexdigest()}
    (ROOT/'research'/f'control_verification_{model}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
