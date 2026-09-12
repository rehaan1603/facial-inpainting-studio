"""Quarantine every member of an exact-file duplicate group; never move source data."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data/manifests'
    audit=json.loads((ROOT/'research/data_audit.json').read_text())
    excluded={p for group in audit['exact_file_duplicate_groups'] for p in group}
    checks=[json.loads(line) for line in (folder/'image_integrity.jsonl').read_text().splitlines()]
    hashes={r['path']:r['sha256'] for r in checks if 'sha256' in r}
    report={'policy':'Quarantine every member of every exact-file duplicate group across both sources; preserve original official manifests.',
            'datasets':{},'quarantined_paths':sorted(excluded)}
    all_clean=[]
    for name in ['celebahq','lapa']:
        rows=list(csv.DictReader((folder/f'{name}.csv').open()))
        clean=[r for r in rows if r['image_path'] not in excluded]
        for r in clean: r['source_sha256']=hashes[r['image_path']]
        target=folder/f'{name}_clean.csv'
        with target.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=clean[0].keys()); writer.writeheader(); writer.writerows(clean)
        report['datasets'][name]={'before':len(rows),'after':len(clean),'removed':len(rows)-len(clean),
            'splits':dict(Counter(r['split'] for r in clean)),
            'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        if name=='celebahq':
            sets={s:{r['identity'] for r in clean if r['split']==s} for s in ['train','val','test']}
            assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
        all_clean.extend(clean)
    assert len({r['source_sha256'] for r in all_clean})==len(all_clean)
    report['limitations']=['Exact file hashes only; perceptual duplicates and identity annotation mistakes may remain.',
                          'Clean manifests differ from official benchmark splits; report this explicitly.',
                          'Pretrained-model training data exposure is not resolved by deduplication.']
    (ROOT/'research/clean_manifest_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['datasets'],indent=2))


if __name__=='__main__': main()
