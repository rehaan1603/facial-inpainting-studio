"""Metadata-only pilot selection; no reserved or validation images are decoded."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from reference_intervention_core import ROOT, sha


def main():
    out = ROOT / 'outputs/reference_intervention_pilot_v1'
    out.mkdir(exist_ok=False)
    source = ROOT / 'data/manifests/celebahq_reviewed_v2.csv'
    rows = list(csv.DictReader(source.open()))
    protocol = json.loads((ROOT / 'research/protocols/unseen_identity_protocol_v1.json').read_text())
    excluded = {str(c['identity']) for c in protocol['cases']}
    excluded |= {r['identity'] for r in rows if r['split'] != 'train'}
    fp_path = ROOT / 'outputs/near_duplicate_audit/fingerprints.jsonl'
    fp = {str(r['hq_id']): r for r in map(json.loads, fp_path.read_text().splitlines()) if r['dataset'] == 'celebahq'}
    protected = [f for f in fp.values() if str(f['identity']) in excluded]
    protected_hash = {f['decoded_rgb_sha256'] for f in protected}
    protected_ph = [int(f['phash'], 16) for f in protected]
    groups = defaultdict(list)
    for r in rows:
        if r['split'] == 'train' and r['identity'] not in excluded:
            groups[r['identity']].append(r)
    rank = lambda value: hashlib.sha256(('intervention-pilot-v1:' + value).encode()).hexdigest()
    cases, retained = [], []
    for identity in sorted(groups, key=rank):
        selected = []
        for row in sorted(groups[identity], key=lambda r: rank(r['hq_id'])):
            f = fp[row['hq_id']]
            if f['decoded_rgb_sha256'] in protected_hash:
                continue
            ph = int(f['phash'], 16)
            if any((ph ^ other).bit_count() <= 6 for other in protected_ph + retained + [int(fp[s['hq_id']]['phash'], 16) for s in selected]):
                continue
            selected.append(row)
            if len(selected) == 5:
                break
        if len(selected) == 5:
            retained += [int(fp[s['hq_id']]['phash'], 16) for s in selected]
            cases.append(dict(identity=identity, role='train' if len(cases) < 16 else 'adapter_holdout', target=selected[0], references=selected[1:]))
        if len(cases) == 20:
            break
    assert len(cases) == 20 and len({c['identity'] for c in cases}) == 20
    document = dict(stage='engineering pilot, not final validation', source_sha256=sha(source), fingerprints_sha256=sha(fp_path),
                    cases=cases, excluded_identities=sorted(excluded, key=int), seed=20260930,
                    limitations=['All images from original training partition; earlier project exposure and pretrained overlap possible.',
                                 'Adapter holdout is disjoint from these 16 adapter training identities only.',
                                 'One target and four references per identity; not publication-scale.'],
                    training=dict(steps_per_arm=64, learning_rate=0.001, arms=['reconstruction', 'intervention'],
                                  intervention_weight=0.1, loss='clean and corrupt epsilon MSE; intervention adds corrupt-to-detached-clean consistency',
                                  timesteps=[100, 300, 500, 700], corruption='replace central 256x256 patch in reference zero with next training identity reference zero',
                                  target_degradation='512 to 64 to 512 bicubic plus Gaussian sigma 8/255',
                                  checkpoint_selection='fixed last step, no heldout selection'))
    (out / 'manifest.json').write_text(json.dumps(document, indent=2))
    print(json.dumps({'train_identities':16, 'adapter_holdout_identities':4, 'source_pixels_read':0, 'manifest_sha256':sha(out/'manifest.json')}))


if __name__ == '__main__':
    main()
