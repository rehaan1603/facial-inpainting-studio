"""Evaluate immutable context-support outputs; clean data enter only this process."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from extended_evaluation_metrics import Metrics, read_rgb

BASE = ROOT / 'outputs/context_support_v1'


def main():
    data = json.loads((BASE / 'comparison.json').read_text())
    cfg = json.loads((ROOT / 'research/protocols/context_support_v1.json').read_text())
    assert len(data['rows']) == len(cfg['cases']) * len(cfg['seeds']) * len(cfg['arms'])
    assert all(r['case_id'] in cfg['cases'] and r['identity'] not in cfg['reserved_identities'] for r in data['rows'])
    manifest = ROOT / 'outputs/distortion_aware_v1'
    cases = {c['case_id']: c for c in json.loads((manifest / 'inference_manifest.json').read_text())['cases']}
    evaluation = {c['case_id']: c for c in json.loads((manifest / 'evaluation_manifest.json').read_text())['cases']}
    signature = {'comparison_sha256': sha(BASE / 'comparison.json'), 'evaluator_sha256': sha(__file__),
                 'metric_source_sha256': sha(ROOT / 'scripts/extended_evaluation_metrics.py'),
                 'evaluation_manifest_sha256': sha(manifest / 'evaluation_manifest.json')}
    lock = BASE / 'evaluation_signature.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == signature
    else:
        write_new(lock, signature)
    if (BASE / 'evaluation.json').exists():
        raise FileExistsError('Completed evaluation is immutable')
    import cv2
    set_threads = cv2.setNumThreads
    cv2.setNumThreads = lambda n: set_threads(min(int(n), 1))
    cv2.setNumThreads(1)
    metrics = Metrics()
    features, scored_cache, rows = {}, {}, []
    for row in data['rows']:
        dest = BASE / 'scores' / (row['key'] + '.json')
        if dest.exists():
            score = json.loads(dest.read_text())
        else:
            score = {'status': 'failed'}
            try:
                if row['status'] != 'complete':
                    raise ValueError(row.get('error', 'Generation failed'))
                c, e = cases[row['case_id']], evaluation[row['case_id']]
                cache_key = (row['case_id'], row['output_sha256'])
                assert sha(row['output']) == row['output_sha256']
                if cache_key in scored_cache:
                    source_key, previous = scored_cache[cache_key]
                    score = dict(previous, reused_identical_output_score=source_key)
                else:
                    if row['identity'] not in features:
                        target = read_rgb(e['target']['path'], e['target']['sha256'])
                        tv, td = metrics.features(target)
                        gallery = [metrics.features(read_rgb(g['path'], g['sha256'])) for g in e['gallery']]
                        features[row['identity']] = (target, tv, td, gallery)
                    target, tv, td, gallery = features[row['identity']]
                    output = read_rgb(row['output'], row['output_sha256'])
                    observed = read_rgb(c['observed'], c['observed_sha256'])
                    mask = read_rgb(c['mask'], c['mask_sha256'])[:, :, 0] >= 128
                    values = metrics.score(output, target, observed, mask, tv)
                    values['known_pixels_unchanged'] = bool(np.array_equal(output[~mask], observed[~mask]))
                    ov, _ = metrics.features(output)
                    for name in ['facenet', 'arcface_conditioning']:
                        sims = [float(np.dot(ov[name], v[name])) for v, s in gallery if v[name] is not None and ov[name] is not None]
                        values[name + '_gallery_cosine'] = float(np.mean(sims)) if sims else None
                        values[name + '_gallery_valid_count'] = len(sims)
                    score = {'status': 'complete', 'metrics': values, 'target_detections': td,
                             'gallery_detections': [s for v, s in gallery], 'output_sha256': row['output_sha256']}
                scored_cache[cache_key] = (row['key'], score)
            except Exception as error:
                score['error'] = f'{type(error).__name__}: {error}'
            write_new(dest, score)
        if score['status'] == 'complete':
            scored_cache[(row['case_id'], row['output_sha256'])] = (row['key'], score)
        rows.append(dict(row, evaluation=score))
        print(f"SCORE {len(rows)}/{len(data['rows'])} {row['key']} {score['status']}", flush=True)
    write_new(BASE / 'evaluation.json', {'signature_sha256': sha(lock), 'rows': rows, 'final_test_used': False})


if __name__ == '__main__':
    main()
