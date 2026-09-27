"""Independent frozen metrics for the completed reference/context factorial screen."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from extended_evaluation_metrics import Metrics, read_rgb

BASE = ROOT / 'outputs/scaffold_conditioning_v1'


def main():
    cfg = json.loads((ROOT / 'research/protocols/scaffold_conditioning_v1.json').read_text())
    data = json.loads((BASE / 'comparison.json').read_text())
    assert len(data['rows']) == 40 and len({r['key'] for r in data['rows']}) == 40
    assert all(r['case_id'] in cfg['cases'] and r['identity'] not in cfg['reserved_identities'] for r in data['rows'])
    origin = ROOT / 'outputs/distortion_aware_v1'
    inference = {c['case_id']: c for c in json.loads((origin / 'inference_manifest.json').read_text())['cases']}
    evaluation = {c['case_id']: c for c in json.loads((origin / 'evaluation_manifest.json').read_text())['cases']}
    signature = {'comparison_sha256': sha(BASE / 'comparison.json'), 'evaluator_sha256': sha(__file__),
                 'metric_source_sha256': sha(ROOT / 'scripts/extended_evaluation_metrics.py'),
                 'evaluation_manifest_sha256': sha(origin / 'evaluation_manifest.json')}
    write_new(BASE / 'evaluation_signature.json', signature)
    import cv2
    set_threads = cv2.setNumThreads; cv2.setNumThreads = lambda n: set_threads(min(int(n), 1)); cv2.setNumThreads(1)
    metrics, features, rows = Metrics(), {}, []
    for row in data['rows']:
        score = {'status': 'failed'}
        try:
            if row['status'] != 'complete': raise ValueError(row.get('error', 'Generation failed'))
            c, e = inference[row['case_id']], evaluation[row['case_id']]
            if row['identity'] not in features:
                target = read_rgb(e['target']['path'], e['target']['sha256']); tv, td = metrics.features(target)
                galleries = [metrics.features(read_rgb(g['path'], g['sha256'])) for g in e['gallery']]
                features[row['identity']] = target, tv, td, galleries
            target, tv, td, galleries = features[row['identity']]
            output = read_rgb(row['output'], row['output_sha256'])
            observed = read_rgb(c['observed'], c['observed_sha256'])
            mask = read_rgb(c['mask'], c['mask_sha256'])[:, :, 0] >= 128
            values = metrics.score(output, target, observed, mask, tv)
            values['known_pixels_unchanged'] = bool(np.array_equal(output[~mask], observed[~mask]))
            ov, _ = metrics.features(output)
            for name in ['facenet', 'arcface_conditioning']:
                sims = [float(np.dot(ov[name], v[name])) for v, s in galleries if v[name] is not None and ov[name] is not None]
                values[name+'_gallery_cosine'] = float(np.mean(sims)) if sims else None
                values[name+'_gallery_valid_count'] = len(sims)
            score = {'status': 'complete', 'metrics': values, 'target_detections': td,
                     'gallery_detections': [s for v, s in galleries], 'output_sha256': row['output_sha256']}
        except Exception as error:
            score['error'] = f'{type(error).__name__}: {error}'
        write_new(BASE / 'scores' / (row['key']+'.json'), score)
        rows.append(dict(row, evaluation=score)); print('REFERENCE CONTEXT SCORE', len(rows), '/40', row['key'], score['status'], flush=True)
    write_new(BASE / 'evaluation.json', {'signature_sha256': sha(BASE / 'evaluation_signature.json'), 'rows': rows, 'final_test_used': False})


if __name__ == '__main__': main()
