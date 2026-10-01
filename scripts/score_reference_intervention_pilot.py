"""Offline whole-image scoring, separated from pilot inference."""
import json
from collections import defaultdict
import numpy as np
import torch
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha


def main():
    out = ROOT/'outputs/reference_intervention_pilot_v1'
    dest = out/'evaluation'
    rows = json.loads((dest/'rows.json').read_text())
    if len(rows) != 28:
        raise ValueError('Expected all 28 outputs before scoring')
    output = out/'scores.json'
    if output.exists():
        raise ValueError('Refusing to overwrite scores')
    metrics = Metrics()
    targets, records = {}, []
    for row in rows:
        identity = row['identity']
        if identity not in targets:
            target = read_rgb(out/'images'/identity/'target.png')
            vectors, detections = metrics.features(target)
            targets[identity] = target, vectors, detections
        target, vectors, detections = targets[identity]
        rgb = read_rgb(dest/row['file'], row['sha256'])
        values = metrics.quality(rgb)
        predicted, detected = metrics.features(rgb)
        for name, vector in predicted.items():
            values[name+'_cosine'] = float(np.dot(vector, vectors[name])) if vector is not None and vectors[name] is not None else None
        a, b = rgb.astype(float)/255, target.astype(float)/255
        values['mae'] = float(np.abs(a-b).mean())
        values['psnr'] = float(-10*np.log10(np.square(a-b).mean()))
        values['ssim'] = float(metrics.ssim(a,b,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11))
        with torch.no_grad():
            values['lpips'] = metrics.lpips(metrics.tensor(rgb)*2-1, metrics.tensor(target)*2-1).item()
        records.append(dict(row, metrics=values, target_detections=detections, output_detections=detected))
        print('Scored', len(records), '/28', flush=True)
    grouped = defaultdict(list)
    for row in records:
        grouped[row['arm']+'/'+row['condition']].append(row)
    summary = {}
    for group, entries in grouped.items():
        summary[group] = {}
        for key in ['facenet_cosine','arcface_conditioning_cosine','lpips','mae','psnr','ssim','niqe','brisque']:
            values = [r['metrics'][key] for r in entries if r['metrics'].get(key) is not None]
            summary[group][key] = dict(mean=float(np.mean(values)) if values else None, n=len(values))
    result = dict(scope='Four adapter-held-out identities, one seed and degradation; exploratory only',
                  evaluator_sha256=sha(__file__), metrics_source_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),
                  generation_rows_sha256=sha(dest/'rows.json'), rows=records, summary=summary,
                  final_test_used=False, statistical_significance_claimed=False)
    output.write_text(json.dumps(result, indent=2))
    public = dict(result)
    (ROOT/'research/reference_intervention_pilot_scores_v1.json').write_text(json.dumps(public, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
