"""Run the unchanged hash-pinned masked scorer in the preserved CPU environment.

The scorer's exact output is retained separately before adding pilot-specific
provenance, coverage and failure counts. Cached, hash-verified control scores are
reused, and every candidate receives the same evaluator and region definitions.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/masked_sdxl_compatibility_pilot_eval_v2'
KEYS = ['hole_psnr', 'hole_ssim', 'hole_lpips', 'eye_lpips', 'facenet_cosine', 'arcface_conditioning_cosine']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def add_coverage(rows, conditions):
    coverage = {}
    for condition in conditions:
        subset = [r for r in rows if r['condition'] == condition]
        complete = [r for r in subset if r['status'] == 'complete']
        metric_counts = {k: dict(valid=sum(r.get('metrics', {}).get(k) is not None for r in subset),
                                 missing=sum(r.get('metrics', {}).get(k) is None for r in subset), expected=3)
                         for k in KEYS}
        detections = {}
        for family in ['facenet', 'arcface_conditioning']:
            detections[family] = {
                field: dict(Counter(r.get(field, {}).get(family, {}).get('status', 'unmeasured') for r in subset))
                for field in ['target_detections', 'output_detections']}
        changed = [r.get('metrics', {}).get('visible_changed') for r in complete]
        coverage[condition] = dict(expected=3, records=len(subset), complete=len(complete),
            generation_attempted=sum(bool(r.get('attempted_generation')) for r in subset),
            reused=sum(bool(r.get('reused')) for r in subset),
            failed=sum(r['status'] == 'failed' for r in subset),
            blocked=sum(r['status'] == 'blocked' for r in subset),
            status_counts=dict(Counter(r['status'] for r in subset)), metric_counts=metric_counts,
            detections=detections,
            visible_preservation=dict(measured=sum(v is not None for v in changed),
                                      exact=sum(v == 0 for v in changed),
                                      changed_pixels_total=sum(v for v in changed if v is not None)))
    return coverage


def compare_supports(rows):
    lookup = {(r['identity'], r['condition']): r for r in rows}
    matches = {}
    fields = ['pixels', 'ssim_valid_centres', 'bbox_xyxy', 'bbox_lpips_status']
    for ident in ['263', '6681', '10149']:
        geometry = {}
        for arm in ['control3', 'local', 'trained']:
            row = lookup.get((ident, arm), {})
            if row.get('status') != 'complete':
                geometry[arm] = None
            else:
                geometry[arm] = {name: {key: region.get(key) for key in fields}
                                 for name, region in row.get('regions', {}).items()}
        available = all(geometry[a] for a in geometry)
        matches[ident] = dict(matched=bool(available and geometry['control3'] == geometry['local'] == geometry['trained']),
                              geometry=geometry)
    return dict(all_matched=all(v['matched'] for v in matches.values()), people=matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    if sha(ROOT / 'scripts/score_masked_sdxl_v1.py') != cfg['scorer_sha256']:
        raise ValueError('Frozen masked scorer source pin changed')
    if sha(__file__) != cfg['scorer_wrapper_sha256']:
        raise ValueError('Pilot scorer wrapper changed after generation was frozen')
    for pin in cfg['metric_source_pins']:
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('Downstream metric source or evaluator provenance changed')
    from score_target_compatibility_v1 import components
    cases = {c['identity']: c for c in json.loads((ROOT / 'outputs/masked_task_v1/cases.json').read_text(encoding='utf-8'))
             if c['location'] == 'eyes' and c['identity'] in ['263', '6681', '10149']}
    component_sources = {ident: components(c['hq_id'], (512, 512))[1] for ident, c in cases.items()}
    import score_masked_sdxl_v1 as scorer
    scorer.OUT = output
    scorer.main()
    frozen_path = output / 'summary_frozen_scorer.json'
    # Preserve the exact byte content produced by the unchanged scorer.
    frozen_bytes = (output / 'summary.json').read_bytes()
    if frozen_path.exists() and frozen_path.read_bytes() != frozen_bytes:
        raise ValueError('Frozen scorer summary changed; prior score snapshot retained')
    frozen_path.write_bytes(frozen_bytes)
    result = json.loads(frozen_bytes)
    after_components = {ident: components(c['hq_id'], (512, 512))[1] for ident, c in cases.items()}
    if component_sources != after_components:
        raise ValueError('Evaluation annotation files changed while scoring')
    result.update(frozen_scorer_output_sha256=sha(frozen_path), scorer_wrapper_sha256=sha(__file__),
        evaluation_original_access='CPU metric evaluator only; generation did not open evaluation originals',
        training=dict(updates=cfg['training_updates'], resolution=cfg['training_resolution'],
                      trainable_parameters=cfg['trainable_parameters'],
                      receipt_path=cfg['training_receipt_path'], receipt_sha256=cfg['training_receipt_sha256']),
        coverage=add_coverage(result['rows'], cfg['conditions']),
        matched_region_supports=compare_supports(result['rows']),
        evaluation_component_sources=component_sources,
        limitations=['Three fixed development identities and inference seed17; no generalization or accuracy percentage claim',
                     'Trained checkpoint is a single fixed 64-update, 512-resolution pilot evaluated at 1024 and exported at 512',
                     'Hole SSIM uses fully contained 11x11 windows; hole/eye LPIPS uses fixed contextual crops',
                     'ArcFace belongs to the generation conditioning family; FaceNet is separate from conditioning',
                     'Visible RGB is preserved by final compositing; retained raw outputs may differ there',
                     'Numerical gate and completed visual review are both required before proposing replication; no website promotion'])
    save(output / 'summary.json', result)
    save(output / 'scoring_receipt.json', dict(status='complete', device='cpu', cuda_visible_devices='-1',
        config_sha256=sha(output / 'config.json'), scorer_sha256=cfg['scorer_sha256'],
        frozen_scorer_output_sha256=sha(frozen_path), summary_sha256=sha(output / 'summary.json'),
        coverage=result['coverage'], matched_region_supports=result['matched_region_supports'],
        evaluation_component_sources=component_sources))


if __name__ == '__main__':
    main()
