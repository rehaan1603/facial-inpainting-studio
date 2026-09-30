"""Immutable numerical report and local visual sheets for structure transport."""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_statistics import identity_means, contrast, holm
from evaluate_structure_transport import BASE, PROTOCOL, ORIGIN, METRICS, FROZEN_SOURCES, assert_frozen, validate_rows


def finite(value):
    return bool(isinstance(value, (int, float)) and np.isfinite(value))


def summary(group, seeds):
    result = {'rows': len(group), 'generation_complete': sum(r['status'] == 'complete' for r in group),
              'evaluation_complete': sum(r['evaluation']['status'] == 'complete' for r in group)}
    for metric in METRICS:
        values, excluded = identity_means(group, metric, seeds)
        result[metric] = {'mean': float(np.mean(list(values.values()))) if values else None,
                          'identity_count': len(values), 'excluded_identities': excluded,
                          'valid_row_count': sum(finite(r['evaluation'].get('metrics', {}).get(metric)) for r in group)}
    return result


def compute_gate(groups, seeds):
    """Frozen thresholds; means require all four identical identity supports."""
    all_ids = {r['identity'] for r in groups['candidate']}
    assert len(all_ids) == 4

    def values(arm, metric, seed=None):
        selected = [r for r in groups[arm] if seed is None or r['seed'] == seed]
        v, _ = identity_means(selected, metric, seeds if seed is None else [seed])
        return v if set(v) == all_ids else None

    def pair(arm, metric, seed=None):
        a, b = values('candidate', metric, seed), values(arm, metric, seed)
        return [float(np.mean(list(x.values()))) for x in [a, b]] if a is not None and b is not None else [None, None]

    checks = []
    gate_metrics = ['facenet_cosine', 'facenet_gallery_cosine', 'hole_mae', 'lpips', 'ssim_rgb', 'psnr_rgb', 'structure_nme']
    for seed in seeds:
        for control in ['scaffold', 'reference_context']:
            pairs = {m: pair(control, m, seed) for m in gate_metrics}
            tests = {'full_paired_support': all(all(finite(v) for v in p) for p in pairs.values())}
            if tests['full_paired_support']:
                a = {m: p[0] for m, p in pairs.items()}
                b = {m: p[1] for m, p in pairs.items()}
                tests.update(facenet_gain_0_005=a['facenet_cosine'] >= b['facenet_cosine'] + .005,
                             gallery_loss_at_most_0_002=a['facenet_gallery_cosine'] >= b['facenet_gallery_cosine'] - .002,
                             hole_mae_regression_at_most_1pct=a['hole_mae'] <= 1.01 * b['hole_mae'],
                             lpips_regression_at_most_1pct=a['lpips'] <= 1.01 * b['lpips'],
                             ssim_loss_at_most_0_005=a['ssim_rgb'] >= b['ssim_rgb'] - .005,
                             psnr_loss_at_most_0_1db=a['psnr_rgb'] >= b['psnr_rgb'] - .1)
                tests['structure_reduction_at_least_5pct' if control == 'scaffold' else 'structure_regression_at_most_1pct'] = (
                    a['structure_nme'] <= (.95 if control == 'scaffold' else 1.01) * b['structure_nme'])
            checks.append({'seed': seed, 'control': control, 'candidate_control_means': pairs,
                           'checks': tests, 'passed': all(tests.values())})
        paired = [values(a, m, seed) for a, m in [('candidate', 'facenet_cosine'), ('scaffold', 'facenet_cosine'),
                                                  ('candidate', 'structure_nme'), ('scaffold', 'structure_nme')]]
        improved = [] if any(x is None for x in paired) else [i for i in sorted(all_ids)
                    if paired[0][i] > paired[1][i] and paired[2][i] < paired[3][i]]
        checks.append({'seed': seed, 'control': 'scaffold', 'check': 'at_least_3_of_4_improve_identity_and_structure',
                       'improved_identities': improved, 'passed': len(improved) >= 3})
    for seed in seeds:
        for control in ['single_reference', 'wrong_identity']:
            pairs = {m: pair(control, m, seed) for m in ['facenet_cosine', 'hole_mae', 'lpips']}
            tests = {'full_paired_support': all(all(finite(v) for v in p) for p in pairs.values())}
            if tests['full_paired_support']:
                tests.update(facenet_gain_0_002=pairs['facenet_cosine'][0] >= pairs['facenet_cosine'][1] + .002,
                             hole_mae_regression_at_most_1pct=pairs['hole_mae'][0] <= 1.01 * pairs['hole_mae'][1],
                             lpips_regression_at_most_1pct=pairs['lpips'][0] <= 1.01 * pairs['lpips'][1])
            checks.append({'seed': seed, 'control': control,
                           'candidate_control_means': pairs, 'checks': tests, 'passed': all(tests.values())})
    reconstructed = [r for a, rows in groups.items() if a != 'observed' for r in rows]
    coverage_failures = []
    for r in reconstructed:
        e, reasons = r['evaluation'], []
        m = e.get('metrics', {})
        if r['status'] != 'complete': reasons.append('generation_failed')
        if e['status'] != 'complete': reasons.append('evaluation_failed')
        reasons += ['missing_metric:' + name for name in METRICS if not finite(m.get(name))]
        for detector in ['facenet', 'arcface_conditioning']:
            if m.get('detections', {}).get(detector, {}).get('status') != 'ok': reasons.append('output_detection:' + detector)
            if e.get('target_detections', {}).get(detector, {}).get('status') != 'ok': reasons.append('target_detection:' + detector)
            if m.get(detector + '_gallery_valid_count') != 3: reasons.append('gallery_coverage:' + detector)
            galleries = e.get('gallery_detections', [])
            if len(galleries) != 3 or any(g.get(detector, {}).get('status') != 'ok' for g in galleries):
                reasons.append('gallery_detection:' + detector)
        if e.get('structure', {}).get('status') != 'ok': reasons.append('structural_metric_failed')
        if m.get('known_pixels_unchanged') is not True: reasons.append('preservation_unverified')
        if reasons: coverage_failures.append({'key': r['key'], 'reasons': reasons})
    complete = not coverage_failures
    return {'passed_numerical_gate': complete and all(c['passed'] for c in checks),
            'full_reconstructed_arm_coverage': complete, 'coverage_failures': coverage_failures,
            'checks': checks, 'visual_review_status': 'pending', 'visual_review_required': True,
            'progression_authorized': False,
            'interpretation': 'A numerical pass needs separate visual review and prospective independent confirmation; it does not establish novelty.'}


def safe_detection(value):
    return {k: v for k, v in value.items() if k in ['status', 'face_count']}


def safe_row(row):
    # Allowlist: never publish coordinates, images, image paths, or raw tracebacks.
    result = {k: row[k] for k in ['key', 'case_id', 'identity', 'seed', 'mode', 'status', 'output_sha256', 'reused_control'] if k in row}
    e = row['evaluation']
    score = {'status': e['status']}
    if 'metrics' in e:
        score['metrics'] = {m: e['metrics'].get(m) for m in METRICS}
        score['metrics'].update({k: e['metrics'].get(k) for k in ['facenet_gallery_valid_count', 'arcface_conditioning_gallery_valid_count', 'known_pixels_unchanged']})
        score['output_detections'] = {k: safe_detection(v) for k, v in e['metrics'].get('detections', {}).items()}
        score['target_detections'] = {k: safe_detection(v) for k, v in e.get('target_detections', {}).items()}
        score['gallery_detections'] = [{k: safe_detection(v) for k, v in g.items()} for g in e.get('gallery_detections', [])]
        structural = e.get('structure', {})
        score['structure'] = {k: structural.get(k) for k in ['status', 'missing_landmark_count']}
        score['structure'].update({k: safe_detection(structural.get(k, {})) for k in ['target', 'output']})
    score['error_recorded_locally'] = 'error' in e
    score['reused_same_case_output_metrics'] = e.get('reused_same_case_output_metrics', False)
    result['generation_error_recorded_locally'] = 'error' in row
    result['evaluation'] = score
    return result


def failures(rows):
    result = []
    for row in rows:
        if row['status'] != 'complete': result.append({'key': row['key'], 'stage': 'generation', 'status': row['status']})
        e = row['evaluation']
        if e['status'] != 'complete': result.append({'key': row['key'], 'stage': 'evaluation', 'status': e['status']})
        sources = [('output', e.get('metrics', {}).get('detections', {})), ('target', e.get('target_detections', {}))]
        sources += [('gallery_' + str(i), g) for i, g in enumerate(e.get('gallery_detections', []))]
        for source, detectors in sources:
            for detector, status in detectors.items():
                if status.get('status') != 'ok': result.append({'key': row['key'], 'stage': source, 'detector': detector, **safe_detection(status)})
        if e.get('structure', {}).get('status') != 'ok':
            result.append({'key': row['key'], 'stage': 'structural_metric', 'status': e.get('structure', {}).get('status', 'not_evaluated')})
        if e['status'] == 'complete':
            result += [{'key': row['key'], 'stage': 'metric', 'metric': m, 'status': 'missing_or_nonfinite'} for m in METRICS if not finite(e.get('metrics', {}).get(m))]
    return result


def public_signature(signature):
    safe = {k: v for k, v in signature.items() if k.endswith('_sha256') or k == 'frozen_at_utc'}
    safe['model_files'] = {Path(k).name: v for k, v in signature['model_files'].items()}
    return safe


def contact_sheets(rows, cfg, evaluation):
    lookup = {(r['case_id'], r['seed'], r['mode']): r for r in rows}
    for cid in cfg['cases']:
        for seed in cfg['seeds']:
            target = evaluation[cid]['target']
            assert sha(target['path']) == target['sha256']
            labels = [('clean scoring target', target['path'])]
            labels += [(arm, lookup[cid, seed, arm].get('output') if lookup[cid, seed, arm]['status'] == 'complete' else None) for arm in cfg['arms']]
            assert len(labels) == 9
            sheet = Image.new('RGB', (3 * 256, 3 * 280), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, (label, path) in enumerate(labels):
                x, y = index % 3 * 256, index // 3 * 280
                draw.text((x + 4, y + 4), label, fill='black')
                if path:
                    with Image.open(path) as image:
                        sheet.paste(image.convert('RGB').resize((256, 256)), (x, y + 24))
                else:
                    draw.text((x + 8, y + 80), 'FAILED / NO OUTPUT', fill='red')
            dest = BASE / 'review' / f'{cid}_s{seed}.png'
            dest.parent.mkdir(exist_ok=True)
            with dest.open('xb') as stream:
                sheet.save(stream, format='PNG')


def main():
    cfg = json.loads(PROTOCOL.read_text())
    signature = assert_frozen()
    comparison = json.loads((BASE / 'comparison.json').read_text())
    validate_rows(comparison, cfg)
    es = json.loads((BASE / 'evaluation_signature.json').read_text())
    for name, path in FROZEN_SOURCES.items(): assert es[name + '_sha256'] == sha(path)
    assert es['generation_signature_sha256'] == sha(BASE / 'signature.json')
    assert es['comparison_sha256'] == sha(BASE / 'comparison.json')
    assert es['evaluation_manifest_sha256'] == sha(ORIGIN / 'evaluation_manifest.json')
    assert es['inference_manifest_sha256'] == sha(ORIGIN / 'inference_manifest.json')
    assert es['model_files'] == signature['model_files']
    data = json.loads((BASE / 'evaluation.json').read_text())
    assert data['signature_sha256'] == sha(BASE / 'evaluation_signature.json')
    rows = data['rows']
    assert len(rows) == 64 and len({r['key'] for r in rows}) == 64
    originals = {r['key']: r for r in comparison['rows']}
    for row in rows:
        assert {k: v for k, v in row.items() if k != 'evaluation'} == originals[row['key']]
    inference = {c['case_id']: c for c in json.loads((ORIGIN / 'inference_manifest.json').read_text())['cases']}
    preserved = 0
    for row in rows:
        if row['status'] != 'complete': continue
        c = inference[row['case_id']]
        assert sha(row['output']) == row['output_sha256']
        assert sha(c['observed']) == c['observed_sha256'] and sha(c['mask']) == c['mask_sha256']
        with Image.open(c['observed']) as im: observed = np.asarray(im.convert('RGB'))
        with Image.open(c['mask']) as im: mask = np.asarray(im.convert('L')) >= 128
        with Image.open(row['output']) as im: output = np.asarray(im.convert('RGB'))
        assert observed.shape == output.shape and mask.shape == output.shape[:2]
        assert np.array_equal(observed[~mask], output[~mask])
        preserved += 1
    groups = {arm: [r for r in rows if r['mode'] == arm] for arm in cfg['arms']}
    summaries = {arm: summary(group, cfg['seeds']) for arm, group in groups.items()}
    primary = {f'candidate_minus_{arm}/{metric}': contrast(groups['candidate'], groups[arm], metric, cfg['seeds'])
               for arm in ['scaffold', 'reference_context', 'wrong_identity']
               for metric in ['facenet_cosine', 'hole_mae', 'structure_nme']}
    assert len(primary) == 9
    holm(primary)
    gate = compute_gate(groups, cfg['seeds'])
    all_failures = failures(rows)
    new_rows = [r for r in rows if r['mode'] in ['candidate', 'single_reference', 'wrong_identity']]
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Four previously observed development identities, two seed slots; structural feasibility only',
              'new_transform_attempts': len(new_rows), 'new_transforms_complete': sum(r['status'] == 'complete' for r in new_rows),
              'new_neural_generations': 0, 'reused_control_slots': 40,
              'evaluation_complete': sum(r['evaluation']['status'] == 'complete' for r in rows),
              'pixel_preservation_verified_rows': preserved, 'summaries': summaries,
              'primary_contrasts': primary, 'progression_gate': gate, 'failures': all_failures,
              'protocol_sha256': sha(PROTOCOL), 'generation_signature_sha256': sha(BASE / 'signature.json'),
              'evaluation_sha256': sha(BASE / 'evaluation.json'), 'reporter_sha256': sha(__file__),
              'statistics_sha256': sha(ROOT / 'src/preservation/context_statistics.py'),
              'statistical_unit': 'Identity; both seed scores required and averaged; deterministic repeated controls do not add sample size',
              'structural_metric_limit': 'Separate 106-point learned proxy, same vendor/detector family; no anatomical-truth claim',
              'visual_review_status': 'pending', 'final_test_used': False, 'novelty_established': False}
    write_new(ROOT / 'research/structure_transport_results_v1.json', result)
    safe = [safe_row(r) for r in rows]
    write_new(ROOT / 'research/structure_transport_evidence_v1.json',
              {'signature': public_signature(signature), 'evaluation_signature': public_signature(es), 'rows': safe})
    with (ROOT / 'research/structure_transport_metrics_v1.csv').open('x', newline='', encoding='utf-8') as stream:
        fields = ['case_id', 'identity', 'seed', 'mode', 'generation_status', 'evaluation_status'] + METRICS
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields[:4]}, 'generation_status': row['status'],
                             'evaluation_status': row['evaluation']['status'],
                             **{m: row['evaluation'].get('metrics', {}).get(m) for m in METRICS}})
    fmt = lambda value: 'NA' if value is None else f'{value:.6f}'
    lines = ['# Structure-only reference transport: feasibility results', '',
             '**Numerical progression gate: ' + ('passed; visual review pending' if gate['passed_numerical_gate'] else 'failed') + '. Novelty is not established.**', '',
             'This frozen screen transports coordinates from supplied references into fixed ResShift reconstructions. No reference texture is transferred and no new neural generation or training occurs. Four previously observed identities and seeds 17/29 are development diagnostics, not an independent or reserved-final test.', '',
             f"{result['new_transforms_complete']}/24 structural transforms completed; 40 control slots reused; {result['evaluation_complete']}/64 rows evaluated. All {preserved} completed outputs were separately checked for exact visible-pixel preservation. Every failed attempt and missing metric is retained. Visual review remains pending.", '',
             '| Arm | FaceNet ↑ (n) | Gallery ↑ (n) | Hole MAE ↓ (n) | LPIPS ↓ (n) | Structure NME ↓ (n) |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, s in summaries.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(f"{fmt(s[m]['mean'])} ({s[m]['identity_count']})" for m in ['facenet_cosine', 'facenet_gallery_cosine', 'hole_mae', 'lpips', 'structure_nme']) + ' |')
    lines += ['', 'Means require both seeds for each included identity. Every metric has its own explicitly reported denominator; progression checks require the same complete support of four identities. Full PSNR, SSIM, diagnostic ArcFace, NIQE, BRISQUE, visible error, coverage, and failure records are in the JSON/CSV.', '',
              '| Prespecified contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |',
              '|---|---:|---:|---|---:|']
    for name, c in primary.items(): lines.append(f"| {name} | {c['identities']} | {fmt(c['mean_delta'])} | {c['ci95']} | {fmt(c['p_holm'])} |")
    lines += ['', 'Nine primary contrasts use seed-averaged identity differences, exact two-sided sign flips and Holm correction. Four identities cannot establish superiority; the smallest attainable unadjusted exact p-value with four nonzero pairs is 0.125. Bootstrap intervals are unstable. Repeated deterministic RefFace/observation controls are not additional samples.', '',
              'The structural metric uses separate InsightFace 106-point weights on the clean target and output. Only target landmarks whose rounded locations fall in the original missing mask contribute. Euclidean distances are normalized by the target detector’s eye separation; no Procrustes alignment is used. Both images must have exactly one detected face. This learned proxy shares a model vendor/detector family with the 68-point inference system and is not anatomical ground truth. Per-person coordinates and images remain local.', '',
              'The machine-checked gate includes both-seed identity, gallery, pixel/perceptual, SSIM, PSNR and structure thresholds; personal-reference and single-reference ablations; complete reconstructed-arm detection/metric coverage; and exact preservation. A numerical pass alone does not authorize progression: every local comparison sheet needs separate visual review for eyes, spacing, nose, lips, teeth, expression and asymmetry. A later pass permits only a separately frozen independent confirmation plus further novelty review.', '',
              'Reproduce in order after the frozen runner finishes all 64 terminal rows:', '',
              '```powershell',
              r'& C:\Users\rehaa\.cache\facial-inpainting\evaluation_env_v1\Scripts\python.exe -X utf8 scripts/evaluate_structure_transport.py',
              r'& C:\Users\rehaa\.cache\facial-inpainting\evaluation_env_v1\Scripts\python.exe -X utf8 scripts/report_structure_transport.py',
              '```', '',
              'Outputs are immutable: do not rerun into an existing result directory. The scripts verify the before-generation source/model hashes, the complete comparison file, evaluation manifests, original image hashes and exact visible-pixel preservation. Eight reserved identities are excluded; no final model generation/evaluation is performed.']
    with (ROOT / 'research/STRUCTURE_TRANSPORT_RESULTS_V1.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    evaluation = {c['case_id']: c for c in json.loads((ORIGIN / 'evaluation_manifest.json').read_text())['cases']}
    contact_sheets(rows, cfg, evaluation)
    print(json.dumps({'numerical_gate': gate['passed_numerical_gate'], 'evaluated': result['evaluation_complete'],
                      'transforms_complete': result['new_transforms_complete'], 'failure_events': len(all_failures),
                      'visual_review': 'pending'}, indent=2))


if __name__ == '__main__':
    main()
