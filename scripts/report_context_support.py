"""Report the prespecified mask-support screen without tuning selection from truth."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_statistics import contrast, holm, identity_means
from report_distortion_aware import METRICS

BASE = ROOT / 'outputs/context_support_v1'


def main():
    cfg = json.loads((ROOT / 'research/protocols/context_support_v1.json').read_text())
    generated = json.loads((BASE / 'comparison.json').read_text())
    data = json.loads((BASE / 'evaluation.json').read_text())
    rows = data['rows']
    assert len(rows) == 80 and len({r['key'] for r in rows}) == 80
    for row in rows:
        if row['status'] == 'complete':
            assert sha(row['output']) == row['output_sha256']
    arms = {a: [r for r in rows if r['mode'] == a] for a in cfg['arms']}
    summaries = {}
    for arm, group in arms.items():
        entry = {'rows': len(group), 'scored': sum(r['evaluation']['status']=='complete' for r in group)}
        for metric in METRICS:
            means, excluded = identity_means(group, metric, cfg['seeds'])
            entry[metric] = {'mean': float(np.mean(list(means.values()))) if means else None,
                             'identities': len(means), 'excluded': excluded}
        entry['preservation_pass'] = sum(r['evaluation'].get('metrics', {}).get('known_pixels_unchanged') is True for r in group)
        summaries[arm] = entry
    contrasts = {f'context_selected_minus_{a}/{m}': contrast(arms['context_selected'], arms[a], m, cfg['seeds'])
                 for a in cfg['primary_controls'] for m in cfg['primary_metrics']}
    holm(contrasts)

    def seed_mean(arm, seed, metric):
        values = [r['evaluation'].get('metrics', {}).get(metric) for r in arms[arm] if r['seed'] == seed]
        return float(np.mean(values)) if len(values) == 4 and all(v is not None for v in values) else None

    controls = [f'fixed_r{r}' for r in cfg['radii']] + ['support_mean5', 'seed_mean5']
    gate_rows = []
    for seed in cfg['seeds']:
        for control in controls:
            pair = {m: [seed_mean(a, seed, m) for a in ['context_selected', control]]
                    for m in ['facenet_cosine', 'hole_mae', 'facenet_gallery_cosine', 'lpips']}
            if any(v is None for p in pair.values() for v in p):
                checks = {'coverage': False}
            else:
                checks = {'facenet_gain_at_least_0.005': pair['facenet_cosine'][0] - pair['facenet_cosine'][1] >= .005,
                          'hole_mae_reduction_at_least_2pct': pair['hole_mae'][0] <= .98 * pair['hole_mae'][1],
                          'gallery_noninferiority_0.002': pair['facenet_gallery_cosine'][0] >= pair['facenet_gallery_cosine'][1] - .002,
                          'lpips_noninferiority_1pct': pair['lpips'][0] <= 1.01 * pair['lpips'][1]}
            gate_rows.append({'seed': seed, 'control': control, 'values_candidate_control': pair, 'checks': checks, 'pass': all(checks.values())})
    complete = all(r['status']=='complete' and r['evaluation']['status']=='complete' for r in rows)
    detections = all(r['evaluation'].get('metrics', {}).get('detections', {}).get(n, {}).get('status')=='ok'
                     for r in rows for n in ['facenet', 'arcface_conditioning'])
    gallery_complete = all(r['evaluation'].get('metrics', {}).get(n+'_gallery_valid_count') == 3
                           for r in rows for n in ['facenet', 'arcface_conditioning'])
    selections_ok = all(s['status']=='selected' for s in generated['selections'])
    preservation = all(r['evaluation'].get('metrics', {}).get('known_pixels_unchanged') is True for r in rows)
    passed = complete and detections and gallery_complete and selections_ok and preservation and all(g['pass'] for g in gate_rows)
    correlations = []
    for selection in generated['selections']:
        group = [r for r in rows if r['case_id'] == selection['case_id'] and r['seed'] == selection['seed']]
        probe = [selection['losses'].get(str(r)) for r in cfg['radii']]
        errors = [next(x['evaluation'].get('metrics', {}).get('hole_mae') for x in group if x['mode']==f'fixed_r{r}') for r in cfg['radii']]
        correlation = None
        if all(v is not None for v in probe+errors) and len(set(probe)) > 1 and len(set(errors)) > 1:
            correlation = float(spearmanr(probe, errors).statistic)
        correlations.append({'case_id': selection['case_id'], 'seed': selection['seed'],
                             'probe_losses': probe, 'actual_hole_errors': errors, 'spearman': correlation})
    timing = [g['seconds'] for g in generated['generation_ledger']]
    result = {'date': '2026-09-25', 'previously_observed_identities': 4, 'seeds': cfg['seeds'],
              'generations_attempted': len(generated['generation_ledger']),
              'generations_complete': sum(g['status']=='complete' for g in generated['generation_ledger']),
              'rows': len(rows), 'scored': sum(r['evaluation']['status']=='complete' for r in rows),
              'summaries': summaries, 'primary_contrasts': contrasts,
              'progression_gate': {'passed': bool(passed), 'complete': complete, 'detections_complete': detections,
                  'gallery_complete': gallery_complete, 'no_abstention': selections_ok, 'preservation_exact': preservation,
                  'per_seed_control_checks': gate_rows},
              'selections': generated['selections'], 'probe_hole_correlations': correlations,
              'generation_seconds_total': float(sum(timing)), 'generation_seconds_median': float(np.median(timing)),
              'protocol_sha256': sha(ROOT / 'research/protocols/context_support_v1.json'),
              'evaluation_sha256': sha(BASE / 'evaluation.json'), 'reporter_sha256': sha(__file__),
              'final_test_used': False,
              'decision': 'Separate frozen unfamiliar-development confirmation permitted; novelty not established' if passed else
                          'Gate failed; do not promote or expand this mechanism or train an adapter on its premise'}
    write_new(ROOT / 'research/context_support_results_v1.json', result)
    # Publish numerical receipts and hashes, never image bytes, embeddings or local paths.
    safe_rows = []
    for row in rows:
        safe = {k: v for k, v in row.items() if k not in ['output', 'evaluation']}
        score = row['evaluation']
        safe['evaluation'] = {k: v for k, v in score.items() if k not in ['metrics', 'target_detections', 'gallery_detections']}
        if 'metrics' in score:
            safe['evaluation']['metrics'] = {k: v for k, v in score['metrics'].items() if k != 'detections'}
            safe['evaluation']['detection_status'] = {k: {x: y for x, y in v.items() if x in ['face_count', 'status']} for k, v in score['metrics']['detections'].items()}
        safe_rows.append(safe)
    write_new(ROOT / 'research/context_support_evidence_v1.json', {
        'signature': json.loads((BASE / 'signature.json').read_text()),
        'evaluation_signature': json.loads((BASE / 'evaluation_signature.json').read_text()),
        'rows': safe_rows, 'generation_ledger': [{k: v for k, v in g.items() if k != 'output'} for g in generated['generation_ledger']]})
    fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
    lines = ['# Context-guided mask support — results, 25 September 2026', '',
             '**Decision: ' + result['decision'] + '.**', '',
             'Implemented and tested after freezing the protocol/source signature. Four previously observed removal cases, two seeds, 104 generator calls and 80 comparison rows. Generator and metric definitions remain frozen; inference never reads clean targets or withheld galleries. This is a single-image mechanism screen, not fresh-identity or pretraining-disjoint validation.', '',
             f"Generation coverage: {result['generations_complete']}/104. Scoring coverage: {result['scored']}/80. Exact original-mask context preservation: {preservation}. Face detection coverage complete: {detections}. No calibration abstentions: {selections_ok}.", '',
             '| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ |', '|---|---:|---:|---:|---:|']
    for arm, entry in summaries.items():
        lines.append('| '+arm+' | '+' | '.join(fmt(entry[m]['mean']) for m in ['facenet_cosine', 'facenet_gallery_cosine', 'hole_mae', 'lpips'])+' |')
    lines += ['', 'Each mean first averages both seeds within each identity. Detection/metric coverage, ArcFace, NIQE, BRISQUE, SSIM and PSNR remain in the numerical report. The support/seed five-call controls match the proposed policy cost; the four-call support average is a cheaper control.', '',
              '| Prespecified contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |', '|---|---:|---:|---|---:|']
    for name, value in contrasts.items():
        lines.append(f"| {name} | {value['identities']} | {fmt(value['mean_delta'])} | {value['ci95']} | {fmt(value['p_holm'])} |")
    lines += ['', 'The statistical unit is identity, not seed or output image. Four identities make the intervals unstable and the minimum two-sided exact p-value 0.125. Eight prespecified contrasts use Holm correction. These tests cannot establish publication-level superiority.', '',
              '| Case | Seed | Chosen radius | Probe error vs hidden-hole MAE Spearman |', '|---|---:|---:|---:|']
    for selection, correlation in zip(generated['selections'], correlations):
        lines.append(f"| {selection['case_id']} | {selection['seed']} | {selection['chosen_radius']} | {fmt(correlation['spearman'])} |")
    lines += ['', 'The correlation is a post-selection diagnostic across four radii within each case/seed, not a calibrated probability, independent trial count or tuning signal. Selection receipts were locked before the final bank was generated. Small visible patches may not predict fidelity of missing facial structures.', '',
              f"Measured summed generator-call time: {sum(timing):.1f} seconds; median call: {np.median(timing):.2f} seconds. Model initialization, scoring and file/report work are excluded. A deployment of the policy requires four calibration calls and one final call, not the entire experimental bank.", '',
              'Local visual sheets contain every fixed-radius arm, selected result and both equal-budget controls for every case/seed. Images remain local. Novelty is unproven: Noise2Self, internal-image inpainting adaptation and mask perturbation have substantial overlap, described in `CONTEXT_SUPPORT_METHOD.md`. No website default, previous experiment, learned adapter or reserved-final identity was changed.']
    (ROOT / 'research/CONTEXT_SUPPORT_RESULTS_V1.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    truth = {c['case_id']: c for c in json.loads((ROOT / 'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for cid in cfg['cases']:
        for seed in cfg['seeds']:
            paths = [('clean scoring target', truth[cid]['target']['path'])]
            for arm in ['observed', 'fixed_r0', 'fixed_r2', 'fixed_r4', 'fixed_r8', 'context_selected', 'support_mean5', 'seed_mean5']:
                row = next(r for r in rows if r['case_id'] == cid and r['seed']==seed and r['mode']==arm)
                paths.append((arm, row.get('output')))
            sheet = Image.new('RGB', (3*256, 3*280), 'white'); draw = ImageDraw.Draw(sheet)
            for i, (label, path) in enumerate(paths):
                x, y = i % 3 * 256, i // 3 * 280
                draw.text((x+4, y+4), label, fill='black')
                if path:
                    with Image.open(path) as im:
                        sheet.paste(im.convert('RGB').resize((256, 256)), (x, y+24))
                else:
                    draw.text((x+4, y+70), 'FAILED', fill='red')
            dest = BASE / 'review' / f'{cid}_s{seed}.jpg'
            dest.parent.mkdir(exist_ok=True)
            sheet.save(dest)
    print(json.dumps({'gate_passed': bool(passed), 'scored': result['scored'], 'rows': 80,
                      'selected': summaries['context_selected'], 'fixed_r4': summaries['fixed_r4']}, indent=2))


if __name__ == '__main__':
    main()
