"""Report the complete reference/context factorial screen and its frozen gate."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_statistics import identity_means, contrast, holm
from report_distortion_aware import METRICS

BASE = ROOT / 'outputs/reference_context_v2'


def main():
    cfg = json.loads((ROOT / 'research/protocols/reference_context_v2.json').read_text())
    data = json.loads((BASE / 'evaluation.json').read_text()); rows = data['rows']
    assert len(rows) == 40
    arms = {a: [r for r in rows if r['mode']==a] for a in cfg['arms']}
    summaries = {}
    for arm, group in arms.items():
        entry = {'rows': len(group), 'complete': sum(r['evaluation']['status']=='complete' for r in group)}
        for metric in METRICS:
            means, excluded = identity_means(group, metric, cfg['seeds'])
            entry[metric] = {'mean': float(np.mean(list(means.values()))) if means else None,
                             'identity_count': len(means), 'excluded': excluded}
        seconds = [r['generation_seconds'] for r in group if 'generation_seconds' in r]
        entry['mean_generation_seconds'] = float(np.mean(seconds)) if seconds else None
        entry['maximum_allocated_bytes'] = max((r.get('peak_allocated_bytes', 0) for r in group), default=0)
        summaries[arm] = entry
    primary = {f'reference_context_minus_{a}/{m}': contrast(arms['reference_context'], arms[a], m, cfg['seeds'])
               for a in cfg['primary_controls'] for m in cfg['primary_metrics']}
    holm(primary)
    secondary = {f'{a}_minus_{b}/{m}': contrast(arms[a], arms[b], m, cfg['seeds'])
                 for a, b in [('reference', 'no_reference'), ('no_reference_context', 'no_reference'), ('reference', 'scaffold')]
                 for m in METRICS}
    interactions = {}
    for metric in METRICS:
        groups = {a: identity_means(arms[a], metric, cfg['seeds'])[0] for a in cfg['arms']}
        ids = set.intersection(*(set(v) for v in groups.values()))
        values = {i: (groups['reference_context'][i]-groups['reference'][i]) -
                     (groups['no_reference_context'][i]-groups['no_reference'][i]) for i in sorted(ids)}
        interactions[metric] = {'identity_values': values, 'mean': float(np.mean(list(values.values()))) if values else None,
                                'scope': 'Descriptive factorial interaction, not a prespecified primary test'}
    def seed_mean(arm, seed, metric):
        vals = [r['evaluation'].get('metrics', {}).get(metric) for r in arms[arm] if r['seed']==seed]
        return float(np.mean(vals)) if len(vals)==4 and all(v is not None for v in vals) else None
    gates = []
    for seed in cfg['seeds']:
        for control in cfg['primary_controls']:
            pairs = {m: [seed_mean(a, seed, m) for a in ['reference_context', control]]
                     for m in ['facenet_cosine', 'hole_mae', 'facenet_gallery_cosine', 'lpips']}
            if any(v is None for p in pairs.values() for v in p):
                checks = {'coverage': False}
            elif control == 'no_reference_context':
                checks = {'reference_identity_benefit': pairs['facenet_cosine'][0] > pairs['facenet_cosine'][1],
                          'reference_pixel_noninferiority': pairs['hole_mae'][0] <= pairs['hole_mae'][1]}
            else:
                checks = {'identity_gain_0.005': pairs['facenet_cosine'][0] >= pairs['facenet_cosine'][1]+.005,
                          'hole_error_reduction_1pct': pairs['hole_mae'][0] <= .99*pairs['hole_mae'][1],
                          'gallery_noninferiority_0.002': pairs['facenet_gallery_cosine'][0] >= pairs['facenet_gallery_cosine'][1]-.002,
                          'lpips_noninferiority_1pct': pairs['lpips'][0] <= pairs['lpips'][1]*1.01}
            gates.append({'seed': seed, 'control': control, 'values_candidate_control': pairs, 'checks': checks, 'pass': all(checks.values())})
    coverage = all(r['evaluation']['status']=='complete' and
                   all(r['evaluation']['metrics']['detections'][n]['status']=='ok' and r['evaluation']['metrics'][n+'_gallery_valid_count']==3
                       for n in ['facenet', 'arcface_conditioning']) for r in rows)
    preservation = all(r['evaluation'].get('metrics', {}).get('known_pixels_unchanged') is True for r in rows)
    passed = coverage and preservation and all(g['pass'] for g in gates)
    result = {'date': '2026-09-25', 'screen': 'four previously observed development identities; two seeds; no final use',
              'attempt_v1': 'Failed EMA preflight before sampling; retained separately, not replaced',
              'new_generations': 32, 'equivalence_generations': 2, 'reused_scaffolds': 8,
              'generation_complete': sum(r['status']=='complete' for r in rows if r['mode']!='scaffold'),
              'scored': sum(r['evaluation']['status']=='complete' for r in rows),
              'gain_zero_equivalence': json.loads((BASE / 'gain_zero_equivalence.json').read_text()),
              'summaries': summaries, 'primary_contrasts': primary, 'secondary_contrasts': secondary,
              'descriptive_factorial_interaction': interactions,
              'progression_gate': {'passed_numerical_gate': bool(passed), 'coverage': coverage, 'preservation': preservation, 'checks': gates,
                                   'visual_review_required': True},
              'protocol_sha256': sha(ROOT / 'research/protocols/reference_context_v2.json'),
              'evaluation_sha256': sha(BASE / 'evaluation.json'), 'reporter_sha256': sha(__file__),
              'statistics_sha256': sha(ROOT / 'src/preservation/context_statistics.py'), 'final_test_used': False}
    write_new(ROOT / 'research/reference_context_results_v2.json', result)
    safe_rows = []
    for r in rows:
        if r['status']=='complete': assert sha(r['output']) == r['output_sha256']
        safe = {k: v for k, v in r.items() if k not in ['output', 'evaluation', 'traceback']}
        score = r['evaluation']
        safe['evaluation'] = {k: v for k, v in score.items() if k not in ['metrics', 'target_detections', 'gallery_detections']}
        if 'metrics' in score:
            safe['evaluation']['metrics'] = {k: v for k, v in score['metrics'].items() if k != 'detections'}
            safe['evaluation']['detection_status'] = {n: {k: v for k, v in d.items() if k in ['face_count','status']} for n, d in score['metrics']['detections'].items()}
        safe_rows.append(safe)
    old = ROOT / 'outputs/reference_context_v1'
    failure = json.loads((old / 'receipts/1306_removal_s17_no_reference.json').read_text())
    write_new(ROOT / 'research/reference_context_evidence_v2.json', {
        'signature': json.loads((BASE / 'signature.json').read_text()),
        'evaluation_signature': json.loads((BASE / 'evaluation_signature.json').read_text()),
        'initial_failure': failure, 'initial_signature': json.loads((old / 'signature.json').read_text()), 'rows': safe_rows})
    fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
    lines = ['# Reference/context factorial screen — 25 September 2026', '',
             '**Numerical progression gate: '+('passed; visual review and separate confirmation still required' if passed else 'failed')+'.**', '',
             'Four already observed removal identities, seeds 17 and 29. Two factors: genuine versus zero reference conditioning; visible-context correction on versus off. ResShift scaffold controls are unchanged. This measures the contribution of each factor, rather than crediting a cascade gain to a new mechanism.', '',
             f"Generation: {result['generation_complete']}/32 new restorations plus two full GPU equivalence runs and eight reused scaffolds. Evaluation: {result['scored']}/40. Disabled-correction latent equality: {result['gain_zero_equivalence']['latent_bit_exact']}. Every output preserves original visible pixels: {preservation}. Detector/gallery coverage complete: {coverage}.", '',
             '| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ | Mean generation seconds |', '|---|---:|---:|---:|---:|---:|']
    for a, s in summaries.items():
        lines.append('| '+a+' | '+' | '.join(fmt(s[m]['mean']) for m in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips'])+' | '+fmt(s['mean_generation_seconds'])+' |')
    lines += ['', 'Correction adds four decoder/eight encoder passes; it is not equal total compute to an uncorrected cascade. Times exclude model loading and conditioning. The scaffold cost was measured separately in the context-support study.', '',
              '| Primary contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |', '|---|---:|---:|---|---:|']
    for name, c in primary.items(): lines.append(f"| {name} | {c['identities']} | {fmt(c['mean_delta'])} | {c['ci95']} | {fmt(c['p_holm'])} |")
    lines += ['', 'Six primary tests were specified before generation. Both seeds are averaged within identity; n=4, not n=8 or n=40. Bootstrap intervals are unstable and the minimum two-sided exact p-value is 0.125. The full JSON retains per-seed gate checks, secondary comparisons, all quality metrics, failures and the descriptive factorial interaction.', '',
              'The first v1 wrapper attempt failed before sampling because blanket parameter freezing conflicted with the author EMA scope. Review also fixed missing empty hook kwargs and corrected the false statement that the author sample function did not forward the hook. Recovery v2 retains cases, seeds, method parameters and gates. Original records remain intact.', '',
              'Prior-art boundaries and equations are in `REFERENCE_CONTEXT_METHOD.md`. Neither cascaded restoration, reference conditioning nor known-context diffusion projection is new by itself. This screen does not establish novelty, reliable unknown-person recovery, a publication-ready method, or permission to open reserved-final identities.']
    (ROOT / 'research/REFERENCE_CONTEXT_RESULTS_V2.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    evaluation = {c['case_id']: c for c in json.loads((ROOT / 'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for cid in cfg['cases']:
        for seed in cfg['seeds']:
            paths = [('clean scoring target', evaluation[cid]['target']['path'])]
            paths += [(a, next(r.get('output') for r in arms[a] if r['case_id']==cid and r['seed']==seed)) for a in cfg['arms']]
            sheet = Image.new('RGB', (3*256, 2*280), 'white'); draw = ImageDraw.Draw(sheet)
            for index, (label, path) in enumerate(paths):
                x, y = index % 3 * 256, index // 3 * 280
                draw.text((x+4, y+4), label, fill='black')
                if path:
                    with Image.open(path) as im: sheet.paste(im.convert('RGB').resize((256,256)), (x,y+24))
                else: draw.text((x+4,y+70), 'FAILED', fill='red')
            dest = BASE / 'review' / f'{cid}_s{seed}.jpg'; dest.parent.mkdir(exist_ok=True); sheet.save(dest)
    print(json.dumps({'gate_passed': bool(passed), 'complete': result['scored'], 'means': {
        a:{m:s[m]['mean'] for m in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips']} for a,s in summaries.items()}}, indent=2))


if __name__ == '__main__': main()
