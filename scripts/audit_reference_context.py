"""Post-run integrity audit; never reads clean targets, galleries or final pixels."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = ROOT / 'outputs/distortion_aware_v1/inference_manifest.json'
    inputs = {c['case_id']: c for c in read(manifest)['cases']}
    old = ROOT / 'outputs/reference_context_v1'
    initial = read(old / 'signature.json')
    for key, path in [('protocol', 'research/protocols/reference_context_v1.json'),
                      ('runner', 'scripts/run_reference_context.py'),
                      ('mechanism', 'src/preservation/reference_context.py')]:
        assert initial[key+'_sha256'] == sha(ROOT / path)
    failure = read(old / 'receipts/1306_removal_s17_no_reference.json')
    assert failure['status'] == 'failed' and 'output' not in failure
    studies = []
    for name, evaluator, reporter, reused in [
            ('reference_context_v2', 'evaluate_reference_context', 'report_reference_context', ['scaffold']),
            ('scaffold_conditioning_v1', 'evaluate_scaffold_conditioning', 'report_scaffold_conditioning', ['scaffold', 'reference_context'])]:
        base = ROOT / 'outputs' / name
        cfg = read(ROOT / 'research/protocols' / (name+'.json'))
        signature = read(base / 'signature.json')
        runner = 'run_reference_context_v2' if name == 'reference_context_v2' else 'run_scaffold_conditioning'
        for key, path in [('protocol', 'research/protocols/'+name+'.json'),
                          ('runner', 'scripts/'+runner+'.py'),
                          ('mechanism', 'src/preservation/reference_context_v2.py')]:
            assert signature[key+'_sha256'] == sha(ROOT / path)
        assert signature['manifest_sha256'] == sha(manifest)
        assert signature['scaffold_comparison_sha256'] == sha(ROOT / 'outputs/context_support_v1/comparison.json')
        if name == 'scaffold_conditioning_v1':
            assert signature['prior_reference_context_comparison_sha256'] == sha(ROOT / 'outputs/reference_context_v2/comparison.json')
        comparison = read(base / 'comparison.json')
        assert comparison['signature_sha256'] == sha(base / 'signature.json')
        es = read(base / 'evaluation_signature.json')
        for key, path in [('comparison', base / 'comparison.json'), ('evaluator', ROOT / 'scripts' / (evaluator+'.py')),
                          ('metric_source', ROOT / 'scripts/extended_evaluation_metrics.py'),
                          ('evaluation_manifest', ROOT / 'outputs/distortion_aware_v1/evaluation_manifest.json')]:
            assert es[key+'_sha256'] == sha(path)
        result_name = 'reference_context_results_v2' if name == 'reference_context_v2' else 'scaffold_conditioning_results_v1'
        summary = read(ROOT / 'research' / (result_name+'.json'))
        assert summary['evaluation_sha256'] == sha(base / 'evaluation.json')
        assert summary['reporter_sha256'] == sha(ROOT / 'scripts' / (reporter+'.py'))
        assert summary['statistics_sha256'] == sha(ROOT / 'src/preservation/context_statistics.py')
        equality = read(base / 'gain_zero_equivalence.json')
        assert equality['latent_bit_exact'] and equality['latent_max_abs_difference'] == 0 and equality['hook_calls'] == 50
        evaluation = read(base / 'evaluation.json')
        assert evaluation['signature_sha256'] == sha(base / 'evaluation_signature.json')
        rows = evaluation['rows']
        expected = {(c, s, a) for c in cfg['cases'] for s in cfg['seeds'] for a in cfg['arms']}
        assert len(rows) == 40 and {(r['case_id'], r['seed'], r['mode']) for r in rows} == expected
        ledger = {r['key']: r for r in comparison['rows']}
        failures, coverage = [], {}
        for row in rows:
            assert row['identity'] not in cfg['reserved_identities']
            assert {k: v for k, v in row.items() if k != 'evaluation'} == ledger[row['key']]
            assert read(base / 'receipts' / (row['key']+'.json')) == ledger[row['key']]
            assert read(base / 'scores' / (row['key']+'.json')) == row['evaluation']
            assert row['status'] == row['evaluation']['status'] == 'complete'
            assert sha(row['output']) == row['output_sha256'] == row['evaluation']['output_sha256']
            c = inputs[row['case_id']]
            assert sha(c['observed']) == c['observed_sha256'] and sha(c['mask']) == c['mask_sha256']
            with Image.open(c['observed']) as image: observed = np.asarray(image.convert('RGB'))
            with Image.open(c['mask']) as image: mask = np.asarray(image.convert('L')) >= 128
            with Image.open(row['output']) as image: output = np.asarray(image.convert('RGB'))
            assert np.array_equal(output[~mask], observed[~mask])
            assert row['evaluation']['metrics']['known_pixels_unchanged']
            if row['mode'] not in reused:
                assert row['hook_calls'] == 50
                indices = [a['step_index'] for a in row['correction_applied']]
                assert indices == (cfg['correction']['steps'] if row['correction_gain'] else [])
                no_ref = 'no_reference' in row['mode']
                assert (row['reference_condition_abs_sum'] == 0) == no_ref
                raw = Path(row['output']).with_name(Path(row['output']).stem+'_raw.png')
                assert sha(raw) == row['raw_sha256']
            for detector, status in row['evaluation']['metrics']['detections'].items():
                if status['status'] != 'ok':
                    failures.append({'key': row['key'], 'detector': detector, 'status': status['status']})
        for arm in cfg['arms']:
            coverage[arm] = {m: summary['summaries'][arm][m]['identity_count'] for m in [
                'facenet_cosine', 'facenet_gallery_cosine', 'arcface_conditioning_cosine', 'hole_mae', 'lpips']}
        studies.append({'experiment': name, 'rows_and_pixel_preservation_verified': len(rows),
                        'new_generation_receipts': sum(r['mode'] not in reused for r in rows),
                        'source_and_evaluation_signatures_verified': True, 'zero_gain_bit_exact': True,
                        'detector_failures': failures, 'metric_identity_counts': coverage,
                        'visual_sheets_reviewed': {p.name: sha(p) for p in sorted((base/'review').glob('*.jpg'))},
                        'review_scope': 'All eight sheets inspected by assistant; not a blinded human study'})
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'studies': studies,
              'initial_v1_failure_preserved': True, 'audit_source_sha256': sha(__file__),
              'clean_targets_or_gallery_pixels_read_by_auditor': False, 'reserved_final_pixels_accessed': False,
              'scope': 'Integrity and coverage, not novelty or statistical superiority'}
    write_new(ROOT / 'research/reference_context_audit_v1.json', result)
    print(json.dumps({'experiments': len(studies), 'rows_verified': 80, 'new_generations': 56,
                      'detector_failure_events': [len(s['detector_failures']) for s in studies]}, indent=2))


if __name__ == '__main__': main()
