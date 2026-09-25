"""Verify the completed screen's sources, images, locked selections and metrics."""
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new


def main():
    base = ROOT / 'outputs/context_support_v1'
    locked = json.loads((base / 'signature.json').read_text())
    signature = locked['signature']
    assert sha(ROOT / 'research/protocols/context_support_v1.json') == signature['protocol_sha256']
    assert sha(ROOT / 'outputs/distortion_aware_v1/inference_manifest.json') == signature['inference_manifest_sha256']
    for path, expected in signature['sources'].items():
        assert sha(ROOT / path) == expected, path
    comparison = json.loads((base / 'comparison.json').read_text())
    assert comparison['signature_sha256'] == sha(base / 'signature.json')
    metrics_signature = json.loads((base / 'evaluation_signature.json').read_text())
    assert metrics_signature['comparison_sha256'] == sha(base / 'comparison.json')
    assert metrics_signature['evaluator_sha256'] == sha(ROOT / 'scripts/evaluate_context_support.py')
    assert metrics_signature['metric_source_sha256'] == sha(ROOT / 'scripts/extended_evaluation_metrics.py')
    assert metrics_signature['evaluation_manifest_sha256'] == sha(ROOT / 'outputs/distortion_aware_v1/evaluation_manifest.json')
    evaluation = json.loads((base / 'evaluation.json').read_text())
    assert evaluation['signature_sha256'] == sha(base / 'evaluation_signature.json')
    summary = json.loads((ROOT / 'research/context_support_results_v1.json').read_text())
    assert summary['evaluation_sha256'] == sha(base / 'evaluation.json')
    assert summary['reporter_sha256'] == sha(ROOT / 'scripts/report_context_support.py')
    assert len(comparison['generation_ledger']) == 104 and len(evaluation['rows']) == 80
    for generated in comparison['generation_ledger']:
        assert generated['status'] == 'complete' and sha(generated['output']) == generated['output_sha256']
    selected_hash_matches = 0
    for selection in comparison['selections']:
        prefix = f"{selection['case_id']}_s{selection['seed']}"
        stored = json.loads((base / 'selections' / (prefix + '.json')).read_text())
        assert stored == selection
        rows = {r['mode']: r for r in evaluation['rows'] if r['case_id'] == selection['case_id'] and r['seed'] == selection['seed']}
        assert rows['context_selected']['output_sha256'] == rows[f"fixed_r{selection['chosen_radius']}"]['output_sha256']
        selected_hash_matches += 1
    inputs = {c['case_id']: c for c in json.loads((ROOT / 'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases']}
    for row in evaluation['rows']:
        assert row['status'] == row['evaluation']['status'] == 'complete'
        assert sha(row['output']) == row['output_sha256'] == row['evaluation']['output_sha256']
        c = inputs[row['case_id']]
        with Image.open(c['observed']) as im:
            observed = np.asarray(im.convert('RGB'))
        with Image.open(c['mask']) as im:
            mask = np.asarray(im.convert('L')) >= 128
        with Image.open(row['output']) as im:
            output = np.asarray(im.convert('RGB'))
        assert np.array_equal(output[~mask], observed[~mask])
    failures = [{ 'key': r['key'], 'evaluator': n, 'status': s['status'] }
                for r in evaluation['rows'] for n, s in r['evaluation']['metrics']['detections'].items() if s['status'] != 'ok']
    assert all('observed' in f['key'] for f in failures)
    result = {'date': '2026-09-25', 'source_signatures_verified': True, 'generation_image_hashes_verified': 104,
              'comparison_image_hashes_and_preservation_verified': 80, 'selection_matches_locked_radius': selected_hash_matches,
              'detector_failures_in_damaged_input_controls': failures, 'reconstructed_detector_failures': 0,
              'review_sheet_hashes': {p.name: sha(p) for p in sorted((base / 'review').glob('*.jpg'))},
              'statistics_source_sha256': sha(ROOT / 'src/preservation/context_statistics.py'),
              'statistics_tests_sha256': sha(ROOT / 'tests/test_context_statistics.py'),
              'mechanism_tests_sha256': sha(ROOT / 'tests/test_context_support.py'),
              'audit_source_sha256': sha(__file__), 'reserved_final_pixels_accessed': False,
              'scope': 'Technical audit of this frozen run; not evidence of novelty or generalization'}
    write_new(ROOT / 'research/context_support_audit_v1.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
