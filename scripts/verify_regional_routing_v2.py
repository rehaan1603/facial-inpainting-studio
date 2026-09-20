"""Verify completed development evidence before reporting or publication."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new


def main():
    base = ROOT / 'outputs/regional_routing_v2'
    cfg_path = ROOT / 'research/protocols/regional_routing_protocol_v2.json'
    cfg = json.loads(cfg_path.read_text())
    signature = json.loads((base / 'signature.json').read_text())
    assert sha(cfg_path) == signature['protocol_sha256'], 'Protocol changed'
    assert sha(ROOT / cfg['cases_manifest']) == signature['cases_sha256'], 'Cases changed'
    for name, digest in signature['sources'].items():
        assert sha(ROOT / name) == digest, f'Source changed: {name}'
    cases = json.loads((ROOT / cfg['cases_manifest']).read_text())['cases']
    assert all(c['split'] == 'validation' for c in cases), 'Non-development cases'
    expected = {(c['case_id'], seed, policy) for c in cases
                for seed in cfg['seeds'] for policy in cfg['policies']}
    comparison = json.loads((base / 'comparison.json').read_text())
    evaluation = json.loads((base / 'evaluation.json').read_text())
    assert comparison['final_test_used'] is False
    assert evaluation['final_test_used'] is False
    assert comparison['signature_sha256'] == sha(base / 'signature.json')
    es = json.loads((base / 'evaluation_signature.json').read_text())
    assert evaluation['signature_sha256'] == sha(base / 'evaluation_signature.json')
    assert es['comparison_sha256'] == sha(base / 'comparison.json')
    assert es['evaluator_sha256'] == sha(ROOT / 'scripts/evaluate_regional_routing_v2.py')
    assert es['metric_source_sha256'] == sha(ROOT / 'scripts/extended_evaluation_metrics.py')
    key = lambda r: (r['case_id'], r['seed'], r['policy'])
    for data in [comparison, evaluation]:
        keys = [key(r) for r in data['rows']]
        assert len(keys) == len(set(keys)) and set(keys) == expected, 'Missing/duplicate rows'
    generated = {key(r): r for r in comparison['rows']}
    checked = 0
    failures = []
    for row in evaluation['rows']:
        record = generated[key(row)]
        assert all(row[k] == v for k, v in record.items()), 'Evaluation provenance mismatch'
        if record['status'] == 'complete':
            output = Path(record['output'])
            assert sha(output) == record['output_sha256'], f'Output changed: {key(row)}'
            if 'metadata_sha256' in record:
                assert sha(output.with_suffix('.json')) == record['metadata_sha256']
                metadata = json.loads(output.with_suffix('.json').read_text())
                assert sha(output.with_name(output.stem + '_routing.npz')) == metadata['regional_routing']['map_sha256']
            checked += 1
        score = row['evaluation']
        if score['status'] == 'complete':
            assert record['status'] == 'complete'
            assert score['output_sha256'] == record['output_sha256']
            assert score['metrics']['visible_mae'] == 0, 'Known pixels changed'
        else:
            failures.append({'case_id': row['case_id'], 'seed': row['seed'],
                             'policy': row['policy'], 'error': score.get('error')})
    result = {'comparison_sha256': sha(base / 'comparison.json'),
              'evaluation_sha256': sha(base / 'evaluation.json'),
              'expected_rows': len(expected), 'verified_outputs': checked,
              'evaluation_failures': failures, 'final_test_used': False,
              'note': 'Integrity verification is not evidence of visual quality or superiority.'}
    write_new(base / 'completion_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
