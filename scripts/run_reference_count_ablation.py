"""Generate fixed nested reference-count ablations without exposing clean targets."""
import json
from pathlib import Path
from reference_inpaint import reconstruct, sha
from atomic_records import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/reference_count_ablation_v1'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ['outputs/reference_diagnostics_v1/manifest.json',
             'research/EXTENDED_EVALUATION_PROTOCOL_V1.md',
             'research/reference-environment-v2-lock.txt',
             'scripts/reference_inpaint.py', 'scripts/reference_blending.py',
             'scripts/atomic_records.py', 'scripts/run_reference_count_ablation.py']
    signature = {p:sha(ROOT/p) for p in paths}
    guard = OUT/'signature.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8')) != signature:
        raise ValueError('Ablation inputs changed; use a new version')
    write_json(guard, signature)
    cases = json.loads((ROOT/paths[0]).read_text(encoding='utf-8'))['cases']
    runtime = {}
    rows = []
    for case in cases:
        observed, mask = ROOT/case['observed'], ROOT/case['mask']
        assert sha(observed) == case['observed_sha256'] and sha(mask) == case['mask_sha256']
        refs = [x for x in case['images'] if x['role'].startswith('reference_')]
        for item in refs:
            assert sha(ROOT/item['file']) == item['processed_sha256']
        for count in [1,2]:
            folder = OUT/f"{case['case_id']}_refs{count}"
            folder.mkdir(exist_ok=True)
            record_path = folder/'record.json'
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding='utf-8'))
                for output in record['outputs']:
                    assert sha(ROOT/output['file']) == output['sha256']
            else:
                try:
                    meta = reconstruct(observed, mask, [ROOT/x['file'] for x in refs[:count]],
                                       folder/'result.png', seed=17, steps=30, scale=.8,
                                       strength=.99, runtime=runtime, progress_path=folder/'progress.json')
                    record = {'case_id':case['case_id'],'identity':case['identity_label'],
                              'reference_count':count,'reference_files':[x['file'] for x in refs[:count]],
                              'scale':.8,'strength':.99,'seed':17,
                              'inference_seconds':meta['inference_seconds_including_offload'],
                              'outputs':[{'compositor':kind,'file':str((folder/name).relative_to(ROOT)).replace('\\','/'),
                                          'sha256':sha(folder/name)} for kind,name in [('hard','result_hard.png'),('poisson','result.png')]]}
                    write_json(record_path, record)
                except Exception as error:
                    write_json(folder/'failure.json', {'error':str(error)})
                    raise
            rows.append(record)
            write_json(OUT/'progress.json', {'completed_candidates':len(rows),'total_candidates':24,
                                           'status':'complete' if len(rows)==24 else 'running'})
            print(f'Count ablation {len(rows)}/24', flush=True)
    write_json(OUT/'manifest.json', {'signature':signature,'candidates':rows})

if __name__ == '__main__':
    main()
