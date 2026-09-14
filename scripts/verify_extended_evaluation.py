"""Audit scored-image integrity and agreement with prior frozen measurements."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    original=json.loads((ROOT/'research/extended_evaluation_original_v1.json').read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'research/identity_diagnostic_results_v1.json').read_text(encoding='utf-8'))
    lookup={x['output_sha256']:x for x in prior['rows']}
    with (ROOT/'outputs/reference_diagnostic_evaluation_v3/metrics.csv').open(encoding='utf-8',newline='') as stream:
        old_quality={x['output_sha256']:x for x in csv.DictReader(stream)}
    discrepancies=[]
    lpips_errors=[]
    for row in original['rows']:
        old=lookup[row['output_sha256']]['cosine_similarity']
        new=row['facenet_cosine']
        assert (old is None)==(new is None), 'Detection coverage changed'
        if old is not None:
            discrepancies.append(abs(old-new))
        lpips_errors.append(abs(row['lpips']-float(old_quality[row['output_sha256']]['full_face_lpips'])))
    assert max(discrepancies)<1e-6
    assert max(lpips_errors)<1e-6
    datasets=[original]
    count_path=ROOT/'research/extended_evaluation_count_v1.json'
    if count_path.exists():
        datasets.append(json.loads(count_path.read_text(encoding='utf-8')))
    for data in datasets:
        for path,expected in data['signature'].items():
            assert sha(ROOT/path)==expected, f'Signature changed: {path}'
        for row in data['rows']:
            assert sha(ROOT/row['file'])==row['output_sha256'], 'Output modified after scoring'
            assert row['visible_mae']==0
    count_manifest_path=ROOT/'outputs/reference_count_ablation_v1/manifest.json'
    generation_checks=0
    if count_manifest_path.exists():
        count_manifest=json.loads(count_manifest_path.read_text(encoding='utf-8'))
        cases={c['case_id']:c for c in json.loads((ROOT/'outputs/reference_diagnostics_v1/manifest.json').read_text(encoding='utf-8'))['cases']}
        for relative,expected in count_manifest['signature'].items():
            assert sha(ROOT/relative)==expected
        for candidate in count_manifest['candidates']:
            folder=ROOT/candidate['outputs'][0]['file']
            metadata=json.loads((folder.parent/'result.json').read_text(encoding='utf-8'))
            old=json.loads((ROOT/f"outputs/reference_diagnostic_evaluation_v3/{candidate['case_id']}_scale0.8_strength0.99/result.json").read_text(encoding='utf-8'))
            for field in ['seed','steps','adapter_scale','sampling_steps','strength','adapter_sha256','environment','model_resolution','input_sha256','mask_sha256']:
                assert metadata[field]==old[field], f'Unmatched generation setting: {field}'
            refs=[x for x in cases[candidate['case_id']]['images'] if x['role'].startswith('reference_')][:candidate['reference_count']]
            assert metadata['reference_count']==candidate['reference_count']
            assert [x['sha256'] for x in metadata['references']]==[x['processed_sha256'] for x in refs]
            for output in candidate['outputs']:
                assert sha(ROOT/output['file'])==output['sha256']
            generation_checks+=1
    report={'verified_rows':sum(len(x['rows']) for x in datasets),
            'max_facenet_difference_from_prior':max(discrepancies),
            'max_lpips_difference_from_prior':max(lpips_errors),
            'all_output_hashes_verified':True,'all_evaluator_signatures_verified':True,
            'matched_generation_candidates':generation_checks,
            'coverage':{data['suite']:{metric:sum(x[metric] is not None for x in data['rows']) for metric in
                                     ['facenet_cosine','arcface_conditioning_cosine','niqe','brisque','ssim_rgb','hole_psnr']} for data in datasets}}
    (ROOT/'research/extended_evaluation_verification_v1.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
