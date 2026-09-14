"""Score all original or count-ablation outputs, retaining failures and hashes."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image
import numpy as np
from atomic_records import write_json
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['original','count'],default='original')
    args = parser.parse_args()
    paths = ['scripts/extended_evaluation_metrics.py','scripts/evaluate_extended_diagnostics.py',
             'research/EXTENDED_EVALUATION_PROTOCOL_V1.md','research/extended_evaluation_provenance_v1.json',
             'research/identity_evaluator_provenance_v1.json','research/extended-evaluation-environment-v1-lock.txt',
             'outputs/reference_diagnostics_v1/manifest.json']
    manifest = json.loads((ROOT/paths[-1]).read_text(encoding='utf-8'))
    items = []
    if args.suite=='original':
        path = ROOT/'outputs/reference_diagnostic_evaluation_v3/metrics.csv'
        paths.append(str(path.relative_to(ROOT)))
        with path.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                file = f"outputs/reference_diagnostic_evaluation_v3/{row['case_id']}_scale{row['scale']}_strength{row['strength']}/"+('result_hard.png' if row['compositor']=='hard' else 'result.png')
                items.append({'case_id':row['case_id'],'identity':row['identity'],'scale':float(row['scale']),
                              'strength':float(row['strength']),'reference_count':4,'compositor':row['compositor'],
                              'file':file,'output_sha256':row['output_sha256']})
        assert len(items)==96
    else:
        path = ROOT/'outputs/reference_count_ablation_v1/manifest.json'
        paths.append(str(path.relative_to(ROOT)))
        for candidate in json.loads(path.read_text(encoding='utf-8'))['candidates']:
            for output in candidate['outputs']:
                items.append({k:candidate[k] for k in ['case_id','identity','scale','strength','reference_count']} |
                             {'compositor':output['compositor'],'file':output['file'],'output_sha256':output['sha256']})
        assert len(items)==48
    signature = {p:sha(ROOT/p) for p in paths}
    out = ROOT/f'outputs/extended_evaluation_v1/{args.suite}'
    out.mkdir(parents=True,exist_ok=True)
    guard = out/'signature.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8')) != signature:
        raise ValueError('Evaluation inputs changed; use a fresh version')
    write_json(guard,signature)
    metric = Metrics()
    rows, controls = [], []
    for case in manifest['cases']:
        t = next(x for x in case['images'] if x['role']=='target')
        target = read_rgb(ROOT/t['file'],t['processed_sha256'])
        observed = read_rgb(ROOT/case['observed'],case['observed_sha256'])
        if sha(ROOT/case['mask'])!=case['mask_sha256']:
            raise ValueError('Mask hash mismatch')
        with Image.open(ROOT/case['mask']) as image:
            mask = np.asarray(image.convert('L'))>=128
        vectors, target_detections = metric.features(target)
        controls.append({'case_id':case['case_id'],'target_quality':metric.quality(target),
                         'target_detections':target_detections,'observed_quality':metric.quality(observed)})
        for item in [x for x in items if x['case_id']==case['case_id']]:
            key = f"{item['case_id']}_scale{item['scale']}_strength{item['strength']}_refs{item['reference_count']}_{item['compositor']}"
            saved = out/(key+'.json')
            rgb = read_rgb(ROOT/item['file'],item['output_sha256'])
            if saved.exists():
                row = json.loads(saved.read_text(encoding='utf-8'))
                if row['output_sha256']!=item['output_sha256']:
                    raise ValueError('Saved score input changed')
            else:
                row = item | metric.score(rgb,target,observed,mask,vectors) | {'target_detections':target_detections}
                write_json(saved,row)
            rows.append(row)
        write_json(out/'progress.json',{'scored':len(rows),'total':len(items),'status':'running'})
        print(f'{args.suite}: {len(rows)}/{len(items)} scored',flush=True)
    result = {'suite':args.suite,'scope':'development_only','signature':signature,'rows':rows,'controls':controls}
    write_json(ROOT/f'research/extended_evaluation_{args.suite}_v1.json',result)
    write_json(out/'progress.json',{'scored':len(rows),'total':len(items),'status':'complete'})

if __name__=='__main__':
    main()
