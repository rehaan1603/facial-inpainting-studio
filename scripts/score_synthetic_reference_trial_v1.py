"""Evaluate all completed arms offline; preserve metric/detection failures."""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from extended_evaluation_metrics import Metrics, read_rgb, sha, ROOT

OUT = ROOT/'outputs/synthetic_references_v1'
PACK = Path.home()/'Downloads/Human dataset/Ready to test - Human Faces v1'
DELIVERY = PACK/'Synthetic references - experimental 2026-10-02'


def main():
    generation = json.loads((OUT/'generation.json').read_text())
    assert len(generation['rows']) == 12, 'Retain all twelve planned attempts before scoring.'
    assert not (OUT/'scores.json').exists(), 'Never replace previous results.'
    protocol = json.loads((OUT/'protocol.json').read_text())
    cases = {c['case_id']: c for c in protocol['cases']}
    evaluations = {c['case_id']: c for c in json.loads((PACK/'evaluation_manifest.json').read_text())['cases']}
    prior = json.loads((ROOT/'outputs/single_image_edges_v1/scores.json').read_text())['rows']
    rows = [dict(r, path=str(OUT/r['result']) if r['status']=='complete' else None) for r in generation['rows']]
    for case_id, c in cases.items():
        rows.append(dict(case_id=case_id, arm='damaged_input', status='complete',
                         path=c['input']['path'], result_sha256=c['input']['sha256']))
        for model in ['lama', 'resshift']:
            r = next(r for r in prior if r['case_id']==case_id and r['model']==model)
            # Match each model's actual current studio default.
            name = 'result_raw.png' if model=='lama' else 'result.png'
            checksum = r['raw_sha256'] if model=='lama' else r['finished_sha256']
            rows.append(dict(case_id=case_id, arm=model, status='complete',
                path=str(ROOT/'outputs/webapp_runs'/r['job_id']/name), result_sha256=checksum))
    metrics = Metrics()
    target_cache, panel_cache = {}, {}
    for row in rows:
        if row['status'] != 'complete':
            continue
        case_id = row['case_id']
        c, e = cases[case_id], evaluations[case_id]
        if case_id not in target_cache:
            target = read_rgb(e['target'], e['target_sha256'])
            vectors, detections = metrics.features(target)
            target_cache[case_id] = target, vectors, detections
        target, vectors, detections = target_cache[case_id]
        observed = read_rgb(c['input']['path'], c['input']['sha256'])
        mask = read_rgb(c['mask']['path'], c['mask']['sha256'])[:,:,0] >= 128
        rgb = read_rgb(row['path'], row['result_sha256'])
        try:
            values = metrics.score(rgb, target, observed, mask, vectors)
            ys, xs = np.nonzero(mask)
            region = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
            with torch.inference_mode():
                values['eye_region_lpips'] = float(metrics.lpips(
                    metrics.tensor(rgb[region])*2-1, metrics.tensor(target[region])*2-1).item())
            row.update(metrics=values, target_detections=detections,
                       outside_mask_exact=bool(np.array_equal(rgb[~mask],observed[~mask])))
        except Exception as error:
            row['evaluation_error'] = str(error)
        panel_cache[(case_id,row['arm'])] = rgb
        print('SCORED', case_id, row['arm'], 'ok' if 'metrics' in row else 'failed', flush=True)
    summary = {}
    keys = ['hole_mae','hole_psnr','lpips','eye_region_lpips','facenet_cosine','arcface_conditioning_cosine','niqe','brisque']
    for arm in ['damaged_input','lama','resshift',*protocol['arms']]:
        selected = [r for r in rows if r['arm']==arm]
        summary[arm] = {}
        for key in keys:
            values = [r['metrics'][key] for r in selected if r.get('metrics',{}).get(key) is not None]
            summary[arm][key] = dict(mean=float(np.mean(values)) if values else None, n=len(values))
    paired = {}
    lookup = {(r['case_id'],r['arm']):r for r in rows}
    for baseline in ['reference_off','first_synthetic','resshift']:
        comparisons = {}
        for key in keys:
            deltas=[]
            for case_id in cases:
                a=lookup[(case_id,'all_synthetic')].get('metrics',{}).get(key)
                b=lookup[(case_id,baseline)].get('metrics',{}).get(key)
                if a is not None and b is not None:deltas.append(a-b)
            lower = key in ['hole_mae','lpips','eye_region_lpips','niqe','brisque']
            comparisons[key]=dict(mean_delta=float(np.mean(deltas)) if deltas else None,
                common_n=len(deltas), improved=sum(d<0 if lower else d>0 for d in deltas))
        paired['all_synthetic_minus_'+baseline]=comparisons
    for case_id,c in cases.items():
        panels=[('Clean comparison',target_cache[case_id][0])]
        for arm in ['damaged_input','resshift','reference_off','first_synthetic','all_synthetic']:
            panels.append((arm, panel_cache.get((case_id,arm))))
        sheet=Image.new('RGB',(1536,406),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,rgb) in enumerate(panels):
            draw.text((i*256+5,8),label,fill='black')
            if rgb is not None:
                image=Image.fromarray(rgb)
                sheet.paste(image.resize((256,256)),(i*256,30))
                sheet.paste(image.crop((165,205,347,295)).resize((256,120)),(i*256,286))
        sheet.save(OUT/(case_id+'_comparison.png'))
        folder=DELIVERY/('Photo '+case_id.split('_')[1])
        shutil.copy2(OUT/(case_id+'_comparison.png'),folder/'Reconstruction comparison.png')
        for arm in protocol['arms']:
            r=lookup[(case_id,arm)]
            if r['status']=='complete':shutil.copy2(r['path'],folder/('Result - '+arm+'.png'))
    public_rows=[{k:v for k,v in r.items() if k!='path'} for r in rows]
    receipt=dict(scope=protocol['scope'], training_started=False, generation_attempts=12,
        generation_completed=sum(r['status']=='complete' for r in generation['rows']),
        evaluator_sha256=sha(__file__), metrics_source_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),
        rows=public_rows, summary=summary, paired_common_support=paired)
    (OUT/'scores.json').write_text(json.dumps(receipt,indent=2))
    (ROOT/'research/synthetic_reference_trial_v1.json').write_text(json.dumps(receipt,indent=2))
    shutil.copy2(OUT/'scores.json',DELIVERY/'RESULTS.json')
    print(json.dumps({'summary':summary,'paired':paired}),flush=True)


if __name__ == '__main__':
    main()
