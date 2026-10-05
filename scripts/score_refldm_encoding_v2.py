"""Paired local metrics for the fixed observed-identity encoding diagnostic."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from extended_evaluation_metrics import Metrics, ROOT, read_rgb, sha
from score_target_compatibility_v1 import components
from region_metrics_v1 import score_regions

OUT = ROOT/'outputs/refldm_encoding_v2'
KEYS = ['psnr', 'ssim', 'lpips', 'facenet_cosine', 'arcface_conditioning_cosine', 'eye_bbox_lpips']
HIGH = {'psnr', 'ssim', 'facenet_cosine', 'arcface_conditioning_cosine'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def contrasts(rows, targets):
    result = {}
    for key in KEYS:
        pairs = []
        for target in targets:
            for seed in [17, 29, 43]:
                pair = {r['arm']:r for r in rows if r['target']==target and r.get('seed')==seed}
                a = pair.get('joint_rgb', {}).get('metrics', {}).get(key)
                b = pair.get('separate_latents', {}).get('metrics', {}).get(key)
                if a is not None and b is not None:
                    pairs.append(dict(target=target, identity=pair['joint_rgb']['identity'], seed=seed,
                                      joint=a, separate=b, delta=a-b))
        identity_units = []
        for identity in sorted({p['identity'] for p in pairs}):
            target_units = []
            for target in sorted({p['target'] for p in pairs if p['identity']==identity}):
                ps = [p for p in pairs if p['target']==target]
                target_units.append(dict(target=target, valid_seeds=len(ps),
                    delta=float(np.mean([p['delta'] for p in ps]))))
            identity_units.append(dict(identity=identity, targets=target_units,
                delta=float(np.mean([p['delta'] for p in target_units]))))
        d = [p['delta'] for p in pairs]
        result[key] = dict(n_pairs=len(pairs), n_identities=len(identity_units), pairs=pairs,
            joint_mean=float(np.mean([p['joint'] for p in pairs])) if pairs else None,
            separate_mean=float(np.mean([p['separate'] for p in pairs])) if pairs else None,
            mean_paired_delta=float(np.mean(d)) if d else None, identity_units=identity_units,
            improved=sum(v>1e-6 if key in HIGH else v < -1e-6 for v in d),
            worsened=sum(v < -1e-6 if key in HIGH else v>1e-6 for v in d),
            tied=sum(abs(v)<=1e-6 for v in d))
    return result


def main():
    assert not (OUT/'scores.json').exists()
    completed = json.loads((OUT/'completion.json').read_text())
    assert completed['status']=='complete' and completed['outputs']==30
    assert sha(OUT/'generation.json')==completed['generation_sha256']
    assert sha(OUT/'config.json')==completed['config_sha256']
    cfg = json.loads((OUT/'config.json').read_text())
    records = json.loads((OUT/'generation.json').read_text())
    assert len(records)==30 and all(r['status']=='complete' for r in records)
    local = json.loads((ROOT/'configs/local.json').read_text())
    metric = Metrics()
    rows = []
    for case in cfg['cases']:
        folder = OUT/Path(case['target']).stem
        target = read_rgb(folder/'hq.png', case['hq_sha256'])
        observed = read_rgb(folder/'lq.png', case['lq_sha256'])
        tv, td = metric.features(target)
        parts, component_sources = components(int(Path(case['target']).stem), target.shape[:2])
        hq_path = Path(local['hq'])/'CelebA-HQ-img'/f"{int(Path(case['target']).stem)}.jpg"
        with Image.open(hq_path) as image:
            local_hq = np.asarray(image.convert('RGB').resize(target.shape[1::-1], Image.Resampling.BICUBIC))
        gray_a = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(local_hq, cv2.COLOR_RGB2GRAY).astype(np.float32)
        shift, response = cv2.phaseCorrelate(gray_a, gray_b)
        mad = float(np.abs(target.astype(float)-local_hq.astype(float)).mean())
        geometry = dict(local_hq_sha256=sha(hq_path), mean_absolute_rgb_difference_255=mad,
            shift_xy=list(shift), phase_response=response,
            verified=mad<2 and max(abs(v) for v in shift)<0.25,
            scope='Same HQ index and near-identical resized RGB support annotation geometry; compression differs')
        input_row = dict(identity=case['identity'], target=case['target'], seed=None, arm='degraded_input',
                         references=len(case['references']), status='complete', file=str((folder/'lq.png').relative_to(OUT)),
                         sha256=case['lq_sha256'])
        for original in [input_row]+[r for r in records if r['target']==case['target']]:
            row = dict(original)
            try:
                rgb = read_rgb(OUT/row['file'], row['sha256'])
                assert rgb.shape==target.shape==(512,512,3)
                vec, det = metric.features(rgb)
                a, b = rgb.astype(np.float64)/255, target.astype(np.float64)/255
                mse = float(np.square(a-b).mean())
                values = dict(psnr=float(-10*np.log10(mse)), ssim=float(metric.ssim(a,b,
                    data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
                with torch.inference_mode():
                    values['lpips'] = float(metric.lpips(metric.tensor(rgb)*2-1,metric.tensor(target)*2-1))
                for k,v in vec.items():
                    values[k+'_cosine'] = float(np.dot(v,tv[k])) if v is not None and tv[k] is not None else None
                region = score_regions(metric,rgb,target,observed,np.ones(target.shape[:2],bool),parts) if geometry['verified'] else None
                values['eye_bbox_lpips'] = region['eyes']['bbox_lpips'] if region else None
                row.update(metrics=values, target_detections=td, output_detections=det, geometry=geometry,
                           eye_region=region['eyes'] if region else {'status':'alignment_unverified'},
                           component_sources=component_sources)
            except Exception as error:
                row['evaluation_error'] = repr(error)
            rows.append(row)
            write(OUT/'scores.partial.json',rows)
            print('SCORED',len(rows),row['target'],row['seed'],row['arm'],
                  'ok' if 'metrics' in row else row['evaluation_error'], flush=True)
    multi = [c['target'] for c in cfg['cases'] if len(c['references'])>1]
    single = [c['target'] for c in cfg['cases'] if len(c['references'])==1]
    checks = dict(smoke_replays_identical=all(r.get('smoke_replay_identical',True) for r in records),
        single_reference_pairs_identical=all(
            len({r['sha256'] for r in records if r['target']==t and r['seed']==s})==1 for t in single for s in cfg['seeds']),
        scored_rows=sum('metrics' in r for r in rows), expected_rows=35)
    result = dict(scope=cfg['scope'], config=cfg, rows=rows,
        contrasts=dict(multi_reference=contrasts(rows,multi),single_reference_control=contrasts(rows,single)),
        checks=checks, script_sha256=sha(__file__), metric_script_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),
        region_script_sha256=sha(ROOT/'scripts/region_metrics_v1.py'),
        limitations=['Only one identity has multi-reference cases; three seeds are not three people',
            'No new reference corruption in this encoding diagnostic',
            'First CSV references are not asserted to match author benchmark selection',
            'Local metrics and EMA loader; not exact paper reproduction',
            'No adapter training or production changes; reserved final data untouched'])
    write(OUT/'scores.json',result)
    write(ROOT/'research/refldm_encoding_v2_results.json',result)
    print('SCORING COMPLETE',json.dumps(checks),flush=True)


if __name__=='__main__':
    main()
