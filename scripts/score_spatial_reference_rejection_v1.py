"""Score spatial interventions without dropping detector failures or changing controls."""
import json
import numpy as np
import torch
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha
from score_target_compatibility_v1 import components
from region_metrics_v1 import score_regions

OUT=ROOT/'outputs/spatial_reference_rejection_v1'
KEYS=['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']
HIGH={'psnr','ssim','facenet_cosine','arcface_conditioning_cosine'}
IDS=['263','6681','7400','10149']


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def compare(rows,arm,control,condition,seed=None):
    lookup={(r['identity'],r['seed'],r['condition'],r['arm']):r for r in rows}
    result={}
    for key in KEYS:
        pairs=[]
        for identity in IDS:
            for s in ([seed] if seed is not None else [17,29,43]):
                a=lookup[(identity,s,condition,arm)].get('metrics',{}).get(key)
                b=lookup[(identity,s,condition,control)].get('metrics',{}).get(key)
                if a is not None and b is not None:pairs.append(dict(identity=identity,seed=s,candidate=a,control=b,delta=a-b))
        units=[dict(identity=i,n_seeds=len(v),delta=float(np.mean(v))) for i in IDS
               for v in [[p['delta'] for p in pairs if p['identity']==i]] if v]
        d=np.array([p['delta'] for p in pairs]);u=np.array([p['delta'] for p in units])
        rng=np.random.default_rng(20261006)
        ci=np.quantile(u[rng.integers(0,len(u),(2000,len(u)))].mean(1),[.025,.975]).tolist() if len(u) else None
        result[key]=dict(n_pairs=len(pairs),n_identities=len(units),pairs=pairs,identity_units=units,
            candidate_mean=float(np.mean([p['candidate'] for p in pairs])) if pairs else None,
            control_mean=float(np.mean([p['control'] for p in pairs])) if pairs else None,
            mean_delta=float(d.mean()) if len(d) else None,identity_mean_delta=float(u.mean()) if len(u) else None,
            identity_bootstrap_95_ci=ci,improved=int(np.sum(d>1e-6 if key in HIGH else d < -1e-6)),
            worsened=int(np.sum(d < -1e-6 if key in HIGH else d>1e-6)))
    return result


def main():
    assert not (OUT/'scores.json').exists()
    complete=json.loads((OUT/'completion.json').read_text())
    assert sha(OUT/'evaluation/rows.json')==complete['rows_sha256'] and sha(OUT/'config.json')==complete['config_sha256']
    records=json.loads((OUT/'evaluation/rows.json').read_text());assert len(records)==72
    old_path=ROOT/'outputs/identity_timestep_v5/timestep_scaled/scores.json'
    old=json.loads(old_path.read_text())['rows']
    cases={r['identity']:r for r in json.loads((ROOT/'outputs/reference_intervention_pilot_v1/manifest.json').read_text())['cases']}
    m=Metrics();cache={};rows=[]
    for record in records:
        row=dict(record)
        try:
            rgb=read_rgb(OUT/'evaluation'/row['file'],row['sha256'])
            if row['arm']=='baseline':
                previous=next(r for r in old if all(r[k]==row[k] for k in ['identity','arm','seed','condition']))
                assert previous['sha256']==row['sha256']
                for key in ['metrics','target_detections','output_detections','component_sources','regions']:row[key]=previous[key]
                row['metrics_reused_from']=str(old_path.relative_to(ROOT))
            else:
                identity=row['identity']
                if identity not in cache:
                    folder=ROOT/'outputs/reference_intervention_pilot_v1/images'/identity
                    target=read_rgb(folder/'target.png');observed=read_rgb(folder/'input.png');tv,td=m.features(target)
                    parts,receipts=components(cases[identity]['target']['hq_id'],target.shape[:2])
                    cache[identity]=(target,observed,tv,td,parts,receipts)
                target,observed,tv,td,parts,receipts=cache[identity]
                vec,det=m.features(rgb);regions=score_regions(m,rgb,target,observed,np.ones(target.shape[:2],bool),parts)
                values={k:regions['whole'][k] for k in ['psnr','ssim','mae']}
                with torch.inference_mode():values['lpips']=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(target)*2-1))
                for key,v in vec.items():values[key+'_cosine']=float(np.dot(v,tv[key])) if v is not None and tv[key] is not None else None
                values['eye_bbox_lpips']=regions['eyes']['bbox_lpips']
                row.update(metrics=values,target_detections=td,output_detections=det,component_sources=receipts,regions=regions)
        except Exception as error:row['evaluation_error']=repr(error)
        rows.append(row);write(OUT/'scores.partial.json',rows)
        print('SCORED',len(rows),row['identity'],row['arm'],row['seed'],row['condition'],'ok' if 'metrics' in row else row['evaluation_error'],flush=True)
    contrasts={}
    for condition in ['clean','wrong_patch']:
        for arm,control in [('oracle_region','baseline'),('random_corner','baseline'),('oracle_region','random_corner')]:
            for seed in [None,17,29,43]:
                contrasts[f'{arm}/{control}/{condition}/{seed or "all"}']=compare(rows,arm,control,condition,seed)
    violations=[];missing=[]
    for condition in ['clean','wrong_patch']:
        for seed in [17,29,43]:
            for key,v in contrasts[f'oracle_region/baseline/{condition}/{seed}'].items():
                label=f'{condition}/{seed}/{key}'
                if v['n_pairs']!=4:missing.append(label)
                delta=v['mean_delta']
                if delta is None or (delta < -1e-6 if key in HIGH else delta>1e-6):violations.append(label)
    gains=all(contrasts[f'oracle_region/baseline/{c}/all']['lpips']['mean_delta'] is not None and
        contrasts[f'oracle_region/baseline/{c}/all']['lpips']['mean_delta'] < -1e-6 for c in ['clean','wrong_patch'])
    gate=dict(passes=not violations and not missing and gains,violations=violations,incomplete_support=missing,
        lpips_improves_both=gains,deployment_candidate=False)
    result=dict(scope='Oracle spatial token rejection; four observed identities, three inference seeds, no training',
        rows=rows,comparisons=contrasts,gate=gate,counts=dict(new_images=48,reused_baselines=24,neutral_replays=8,
        scored_rows=sum('metrics' in r for r in rows)),script_sha256=sha(__file__),config_sha256=sha(OUT/'config.json'),
        metrics_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),region_metrics_sha256=sha(ROOT/'scripts/region_metrics_v1.py'))
    write(OUT/'scores.json',result);write(ROOT/'research/spatial_reference_rejection_v1_results.json',result)
    lines=['# Spatial reference rejection results','', '6 October 2026. Fixed development diagnostic, no training or deployment.','',
        '48 intervention outputs, 24 reused baseline outputs and eight neutral baseline replays. Fixed central-quarter rejection versus a seeded equal-area corner control. Oracle location knowledge is unavailable to an automatic client-image method.','']
    for condition in ['clean','wrong_patch']:
        for arm,control in [('oracle_region','baseline'),('random_corner','baseline'),('oracle_region','random_corner')]:
            lines += [f'## {arm} versus {control}: {condition}','','| Metric | Intervention | Control | Delta | Pairs | Wins / losses |','|---|---:|---:|---:|---:|---:|']
            for k,v in contrasts[f'{arm}/{control}/{condition}/all'].items():
                fmt=lambda x:'unavailable' if x is None else f'{x:.6f}'
                lines.append(f"| {k} | {fmt(v['candidate_mean'])} | {fmt(v['control_mean'])} | {fmt(v['mean_delta'])} | {v['n_pairs']}/12 | {v['improved']} / {v['worsened']} |")
            lines.append('')
    lines += ['## Decision and limits','',f"Complete joint screen passed: **{gate['passes']}**. LPIPS improves in both conditions: **{gains}**.",
        '', 'Per-seed regressions and missing support remain in JSON. Four observed identities do not establish population generalization. Corrupt information can spread outside the rejected token window through receptive fields. No production change follows this diagnostic. Visual inspection is recorded separately.',
        '', '[Fixed protocol](SPATIAL_REFERENCE_REJECTION_PREFLIGHT_V1.md) · [Full values, pairs and bootstrap intervals](spatial_reference_rejection_v1_results.json)']
    (ROOT/'research/SPATIAL_REFERENCE_REJECTION_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE',json.dumps(gate),flush=True)


if __name__=='__main__':main()
