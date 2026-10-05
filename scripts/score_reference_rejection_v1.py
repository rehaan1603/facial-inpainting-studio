"""All oracle-removal and zero-reference outcomes with matched available support."""
import json
import sys
import numpy as np
import torch
from extended_evaluation_metrics import ROOT, Metrics, read_rgb, sha
from score_target_compatibility_v1 import components
from region_metrics_v1 import score_regions

OUT=ROOT/'outputs/reference_rejection_v1'
KEYS=['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']
HIGH={'psnr','ssim','facenet_cosine','arcface_conditioning_cosine'}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def paired(rows,arm,control,condition):
    result={}
    for key in KEYS:
        pairs=[]
        for identity in ['263','6681','7400','10149']:
            for seed in [17,29,43]:
                a=next(r for r in rows if r['identity']==identity and r['seed']==seed and r['arm']==arm)
                b=next(r for r in rows if r['identity']==identity and r['seed']==seed and r['arm']==control
                       and (control!='baseline' or r['condition']==condition))
                x=a.get('metrics',{}).get(key);y=b.get('metrics',{}).get(key)
                if x is not None and y is not None:
                    pairs.append(dict(identity=identity,seed=seed,candidate=x,control=y,delta=x-y))
        units=[dict(identity=i,n_seeds=len(v),mean_delta=float(np.mean(v)))
               for i in sorted({p['identity'] for p in pairs})
               for v in [[p['delta'] for p in pairs if p['identity']==i]]]
        d=[p['delta'] for p in pairs]
        result[key]=dict(n_pairs=len(pairs),n_identities=len(units),pairs=pairs,identity_units=units,
            candidate_mean=float(np.mean([p['candidate'] for p in pairs])) if pairs else None,
            control_mean=float(np.mean([p['control'] for p in pairs])) if pairs else None,
            mean_delta=float(np.mean(d)) if d else None,
            identity_mean_delta=float(np.mean([u['mean_delta'] for u in units])) if units else None,
            improved=int(sum(x>1e-6 if key in HIGH else x < -1e-6 for x in d)),
            worsened=int(sum(x < -1e-6 if key in HIGH else x>1e-6 for x in d)))
    return result


def main():
    assert not (OUT/'scores.json').exists()
    complete=json.loads((OUT/'completion.json').read_text())
    assert sha(OUT/'evaluation/rows.json')==complete['rows_sha256']
    assert sha(OUT/'config.json')==complete['config_sha256']
    records=json.loads((OUT/'evaluation/rows.json').read_text())
    assert len(records)==48 and all(r['status']=='complete' for r in records)
    old_path=ROOT/'outputs/identity_timestep_v5/timestep_scaled/scores.json'
    old=json.loads(old_path.read_text())['rows']
    cases={r['identity']:r for r in json.loads((ROOT/'outputs/reference_intervention_pilot_v1/manifest.json').read_text())['cases']}
    resumed='--resume-completed-metrics' in sys.argv
    rows=[]
    if resumed:
        rows=json.loads((OUT/'scores.partial.json').read_text())
        assert len(rows)==len(records)==48 and all('metrics' in r for r in rows)
        for row,record in zip(rows,records):
            for key in ['identity','arm','condition','seed','sha256','file']:
                assert row[key]==record[key]
            assert sha(OUT/'evaluation'/row['file'])==row['sha256']
        records=[]
    m=None if resumed else Metrics();cache={}
    for record in records:
        row=dict(record)
        try:
            rgb=read_rgb(OUT/'evaluation'/row['file'],row['sha256'])
            if row['arm']=='baseline':
                previous=next(r for r in old if all(r[k]==row[k] for k in ['identity','arm','seed','condition']))
                assert previous['sha256']==row['sha256']
                for key in ['metrics','target_detections','output_detections','component_sources','regions']:
                    row[key]=previous[key]
                row['metrics_reused_from']=str(old_path.relative_to(ROOT))
            else:
                identity=row['identity']
                if identity not in cache:
                    folder=ROOT/'outputs/reference_intervention_pilot_v1/images'/identity
                    target=read_rgb(folder/'target.png');observed=read_rgb(folder/'input.png')
                    tv,td=m.features(target)
                    parts,receipts=components(cases[identity]['target']['hq_id'],target.shape[:2])
                    cache[identity]=(target,observed,tv,td,parts,receipts)
                target,observed,tv,td,parts,receipts=cache[identity]
                vec,det=m.features(rgb)
                regions=score_regions(m,rgb,target,observed,np.ones(target.shape[:2],bool),parts)
                values={k:regions['whole'][k] for k in ['psnr','ssim','mae']}
                with torch.inference_mode():
                    values['lpips']=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(target)*2-1))
                for key,v in vec.items():
                    values[key+'_cosine']=float(np.dot(v,tv[key])) if v is not None and tv[key] is not None else None
                values['eye_bbox_lpips']=regions['eyes']['bbox_lpips']
                row.update(metrics=values,target_detections=td,output_detections=det,component_sources=receipts,regions=regions)
        except Exception as e:
            row['evaluation_error']=repr(e)
        rows.append(row);write(OUT/'scores.partial.json',rows)
        print('SCORED',len(rows),row['identity'],row['arm'],row['seed'],'ok' if 'metrics' in row else row['evaluation_error'],flush=True)
    comparisons={}
    for arm in ['drop_first','zero_references']:
        for condition in ['clean','wrong_patch']:
            comparisons[f'{arm}/baseline/{condition}']=paired(rows,arm,'baseline',condition)
    comparisons['drop_first/zero_references']=paired(rows,'drop_first','zero_references',None)
    primary=comparisons['drop_first/baseline/wrong_patch']
    violations=[key for key,v in primary.items() if v['mean_delta'] is None or
                (v['mean_delta'] < -1e-6 if key in HIGH else v['mean_delta']>1e-6)]
    incomplete=[key for key,v in primary.items() if v['n_pairs']!=12]
    gate=dict(violations=violations,incomplete_metrics=incomplete,
        lpips_improves=primary['lpips']['mean_delta'] is not None and primary['lpips']['mean_delta'] < -1e-6,
        pass_complete_joint_diagnostic=not violations and not incomplete and primary['lpips']['mean_delta'] < -1e-6,
        deployment_candidate=False,reason='Oracle corruption location is unavailable to an automatic client-image method')
    result=dict(scope='Four observed identities and three seeds; oracle rejection feasibility, not a learned method',
        rows=rows,comparisons=comparisons,oracle_gate=gate,
        counts=dict(new_images=24,baseline_images_reused=24,replay_calls=8,scored_rows=sum('metrics' in r for r in rows),
                    new_metric_rows=24,metric_rows_reused=24),
        script_sha256=sha(__file__),previous_scores_sha256=sha(old_path),config_sha256=sha(OUT/'config.json'),
        metric_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),region_metric_sha256=sha(ROOT/'scripts/region_metrics_v1.py'),
        resumed_completed_metrics=resumed)
    write(OUT/'scores.json',result)
    write(ROOT/'research/reference_rejection_v1_results.json',result)
    print('SCORING COMPLETE',json.dumps(gate),flush=True)


if __name__=='__main__':main()
