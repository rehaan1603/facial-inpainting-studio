"""Score external native/composited outputs; frozen metric definitions remain unchanged.

For native whole-image restoration, the frozen preservation assertion is inapplicable.
Passing output as the assertion reference changes no metric arithmetic; actual observed
image differences are recorded separately. Composited outputs retain the original check.
"""
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from extended_evaluation_metrics import Metrics,read_rgb
from distortion_aware_study import BASE,RESERVED
OUT=ROOT/'outputs/reference_risk_diagnostic_v1'


def main():
    deadline=time.monotonic()+7200
    while not (OUT/'comparison.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('ReF-LDM generation incomplete')
        time.sleep(5)
    metrics=Metrics();data=json.loads((OUT/'comparison.json').read_text())
    cases={c['case_id']:c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases']}
    evaluation={c['case_id']:c for c in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    signature={'comparison_sha256':sha(OUT/'comparison.json'),'evaluator_sha256':sha(__file__),'metric_source_sha256':sha(ROOT/'scripts/extended_evaluation_metrics.py'),'evaluation_manifest_sha256':sha(BASE/'evaluation_manifest.json')}
    lock=OUT/'evaluation_signature.json'
    if lock.exists():assert json.loads(lock.read_text())==signature
    else:write_new(lock,signature)
    features={};rows=[]
    for row in data['rows']:
        assert row['identity'] not in RESERVED
        dest=OUT/'scores'/(row['key']+'.json')
        if dest.exists():score=json.loads(dest.read_text())
        else:
            score={'status':'failed'}
            try:
                if row['status']!='complete':raise ValueError(row.get('error'))
                c=cases[row['case_id']];e=evaluation[row['case_id']]
                if row['identity'] not in features:
                    target=read_rgb(e['target']['path'],e['target']['sha256']);tv,td=metrics.features(target)
                    galleries=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in e['gallery']]
                    features[row['identity']]=(target,tv,td,galleries)
                target,tv,td,galleries=features[row['identity']]
                output=read_rgb(row['output'],row['output_sha256']);observed=read_rgb(c['observed'],c['observed_sha256']);mask=read_rgb(c['mask'],c['mask_sha256'])[:,:,0]>=128
                values=metrics.score(output,target,output if row['mode']=='native' else observed,mask,tv)
                values['outside_observation_mae']=float(np.abs(output.astype(float)[~mask]-observed.astype(float)[~mask]).mean()/255)
                values['known_pixels_unchanged']=bool(np.array_equal(output[~mask],observed[~mask]))
                ov,od=metrics.features(output)
                for name in ['facenet','arcface_conditioning']:
                    sims=[float(np.dot(ov[name],v[name])) for v,s in galleries if v[name] is not None and ov[name] is not None]
                    values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None;values[name+'_gallery_valid_count']=len(sims)
                score={'status':'complete','metrics':values,'target_detections':td,'gallery_detections':[s for v,s in galleries],'output_sha256':row['output_sha256']}
            except Exception as error:score['error']=f'{type(error).__name__}: {error}'
            write_new(dest,score)
        rows.append(dict(row,evaluation=score));print('RISK SCORE',len(rows),'/12',score['status'],flush=True)
    write_new(OUT/'evaluation.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})


if __name__=='__main__':main()
