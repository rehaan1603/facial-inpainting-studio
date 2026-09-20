"""Score every frozen grid output, including the no-generation input control."""
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from extended_evaluation_metrics import Metrics,read_rgb
from distortion_aware_study import BASE,RESERVED


def main():
    deadline=time.monotonic()+10800
    while not (BASE/'comparison.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Generation incomplete after three hours')
        time.sleep(5)
    metrics=Metrics()
    comparison=json.loads((BASE/'comparison.json').read_text())
    cases={c['case_id']:c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases']}
    evaluation={c['case_id']:c for c in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    signature={'comparison_sha256':sha(BASE/'comparison.json'),'inference_manifest_sha256':sha(BASE/'inference_manifest.json'),
               'evaluation_manifest_sha256':sha(BASE/'evaluation_manifest.json'),'metric_source_sha256':sha(ROOT/'scripts/extended_evaluation_metrics.py'),'evaluator_sha256':sha(__file__)}
    lock=BASE/'evaluation_signature.json'
    if lock.exists():assert json.loads(lock.read_text())==signature
    else:write_new(lock,signature)
    features={};scored={};rows=[]
    for row in comparison['rows']:
        assert row['identity'] not in RESERVED
        dest=BASE/'scores'/(row['key']+'.json');c=cases[row['case_id']];e=evaluation[row['case_id']]
        if dest.exists():score=json.loads(dest.read_text())
        else:
            score={'status':'failed'}
            try:
                if row['status']!='complete':raise ValueError(row.get('error','Generation failed'))
                identity=row['identity']
                if identity not in features:
                    target=read_rgb(e['target']['path'],e['target']['sha256']);vectors,detection=metrics.features(target)
                    galleries=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in e['gallery']]
                    features[identity]=(target,vectors,detection,galleries)
                target,vectors,detection,galleries=features[identity]
                output=read_rgb(row['output'],row['output_sha256']);observed=read_rgb(c['observed'],c['observed_sha256'])
                mask=read_rgb(c['mask'],c['mask_sha256'])[:,:,0]>=128
                cache_key=(identity,c['mask_sha256'],row['output_sha256'])
                if cache_key in scored:score=dict(scored[cache_key],reused_identical_output=True)
                else:
                    values=metrics.score(output,target,observed,mask,vectors);ov,od=metrics.features(output)
                    for name in ['facenet','arcface_conditioning']:
                        sims=[float(np.dot(ov[name],v[name])) for v,s in galleries if v[name] is not None and ov[name] is not None]
                        values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None;values[name+'_gallery_valid_count']=len(sims)
                    score={'status':'complete','metrics':values,'target_detections':detection,'gallery_detections':[s for v,s in galleries],'output_sha256':row['output_sha256']}
                    scored[cache_key]=score
            except Exception as error:score['error']=f'{type(error).__name__}: {error}'
            write_new(dest,score)
        rows.append(dict(row,evaluation=score));print('DISTORTION SCORE',len(rows),'/',len(comparison['rows']),score['status'],flush=True)
    write_new(BASE/'evaluation.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})


if __name__=='__main__':main()
