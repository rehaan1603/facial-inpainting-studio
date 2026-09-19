"""Evaluate regional-routing development outputs after generation completes."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.research_integrity import ROOT,sha,write_new
from extended_evaluation_metrics import Metrics,read_rgb

def main():
    base=ROOT/'outputs/regional_routing_v1';old=ROOT/'outputs/generalization_v1'
    metrics=Metrics();deadline=time.monotonic()+3600
    print('Evaluator ready; waiting for completed generation manifest',flush=True)
    while not (base/'comparison.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Generation did not finish within one hour')
        time.sleep(5)
    comparison=json.loads((base/'comparison.json').read_text());case_path=old/'cases/manifest.json'
    cases={c['case_id']:c for c in json.loads(case_path.read_text())['cases']}
    signature={'comparison_sha256':sha(base/'comparison.json'),'cases_sha256':sha(case_path),'evaluator_sha256':sha(__file__),
               'metric_source_sha256':sha(ROOT/'scripts/extended_evaluation_metrics.py'),'previous_evaluation_sha256':sha(old/'evaluation.json')}
    lock=base/'evaluation_signature.json'
    if lock.exists():
        if json.loads(lock.read_text())!=signature:raise ValueError('Evaluation changed')
    else:write_new(lock,signature)
    oldrows=json.loads((old/'evaluation.json').read_text())['rows'];features={};results=[]
    for r in comparison['rows']:
        dest=base/'scores'/f"{r['case_id']}_{r['seed']}_{r['policy']}.json"
        if dest.exists():score=json.loads(dest.read_text())
        elif r['status']!='complete':score={'status':'failed','error':r.get('error')};write_new(dest,score)
        elif r['policy']=='concat':
            previous=next(x for x in oldrows if x['generation_key']==r['reused_generation_key'])
            score=dict(previous['evaluation'],reused_from='generalization_v1')
            if score['output_sha256']!=r['output_sha256'] or sha(r['output'])!=r['output_sha256']:raise ValueError('Old scored output changed')
            write_new(dest,score)
        else:
            c=cases[r['case_id']];identity=c['identity'];score={'status':'failed'}
            try:
                if identity not in features:
                    target=read_rgb(c['evaluation_only_target']['path'],c['evaluation_only_target']['sha256'])
                    vectors,status=metrics.features(target)
                    galleries=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in c['evaluation_only_gallery']]
                    features[identity]=(target,vectors,status,galleries)
                target,vectors,status,galleries=features[identity]
                observed=read_rgb(c['observed'],c['distortion']['observed_sha256'])
                if sha(c['mask'])!=c['distortion']['mask_sha256']:raise ValueError('Mask changed')
                mask=np.asarray(Image.open(c['mask']).convert('L'))>=128
                rgb=read_rgb(r['output'],r['output_sha256']);values=metrics.score(rgb,target,observed,mask,vectors)
                output_vec,output_detection=metrics.features(rgb)
                for name in ['facenet','arcface_conditioning']:
                    sims=[float(np.dot(output_vec[name],v[name])) for v,s in galleries if v[name] is not None and output_vec[name] is not None]
                    values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None;values[name+'_gallery_valid_count']=len(sims)
                score.update(status='complete',metrics=values,target_detections=status,gallery_detections=[s for v,s in galleries],output_sha256=r['output_sha256'])
            except Exception as error:score['error']=f'{type(error).__name__}: {error}'
            write_new(dest,score)
        results.append(dict(r,evaluation=score));print('SCORED ROUTING',len(results),'/144',score['status'],flush=True)
    write_new(base/'evaluation.json',{'signature_sha256':sha(lock),'rows':results,'final_test_used':False})

if __name__=='__main__':main()
