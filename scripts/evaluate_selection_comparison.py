"""Post-generation scoring only. Clean targets and galleries never enter inference."""
import json
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.research_integrity import ROOT,sha,write_new
from extended_evaluation_metrics import Metrics,read_rgb

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-generation',action='store_true');args=parser.parse_args()
    base=ROOT/'outputs/generalization_v1'
    metrics=Metrics()
    if args.wait_for_generation:
        print('Metrics loaded; waiting for the complete generation manifest.',flush=True)
        deadline=time.monotonic()+1800
        while not (base/'comparison.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Generation manifest did not finish within 30 minutes')
            time.sleep(5)
    comparison=json.loads((base/'comparison.json').read_text())
    cases={c['case_id']:c for c in json.loads((base/'cases/manifest.json').read_text())['cases']}
    provenance={'comparison_sha256':sha(base/'comparison.json'),'evaluator_sha256':sha(__file__),
                'frozen_metric_source_sha256':sha(ROOT/'scripts/extended_evaluation_metrics.py'),'clean_target_scope':'post-generation evaluation only'}
    path=base/'evaluation_signature.json'
    if path.exists():
        if json.loads(path.read_text())!=provenance:raise ValueError('Evaluation signature changed')
    else:write_new(path,provenance)
    features={};case_cache={};scores={};rows=[]
    for logical in comparison['logical_rows']:
        case=cases[logical['case_id']];key=logical['generation_key'];identity=case['identity']
        if identity not in features:
            target=read_rgb(case['evaluation_only_target']['path'],case['evaluation_only_target']['sha256'])
            target_vec,target_status=metrics.features(target)
            galleries=[]
            for g in case['evaluation_only_gallery']:
                vector,status=metrics.features(read_rgb(g['path'],g['sha256']));galleries.append((vector,status))
            features[identity]=(target,target_vec,target_status,galleries)
        if logical['case_id'] not in case_cache:
            obs=read_rgb(case['observed'],case['distortion']['observed_sha256'])
            if sha(case['mask'])!=case['distortion']['mask_sha256']:raise ValueError('Mask changed')
            mask=np.asarray(Image.open(case['mask']).convert('L'))>=128
            case_cache[logical['case_id']]=(obs,mask)
        if key not in scores:
            path=base/'scores'/(key+'.json')
            if path.exists():score=json.loads(path.read_text())
            else:
                record=json.loads((base/'generations'/key/'record.json').read_text());score={'generation_key':key,'status':record['status']}
                if record['status']=='complete':
                    try:
                        rgb=read_rgb(record['output'],record['output_sha256']);target,vec,det,galleries=features[identity];obs,mask=case_cache[logical['case_id']]
                        result=metrics.score(rgb,target,obs,mask,vec);output_vec,output_status=metrics.features(rgb)
                        for name in ['facenet','arcface_conditioning']:
                            similarities=[float(np.dot(output_vec[name],v[name])) for v,s in galleries if v[name] is not None and output_vec[name] is not None]
                            result[name+'_gallery_cosine']=float(np.mean(similarities)) if similarities else None
                            result[name+'_gallery_valid_count']=len(similarities)
                        score.update(status='complete',metrics=result,target_detections=det,gallery_detections=[s for v,s in galleries],output_sha256=record['output_sha256'])
                    except Exception as error:score.update(status='failed',error=f'{type(error).__name__}: {error}')
                else:score['error']=record.get('error')
                write_new(path,score)
            scores[key]=score
        rows.append(dict(logical,evaluation=scores[key]));print('SCORED',len(rows),'/168',scores[key]['status'],flush=True)
    write_new(base/'evaluation.json',{'signature_sha256':sha(base/'evaluation_signature.json'),'rows':rows,'final_test_used':False})

if __name__=='__main__':main()
