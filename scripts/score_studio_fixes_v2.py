"""Frozen evaluator for separately recorded client-input/runtime verification."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from extended_evaluation_metrics import Metrics,read_rgb
from unfamiliar_studio_audit_v2 import BASE,cases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--correct-effective-masks',action='store_true');args=parser.parse_args()
    sources=[ROOT/'research/studio_fixes_verification_v2.json',ROOT/'research/studio_client_inputs_verification_v2.json',ROOT/'research/studio_client_inputs_verification_v2_recovery.json']
    planned=[]
    for file in sources:
        for row in json.loads(file.read_text())['rows']:
            if row['status']!='complete':continue
            if row.get('check') in ['portrait_api','thin_mask']:continue
            if args.correct_effective_masks and row.get('mode','painted')=='painted':continue
            planned.append(dict(row,source_receipt=file.name))
    inputs={c['case_id']:c for c in cases()}
    evaluation={e['case_id']:e for e in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    audit=json.loads((ROOT/'research/unfamiliar_studio_accuracy_v2.json').read_text())['rows']
    old={(r['case_id'],r['arm']):r['evaluation'] for r in audit}
    metrics=Metrics();cache={};rows=[]
    for row in planned:
        c=inputs[row['case_id']];e=evaluation[row['case_id']]
        folder=ROOT/'outputs/webapp_runs'/row['job_id']
        output=folder/('preserved.png' if row['backbone']=='refldm' else 'result.png')
        score={'status':'failed'}
        try:
            if c['identity'] not in cache:
                target=read_rgb(e['target']['path'],e['target']['sha256']);tv,td=metrics.features(target)
                gallery=[metrics.features(read_rgb(g['path'],g['sha256'])) for g in e['gallery']]
                cache[c['identity']]=(target,tv,td,gallery)
            target,tv,td,gallery=cache[c['identity']]
            rgb=read_rgb(output,row['output_sha256']);observed=read_rgb(c['observed'],c['observed_sha256'])
            mask=read_rgb(c['mask'],c['mask_sha256'])[:,:,0]>=128
            if args.correct_effective_masks:
                from PIL import Image
                effective=np.asarray(Image.open(folder/'effective_mask.png').convert('L'))>=128
            else:effective=mask
            values=metrics.score(rgb,target,observed,effective,tv);ov,od=metrics.features(rgb)
            if args.correct_effective_masks:
                values['original_damage_mae']=float(np.abs(rgb.astype(float)-target.astype(float))[mask].mean()/255)
                values['original_visible_change_mae']=float(np.abs(rgb.astype(float)-observed.astype(float))[~mask].mean()/255)
                row['evaluation_mask_sha256']=sha(folder/'effective_mask.png')
            for name in ['facenet','arcface_conditioning']:
                sims=[float(np.dot(ov[name],v[name])) for v,s in gallery if ov[name] is not None and v[name] is not None]
                values[name+'_gallery_cosine']=float(np.mean(sims)) if sims else None
                values[name+'_gallery_valid_count']=len(sims)
            score=dict(status='complete',metrics=values,output_detections=od,target_detections=td)
            if row.get('mode','painted')=='painted':
                arm='refldm_partial' if row['backbone']=='refldm' else row['backbone']
                if row['backbone']=='reference':arm='reference_'+('1024' if row['detail']=='detailed' else '512')
                control=old.get((c['case_id'],arm),{})
                if control.get('status')=='complete':
                    score['delta_vs_original_same_method']={key:values[key]-control['metrics'][key] for key in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips'] if values.get(key) is not None and control['metrics'].get(key) is not None}
        except Exception as error:score['error']=str(error)
        row.pop('reference_preparation',None);rows.append(dict(row,evaluation=score))
        print(row['case_id'],row.get('check',row.get('framing')),score['status'],flush=True)
    suffix='_mask_corrected' if args.correct_effective_masks else ''
    scope='Post-fix descriptive development scoring. No new independent identities, no novelty/generalization claim. Initial failures remain in original receipts.'
    scope+=(' Effective mask is passed to the strict preservation evaluator; original_damage_mae separately measures the original damage region. This corrects the initial caller mask mismatch for expanded/refined runs without changing model outputs or frozen evaluator.' if args.correct_effective_masks else ' Mask error uses original damage; expanded/refined edits may change original visible pixels.')
    write_new(ROOT/('research/studio_fixes_quality_v2'+suffix+'.json'),dict(scope=scope,metric_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),source_receipt_sha256={f.name:sha(f) for f in sources},rows=rows,final_test_used=False))


if __name__=='__main__':main()
