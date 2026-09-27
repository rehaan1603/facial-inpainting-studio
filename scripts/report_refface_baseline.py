"""Report all reference choices and prespecified comparisons for author baseline."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.context_statistics import identity_means, contrast, holm
from report_distortion_aware import METRICS

BASE=ROOT/'outputs/refface_baseline_v1'


def main():
    cfg=json.loads((ROOT/'research/protocols/refface_baseline_v1.json').read_text())
    signature=json.loads((BASE/'signature.json').read_text())
    for key,path in [('runner','scripts/run_refface_baseline.py'),('wrapper','src/preservation/refface_baseline.py'),('protocol','research/protocols/refface_baseline_v1.json')]:assert signature[key+'_sha256']==sha(ROOT/path)
    es=json.loads((BASE/'evaluation_signature.json').read_text())
    assert es['evaluator_sha256']==sha(ROOT/'scripts/evaluate_refface_baseline.py')
    assert es['comparison_sha256']==sha(BASE/'comparison.json')
    data=json.loads((BASE/'evaluation.json').read_text());rows=data['rows']
    assert len(rows)==28 and len({r['key'] for r in rows})==28
    groups={a:[r for r in rows if r['mode']==a] for a in cfg['arms']}
    summaries={}
    for arm,group in groups.items():
        item={'generation_complete':sum(r['status']=='complete' for r in group),
              'scored':sum(r['evaluation']['status']=='complete' for r in group)}
        for metric in METRICS:
            values,excluded=identity_means(group,metric,[17])
            item[metric]={'mean':float(np.mean(list(values.values()))) if values else None,
                          'identity_count':len(values),'excluded':excluded}
        summaries[arm]=item
    primary={f'reference_0_minus_{a}/{m}':contrast(groups['reference_0'],groups[a],m,[17])
             for a in cfg['primary_controls'] for m in cfg['primary_metrics']}
    holm(primary)
    sensitivity={}
    for cid in cfg['cases']:
        selected=[r for r in rows if r['case_id']==cid and r['mode'].startswith('reference_') and r['mode']!='reference_context']
        sensitivity[cid]={}
        for metric in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips']:
            values=[r['evaluation'].get('metrics',{}).get(metric) for r in selected]
            valid=[v for v in values if v is not None]
            sensitivity[cid][metric]={'valid_references':len(valid),'mean':float(np.mean(valid)) if valid else None,
                'range':float(max(valid)-min(valid)) if valid else None,
                'interpretation':'Sensitivity only; no best-reference selection'}
    failures=[];preserved=0
    inference={c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases']}
    safe=[]
    for r in rows:
        if r['status']=='complete':
            assert sha(r['output'])==r['output_sha256']
            c=inference[r['case_id']]
            with Image.open(c['observed']) as im:observed=np.asarray(im.convert('RGB'))
            with Image.open(c['mask']) as im:mask=np.asarray(im.convert('L'))>=128
            with Image.open(r['output']) as im:output=np.asarray(im.convert('RGB'))
            assert np.array_equal(observed[~mask],output[~mask]);preserved+=1
        for detector,status in r['evaluation'].get('metrics',{}).get('detections',{}).items():
            if status['status']!='ok':failures.append({'key':r['key'],'detector':detector,'status':status['status']})
        item={k:v for k,v in r.items() if k not in ['output','traceback','evaluation']}
        score=r['evaluation'];item['evaluation']={k:v for k,v in score.items() if k not in ['metrics','target_detections','gallery_detections']}
        if 'metrics' in score:
            item['evaluation']['metrics']={k:v for k,v in score['metrics'].items() if k!='detections'}
            item['evaluation']['detections']={d:{k:v for k,v in s.items() if k in ['status','face_count']} for d,s in score['metrics']['detections'].items()}
        safe.append(item)
    new=[r for r in rows if not r.get('reused_control')]
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'previously_observed_identities':4,
        'new_generations':len(new),'new_generations_complete':sum(r['status']=='complete' for r in new),
        'reused_controls':12,'scored':sum(r['evaluation']['status']=='complete' for r in rows),
        'pixel_preservation_verified_rows':preserved,'detector_failures':failures,'summaries':summaries,
        'primary_contrasts':primary,'reference_sensitivity':sensitivity,
        'new_output_hashes_unique':len({r.get('output_sha256') for r in new if r['status']=='complete'}),
        'mean_generation_seconds_including_parser':float(np.mean([r['generation_seconds'] for r in new if r['status']=='complete'])),
        'peak_gpu_allocated_bytes':max(r.get('peak_allocated_bytes',0) for r in new),
        'protocol_sha256':sha(ROOT/'research/protocols/refface_baseline_v1.json'),
        'evaluation_sha256':sha(BASE/'evaluation.json'),'reporter_sha256':sha(__file__),
        'statistics_sha256':sha(ROOT/'src/preservation/context_statistics.py'),
        'final_test_used':False,'novelty_established':False}
    write_new(ROOT/'research/refface_results_v1.json',result)
    write_new(ROOT/'research/refface_evidence_v1.json',{'signature':signature,'evaluation_signature':es,'rows':safe})
    f=lambda v:'NA' if v is None else f'{v:.6f}'
    lines=['# Published reference-guided inpainting baseline — 27 September 2026','',
        'This reproduces an external author model locally; it is not a new project contribution. Four previously observed development cases, four separately supplied reference choices per case, native256 inference and exact outside-mask512 composition. Reference0 was fixed as the primary choice before generation; no best-of-four selection.','',
        f"{result['new_generations_complete']}/16 new predictions; 12 reused controls; {result['scored']}/28 scored rows. All {preserved} successful rows preserve known pixels. {len(failures)} detector-failure events are retained in JSON. Means show their complete identity denominators; scores are not accuracy percentages.",'',
        '| Arm | FaceNet ↑ (n) | Gallery ↑ (n) | Hole MAE ↓ (n) | LPIPS ↓ (n) |',
        '|---|---:|---:|---:|---:|']
    for arm,s in summaries.items():lines.append('| '+arm+' | '+' | '.join(f"{f(s[m]['mean'])} ({s[m]['identity_count']})" for m in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips'])+' |')
    lines+=['','| Prespecified paired contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |','|---|---:|---:|---|---:|']
    for k,c in primary.items():lines.append(f"| {k} | {c['identities']} | {f(c['mean_delta'])} | {c['ci95']} | {f(c['p_holm'])} |")
    lines+=['','Four tests were specified before generation; identity is the statistical unit. Four references are not four independent people, and deterministic inference is not replicated diffusion sampling. With four identities, the smallest two-sided exact p is0.125; bootstrap intervals are unstable. Conditioning reference count differs from the four-reference ReF-LDM comparator. No equal-compute or pretraining-disjoint claim is made.', '',
        f"Mean new generation time including reference parsing, conversion and saving is {result['mean_generation_seconds_including_parser']:.3f}s (model loading excluded); peak allocated GPU memory is {result['peak_gpu_allocated_bytes']/1024**3:.3f}GiB. These are this laptop's screen measurements, not a general speed benchmark.",'',
        'Both author model and author ArcFace R101 plus BiSeNet reference parser were strict-loaded. An adapter check matches the author native inference method bit-for-bit on synthetic inputs. Only reference pixels enter parsing; no target truth, target parsing, target identity feature or gallery enters generation. Eight reserved final identities remain unopened. Native256 synthesis is upsampled, not a native512 model.', '',
        'Official sources: [paper](https://arxiv.org/abs/2303.07014), [author code](https://github.com/WuyangLuo/RefFaceInpainting), [reference parser](https://github.com/zllrunning/face-parsing.PyTorch). Exact commits, weight hashes, frozen inputs and numerical evidence are retained. See `REFFACE_BASELINE_REVIEW_V1.md` for visual findings and the resulting research decision.']
    (ROOT/'research/REFFACE_BASELINE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    evaluation={c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for cid in cfg['cases']:
        labels=[('clean scoring target',evaluation[cid]['target']['path'])]+[(a,next(r.get('output') for r in groups[a] if r['case_id']==cid)) for a in ['observed','scaffold','reference_context','reference_0','reference_1','reference_2','reference_3']]
        sheet=Image.new('RGB',(1024,560),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate(labels):
            x,y=i%4*256,i//4*280;draw.text((x+3,y+4),label,fill='black')
            if path:
                with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
            else:draw.text((x+4,y+70),'FAILED',fill='red')
        folder=BASE/'review';folder.mkdir(exist_ok=True);sheet.save(folder/(cid+'.jpg'))
    print(json.dumps({'scored':result['scored'],'primary_means':{a:{m:s[m] for m in ['facenet_cosine','hole_mae','lpips']} for a,s in summaries.items()},'detector_failures':failures},indent=2))


if __name__=='__main__':main()
