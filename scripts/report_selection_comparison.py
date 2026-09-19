"""Identity-level paired analysis; no image assets are published to the repository."""
import html
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from src.research_integrity import ROOT,sha,write_new

def paired_summary(rows,other,metric):
    indexed={(r['identity'],r['condition'],r['seed'],r['policy']):r for r in rows}
    per_identity={};pairs=0
    for key,r in indexed.items():
        identity,condition,seed,policy=key
        if policy!='mask_aware':continue
        b=indexed.get((identity,condition,seed,other))
        if b is None:continue
        a_value=r['evaluation'].get('metrics',{}).get(metric);b_value=b['evaluation'].get('metrics',{}).get(metric)
        if a_value is None or b_value is None:continue
        per_identity.setdefault(identity,[]).append(a_value-b_value);pairs+=1
    deltas=np.array([np.mean(v) for v in per_identity.values()])
    if not len(deltas):return {'paired_rows':0,'identities':0,'mean_delta':None,'p_exact':None,'interval95':None}
    mean=float(deltas.mean());permuted=[abs(float(np.mean(deltas*np.array(sign)))) for sign in itertools.product([-1,1],repeat=len(deltas))]
    p=float(np.mean(np.array(permuted)>=abs(mean)-1e-12))
    rng=np.random.default_rng(20260919);boot=rng.choice(deltas,(10000,len(deltas)),replace=True).mean(1)
    return {'paired_rows':pairs,'identities':len(deltas),'mean_delta':mean,'p_exact':p,'interval95':np.quantile(boot,[.025,.975]).tolist(),'identity_deltas':{k:float(np.mean(v)) for k,v in per_identity.items()}}

def main():
    base=ROOT/'outputs/generalization_v1';data=json.loads((base/'evaluation.json').read_text());rows=data['rows']
    policies=['first','random','identity_only','quality_only','all','mask_aware','zero_scale']
    metrics=['facenet_cosine','facenet_gallery_cosine','arcface_conditioning_cosine','arcface_conditioning_gallery_cosine','niqe','brisque','hole_mae','visible_mae','psnr_rgb','hole_psnr','ssim_rgb','lpips']
    summary={}
    for policy in policies:
        selected=[r for r in rows if r['policy']==policy];result={'logical_rows':len(selected),'evaluation_failures':sum(r['evaluation']['status']!='complete' for r in selected)}
        for metric in metrics:
            values=[r['evaluation'].get('metrics',{}).get(metric) for r in selected];values=[v for v in values if v is not None]
            result[metric]={'mean':float(np.mean(values)) if values else None,'valid':len(values)}
        summary[policy]=result
    paired={p:paired_summary(rows,p,'facenet_cosine') for p in ['random','identity_only','quality_only','all']}
    ordered=sorted(paired,key=lambda p:paired[p]['p_exact'] if paired[p]['p_exact'] is not None else 1)
    previous=0.
    for i,p in enumerate(ordered):
        value=paired[p]['p_exact'];adjusted=max(previous,min(1.,(4-i)*(value if value is not None else 1.)))
        paired[p]['p_holm']=adjusted;previous=adjusted
    breakdown={}
    for condition in sorted({r['condition'] for r in rows}):
        breakdown[condition]={}
        for policy in policies:
            group=[r for r in rows if r['condition']==condition and r['policy']==policy]
            values=[r['evaluation'].get('metrics',{}).get('facenet_cosine') for r in group]
            values=[v for v in values if v is not None]
            breakdown[condition][policy]={'mean_facenet_cosine':float(np.mean(values)) if values else None,'valid':len(values),'rows':len(group)}
    failures=[{'case_id':r['case_id'],'seed':r['seed'],'policy':r['policy'],'error':r['evaluation'].get('error'),
               'detections':r['evaluation'].get('metrics',{}).get('detections')} for r in rows
              if r['evaluation']['status']!='complete' or r['evaluation'].get('metrics',{}).get('facenet_cosine') is None or r['evaluation'].get('metrics',{}).get('arcface_conditioning_cosine') is None]
    evidence={'evaluation_sha256':sha(base/'evaluation.json'),'summary':summary,'primary_paired_facenet':paired,'condition_breakdown':breakdown,'failures_or_missing_identity_metrics':failures,
              'interpretation':'Four validation identities, not a final test. Bootstrap intervals unstable at n=4. ArcFace reuses the conditioning encoder. Missing detections remain explicit. No publication-ready superiority claim.'}
    write_new(base/'summary.json',evidence);write_new(ROOT/'research/generalization_selection_results_v1.json',evidence)
    def fmt(v):return 'NA' if v is None else f'{v:.4f}'
    text=['# Generalization results — selection v1','', 'Status: completed small local validation experiment; final-test identities remain reserved.','',
          'Four locally fresh identity groups × three damage conditions × two seeds × seven policies = 168 logical rows. Identical generation requests are cached, not represented as independent samples. No model was trained in this cycle.','',
          '| Policy | FaceNet cosine ↑ (valid/24) | Gallery FaceNet ↑ | NIQE ↓ | BRISQUE ↓ | Hole MAE ↓ |','|---|---:|---:|---:|---:|---:|']
    for p,r in summary.items():text.append(f"| {p} | {fmt(r['facenet_cosine']['mean'])} ({r['facenet_cosine']['valid']}/24) | {fmt(r['facenet_gallery_cosine']['mean'])} | {fmt(r['niqe']['mean'])} | {fmt(r['brisque']['mean'])} | {fmt(r['hole_mae']['mean'])} |")
    text+=['','## Primary paired comparisons','', 'Identity means of jointly valid condition/seed pairs. Positive delta favors mask-aware selection. Holm correction covers these four comparisons.','', '| Compared with | Identities | Paired rows | Mean delta | Bootstrap 95% interval | Exact p / Holm p |','|---|---:|---:|---:|---|---|']
    for p,r in paired.items():text.append(f"| {p} | {r['identities']} | {r['paired_rows']} | {fmt(r['mean_delta'])} | {r['interval95']} | {fmt(r['p_exact'])} / {fmt(r['p_holm'])} |")
    text+=['','## Condition breakdown','', '| Damage | Policy | FaceNet mean | Valid / 8 |','|---|---|---:|---:|']
    for condition,policies_data in breakdown.items():
        for p,r in policies_data.items():text.append(f"| {condition} | {p} | {fmt(r['mean_facenet_cosine'])} | {r['valid']} / 8 |")
    text+=['',f'Rows with failed generation/evaluation or missing FaceNet/ArcFace target similarity: **{len(failures)}**. Full statuses, metric-specific coverage and all complementary metric means are in `generalization_selection_results_v1.json`; the numerical evidence ledger retains every row.']
    text+=['','## Interpretation and limitations','',evidence['interpretation'],'',
           'Selection uses damaged input, mask and runtime reference photos only. Dataset identity labels are used for experimental grouping, never by the inference scorer. Unknown pretrained exposure prevents a pretraining-independent generalization claim.',
           '', 'The scorer selects one global FaceID embedding; it does not inject spatial facial details. The all-reference arm has a larger information budget. Zero-scale disables identity contribution within the same adapter graph; it is not a separate LaMa or no-adapter baseline.',
           '', 'Noise may receive an artificially high sharpness score. Pose and visibility are five-landmark proxies, not measured occlusion. Synthetic regional degradation with a supplied mask does not establish blind restoration, natural occlusion robustness, or global deblurring.',
           '', 'Inspect local outputs/generalization_v1/contact_sheets and every failed row before drawing quality conclusions. Final evaluation, external methods, broader identities/severity, human assessment and a justified research contribution remain outstanding.']
    (ROOT/'research/GENERALIZATION_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    # All cases, both seeds, every policy: no cherry-picked local gallery.
    cases={c['case_id']:c for c in json.loads((base/'cases/manifest.json').read_text())['cases']}
    folder=base/'contact_sheets';folder.mkdir(exist_ok=True)
    for cid,case in cases.items():
        for seed in [17,29]:
            images=[('Clean evaluation target',case['evaluation_only_target']['path']),('Damaged input',case['observed'])]
            for policy in policies:
                r=next(r for r in rows if r['case_id']==cid and r['seed']==seed and r['policy']==policy)
                record=json.loads((base/'generations'/r['generation_key']/'record.json').read_text())
                images.append((policy,record.get('output')))
            sheet=Image.new('RGB',(3*256,3*280),'white');draw=ImageDraw.Draw(sheet)
            for i,(label,path) in enumerate(images):
                x,y=(i%3)*256,(i//3)*280;draw.text((x+5,y+4),label,fill='black')
                if path:
                    with Image.open(path) as image:sheet.paste(image.convert('RGB').resize((256,256)),(x,y+24))
                else:draw.text((x+10,y+80),'GENERATION FAILED',fill='red')
            sheet.save(folder/f'{cid}_seed{seed}.png')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
