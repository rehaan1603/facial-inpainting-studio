"""Offline paired assessment after adapter selection; no automatic model promotion."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from extended_evaluation_metrics import Metrics, read_rgb, sha
from train_human_resshift_v1 import OUT, PACK, ROOT


def main():
    generation=json.loads((OUT/'generation.json').read_text())
    assert len(generation['rows'])==72, 'All 36 pairs must finish before evaluation'
    assert not (OUT/'scores.json').exists()
    metric=Metrics();rows=[];target_cache={}
    for record in generation['rows']:
        row=dict(record);folder=Path(row['result']).parent
        target=read_rgb(folder/'target.png');observed=read_rgb(folder/'input.png')
        m=read_rgb(folder/'mask.png')[:,:,0]>127
        if row['case_id'] not in target_cache:
            target_cache[row['case_id']]=metric.features(target)
        vectors,detections=target_cache[row['case_id']]
        rgb=read_rgb(row['result'],row['sha256'])
        try:
            values=metric.score(rgb,target,observed,m,vectors)
            ys,xs=np.nonzero(m)
            region=np.s_[ys.min():ys.max()+1,xs.min():xs.max()+1]
            # Expand a tiny crop to >=32 px to support AlexNet receptive fields.
            def crop(a):
                im=Image.fromarray(a[region]);w,h=im.size
                if min(w,h)<32:
                    im=im.resize((max(32,w),max(32,h)),Image.Resampling.BICUBIC)
                return np.asarray(im).copy()
            with torch.inference_mode():
                values['hole_bbox_lpips']=float(metric.lpips(metric.tensor(crop(rgb))*2-1,
                                                               metric.tensor(crop(target))*2-1).item())
            row.update(metrics=values,target_detections=detections)
        except Exception as error:
            row['evaluation_error']=str(error)
        rows.append(row)
        print('SCORED',row['case_id'],row['arm'],'ok' if 'metrics' in row else row['evaluation_error'],flush=True)
    keys=['hole_mae','hole_psnr','lpips','hole_bbox_lpips','facenet_cosine','arcface_conditioning_cosine','niqe','brisque']
    lookup={(r['case_id'],r['arm']):r for r in rows}
    summaries={}
    for group in ['new_holdout','previous_diagnostic']:
        selected=[r for r in rows if r['group']==group]
        groupsummary={}
        for key in keys:
            pairs=[]
            for case in sorted({r['case_id'] for r in selected}):
                values=[lookup[case,arm].get('metrics',{}).get(key) for arm in ['baseline','adapted']]
                if all(v is not None and np.isfinite(v) for v in values):
                    pairs.append((case,*values))
            if not pairs:
                groupsummary[key]={'common_n':0};continue
            diff=np.array([a-b for _,b,a in pairs]);lower=key not in ['hole_psnr','facenet_cosine','arcface_conditioning_cosine']
            # Resample whole photographs, keeping both mask conditions together.
            by_photo={}
            for case,b,a in pairs:
                photo='_'.join(case.split('_')[:2])
                by_photo.setdefault(photo,[]).append(a-b)
            clusters=np.array([np.mean(v) for v in by_photo.values()])
            rng=np.random.default_rng(20261002)
            boots=clusters[rng.integers(0,len(clusters),size=(2000,len(clusters)))].mean(1)
            groupsummary[key]=dict(common_n=len(pairs),photograph_n=len(clusters),
                baseline_mean=float(np.mean([b for _,b,_ in pairs])),adapted_mean=float(np.mean([a for _,_,a in pairs])),
                mean_delta=float(diff.mean()),improved=int(np.sum(diff<0 if lower else diff>0)),
                photo_bootstrap_95_interval=np.quantile(boots,[.025,.975]).tolist())
        summaries[group]=groupsummary
    delivery=PACK/('GPU adaptation trial - 2026-10-02' if OUT.name.endswith('_v1') else 'GPU perceptual adaptation trial - 2026-10-02');delivery.mkdir(exist_ok=True)
    for case_id in sorted({r['case_id'] for r in rows}):
        folder=Path(lookup[case_id,'baseline']['result']).parent
        sheet=Image.new('RGB',(1024,400),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,name) in enumerate([('Clean comparison only','target'),('Damaged input','input'),('Original ResShift','baseline'),('GPU adapted ResShift','adapted')]):
            im=Image.open(folder/f'{name}.png').convert('RGB')
            draw.text((i*256+5,7),label,fill='black');sheet.paste(im,(i*256,27))
            sheet.paste(im.crop((75,90,185,155)).resize((220,110)),(i*256+18,287))
        sheet.save(delivery/f'{case_id}.png')
    report=dict(scope='Small local development pilot; unknown person overlap and source rights; not independent publication evidence',
        selected_step=generation['selected_step'],summary=summaries,rows=rows,
        decision='Not automatically promoted. Read paired results and inspect failures; low-resolution targets do not validate high-resolution facial detail.',
        protocol_sha256=sha(OUT/'protocol.json'),training_sha256=sha(OUT/'training.json'),
        script_sha256=sha(__file__),delivery=str(delivery))
    (OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (delivery/'RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    public=dict(report)
    public['rows']=[{k:v for k,v in row.items() if k!='result'} for row in rows]
    public.pop('delivery')
    (ROOT/'research'/f'{OUT.name}.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
