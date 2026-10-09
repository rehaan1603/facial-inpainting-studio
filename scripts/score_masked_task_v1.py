"""Score every available masked diagnostic output with paired coverage reporting."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
from extended_evaluation_metrics import ROOT,Metrics,read_rgb,sha
from region_metrics_v1 import score_regions
from score_target_compatibility_v1 import components

OUT=ROOT/'outputs/masked_task_v1'


def main():
    cases={(c['identity'],c['location']):c for c in json.loads((OUT/'cases.json').read_text())}
    generated=json.loads((OUT/'evaluation/rows.json').read_text())
    primary_count=len(generated)
    native=OUT/'native_resshift/rows.json'
    if native.exists():
        generated += [dict(r,arm='native_resshift',condition='no_references',
                           file='../native_resshift/'+r['file']) for r in json.loads(native.read_text())]
    dest=OUT/'scores';dest.mkdir(exist_ok=True)
    metric=Metrics();targets={};rows=[]
    for r in generated:
        c=cases[r['identity'],r['location']];p=dest/(r['arm']+'_'+r['file'].replace('../','').replace('/','_')+'.json')
        assert sha(OUT/'evaluation'/r['file'])==r['sha256']
        if p.exists():
            cached=json.loads(p.read_text());assert cached['sha256']==r['sha256'];rows.append(cached);continue
        target=read_rgb(c['target'],c['target_sha256']);observed=read_rgb(c['input'],c['input_sha256'])
        assert sha(c['mask'])==c['mask_sha256'];mask=np.asarray(Image.open(c['mask']))>127
        rgb=read_rgb(OUT/'evaluation'/r['file']);assert np.array_equal(rgb[~mask],observed[~mask])
        if r['identity'] not in targets:targets[r['identity']]=metric.features(target)
        tv,td=targets[r['identity']];v,d=metric.features(rgb);parts,_=components(c['hq_id'],mask.shape)
        regions=score_regions(metric,rgb,target,observed,mask,parts)
        with torch.inference_mode():lp=float(metric.lpips(metric.tensor(rgb)*2-1,metric.tensor(target)*2-1))
        values=dict(psnr=regions['whole']['psnr'],ssim=regions['whole']['ssim'],lpips=lp,
                    hole_psnr=regions['masked']['psnr'],hole_ssim=regions['masked']['ssim'],hole_lpips=regions['masked']['bbox_lpips'],
                    eye_lpips=regions['eyes']['bbox_lpips'],visible_changed=regions['visible']['changed_pixels_vs_input'])
        for k in tv:values[k+'_cosine']=float(np.dot(tv[k],v[k])) if tv[k] is not None and v[k] is not None else None
        row=dict(r,metrics=values,regions=regions,target_detections=td,output_detections=d)
        p.write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row)
        print('SCORED',len(rows),r['file'],flush=True)
    contrasts={}
    for condition in ['clean','wrong_patch']:
      for location in ['eyes','mouth']:
        selected=[r for r in rows if r['condition']==condition and r['location']==location]
        lookup={(r['identity'],r['seed'],r['arm']):r for r in selected}
        result={}
        for key in ['psnr','ssim','lpips','hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']:
            pairs=[]
            for ident,seed,arm in lookup:
                if arm!='baseline':continue
                b=lookup[ident,seed,arm]['metrics'][key];a=lookup.get((ident,seed,'timestep_scaled'),{}).get('metrics',{}).get(key)
                if a is not None and b is not None:pairs.append(dict(identity=ident,seed=seed,baseline=b,candidate=a,delta=a-b))
            result[key]=dict(n=len(pairs),baseline=float(np.mean([p['baseline'] for p in pairs])) if pairs else None,
                             candidate=float(np.mean([p['candidate'] for p in pairs])) if pairs else None,pairs=pairs)
        contrasts[f'{condition}/{location}']=result
    native_rows=[r for r in rows if r['arm']=='native_resshift']
    native_summary={loc:{k:dict(n=len(v),mean=float(np.mean(v)) if v else None)
                         for k in ['hole_psnr','hole_ssim','hole_lpips','facenet_cosine','arcface_conditioning_cosine']
                         for v in [[r['metrics'][k] for r in native_rows if r['location']==loc and r['metrics'][k] is not None]]}
                    for loc in ['eyes','mouth']}
    report=dict(rows=primary_count,expected=96,complete=primary_count==96,paired=contrasts,native_rows=len(native_rows),native_summary=native_summary,
                scope='Erased-region development diagnostic, distinct from restoration. Eye LPIPS unavailable when eyes are outside hole.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    sheets=OUT/'sheets';sheets.mkdir(exist_ok=True)
    for (ident,loc),c in cases.items():
        images=[('Original',read_rgb(c['target'])),('Erased input',read_rgb(c['input']))]
        for arm in ['baseline','timestep_scaled']:
            match=[r for r in rows if r['identity']==ident and r['location']==loc and r['seed']==17 and r['condition']=='clean' and r['arm']==arm]
            if match:images.append((arm,read_rgb(OUT/'evaluation'/match[0]['file'])))
        match=[r for r in native_rows if r['identity']==ident and r['location']==loc]
        if match:images.append(('ResShift native256',read_rgb(OUT/'evaluation'/match[0]['file'])))
        canvas=Image.new('RGB',(256*len(images),286),'white');draw=ImageDraw.Draw(canvas)
        for i,(name,rgb) in enumerate(images):
            canvas.paste(Image.fromarray(rgb).resize((256,256)),(i*256,30));draw.text((i*256+5,8),name,fill='black')
        canvas.save(sheets/f'{ident}_{loc}.png')
    print(json.dumps({k:{m:{a:b for a,b in v.items() if a!='pairs'} for m,v in x.items()} for k,x in contrasts.items()},indent=2))


if __name__=='__main__':main()
