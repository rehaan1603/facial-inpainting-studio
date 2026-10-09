"""Frozen masked metrics and visual comparisons; retain failure and coverage counts."""
import argparse,json
import numpy as np
import torch
from PIL import Image,ImageDraw
from extended_evaluation_metrics import ROOT,Metrics,sha,read_rgb
from region_metrics_v1 import score_regions
from score_target_compatibility_v1 import components

BASE=ROOT/'outputs/masked_task_v1';OUT=ROOT/'outputs/masked_sdxl_v1'


def main(resolution=512,finish=False,scale_probe=False):
    global OUT
    if resolution==1024:OUT=ROOT/'outputs/masked_sdxl_resolution_v1'
    if finish:OUT=ROOT/'outputs/focused_reference_finish_v1'
    if scale_probe:OUT=ROOT/'outputs/focused_reference_strength_v1'
    cases={(c['identity'],c['location']):c for c in json.loads((BASE/'cases.json').read_text())}
    config=json.loads((OUT/'config.json').read_text());assert sha(BASE/'cases.json')==config['cases_sha256']
    generated=json.loads((OUT/'rows.json').read_text());dest=OUT/'scores';dest.mkdir(exist_ok=True)
    m=Metrics();targets={};rows=[]
    for r in generated:
        if r['status']!='complete':rows.append(r);continue
        path=dest/(r['file']+'.json');assert sha(OUT/r['file'])==r['sha256']
        assert sha(OUT/r['file'].replace('.png','_raw.png'))==r['raw_sha256']
        if path.exists():
            row=json.loads(path.read_text());assert row['sha256']==r['sha256'];rows.append(row);continue
        c=cases[r['identity'],r['location']];target=read_rgb(c['target'],c['target_sha256']);observed=read_rgb(c['input'],c['input_sha256'])
        assert sha(c['mask'])==c['mask_sha256'];mask=np.asarray(Image.open(c['mask']))>127
        rgb=read_rgb(OUT/r['file']);assert np.array_equal(rgb[~mask],observed[~mask])
        if r['identity'] not in targets:targets[r['identity']]=m.features(target)
        tv,td=targets[r['identity']];v,d=m.features(rgb);parts,_=components(c['hq_id'],mask.shape)
        regions=score_regions(m,rgb,target,observed,mask,parts)
        with torch.inference_mode():lp=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(target)*2-1))
        values=dict(psnr=regions['whole']['psnr'],ssim=regions['whole']['ssim'],lpips=lp,
                    hole_psnr=regions['masked']['psnr'],hole_ssim=regions['masked']['ssim'],hole_lpips=regions['masked']['bbox_lpips'],
                    eye_lpips=regions['eyes']['bbox_lpips'],visible_changed=regions['visible']['changed_pixels_vs_input'])
        for k in tv:values[k+'_cosine']=float(np.dot(tv[k],v[k])) if tv[k] is not None and v[k] is not None else None
        row=dict(r,metrics=values,regions=regions,target_detections=td,output_detections=d)
        path.write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row);print('SCORED',len(rows),r['file'],flush=True)
    summary={}
    for condition in config['conditions']:
      for location in ['eyes','mouth']:
        subset=[r for r in rows if r['condition']==condition and r['location']==location]
        summary[condition+'/'+location]=dict(attempted=len(subset),complete=sum(r['status']=='complete' for r in subset),
            metrics={k:dict(n=len(v),mean=float(np.mean(v)) if v else None) for k in ['hole_psnr','hole_ssim','hole_lpips','facenet_cosine','arcface_conditioning_cosine','eye_lpips']
                     for v in [[r.get('metrics',{}).get(k) for r in subset if r.get('metrics',{}).get(k) is not None]]})
    result=dict(attempted=len(rows),expected=config['expected'],all_attempted=len(rows)==config['expected'],all_completed=len(rows)==config['expected'] and all(r['status']=='complete' for r in rows),
                summary=summary,rows=rows,config_sha256=sha(OUT/'config.json'),scorer_sha256=sha(__file__),
                limitations=['Development screen only','Different architectures, sampling settings and reference conditioning across backends',
                             'ArcFace embeddings condition this model; FaceNet is separate from conditioning','No training in this experiment'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    sheets=OUT/'sheets';sheets.mkdir(exist_ok=True)
    for (ident,loc),c in cases.items():
        match=[r for r in rows if r['identity']==ident and r['location']==loc and r['seed']==17 and r['condition']=='clean' and r['status']=='complete']
        if not match:continue
        images=[('Original',read_rgb(c['target'])),('Erased input',read_rgb(c['input'])),
                ('ResShift native256',read_rgb(BASE/'native_resshift'/f'{ident}_{loc}.png')),
                ('SDXL + 4 references',read_rgb(OUT/match[0]['file']))]
        canvas=Image.new('RGB',(1024,286),'white');draw=ImageDraw.Draw(canvas)
        for i,(name,rgb) in enumerate(images):canvas.paste(Image.fromarray(rgb).resize((256,256)),(i*256,30));draw.text((i*256+5,8),name,fill='black')
        canvas.save(sheets/f'{ident}_{loc}.png')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,choices=[512,1024],default=512);p.add_argument('--finish',action='store_true');p.add_argument('--scale-probe',action='store_true');a=p.parse_args();main(a.resolution,a.finish,a.scale_probe)
