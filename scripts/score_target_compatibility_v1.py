"""Frozen regional evaluation and identity-cluster summaries; no outcome selection."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from extended_evaluation_metrics import Metrics,read_rgb,sha,ROOT
from region_metrics_v1 import score_regions
from prepare_target_compatibility_v1 import OUT,BASE


def components(hq_id,shape):
    source=Path.home()/'Downloads/archive/CelebAMask-HQ/CelebAMask-HQ-mask-anno'/str(int(hq_id)//2000)
    result={};used=np.zeros(shape,bool);receipts={}
    for name,parts in {'left_eye':['l_eye'],'right_eye':['r_eye'],'nose':['nose'],'mouth':['mouth','u_lip','l_lip']}.items():
        mask=np.zeros(shape,bool);found=[]
        for part in parts:
            p=source/f'{int(hq_id):05d}_{part}.png'
            if p.exists():
                with Image.open(p) as im:mask|=np.asarray(im.convert('L').resize(shape[::-1],Image.Resampling.NEAREST))>127
                found.append(dict(file=p.name,sha256=sha(p)))
        result[name]=mask;used|=mask;receipts[name]=found
    result['eyes']=result['left_eye']|result['right_eye'];result['other']=~used
    return result,receipts


def paired(rows,arm,control,conditions,key):
    lookup={(r['identity'],r['arm'],r['condition']):r for r in rows}
    units=[]
    for identity in sorted({r['identity'] for r in rows}):
        a=[];b=[]
        for condition in conditions:
            left=lookup.get((identity,arm,condition),{}).get('metrics',{}).get(key)
            right=lookup.get((identity,control,condition),{}).get('metrics',{}).get(key)
            if left is not None and right is not None and np.isfinite(left) and np.isfinite(right):a.append(left);b.append(right)
        if a:units.append(dict(identity=identity,candidate=float(np.mean(a)),control=float(np.mean(b)),paired_conditions=len(a)))
    if not units:return dict(n_identities=0)
    d=np.array([r['candidate']-r['control'] for r in units]);rng=np.random.default_rng(20261002)
    bootstrap=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
    lower=key not in ['psnr','ssim','facenet_cosine','arcface_conditioning_cosine']
    result=dict(n_identities=len(d),units=units,mean_delta=float(d.mean()),median_delta=float(np.median(d)),
        sd_delta=float(d.std(ddof=1)) if len(d)>1 else None,descriptive_95_ci=np.quantile(bootstrap,[.025,.975]).tolist(),
        improved=int(np.sum(d< -1e-9 if lower else d>1e-9)),worsened=int(np.sum(d>1e-9 if lower else d< -1e-9)),
        unchanged=int(np.sum(np.abs(d)<=1e-9)),candidate_mean=float(np.mean([r['candidate'] for r in units])),
        control_mean=float(np.mean([r['control'] for r in units])))
    if key in ['lpips','mae'] and result['control_mean']>0:
        result['relative_error_reduction_percent']=-100*result['mean_delta']/result['control_mean']
    return result


def main():
    generation=json.loads((OUT/'evaluation/rows.json').read_text());assert len(generation)==96
    assert not (OUT/'scores.json').exists()
    protocol=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((BASE/'manifest.json').read_text())
    cases={r['identity']:r for r in manifest['cases']};metrics=Metrics();cache={};rows=[]
    for record in generation:
        row=dict(record)
        if row['status']!='complete':rows.append(row);continue
        identity=row['identity']
        if identity not in cache:
            target=read_rgb(BASE/'images'/identity/'target.png');observed=read_rgb(BASE/'images'/identity/'input.png')
            vectors,detections=metrics.features(target);parts,receipts=components(cases[identity]['target']['hq_id'],target.shape[:2])
            cache[identity]=(target,observed,vectors,detections,parts,receipts)
        target,observed,vectors,detections,parts,receipts=cache[identity]
        try:
            rgb=read_rgb(OUT/'evaluation'/row['file'],row['sha256'])
            predicted,pdetections=metrics.features(rgb);values=metrics.quality(rgb)
            for name,vector in predicted.items():
                values[name+'_cosine']=float(np.dot(vector,vectors[name])) if vector is not None and vectors[name] is not None else None
            region=score_regions(metrics,rgb,target,observed,np.ones(target.shape[:2],bool),parts)
            for key in ['psnr','ssim','mae']:values[key]=region['whole'][key]
            with torch.inference_mode():values['lpips']=float(metrics.lpips(metrics.tensor(rgb)*2-1,metrics.tensor(target)*2-1).item())
            row.update(metrics=values,regions=region,target_detections=detections,output_detections=pdetections,component_sources=receipts,
                region_scope='Entire input is degraded. Masked=whole image; visible is empty. Component scores use official target annotations only offline.')
        except Exception as error:row['evaluation_error']=str(error)
        rows.append(row);(OUT/'scores.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        print('SCORED',identity,row['arm'],row['condition'],'ok' if 'metrics' in row else row.get('evaluation_error'),flush=True)
    contrasts={};keys=['lpips','psnr','ssim','mae','facenet_cosine','arcface_conditioning_cosine','niqe','brisque']
    for control in ['baseline','no_target']:
        for stratum,conditions in [('clean',['clean']),('corrupted',protocol['conditions'][1:])]:
            contrasts[control+'/'+stratum]={key:paired(rows,'full',control,conditions,key) for key in keys}
    gate={}
    for control in ['baseline','no_target']:
        r=contrasts[control+'/corrupted'];p=protocol['engineering_gate']
        available=all(r[k].get('n_identities')==4 and all(u['paired_conditions']==len(protocol['conditions'])-1 for u in r[k]['units']) for k in ['lpips','psnr','facenet_cosine'])
        passed=available and r['lpips']['mean_delta']<=p['lpips_delta_max'] and r['lpips']['improved']>=p['minimum_identities_improved'] and r['psnr']['mean_delta']>=p['psnr_delta_min'] and r['facenet_cosine']['mean_delta']>=p['facenet_delta_min']
        gate[control]='PASS_ENGINEERING_ONLY' if passed else 'FAIL' if available else 'INCONCLUSIVE'
    summary={}
    for arm in ['baseline','no_target','full']:
        for condition in protocol['conditions']:
            rs=[r for r in rows if r['arm']==arm and r['condition']==condition]
            summary[arm+'/'+condition]={key:dict(n=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,sd=float(np.std(v,ddof=1)) if len(v)>1 else None)
                for key in keys for v in [[r['metrics'][key] for r in rs if r.get('metrics',{}).get(key) is not None]]}
    report=dict(status='development only',rows=rows,summary=summary,contrasts=contrasts,engineering_gate=gate,
        primary='Corrupted LPIPS vs baseline and equal-capacity no-target control, subject to PSNR/FaceNet tolerances',
        limitations=['Four observed identities and one seed, not independent confirmation','Identity failures retained; paired support reported',
            'Canonical adapter rectangles are not anatomical alignment','no_target ablates explicit target similarity and pose features but retains minimum target/reference detection confidence; it is not a strictly target-free control',
            'Conditions and timestep blocks are coupled during this pilot training; broader independent corruption/timestep sampling remains',
            'Source pretraining overlap unresolved','No frozen final evaluation or new dataset used'],
        protocol_sha256=sha(OUT/'protocol.json'),generation_sha256=sha(OUT/'evaluation/rows.json'),
        metric_sha256=sha(ROOT/'scripts/extended_evaluation_metrics.py'),regional_metric_sha256=sha(ROOT/'scripts/region_metrics_v1.py'))
    (OUT/'scores.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (ROOT/'research/target_compatibility_results_v1.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    # Every identity/condition receives a sheet; no cherry-picked visual subset.
    sheets=OUT/'comparisons';sheets.mkdir(exist_ok=True)
    for identity,(target,observed,*_) in cache.items():
        for condition in protocol['conditions']:
            sheet=Image.new('RGB',(1280,292),'white');draw=ImageDraw.Draw(sheet)
            panels=[('Target / scoring only',Image.fromarray(target)),('Damaged input',Image.fromarray(observed))]
            for arm in ['baseline','no_target','full']:
                p=OUT/'evaluation'/f'{identity}_{arm}_{condition}.png'
                panels.append((arm,Image.open(p).convert('RGB') if p.exists() else Image.new('RGB',(256,256),'gray')))
            for i,(label,im) in enumerate(panels):draw.text((i*256+4,8),label,fill='black');sheet.paste(im.resize((256,256)),(i*256,28))
            sheet.save(sheets/f'{identity}_{condition}.png')
    print(json.dumps(dict(gate=gate,contrasts=contrasts),indent=2),flush=True)


if __name__=='__main__':main()
