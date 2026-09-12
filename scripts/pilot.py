"""Validation-only diagnostic. Synthetic rectangles; not final paper evaluation."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

ROOT = Path(__file__).resolve().parents[1]


def morph(mask, radius):
    if not radius: return mask.copy()
    filt = ImageFilter.MaxFilter(2*radius+1) if radius > 0 else ImageFilter.MinFilter(2*abs(radius)+1)
    return np.array(Image.fromarray((mask*255).astype('uint8')).filter(filt)) > 0


def make_case(image, seed):
    rng = np.random.default_rng(seed)
    h,w = image.shape[:2]
    x0,y0 = rng.integers(w//5, w//2, size=2)
    bw,bh = rng.integers(w//4, w//2, size=2)
    true = np.zeros((h,w), dtype=bool)
    true[y0:min(h,y0+bh),x0:min(w,x0+bw)] = True
    # Target is never used to generate occluder pixels.
    texture = rng.integers(0,256,size=(8,8,3),dtype=np.uint8)
    occ = np.array(Image.fromarray(texture).resize((w,h), Image.Resampling.BILINEAR))/255.0
    observed = np.where(true[...,None],occ,image).astype(np.float32)
    shifted = np.zeros_like(true); shifted[:,8:] = true[:,:-8]
    return observed, true, {'accurate': true.copy(), 'under_4px': morph(true,-4),
                            'over_4px': morph(true,4), 'shift_8px': shifted}


def metrics(pred, target, observed, true, supplied):
    def mae(region, ref):
        return float(np.abs(pred-ref)[region].mean()) if region.any() else None
    mse = float(((pred-target)[true]**2).mean())
    return {'hole_mae':mae(true,target), 'hole_psnr':float(-10*np.log10(mse)) if mse>0 else None,
        'visible_mae':mae(~true,observed), 'missed_mae':mae(true & ~supplied,target),
        'overcovered_mae':mae(~true & supplied,target), 'true_fraction':float(true.mean()),
        'mask_fp_fraction':float((~true & supplied).mean()), 'mask_fn_fraction':float((true & ~supplied).mean())}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--images',type=int,default=32); args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/local.json').read_text())
    manifest=ROOT/'data/manifests/celebahq_clean.csv'
    rows=list(csv.DictReader(manifest.open()))
    rows=[r for r in rows if r['split']=='val']
    rows.sort(key=lambda r:hashlib.sha256(f"{cfg['seed']}:{r['hq_id']}".encode()).hexdigest())
    # One photograph per identity in this small diagnostic.
    selected=[]; seen=set()
    for r in rows:
        if r['identity'] not in seen:
            selected.append(r); seen.add(r['identity'])
        if len(selected)>=args.images: break
    assert len(selected)==args.images
    out=ROOT/'outputs/pilot_clean_v1'; out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(cfg['cache'])/'models/big-lama.pt'
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    model=torch.jit.load(str(checkpoint),map_location='cuda').eval()
    @torch.inference_mode()
    def infer(obs, mask):
        x=torch.from_numpy(obs.transpose(2,0,1).copy()).unsqueeze(0).cuda()
        m=torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        torch.cuda.synchronize(); start=time.perf_counter()
        result=model(x,m)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        raw=result[0].permute(1,2,0).cpu().numpy().clip(0,1)
        assert raw.shape==obs.shape and np.isfinite(raw).all()
        return np.where(mask[...,None],raw,obs), elapsed
    dummy=np.zeros((256,256,3),np.float32); dm=np.zeros((256,256),bool); dm[80:160,80:160]=True
    for _ in range(3): infer(dummy,dm)
    torch.cuda.reset_peak_memory_stats()
    results=[]; timings=[]; visuals=[]; cases=[]
    for i,r in enumerate(selected):
        with Image.open(r['image_path']) as im:
            target=np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype(np.float32)/255
        seed=cfg['seed']+int(r['hq_id'])
        obs,true,conditions=make_case(target,seed)
        folder=out/'cases'/r['hq_id']; folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray((obs*255).round().astype('uint8')).save(folder/'observed.png')
        Image.fromarray(true.astype('uint8')*255).save(folder/'true_mask.png')
        oracle,elapsed=infer(obs,true); timings.append(elapsed)
        cases.append({'hq_id':r['hq_id'],'identity':r['identity'],'seed':seed,'split':'val',
                      'source_sha256':hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()})
        for name,supplied in conditions.items():
            Image.fromarray(supplied.astype('uint8')*255).save(folder/f'{name}_mask.png')
            methods={'input_unchanged':(obs,0.0),'exact_mask_diagnostic':(oracle,elapsed)}
            if name=='accurate': methods['supplied_mask']=(oracle,elapsed)
            else:
                methods['supplied_mask']=infer(obs,supplied); timings.append(methods['supplied_mask'][1])
            methods['dilate_4px']=infer(obs,morph(supplied,4)); timings.append(methods['dilate_4px'][1])
            for method,(pred,seconds) in methods.items():
                results.append({'hq_id':r['hq_id'],'identity':r['identity'],'condition':name,'method':method,
                    'gpu_forward_seconds':seconds, **metrics(pred,target,obs,true,supplied)})
                Image.fromarray((pred*255).round().astype('uint8')).save(folder/f'{name}_{method}.png')
            if i<4 and name=='under_4px':
                visuals.append([target,obs,supplied[...,None].repeat(3,2).astype(float),
                    methods['supplied_mask'][0],methods['dilate_4px'][0],oracle])
        print(f'Pilot {i+1}/{len(selected)}',flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    (out/'cases.json').write_text(json.dumps(cases,indent=2))
    groups=defaultdict(list)
    for r in results: groups[(r['condition'],r['method'])].append(r)
    summary=[]
    for (condition,method),rs in groups.items():
        summary.append({'condition':condition,'method':method,'n':len(rs),
            **{k:float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
               for k in ['hole_mae','hole_psnr','visible_mae','missed_mae','overcovered_mae']}})
    report={'images':len(selected),'split':'validation only','resolution':256,'results':summary,
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'seed':cfg['seed'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'gpu_forward_median_seconds':float(np.median(timings)), 'gpu_forward_p95_seconds':float(np.percentile(timings,95)),
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'limits':['Small diagnostic, no statistical or novelty claims.','Rectangular synthetic textured occlusions only.',
            'Third-party LaMa export, no official reproduction.','Dilation radius not tuned.','No LPIPS or identity metrics yet.',
            'Forward timings exclude file loading, CPU preprocessing, and compositing.']}
    (ROOT/'research/pilot_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if visuals:
        canvas=Image.new('RGB',(256*6,280*len(visuals)),'white'); draw=ImageDraw.Draw(canvas)
        labels=['Target (evaluation only)','Observed input','Supplied mask','LaMa supplied mask','LaMa + dilation 4px','Exact mask diagnostic']
        for row,ims in enumerate(visuals):
            for col,im in enumerate(ims):
                canvas.paste(Image.fromarray((im*255).round().astype('uint8')),(col*256,row*280+24))
                draw.text((col*256+4,row*280+4),labels[col],fill='black')
        canvas.save(out/'preview.png')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__': main()
