"""Report every CFG2 pair using the frozen metrics and acceptance rule."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/focused_eye_guidance2_20261102'
BASE=ROOT/'outputs/focused_eye_guidance_v1'
KEYS=['psnr','ssim','lpips','hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']
LOWER={'lpips','hole_lpips','eye_lpips'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=json.loads((OUT/'summary.json').read_text());b=json.loads((BASE/'summary.json').read_text())
    assert a['all_completed'] and len(a['rows'])==3
    aa={r['identity']:r for r in a['rows']};bb={r['identity']:r for r in b['rows']}
    assert set(aa)==set(bb)=={'263','6681','10149'}
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    means={};fails=[];pairs=[]
    lines=['# CFG2 versus CFG3 — 2 November 2026','','Fixed three-person development comparison; seed17, four real references,30DDIM steps,1024 inference/512 export, FaceID0.8 and Poisson finishing. Only guidance3-to2 changes. No training or website promotion. Raw and final neutral CFG3 replay is pixel-exact.','','| Metric | CFG3 | CFG2 | Change |','|---|---:|---:|---:|']
    for ident in aa:
        ar,br=aa[ident],bb[ident]
        for r,folder in [(ar,OUT),(br,BASE)]:
            assert sha(folder/r['file'])==r['sha256']
            assert sha(folder/r['file'].replace('.png','_raw.png'))==r['raw_sha256']
            assert r['metrics']['visible_changed']==0
            assert all(r['metrics'].get(k) is not None for k in KEYS)
        assert ar['seed']==br['seed']==17
        # Reject changes to metric region support rather than comparing unlike crops.
        for region in ['masked','eyes']:
            for key in ['pixels','ssim_valid_centres','bbox_xyxy']:
                assert ar['regions'][region][key]==br['regions'][region][key],(ident,region,key)
        pairs.append(dict(identity=ident,control=br['metrics'],candidate=ar['metrics']))
        for k in ['facenet_cosine','arcface_conditioning_cosine']:
            if ar['metrics'][k]-br['metrics'][k]<-.005:fails.append(ident+'/'+k+'/identity_guard')
    for k in KEYS:
        bv=float(np.mean([p['control'][k] for p in pairs]));av=float(np.mean([p['candidate'][k] for p in pairs]));delta=av-bv
        good=delta<=0 if k in LOWER else delta>0 if k=='hole_ssim' else delta>=0
        if not good:fails.append(k+'/mean')
        means[k]=dict(control=bv,candidate=av,delta=delta,pass_gate=good,n=3)
        lines.append(f'| {k} | {bv:.6f} | {av:.6f} | {delta:+.6f} |')
    lines+=['','## Every person','','| Person | Metric | CFG3 | CFG2 | Change |','|---|---|---:|---:|---:|']
    for p in pairs:
        for k in KEYS:lines.append(f"| {p['identity']} | {k} | {p['control'][k]:.6f} | {p['candidate'][k]:.6f} | {p['candidate'][k]-p['control'][k]:+.6f} |")
    lines+=['','Numerical gate: '+('FAIL' if fails else 'PASS')+'. Visual review pending; no automatic promotion.','',*['- '+x for x in fails],'','All three are repeated development cases. Whole SSIM includes preserved visible pixels; cosine is not accuracy. These results do not estimate unfamiliar-person reconstruction performance. If any gate fails, stop this recipe; no per-person parameter choice or extra seed search.']
    review_path=OUT/'visual_review.json'
    reviewed=review_path.exists()
    if reviewed:
        review=json.loads(review_path.read_text())
        lines=[line.replace('Visual review pending; no automatic promotion.','All six raw/finished comparison sheets reviewed; no promotion.') for line in lines]
        lines+=['','## Visual review','',*['- '+k+': '+v for k,v in review['observations'].items()],'',review['decision']]
    (ROOT/'research/FOCUSED_GUIDANCE2_RESULTS_20261102.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'paired_results.json').write_text(json.dumps(dict(means=means,pairs=pairs,failed_gates=fails,numerical_pass=not fails,visual_review='reviewed' if reviewed else 'pending',promoted=False),indent=2))
    dest=OUT/'comparisons';dest.mkdir(exist_ok=True)
    for ident in aa:
        c=cases[ident]
        for raw in [False,True]:
            paths=[Path(c['target']),Path(c['input']),BASE/bb[ident]['file'],OUT/aa[ident]['file']]
            if raw:paths[2:]=[p.with_name(p.stem+'_raw.png') for p in paths[2:]]
            mask=np.asarray(Image.open(c['mask']))>127;ys,xs=np.nonzero(mask);box=(max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(512,int(xs.max())+13),min(512,int(ys.max())+13))
            sheet=Image.new('RGB',(1280,560),'white');draw=ImageDraw.Draw(sheet)
            for j,(path,label) in enumerate(zip(paths,['Original: evaluation only','Damaged input','CFG3 control','CFG2 candidate'])):
                im=Image.open(path).convert('RGB');sheet.paste(im.resize((320,320)),(320*j,30));crop=im.crop(box);crop.thumbnail((310,180));sheet.paste(crop,(320*j+(320-crop.width)//2,370));draw.text((320*j+5,8),label,fill='black')
            sheet.save(dest/f'{ident}_{"raw" if raw else "finished"}.png')
    print(json.dumps(dict(means=means,failed_gates=fails),indent=2))
if __name__=='__main__':main()
