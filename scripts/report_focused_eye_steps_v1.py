"""Report fixed step-budget comparison with every person's metrics."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from report_focused_eye_guidance_replication_v2 import summarize,gate,KEYS
from report_focused_eye_freeu_v1 import sha

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/focused_eye_steps_v1';IDS=['263','6681','10149']


def main():
    summary=json.loads((OUT/'summary.json').read_text());rows=summary['rows']
    visual=json.loads((OUT/'visual_review.json').read_text()) if (OUT/'visual_review.json').exists() else None
    lookup={(r['identity'],r['condition']):r for r in rows}
    contrasts={}
    for arm in ['control3','control5']:
        pairs=[dict(identity=i,control=lookup.get((i,arm),{}).get('metrics',{}),candidate=lookup.get((i,'steps50'),{}).get('metrics',{})) for i in IDS]
        stats=summarize(pairs)
        guard=all(p['control'].get(k) is not None and p['candidate'].get(k) is not None and p['candidate'][k]-p['control'][k]>=-.005
            for p in pairs for k in ['facenet_cosine','arcface_conditioning_cosine'])
        contrasts[arm]=dict(stats=stats,mean_gate=gate(stats,3),identity_guard=guard)
    numerical=all(c['mean_gate'] for c in contrasts.values()) and contrasts['control3']['identity_guard']
    result=dict(contrasts=contrasts,numerical_gate=numerical,replicate=False,website_promotion=False,visual_review_pending=visual is None,visual_review=visual,
                summary_sha256=sha(OUT/'summary.json'))
    (OUT/'paired_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Fixed50-step erased-eye reconstruction check','',
        '10 October 2026. Three fixed development people, seed17. Only DDIM30 to50 changes against CFG3; references, masks, '
        'FaceID0.8, strength1, 1024/512, precision and Poisson finishing stay fixed. No FreeU, appearance branch, training or website change.', '',
        f"Completed {sum(r['status']=='complete' for r in rows)}/9 image records, including six reused controls; one additional byte-exact replay.", '',
        '| Metric | CFG5 /30 | CFG3 /30 | CFG3 /50 |', '|---|---:|---:|---:|']
    for key in KEYS:
        cells=[]
        for arm in ['control5','control3','steps50']:
            vals=[r['metrics'][key] for r in rows if r['condition']==arm and r.get('metrics',{}).get(key) is not None]
            cells.append(f'{np.mean(vals):.6f} ({len(vals)}/3)' if vals else 'NA')
        lines.append('| '+key+' | '+' | '.join(cells)+' |')
    lines+=['','## Every person','', '| Person | Arm | Hole SSIM | PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ident in IDS:
        for arm in ['control5','control3','steps50']:
            m=lookup.get((ident,arm),{}).get('metrics',{})
            cells=[f'{m[k]:.6f}' if m.get(k) is not None else 'NA' for k in ['hole_ssim','hole_psnr','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']]
            lines.append(f'| {ident} | {arm} | '+' | '.join(cells)+' |')
    lines+=['','## Decision','',f"Numerical replication gate: {'PASS' if numerical else 'FAIL'}.",
        'Visual inspection is recorded separately. A failed gate stops this fixed recipe. Keep every negative result; do not choose50 steps '
        'for a favored person or mix the new seed17 average with the previous three-seed average. The previous identity regression for10149 remains explicit.', '',
        'The frozen hole SSIM protocol and contextual crop LPIPS remain unchanged. FaceNet is separate from generation conditioning; ArcFace is '
        'conditioning-family similarity. Whole-image SSIM and cosine values are not reconstruction accuracy percentages. More steps change the DDIM timestep '
        'grid and add computation; they do not add missing reference spatial information. These three people have repeatedly been used for development.', '',
        '[Frozen protocol](FOCUSED_EYE_STEPS_PREFLIGHT_V1.md). Raw and finished images, timings, hashes and every attempted result are retained locally.']
    if visual:lines+=['','## Completed visual review','',*[f'- Person {i}: {note}' for i,note in visual['notes'].items()],'',visual['decision']]
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for number,ident in enumerate(IDS,1):
        c=cases[ident];mask=np.asarray(Image.open(c['mask']))>127;y,x=np.nonzero(mask)
        box=(max(0,int(x.min())-10),max(0,int(y.min())-10),min(512,int(x.max())+11),min(512,int(y.max())+11))
        local=pack/f'Set 0{number} - Person {ident}'/'Experimental - 50 steps';local.mkdir(exist_ok=True)
        pictures=[('Original - evaluation only',Image.open(c['target']).convert('RGB'))]
        for arm,label in [('control5','CFG5 -30 steps'),('control3','CFG3 -30 steps'),('steps50','CFG3 -50 steps')]:
            r=lookup.get((ident,arm),{})
            if r.get('status')=='complete':
                p=OUT/r['file'];assert sha(p)==r['sha256'];pictures.append((label,Image.open(p).convert('RGB')));shutil.copy2(p,local/(label+'.png'))
            else:pictures.append((label+' FAILED',Image.new('RGB',(512,512),'gray')))
        sheet=Image.new('RGB',(320*len(pictures),560),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,img) in enumerate(pictures):
            sheet.paste(img.resize((320,320)),(col*320,30));sheet.paste(img.crop(box).resize((320,180)),(col*320,375));draw.text((col*320+5,8),label,fill='black')
        p=OUT/f'{ident}_comparison.png';sheet.save(p);shutil.copy2(p,local/'Comparison.png')
        (local/'READ ME.txt').write_text('Fixed50-step development test. Same input, mask and four references.\n'
            'Experimental offline result, not a changed website default. All controls and outcomes retained.\n'
            'See research/FOCUSED_EYE_STEPS_RESULTS_V1.md for metrics and decision.\n')
    (ROOT/'research/FOCUSED_EYE_STEPS_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
