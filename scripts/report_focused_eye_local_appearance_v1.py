"""Disclose all local-routing results and produce comparable local test images."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from report_focused_eye_guidance_replication_v2 import summarize, gate, KEYS
from report_focused_eye_freeu_v1 import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/focused_eye_local_appearance_v1'
IDS=['263','6681','10149']
ARMS={'control5':'Original CFG5','control3':'Current CFG3','global':'Global appearance','local':'Local appearance'}


def main():
    source=json.loads((OUT/'summary.json').read_text())
    cfg=json.loads((OUT/'config.json').read_text())
    visual=json.loads((OUT/'visual_review.json').read_text()) if (OUT/'visual_review.json').exists() else None
    rows=source['rows'];lookup={(r['identity'],r['condition']):r for r in rows}
    comparisons={}
    for arm in ['control3','global','control5']:
        pairs=[dict(identity=i,control=lookup.get((i,arm),{}).get('metrics',{}),
                    candidate=lookup.get((i,'local'),{}).get('metrics',{})) for i in IDS]
        stats=summarize(pairs)
        identity_guard=all(p['control'].get(k) is not None and p['candidate'].get(k) is not None
            and p['candidate'][k]-p['control'][k]>=-.005 for p in pairs
            for k in ['facenet_cosine','arcface_conditioning_cosine'])
        comparisons[arm]=dict(stats=stats,mean_gate=gate(stats,3),identity_guard=identity_guard)
    numerical=all(comparisons[a]['mean_gate'] and comparisons[a]['identity_guard'] for a in ['control3','global'])
    result=dict(comparisons=comparisons,numerical_gate=numerical,non_worsening_tolerance_implemented=0.,non_worsening_tolerance_preflight=1e-6,
        visual_review_pending=visual is None,visual_review=visual,replication_authorized=False,
        website_promotion=False,summary_sha256=sha(OUT/'summary.json'))
    (OUT/'paired_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Hole-local reference appearance: fixed three-person screen','',
        '10 October 2026. Same inputs, four references, cached appearance features, CFG3, DDIM30, identity0.8, appearance0.2, '
        '1024 inference / 512 export and Poisson finishing. Only the appearance query routing changes versus the global-appearance control. '
        'The full joint-reference attention is retained; no native per-reference mask path, RGB transfer, training or website change.', '',
        f"Completed {sum(r['status']=='complete' for r in rows)}/{cfg['expected']} records, including {sum(r.get('reused',False) for r in rows)} hash-verified reused controls. "
        'The two raw/finished full-mask and appearance-off replays are additional verification calls.', '',
        '| Metric | Original CFG5 | Current CFG3 | Global appearance | Local appearance |', '|---|---:|---:|---:|---:|']
    for key in KEYS:
        cells=[]
        for arm in ARMS:
            vals=[r['metrics'][key] for r in rows if r['condition']==arm and r.get('metrics',{}).get(key) is not None]
            cells.append(f'{np.mean(vals):.6f} ({len(vals)}/3)' if vals else 'NA')
        lines.append('| '+key+' | '+' | '.join(cells)+' |')
    for arm,comparison in comparisons.items():
        lines+=['',f'## Matched differences against {ARMS[arm]}','',
            '| Metric | Control | Local candidate | Delta | Pairs |','|---|---:|---:|---:|---:|']
        for key,v in comparison['stats'].items():
            numbers=f"{v['control']:.6f} | {v['candidate']:.6f} | {v['delta']:+.6f}" if v['delta'] is not None else 'NA | NA | NA'
            lines.append(f"| {key} | {numbers} | {v['n']}/3 |")
    lines+=['','## Every person','',
        '| Person | Arm | Hole SSIM | PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ident in IDS:
        for arm,label in ARMS.items():
            m=lookup.get((ident,arm),{}).get('metrics',{})
            values=[f'{m[k]:.6f}' if m.get(k) is not None else 'NA' for k in ['hole_ssim','hole_psnr','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']]
            lines.append(f'| {ident} | {label} | '+' | '.join(values)+' |')
    lines+=['','## Decision and limits','',f"Numerical replication gate: {'PASS' if numerical else 'FAIL'}.",
        'The report reuses the existing strict non-worsening gate (zero tolerance), whereas the frozen preflight allows 1e-6 numerical tolerance. '
        'All failed mean differences exceed 1e-6; either rule rejects this recipe. The protocol and results have not been adjusted.', '',
        'The decision also requires visual review. A numerical failure stops replication of this fixed recipe and prevents website promotion. '
        'Every failed attempt and per-person regression remains in the source record; no best-person or parameter selection follows.', '',
        'The unchanged scorer measures hole SSIM over fully contained11x11 windows and LPIPS on contextual crops. '
        'Whole-image SSIM is dominated by preserved pixels. FaceNet is separate from generation conditioning; ArcFace belongs to the conditioning family. '
        'These three development people and seed17 are not an unknown-client accuracy estimate.', '',
        'Routing is local at each attention operation, but subsequent UNet mixing can propagate its influence. It does not provide spatial correspondence '
        'between reference eyes and target eyes. Original visible RGB is preserved by final compositing, with raw images retained separately. '
        'No checkpoints, dataset or evaluation settings changed.', '',
        'See [fixed preflight](FOCUSED_EYE_LOCAL_APPEARANCE_PREFLIGHT_V1.md). '
        'Joint-reference processor calculations use the installed Diffusers source recorded in config.json. '
        'This differs from [native per-reference IP-Adapter masks](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter#masking).']
    if visual:lines+=['','## Completed visual review','',*[f'- Person {i}: {note}' for i,note in visual['notes'].items()],'',visual['decision']]
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for number,ident in enumerate(IDS,1):
        c=cases[ident];assert sha(c['target'])==c['target_sha256']
        mask=np.asarray(Image.open(c['mask']))>127;y,x=np.nonzero(mask)
        box=(max(0,int(x.min())-10),max(0,int(y.min())-10),min(512,int(x.max())+11),min(512,int(y.max())+11))
        images=[('Original - evaluation only',Image.open(c['target']).convert('RGB'))]
        local=pack/f'Set 0{number} - Person {ident}'/'Experimental - Local appearance';local.mkdir(exist_ok=True)
        for arm,label in ARMS.items():
            r=lookup.get((ident,arm),{})
            if r.get('status')=='complete':
                p=OUT/r['file'];assert sha(p)==r['sha256']
                images.append((label,Image.open(p).convert('RGB')));shutil.copy2(p,local/(label+'.png'))
            else:images.append((label+' FAILED',Image.new('RGB',(512,512),'gray')))
        sheet=Image.new('RGB',(320*len(images),560),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,img) in enumerate(images):
            sheet.paste(img.resize((320,320)),(col*320,30));sheet.paste(img.crop(box).resize((320,180)),(col*320,375))
            draw.text((col*320+5,8),label,fill='black');draw.text((col*320+5,356),'Same hole bbox + context',fill='black')
        p=OUT/f'{ident}_comparison.png';sheet.save(p);shutil.copy2(p,local/'Comparison.png')
        (local/'READ ME.txt').write_text('Fixed experimental query-local appearance. Same references and mask.\n'
            'Not added to the website. Controls and every outcome retained.\n'
            'See research/FOCUSED_EYE_LOCAL_APPEARANCE_RESULTS_V1.md for scores and decision.\n')
    (ROOT/'research/FOCUSED_EYE_LOCAL_APPEARANCE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
