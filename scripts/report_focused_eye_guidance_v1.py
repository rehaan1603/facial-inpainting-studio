"""Paired descriptive report and local test images; no per-person selection."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/focused_eye_guidance_v1'
IDS=['263','6681','10149']
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']

def main():
    candidate=json.loads((OUT/'summary.json').read_text())
    control=json.loads((ROOT/'outputs/focused_reference_finish_v1/summary.json').read_text())
    candidates={r['identity']:r for r in candidate['rows'] if r['location']=='eyes' and r['status']=='complete'}
    controls={r['identity']:r for r in control['rows'] if r['location']=='eyes' and r['status']=='complete'}
    assert set(candidates)==set(IDS)==set(controls)
    pairs=[dict(identity=i,control=controls[i]['metrics'],candidate=candidates[i]['metrics']) for i in IDS]
    means={};lines=['# Focused eye reconstruction: lower guidance diagnostic','',
        '9 October 2026. Three repeated-development people, eyes only, clean four-reference sets, seed17. Only diffusion guidance changes5 to3. FaceID scale stays0.8;1024 inference,30DDIM steps and Poisson finishing remain fixed. No training, new references or checkpoint changes.','',
        'The neutral guidance5 replay precedes the three candidates. Original generator and website defaults remain unchanged.','',
        '## Matched means','',
        '| Metric | Guidance5 control | Guidance3 candidate | Delta | Valid pairs |',
        '|---|---:|---:|---:|---:|']
    for k in KEYS:
        valid=[r for r in pairs if r['control'].get(k) is not None and r['candidate'].get(k) is not None]
        b=float(np.mean([r['control'][k] for r in valid])) if valid else None
        a=float(np.mean([r['candidate'][k] for r in valid])) if valid else None
        means[k]=dict(control=b,candidate=a,delta=a-b if valid else None,n=len(valid))
        lines.append(f'| {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} | {len(valid)} |' if valid else f'| {k} | NA | NA | NA |0|')
    lines+=['','## Every person','', '| Person | Metric | Control | Candidate | Delta |','|---|---|---:|---:|---:|']
    for r in pairs:
        for k in KEYS:
            b,a=r['control'].get(k),r['candidate'].get(k)
            lines.append(f"| {r['identity']} | {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} |" if a is not None and b is not None else f"| {r['identity']} | {k} | {b} | {a} | NA |")
    lower={'hole_lpips','eye_lpips'}
    numeric_gate=all(v['n']==3 and (v['delta']<=0 if k in lower else v['delta']>=0) for k,v in means.items()) and means['hole_ssim']['delta']>0
    lines+=['','## Decision','',f'Combined mean numerical gate: **{"PASS" if numeric_gate else "FAIL"}**. Visual inspection and per-person identity errors remain separate. This report does not automatically select the candidate or change website defaults.',
        '', 'Whole-face similarity is not iris-colour accuracy. ArcFace conditions the generator; FaceNet is separate. LPIPS uses the fixed contextual crops from the existing evaluator. Detection failures remain explicit.',
        '', 'Raw guidance5 controls already show iris-colour/gaze changes before Poisson finishing; boundary blending alone cannot solve the observed generation errors.',
        '', 'Visual comparisons are local in `outputs/focused_eye_guidance_v1/comparisons`. Candidate test files are labelled Experimental in each existing three-person folder.']
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    comparisons=OUT/'comparisons';comparisons.mkdir(exist_ok=True)
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for i,ident in enumerate(IDS,1):
        c=cases[ident]
        paths=[Path(c['target']),Path(c['input']),ROOT/'outputs/focused_reference_finish_v1'/candidates[ident]['file'],OUT/candidates[ident]['file']]
        labels=['Original - evaluation only','Damaged input','Control - CFG5','Candidate - CFG3']
        images=[Image.open(p).convert('RGB') for p in paths]
        sheet=Image.new('RGB',(1280,355),'white');d=ImageDraw.Draw(sheet)
        for n,(label,img) in enumerate(zip(labels,images)):
            sheet.paste(img.resize((320,320)),(n*320,35));d.text((n*320+6,10),label,fill='black')
        dest=comparisons/f'{ident}_eyes.png';sheet.save(dest)
        mask=np.asarray(Image.open(c['mask']))>127;ys,xs=np.nonzero(mask)
        bounds=(max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(512,int(xs.max())+13),min(512,int(ys.max())+13))
        crop=Image.new('RGB',(1280,260),'white');dc=ImageDraw.Draw(crop)
        for n,img in enumerate(images):
            eye=img.crop(bounds);eye.thumbnail((308,208));crop.paste(eye,(n*320+(320-eye.width)//2,40+(208-eye.height)//2));dc.text((n*320+6,10),labels[n],fill='black')
        crop.save(comparisons/f'{ident}_eye_detail.png')
        local=pack/f'Set 0{i} - Person {ident}'/'Experimental - CFG 3'
        local.mkdir(exist_ok=True)
        shutil.copy2(paths[-1],local/'Eyes - CFG3 - experimental.png');shutil.copy2(dest,local/'Comparison.png')
        (local/'READ ME.txt').write_text('Experimental guidance3 result. This is not an automatic replacement for the tested control.\nCompare with guidance5 and the original. Do not use the clean original as a reference.\nSee research/FOCUSED_EYE_GUIDANCE_RESULTS_V1.md for all metrics.\n')
    (ROOT/'research/FOCUSED_EYE_GUIDANCE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    (OUT/'paired_results.json').write_text(json.dumps(dict(means=means,pairs=pairs,numeric_gate=numeric_gate,selected=False,visual_review='pending'),indent=2))
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
