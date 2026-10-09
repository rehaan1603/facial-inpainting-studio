"""Report both isolated eye-fidelity experiments with their common fixed control."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
IDS=['263','6681','10149']
ARMS={'DDIM CFG5':'focused_reference_finish_v1','DDIM CFG3':'focused_eye_guidance_v1','Euler CFG5':'focused_eye_scheduler_v1'}
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']

def main():
    out=ROOT/'outputs/focused_eye_fidelity_v1';out.mkdir(exist_ok=True)
    data={a:{r['identity']:r for r in json.loads((ROOT/'outputs'/d/'summary.json').read_text())['rows'] if r['location']=='eyes' and r['status']=='complete' and r['identity'] in IDS} for a,d in ARMS.items()}
    assert all(set(d)==set(IDS) for d in data.values())
    means={a:{k:float(np.mean([r['metrics'][k] for r in d.values()])) if all(r['metrics'].get(k) is not None for r in d.values()) else None for k in KEYS} for a,d in data.items()}
    lower={'hole_lpips','eye_lpips'};gates={}
    for a in list(ARMS)[1:]:
        gates[a]=all(means[a][k] is not None and (means[a][k]<=means['DDIM CFG5'][k] if k in lower else means[a][k]>=means['DDIM CFG5'][k]) for k in KEYS) and means[a]['hole_ssim']>means['DDIM CFG5']['hole_ssim']
    lines=['# Three-person eye fidelity: guidance and sampler experiments','',
        '9 October 2026. Same persons 263, 6681 and 10149, erased-eye masks, four clean references and seed 17. All arms use 1024 inference / 512 output, 30 steps, FaceID scale 0.8, strength 1 and Poisson finishing. No training.','',
        'Each candidate changes ONE setting relative to the saved DDIM guidance 5 control. The guidance 3 candidate retains DDIM. The Euler candidate retains guidance 5. No per-person or best-seed selection.','',
        '## Mean values, three matched cases','',
        '| Method | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:lines.append('| '+a+' | '+' | '.join('NA' if means[a][k] is None else f'{means[a][k]:.6f}' for k in KEYS)+' |')
    lines+=['','## Per-person comparisons','', '| Person | Method | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |','|---|---|---:|---:|---:|---:|---:|---:|']
    for ident in IDS:
        for a in ARMS:lines.append('| '+ident+' | '+a+' | '+' | '.join('NA' if data[a][ident]['metrics'].get(k) is None else f"{data[a][ident]['metrics'][k]:.6f}" for k in KEYS)+' |')
    lines+=['','## Combined numerical gate']+[f'- {a}: {"PASS" if ok else "FAIL"} on matched means.' for a,ok in gates.items()]
    visual_notes={
        '263':'CFG 3 reduces exaggerated contrast, but generated iris colour and gaze still differ from the original. Euler also retains these discrepancies.',
        '6681':'CFG 3 is closer in tone than CFG 5, but the eyes remain more saturated and eyebrow/eye structure is imperfect.',
        '10149':'All arms retain a visibly incorrect gaze/eye appearance. CFG 3 lowers FaceNet by 0.035830 despite higher hole SSIM and ArcFace.'}
    lines+=['','## Visual review and selection','',
        'All three comparison sheets were inspected. **Neither candidate is promoted to a universal website default.** Lower guidance is promising for masked-region structure, but it does not solve exact iris colour, gaze or brow recovery. Euler fails the combined identity requirement.']
    lines += [f'- Person {ident}: {visual_notes[ident]}' for ident in IDS]
    lines += ['','## Anatomical eye SSIM diagnostic','',
        'These values already exist in the frozen scorer. They refer to annotated eye pixels inside the damage mask, rather than the larger erased rectangle that also contains brows and skin. Only fully contained 11-by-11 SSIM windows contribute. Very small valid-window counts make these local values unstable; they are supplementary and do not replace the primary frozen hole SSIM.', '',
        '| Person | Method | Anatomical eye SSIM | Valid window centres |',
        '|---|---|---:|---:|']
    for ident in IDS:
        for a in ARMS:
            region=data[a][ident]['regions']['eyes']; value=region['ssim']
            lines.append(f'| {ident} | {a} | '+('NA' if value is None else f'{value:.6f}')+f" | {region['ssim_valid_centres']} |")
    lines+=['','These gates do not replace visual review or per-person identity checks. Neither research runner modifies the website. SSIM is not a colour-specific or identity-accuracy percentage. ArcFace also conditions FaceID.','',
        'The guidance5 neutral replay is raw- and finished-byte-exact. All original controls remain available. Evaluation-only targets never enter generation. Failures and missing detector measurements remain in source results.','',
        'Preflights: FOCUSED_EYE_GUIDANCE_PREFLIGHT_V1.md and FOCUSED_EYE_SCHEDULER_PREFLIGHT_V1.md. Local image comparisons: outputs/focused_eye_fidelity_v1.']
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for i,ident in enumerate(IDS,1):
        c=cases[ident];images=[Image.open(c['target']).convert('RGB')]+[Image.open(ROOT/'outputs'/d/data[a][ident]['file']).convert('RGB') for a,d in ARMS.items()]
        sheet=Image.new('RGB',(1280,355),'white');draw=ImageDraw.Draw(sheet)
        labels=['Original - evaluation only',*ARMS]
        for n,(label,img) in enumerate(zip(labels,images)):
            sheet.paste(img.resize((320,320)),(n*320,35));draw.text((n*320+6,10),label,fill='black')
        dest=out/f'{ident}_comparison.png';sheet.save(dest)
        local=pack/f'Set 0{i} - Person {ident}'/'Experimental - Eye fidelity'
        local.mkdir(exist_ok=True);shutil.copy2(dest,local/'All methods comparison.png')
        for a,d in list(ARMS.items())[1:]:shutil.copy2(ROOT/'outputs'/d/data[a][ident]['file'],local/(a+' - eyes.png'))
        (local/'READ ME.txt').write_text('Two research comparisons with the fixed DDIM CFG5 control.\nDo not treat the sharper-looking result as proven identity recovery.\nOriginals are evaluation-only. Website defaults remain unchanged.\nFull measured results: research/FOCUSED_EYE_FIDELITY_RESULTS_V1.md\n')
    receipt=dict(means=means,per_person=data,numerical_gates=gates,selected=False,visual_review=visual_notes,cohort=IDS,seed=17,
                 decision='Do not promote either candidate: eye appearance errors persist; CFG3 has a person-level FaceNet regression; Euler fails mean identity.')
    (out/'results.json').write_text(json.dumps(receipt,indent=2))
    (ROOT/'research/FOCUSED_EYE_FIDELITY_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(means=means,gates=gates),indent=2))

if __name__=='__main__':main()
