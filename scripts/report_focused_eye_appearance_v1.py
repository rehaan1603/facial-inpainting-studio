"""Fixed-control appearance report, retain failures and all per-person tradeoffs."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/focused_eye_appearance_v1'
IDS=['263','6681','10149']
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']
LOWER={'hole_lpips','eye_lpips'}

def main():
    source=json.loads((OUT/'summary.json').read_text());assert source['all_attempted']
    control={r['identity']:r for r in json.loads((ROOT/'outputs/focused_reference_finish_v1/summary.json').read_text())['rows'] if r['identity'] in IDS and r['location']=='eyes'}
    candidate={r['identity']:r for r in source['rows']};assert set(control)==set(candidate)==set(IDS)
    pairs=[dict(identity=i,control=control[i]['metrics'],candidate=candidate[i].get('metrics',{}),status=candidate[i]['status']) for i in IDS]
    means={}
    lines=['# Fixed reference appearance branch: three-person results','',
           'Same frozen three-person erased-eye inputs, four real references in unchanged order, seed 17, CFG 5, DDIM 30, scale 0.8, strength 1, 1024 inference / 512 output and Poisson finishing. Only a fixed-scale (0.2) pretrained Plus Face visual-reference branch is added. No training; baseline models/checkpoints and website defaults are preserved.','',
           f"Attempted {source['attempted']}/3; completed {sum(r['status']=='complete' for r in source['rows'])}/3. Failures remain in the source records. Original targets are evaluation-only.",'',
           '## Matched means','', '| Metric | Control | Appearance addition | Delta | Valid pairs |','|---|---:|---:|---:|---:|']
    for k in KEYS:
        valid=[p for p in pairs if p['control'].get(k) is not None and p['candidate'].get(k) is not None]
        b=float(np.mean([p['control'][k] for p in valid])) if valid else None
        a=float(np.mean([p['candidate'][k] for p in valid])) if valid else None
        means[k]=dict(control=b,candidate=a,delta=a-b if valid else None,n=len(valid))
        lines.append(f'| {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} | {len(valid)}/3 |' if valid else f'| {k} | NA | NA | NA | 0/3 |')
    lines+=['','## Every person','', '| Person | Metric | Control | Appearance addition | Delta |','|---|---|---:|---:|---:|---:|']
    gates={}
    for p in pairs:
        gates[p['identity']]=all(p['candidate'].get(k) is not None and (p['candidate'][k]<=p['control'][k] if k in LOWER else p['candidate'][k]>=p['control'][k]) for k in KEYS)
        for k in KEYS:
            b,a=p['control'].get(k),p['candidate'].get(k)
            lines.append(f"| {p['identity']} | {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} |" if b is not None and a is not None else f"| {p['identity']} | {k} | {b} | {a} | NA |")
    mean_gate=all(v['n']==3 and (v['delta']<=0 if k in LOWER else v['delta']>=0) for k,v in means.items()) and means['hole_ssim']['delta']>0
    replay=json.loads((OUT/'replay.json').read_text());assert all(v['pixel_exact'] for v in replay.values())
    visual_notes={
        '263':'Eye colour and gaze mismatch persist; eyebrow texture changes but is not a faithful match.',
        '6681':'Eyes remain oversaturated; the appearance addition does not restore matching original brow/eye structure.',
        '10149':'All six numerical measures improve, but gaze and eyebrow shape still visibly differ from the original.'}
    lines+=['','## Decision and scope','',f"Combined mean numerical gate: {'PASS' if mean_gate else 'FAIL'}.",
            *[f"- Person {i}: {'PASS' if ok else 'FAIL'} on all six non-worsening measures." for i,ok in gates.items()], '',
            'Visual review remains a separate requirement. No automatic production selection. The zero-appearance replay must match both original raw and finished control images byte-for-byte and has passed. All completed outputs preserve every visible pixel. SSIM/identity cosine are not accuracy percentages; three development people and one seed are insufficient for generalization. ArcFace is also used for conditioning. CLIP face-crop features are appearance cues, not exact spatial iris-pixel correspondence.','',
            'Official model assets and feature provenance are stored in outputs/focused_eye_appearance_v1/config.json. See [frozen protocol](FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md), [official model repository](https://huggingface.co/h94/IP-Adapter) and [adapter documentation](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter).']
    lines+=['','## Visual review and recorded encoding failure','',
            'All three original/control/candidate comparisons were inspected. No universal selection: average SSIM decreases and matching iris/gaze/brow reconstruction remains unresolved.']
    lines += [f'- Person {i}: {visual_notes[i]}' for i in IDS]
    lines += ['',
              'The first attempt failed during reference encoding with a Half/Float mismatch before producing a candidate. Its config, runner snapshot, cropped references and log are preserved in outputs/focused_eye_appearance_v1_encoding_failure. The retry explicitly converts the CPU encoder to float32 and uses the required vision hidden states; neutral raw/final replay then passes byte-exactly. No prior checkpoint or experiment was overwritten.','',
              'Next, complete the fixed CFG 3 / appearance 0.2 interaction arm specified in FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md. This is not a new scale search or per-person selection.']
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for index,ident in enumerate(IDS,1):
        if candidate[ident]['status']!='complete':continue
        original=Path(cases[ident]['target']);prior=ROOT/'outputs/focused_reference_finish_v1'/control[ident]['file'];new=OUT/candidate[ident]['file']
        images=[Image.open(p).convert('RGB') for p in [original,prior,new]]
        sheet=Image.new('RGB',(960,355),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,img) in enumerate(zip(['Original - evaluation only','Control - FaceID','Candidate - FaceID + appearance'],images)):
            sheet.paste(img.resize((320,320)),(col*320,35));draw.text((col*320+5,10),label,fill='black')
        sheet.save(OUT/f'{ident}_comparison.png')
        local=pack/f'Set 0{index} - Person {ident}'/'Experimental - Reference appearance';local.mkdir(exist_ok=True)
        shutil.copy2(new,local/'Eyes - appearance addition - experimental.png');shutil.copy2(OUT/f'{ident}_comparison.png',local/'Comparison.png')
        (local/'READ ME.txt').write_text('Fixed reference-appearance addition alongside existing FaceID. Same references; seed17; CFG5.\nExperimental, not a website default. Original remains comparison-only, never a reference.\nFull results: research/FOCUSED_EYE_APPEARANCE_RESULTS_V1.md\n')
    (ROOT/'research/FOCUSED_EYE_APPEARANCE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    (OUT/'paired_results.json').write_text(json.dumps(dict(means=means,pairs=pairs,mean_gate=mean_gate,per_person_gates=gates,selected=False,visual_review=visual_notes),indent=2))
    print(json.dumps(dict(means=means,mean_gate=mean_gate,per_person_gates=gates),indent=2))

if __name__=='__main__':main()
