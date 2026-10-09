"""Verify and disclose a rejected registered-reference detail prototype."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_focused_eye_texture_v1 import apply

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/focused_eye_texture_v1'
IDS=['263','6681','10149']
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))

def main():
    config=read(OUT/'config.json')
    assert sha(ROOT/'scripts/run_focused_eye_texture_v1.py')==config['runner_sha256']
    assert sha(ROOT/'research/FOCUSED_EYE_TEXTURE_PREFLIGHT_V1.md')==config['protocol_sha256']
    summary=read(OUT/'summary.json');assert summary['all_completed'] and summary['attempted']==3
    cases={c['identity']:c for c in read(ROOT/'outputs/masked_task_v1/cases.json') if c['location']=='eyes'}
    controls={r['identity']:r for r in read(ROOT/'outputs/focused_eye_guidance_v1/summary.json')['rows']}
    candidates={r['identity']:r for r in summary['rows']};assert set(candidates)==set(IDS)
    generated={r['identity']:r for r in read(OUT/'rows.json')}
    notes={
        '263':'Registered detail produces misaligned/double eyebrow structure and mismatched gaze.',
        '6681':'Transferred eye detail is clipped or displaced; local eye geometry does not match the target.',
        '10149':'Transferred eyelid expression and eye/brow positioning remain inconsistent with the original.'}
    checks=[]
    lines=['# Registered reference eye/brow texture: rejected prototype','',
           '10 October 2026. Three development people, seed 17 and the unchanged CFG 3 generated scaffold. Four real reference photos; the evaluation original is never a generation input. Fixed visible-anchor alignment, open-eye threshold, residual threshold, bounded colour offset and feathered transfer are specified in FOCUSED_EYE_TEXTURE_PREFLIGHT_V1.md. No training or new diffusion generation occurred. The files labelled raw contain the post-transfer image.', '',
           '## Matched three-person means','',
           '| Metric | CFG 3 control | Texture transfer | Delta | Valid pairs |',
           '|---|---:|---:|---:|---:|']
    means={}
    for k in KEYS:
        assert all(candidates[i]['metrics'][k] is not None and controls[i]['metrics'][k] is not None for i in IDS)
        b=float(np.mean([controls[i]['metrics'][k] for i in IDS]));a=float(np.mean([candidates[i]['metrics'][k] for i in IDS]))
        means[k]=dict(control=b,candidate=a,delta=a-b,n=3)
        lines.append(f'| {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} | 3/3 |')
    lines+=['','## Every person','',
            '| Person | Metric | CFG 3 control | Transfer | Delta |','|---|---|---:|---:|---:|']
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for index,i in enumerate(IDS,1):
        c=cases[i];row=generated[i];details=row['details'];p=OUT/row['file']
        assert sha(p)==row['sha256']==candidates[i]['sha256']
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        observed=rgb(c['input']);mask=np.asarray(Image.open(c['mask']))>127
        scaffold=rgb(ROOT/'outputs/focused_eye_guidance_v1'/controls[i]['file'])
        assert sha(ROOT/'outputs/focused_eye_guidance_v1'/controls[i]['file'])==details['scaffold_sha256']
        assert row['mechanism_status']=='transferred' and details['zero_exact']
        geometry=np.load(OUT/f'{i}_geometry.npz')
        warped=rgb(OUT/f'{i}_registered_reference.png')
        zero=apply(scaffold,observed,mask,warped,geometry['support'],details['colour_offset'],0.)
        regenerated=apply(scaffold,observed,mask,warped,geometry['support'],details['colour_offset'],1.)
        assert np.array_equal(zero,scaffold) and np.array_equal(regenerated,rgb(p))
        assert np.array_equal(rgb(p)[~mask],observed[~mask])
        checks.append(dict(identity=i,zero_exact=True,regeneration_exact=True,visible_exact=True,sha256=sha(p)))
        for k in KEYS:
            b,a=controls[i]['metrics'][k],candidates[i]['metrics'][k]
            lines.append(f'| {i} | {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} |')
        sheet=Image.new('RGB',(960,355),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,img) in enumerate(zip(['Original - evaluation only','CFG 3 scaffold','Rejected texture prototype'],[rgb(c['target']),scaffold,rgb(p)])):
            sheet.paste(Image.fromarray(img).resize((320,320)),(320*col,35));draw.text((320*col+5,10),label,fill='black')
        sheet.save(OUT/f'{i}_comparison.png')
        local=pack/f'Set 0{index} - Person {i}'/'Rejected - Registered texture';local.mkdir(exist_ok=True)
        shutil.copy2(p,local/'Eyes - REJECTED prototype.png');shutil.copy2(OUT/f'{i}_comparison.png',local/'Comparison - REJECTED.png')
        (local/'READ ME.txt').write_text('REJECTED RESEARCH PROTOTYPE. Do not use this as an improved reconstruction.\nAll six mean quality/identity measures worsen against CFG 3; alignment artifacts remain.\nRetained to disclose negative results. Original is comparison-only, never a reference.\nSee research/FOCUSED_EYE_TEXTURE_RESULTS_V1.md\n')
    lines+=['','## Decision and explanation','',
            'Rejected; not added to the website. All six mean measures worsen against CFG 3. All three comparisons were inspected and contain local alignment/appearance errors. A good fit to visible global anchors does not establish correspondence for the hidden eye or eyebrow. An open-eye test also does not verify pose, expression, makeup or local occlusion compatibility.', '',
            *[f'- Person {i}: {notes[i]}' for i in IDS], '',
            'Zero-transfer bitmap equivalence, exact regeneration from saved geometry and exact visible-pixel preservation pass for all three. FaceNet and ArcFace are available for all three; no failed measurement was omitted. The original checkpoints, inputs, references and raw score files remain unchanged.', '',
            'Next spatial work must establish local pose/expression correspondence and confidence before transferring reference detail, with a safe fallback. Keep the optional CFG 3 comparison available while this is investigated; do not claim hidden iris colour or gaze has been recovered exactly. Corrupted-reference and independent-client evaluation remain pending. SSIM and identity cosine are not accuracy percentages.']
    (ROOT/'research/FOCUSED_EYE_TEXTURE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(selected=False,means=means,checks=checks,visual_review=notes,summary_sha256=sha(OUT/'summary.json'))
    (OUT/'paired_results.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    (ROOT/'research/focused_eye_texture_verification_v1.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(json.dumps(dict(selected=False,means=means,checks=checks),indent=2))

if __name__=='__main__':main()
