"""All four fixed arms: disclose paired and per-person gates and visual results."""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/focused_eye_factorial_v1'
IDS=['263','6681','10149']
ARMS={'CFG 5':'focused_reference_finish_v1','CFG 3':'focused_eye_guidance_v1',
      'CFG 5 + appearance':'focused_eye_appearance_v1','CFG 3 + appearance':'focused_eye_factorial_v1'}
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine'];LOWER={'hole_lpips','eye_lpips'}

def main():
    data={a:{r['identity']:r for r in json.loads((ROOT/'outputs'/folder/'summary.json').read_text())['rows'] if r['identity'] in IDS and r['location']=='eyes'} for a,folder in ARMS.items()}
    assert all(set(rows)==set(IDS) for rows in data.values())
    assert all(r['seed']==17 for rows in data.values() for r in rows.values())
    assert all(v['pixel_exact'] for v in json.loads((OUT/'replay.json').read_text()).values())
    comparisons={}
    for control in ['CFG 5','CFG 3']:
        stats={}
        for k in KEYS:
            valid=[i for i in IDS if data[control][i].get('metrics',{}).get(k) is not None and data['CFG 3 + appearance'][i].get('metrics',{}).get(k) is not None]
            b=float(np.mean([data[control][i]['metrics'][k] for i in valid])) if valid else None
            a=float(np.mean([data['CFG 3 + appearance'][i]['metrics'][k] for i in valid])) if valid else None
            stats[k]=dict(control=b,candidate=a,delta=a-b if valid else None,n=len(valid))
        comparisons[control]=dict(metrics=stats,mean_gate=all(v['n']==3 and (v['delta']<=0 if k in LOWER else v['delta']>=0) for k,v in stats.items()) and stats['hole_ssim']['delta']>0)
    means={a:{k:float(np.mean([r['metrics'][k] for r in rows.values()])) if all(r.get('metrics',{}).get(k) is not None for r in rows.values()) else None for k in KEYS} for a,rows in data.items()}
    visual_notes={
        '263':'Appearance at CFG 3 changes iris colour/gaze and does not recover original eyebrow shape; SSIM falls relative to CFG 3 alone.',
        '6681':'Brow/eye structure and colour remain mismatched; added appearance lowers SSIM relative to CFG 3 alone.',
        '10149':'Eyelid, gaze and eyebrow discrepancies persist; FaceNet remains below the CFG 5 control.'}
    def fmt(values):return ' | '.join('NA' if values.get(k) is None else f'{values[k]:.6f}' for k in KEYS)
    lines=['# Fixed guidance and appearance interaction: all four arms','',
           '10 October 2026. Same three development people, four clean real references, seed 17, masks, DDIM 30, FaceID scale 0.8, strength 1, 1024 inference / 512 output and Poisson finish. Guidance is fixed to 5 or 3; appearance branch is absent or fixed at 0.2. The last arm adds only appearance relative to the already tested CFG 3 control. No training or per-person/seed/scale selection.','',
           '## Three-person means','', '| Arm | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:lines.append(f'| {a} | {fmt(means[a])} |')
    lines += ['', '## Every person and every arm','', '| Person | Arm | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for i in IDS:
        for a in ARMS:lines.append(f'| {i} | {a} | {fmt(data[a][i].get("metrics",{}))} |')
    lines += ['', '## Paired final-arm differences','', '| Control | Metric | Control mean | Final mean | Delta | Valid pairs |',
              '|---|---|---:|---:|---:|---:|']
    for control,receipt in comparisons.items():
        for k,v in receipt['metrics'].items():
            lines.append(f"| {control} | {k} | {v['control']:.6f} | {v['candidate']:.6f} | {v['delta']:+.6f} | {v['n']}/3 |" if v['delta'] is not None else f'| {control} | {k} | NA | NA | NA | 0/3 |')
    lines+=['','## Numerical decision','', *[f"- Final arm against {c}: {'PASS' if r['mean_gate'] else 'FAIL'} on matched mean requirements." for c,r in comparisons.items()], '',
            'Keep all per-person tradeoffs visible. A mean pass is not sufficient for universal deployment. The neutral CFG 3 / appearance-off replay matches the saved CFG 3 raw and final images byte-for-byte. Failed attempts, missing identity measurements and original controls remain available. All completed outputs preserve visible pixels exactly.','',
            'All three original/four-arm contact sheets were inspected. The combined arm is not selected: SSIM and ArcFace worsen against CFG 3 alone, and local eye fidelity is unresolved. Three development people and one inference seed cannot establish generalization; SSIM and identity cosines are not accuracy percentages. Whole-image restoration ReF-LDM scores belong to a different task/protocol. See FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md and the separate three-seed guidance-only results.', '',
            *[f'- Person {i}: {visual_notes[i]}' for i in IDS]]
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    for index,ident in enumerate(IDS,1):
        images=[Image.open(cases[ident]['target']).convert('RGB')]
        for a,folder in ARMS.items():
            row=data[a][ident];images.append(Image.open(ROOT/'outputs'/folder/row['file']).convert('RGB') if row['status']=='complete' else Image.new('RGB',(512,512),'gray'))
        labels=['Original - evaluation only',*ARMS];sheet=Image.new('RGB',(1600,355),'white');draw=ImageDraw.Draw(sheet)
        for col,(label,img) in enumerate(zip(labels,images)):
            sheet.paste(img.resize((320,320)),(col*320,35));draw.text((col*320+5,10),label,fill='black')
        sheet.save(OUT/f'{ident}_comparison.png')
        local=pack/f'Set 0{index} - Person {ident}'/'Experimental - Guidance and appearance';local.mkdir(exist_ok=True)
        shutil.copy2(OUT/f'{ident}_comparison.png',local/'All four methods.png')
        if data['CFG 3 + appearance'][ident]['status']=='complete':shutil.copy2(OUT/data['CFG 3 + appearance'][ident]['file'],local/'Eyes - CFG 3 plus appearance - experimental.png')
        (local/'READ ME.txt').write_text('Four fixed research arms, same person/references/input/mask/seed17.\nNot a universal website default. Original is evaluation-only; never include it as a reference.\nFull measured comparison: research/FOCUSED_EYE_FACTORIAL_RESULTS_V1.md\n')
    (ROOT/'research/FOCUSED_EYE_FACTORIAL_RESULTS_V1.md').write_text('\n'.join(lines)+'\n')
    (OUT/'paired_results.json').write_text(json.dumps(dict(means=means,comparisons=comparisons,per_person=data,selected=False,visual_review=visual_notes),indent=2,allow_nan=False))
    print(json.dumps(dict(means=means,gates={k:v['mean_gate'] for k,v in comparisons.items()}),indent=2))

if __name__=='__main__':main()
