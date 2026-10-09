"""Paired multiseed development report; no best-case or seed selection."""
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/focused_eye_guidance_replication_v2'
IDS=['263','6681','10149']; SEEDS=[17,29,43]
KEYS=['hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']
LOWER={'hole_lpips','eye_lpips'}

def summarize(pairs):
    result={}
    for key in KEYS:
        valid=[p for p in pairs if p['control'].get(key) is not None and p['candidate'].get(key) is not None]
        a=np.array([p['candidate'][key] for p in valid]);b=np.array([p['control'][key] for p in valid])
        d=a-b;good=d<0 if key in LOWER else d>0
        result[key]=dict(n=len(valid),control=float(b.mean()) if len(b) else None,candidate=float(a.mean()) if len(a) else None,
                         delta=float(d.mean()) if len(d) else None,improved=int(good.sum()),worsened=int(((d>0) if key in LOWER else (d<0)).sum()))
    return result

def gate(stats,n):
    return all(v['n']==n and v['delta'] is not None and (v['delta']<=0 if k in LOWER else v['delta']>=0) for k,v in stats.items()) and stats['hole_ssim']['delta']>0

def main():
    source=json.loads((OUT/'summary.json').read_text());assert source['all_attempted']
    data={(r['identity'],r['seed'],r['condition']):r for r in source['rows']}
    assert len(data)==18
    pairs=[]
    for ident in IDS:
        for seed in SEEDS:
            c=data[ident,seed,'control'];a=data[ident,seed,'candidate']
            pairs.append(dict(identity=ident,seed=seed,control=c.get('metrics',{}),candidate=a.get('metrics',{}),
                              control_status=c['status'],candidate_status=a['status']))
    overall=summarize(pairs)
    people={i:summarize([p for p in pairs if p['identity']==i]) for i in IDS}
    seeds={str(s):summarize([p for p in pairs if p['seed']==s]) for s in SEEDS}
    visual_notes={
        '263':'Lower guidance usually softens exaggerated contrast, but iris colour and gaze remain different from the original across all three seeds.',
        '6681':'Lower guidance reduces oversaturation in some outputs and improves mean identity; brow/eye structure still varies with seed.',
        '10149':'Iris/gaze and eyebrow-shape errors persist across all seeds; mean FaceNet drops 0.013618 despite higher mean SSIM.'}
    result=dict(overall=overall,per_person=people,per_seed=seeds,pairs=pairs,
                mean_gate=gate(overall,9),per_person_gates={i:gate(v,3) for i,v in people.items()},selected=False,
                visual_review=visual_notes,created_utc=datetime.now(timezone.utc).isoformat())
    lines=['# Three-person lower-guidance replication','',
           'Fixed persons 263, 6681 and 10149; seeds 17, 29 and 43; four real clean references; identical erased-eye masks. DDIM 30, FaceID scale 0.8, strength 1, 1024 inference / 512 output and Poisson finishing. Only CFG 5 versus 3 changes. No training or website default changes. Six seed-17 outputs are hash-verified reused controls/candidates; twelve new GPU calls.','',
           f"Generation: {sum(r['status']=='complete' for r in source['rows'])}/18 complete. Every failed attempt remains in the source summary. All means below use identical valid pairs for each metric; coverage is explicit.",'',
           '## Nine-pair means','',
           '| Metric | CFG 5 control | CFG 3 candidate | Delta | Valid pairs | Improved / worsened |',
           '|---|---:|---:|---:|---:|---:|']
    def row(v):return ('NA | NA | NA' if v['delta'] is None else f"{v['control']:.6f} | {v['candidate']:.6f} | {v['delta']:+.6f}")
    for k,v in overall.items():lines.append(f"| {k} | {row(v)} | {v['n']}/9 | {v['improved']} / {v['worsened']} |")
    lines+=['','## Every person: three-seed means','', '| Person | Metric | Control | Candidate | Delta | Valid pairs |','|---|---|---:|---:|---:|---:|']
    for i,stats in people.items():
        for k,v in stats.items():lines.append(f"| {i} | {k} | {row(v)} | {v['n']}/3 |")
    lines+=['','## Every seed: three-person means','', '| Seed | Metric | Control | Candidate | Delta | Valid pairs |','|---|---|---:|---:|---:|---:|']
    for s,stats in seeds.items():
        for k,v in stats.items():lines.append(f"| {s} | {k} | {row(v)} | {v['n']}/3 |")
    lines+=['','## Every matched image','', '| Person | Seed | Metric | Control | Candidate | Delta |','|---|---:|---|---:|---:|---:|']
    for p in pairs:
        for k in KEYS:
            b,a=p['control'].get(k),p['candidate'].get(k)
            lines.append(f"| {p['identity']} | {p['seed']} | {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} |" if b is not None and a is not None else f"| {p['identity']} | {p['seed']} | {k} | {b} | {a} | NA |")
    lines+=['','## Numerical decision','',f"Combined mean gate: {'PASS' if result['mean_gate'] else 'FAIL'}.",
            *[f"- Person {i}, three-seed gate: {'PASS' if ok else 'FAIL'}." for i,ok in result['per_person_gates'].items()], '',
            'These gates do not authorize universal promotion. Visual eye/gaze/brow fidelity and per-image identity regressions remain separate. This is repeated development on three people, not nine independent people, a held-out test, population accuracy, or an iris-colour percentage. ArcFace also conditions generation; FaceNet is separate. Anatomical eye crops and all detector records remain in the frozen score files.','',
            'Preflight: FOCUSED_EYE_GUIDANCE_REPLICATION_PREFLIGHT_V2.md. Summary and full provenance: outputs/focused_eye_guidance_replication_v2.']
    lines+=['','## Visual review and next intervention','',
            'All nine matched control/candidate images were inspected in the three contact sheets. Neither the mean gate nor higher whole-face cosine establishes correct iris texture or gaze. The setting remains experimental and is not promoted universally.']
    lines += [f'- Person {i}: {visual_notes[i]}' for i in IDS]
    lines += ['',
              'Stop tuning the guidance setting on these same targets. Test a separate visual-reference appearance branch at fixed scale, with a neutral byte-exact replay and otherwise unchanged controls. See FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md. No appearance results are claimed in this guidance report.']
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    pack=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for number,ident in enumerate(IDS,1):
        original=Image.open(cases[ident]['target']).convert('RGB')
        sheet=Image.new('RGB',(960,355*3),'white');draw=ImageDraw.Draw(sheet)
        local=pack/f'Set 0{number} - Person {ident}'/'Experimental - CFG 3 repeated seeds';local.mkdir(exist_ok=True)
        for row_no,seed in enumerate(SEEDS):
            images=[original];labels=[f'Original / seed {seed}']
            for arm,label in [('control','Control CFG 5'),('candidate','Candidate CFG 3')]:
                r=data[ident,seed,arm]
                if r['status']=='complete':
                    p=OUT/r['file'];shutil.copy2(p,local/f'{label} - seed {seed}.png');images.append(Image.open(p).convert('RGB'))
                else:images.append(Image.new('RGB',(512,512),'gray'))
                labels.append(label if r['status']=='complete' else f'{label} FAILED')
            for col,(label,img) in enumerate(zip(labels,images)):
                y=row_no*355;sheet.paste(img.resize((320,320)),(col*320,y+35));draw.text((col*320+5,y+10),label,fill='black')
        sheet.save(OUT/f'{ident}_comparison.png');shutil.copy2(OUT/f'{ident}_comparison.png',local/'All seeds comparison.png')
        (local/'READ ME.txt').write_text('Fixed three-seed guidance comparison. Every seed is included; no best-seed selection.\nOriginal remains evaluation-only. No automatic website default change.\nSee research/FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md for all measured scores and limitations.\n')
    (OUT/'paired_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (ROOT/'research/FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['overall','mean_gate','per_person_gates']},indent=2))

if __name__=='__main__':main()
