"""Matched statistics and local visual comparisons for the frozen FreeU study."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from report_focused_eye_guidance_replication_v2 import summarize, gate, KEYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/focused_eye_freeu_v1'
IDS = ['263', '6681', '10149']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main(visual_pass=False):
    source = json.loads((OUT / 'summary.json').read_text())
    cfg = json.loads((OUT / 'config.json').read_text())
    visual = json.loads((OUT / 'visual_review.json').read_text()) if (OUT / 'visual_review.json').exists() else None
    seeds = cfg['seeds']
    rows = source['rows']
    lookup = {(r['identity'], r['seed'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows), 'Duplicate attempted output'
    comparisons = {}
    for control in ['control3', 'control5']:
        pairs = []
        for ident in IDS:
            for seed in seeds:
                c = lookup.get((ident, seed, control), {})
                a = lookup.get((ident, seed, 'freeu'), {})
                pairs.append(dict(identity=ident, seed=seed, control=c.get('metrics', {}),
                    candidate=a.get('metrics', {}), control_status=c.get('status', 'not_attempted'),
                    candidate_status=a.get('status', 'not_attempted')))
        overall = summarize(pairs)
        people = {i: summarize([p for p in pairs if p['identity'] == i]) for i in IDS}
        comparisons[control] = dict(overall=overall, per_person=people, pairs=pairs,
            mean_gate=gate(overall, len(pairs)),
            per_person_gates={i: gate(stats, len(seeds)) for i, stats in people.items()})
    identity_guard = all(
        stats[k]['n'] == len(seeds) and stats[k]['delta'] is not None and stats[k]['delta'] >= -.005
        for stats in comparisons['control3']['per_person'].values()
        for k in ['facenet_cosine', 'arcface_conditioning_cosine'])
    numeric_pass = all(c['mean_gate'] for c in comparisons.values()) and identity_guard
    decision = dict(numeric_pass=numeric_pass, per_person_identity_guard=identity_guard,
        visual_review_pass=visual_pass, replicate=numeric_pass and visual_pass and seeds == [17],
        website_promotion=False, source_summary_sha256=sha(OUT / 'summary.json'),
        definition='All six matched mean metrics pass against CFG3 and CFG5; each person identity delta vs CFG3 >= -0.005; visual eye/brow gate also required.')
    save(OUT / ('screen_decision.json' if seeds == [17] else 'replication_decision.json'), decision)
    save(OUT / 'paired_results.json', dict(comparisons=comparisons, decision=decision))
    lines = ['# Fixed FreeU decoder experiment on three development people', '',
        '10 October 2026. Installed Diffusers FreeU with author SDXL tuple b1=1.3, b2=1.4, s1=0.9, s2=0.2. '
        'Only decoder feature reweighting changes against CFG3. Existing masks, four references, DDIM30, FaceID0.8, full strength, '
        '1024 model / 512 export and Poisson finishing are unchanged. No training or production changes.', '',
        f"Seeds: {seeds}. Completed {sum(r['status']=='complete' for r in rows)}/{cfg['expected']} records, including "
        f"{sum(r.get('reused',False) for r in rows)} preserved controls. Every failed attempt and missing detection remains explicit.", '',
        '| Metric | Original CFG5 | Current CFG3 | FreeU candidate |', '|---|---:|---:|---:|']
    def average(arm, key):
        v = [r['metrics'][key] for r in rows if r['condition'] == arm and r.get('metrics', {}).get(key) is not None]
        return f'{np.mean(v):.6f} ({len(v)}/{len(IDS)*len(seeds)})' if v else 'NA'
    for key in KEYS:
        lines.append('| ' + key + ' | ' + ' | '.join(average(arm, key) for arm in ['control5', 'control3', 'freeu']) + ' |')
    # Tables above disclose arm-wise coverage; decisions below always use matched pairs.
    for arm, comparison in comparisons.items():
        lines += ['', f'## Matched differences against {arm}', '',
            '| Metric | Control | Candidate | Delta | Valid pairs |', '|---|---:|---:|---:|---:|']
        for key, v in comparison['overall'].items():
            numbers = f"{v['control']:.6f} | {v['candidate']:.6f} | {v['delta']:+.6f}" if v['delta'] is not None else 'NA | NA | NA'
            lines.append(f"| {key} | {numbers} | {v['n']}/{len(IDS)*len(seeds)} |")
    lines += ['', '## Every attempted image', '',
        '| Person | Seed | Arm | Hole SSIM | Hole PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for ident in IDS:
        for seed in seeds:
            for arm in ['control5', 'control3', 'freeu']:
                row = lookup.get((ident, seed, arm), {})
                values = row.get('metrics', {})
                ordered = ['hole_ssim', 'hole_psnr', 'hole_lpips', 'eye_lpips', 'facenet_cosine', 'arcface_conditioning_cosine']
                entries = [f'{values[k]:.6f}' if values.get(k) is not None else 'NA' for k in ordered]
                lines.append(f'| {ident} | {seed} | {arm} | ' + ' | '.join(entries) + ' |')
    lines += ['', '## Decision and scope', '',
        f"Combined numerical gate: {'PASS' if numeric_pass else 'FAIL'}. Per-person identity guard: {'PASS' if identity_guard else 'FAIL'}.",
        'The fixed screen does not authorize website promotion. Visual review is recorded separately in visual_review.json. '
        'If the screen fails, seeds29/43 are not spent on this recipe. No best-person, best-seed or parameter selection is used.', '',
        'Hole SSIM uses the unchanged Gaussian RGB map and fully contained11x11 windows. Whole-image SSIM is dominated by unchanged visible pixels; '
        'it is not missing-eye accuracy. Contextual LPIPS is a bounding-box crop, not a pixel-masked perceptual measure. '
        'Small anatomical-eye masks may have very few SSIM windows. FaceNet is independent of generation conditioning; '
        'ArcFace uses the conditioning recognition model family. This development test does not measure unseen-client accuracy.', '',
        'FreeU adds no local reference RGB correspondence. Eyelid, iris, gaze and eyebrow fidelity still require visual inspection. '
        'CPU offload allows the frozen decoder to run on the laptop GPU; actual runtime and peak memory are retained, with no overclocking.', '',
        'Sources: [FreeU paper](https://arxiv.org/abs/2309.11497), '
        '[author SDXL recipe](https://github.com/ChenyangSi/FreeU#sdxl). '
        'See [audit and frozen preflight](FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md).']
    if visual:
        lines += ['', '## Completed visual review', '', *[f'- Person {i}: {note}' for i, note in visual['notes'].items()], '', visual['decision']]
    cases = {c['identity']: c for c in json.loads((ROOT / 'outputs/masked_task_v1/cases.json').read_text()) if c['location'] == 'eyes'}
    pack = Path.home() / 'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
    for number, ident in enumerate(IDS, 1):
        c = cases[ident]
        assert sha(c['target']) == c['target_sha256'] and sha(c['input']) == c['input_sha256']
        mask = np.asarray(Image.open(c['mask'])) > 127
        y, x = np.nonzero(mask)
        crop = (max(0,int(x.min())-10),max(0,int(y.min())-10),min(512,int(x.max())+11),min(512,int(y.max())+11))
        local = pack / f'Set 0{number} - Person {ident}' / 'Experimental - FreeU decoder'
        local.mkdir(exist_ok=True)
        for seed in seeds:
            images = [('Original - evaluation only', Image.open(c['target']).convert('RGB')),
                      ('Erased input', Image.open(c['input']).convert('RGB'))]
            for arm, label in [('control5', 'Original CFG5'), ('control3', 'Current CFG3'), ('freeu', 'Experimental FreeU')]:
                row = lookup.get((ident, seed, arm), {})
                if row.get('status') == 'complete':
                    path = OUT / row['file']
                    assert sha(path) == row['sha256']
                    images.append((label, Image.open(path).convert('RGB')))
                    shutil.copy2(path, local / f'{label} - seed {seed}.png')
                else:
                    images.append((label + ' FAILED', Image.new('RGB',(512,512),'gray')))
            sheet = Image.new('RGB',(320*len(images),560),'white')
            draw = ImageDraw.Draw(sheet)
            for column, (label, image) in enumerate(images):
                sheet.paste(image.resize((320,320)),(column*320,30))
                sheet.paste(image.crop(crop).resize((320,180)),(column*320,375))
                draw.text((column*320+5,8),label,fill='black')
                draw.text((column*320+5,356),'Same mask bbox + context',fill='black')
            path = OUT / f'{ident}_comparison_{seed}.png'
            sheet.save(path)
            shutil.copy2(path, local / f'Comparison - seed {seed}.png')
        (local / 'READ ME.txt').write_text('Fixed experimental FreeU decoder test. Same references and mask.\n'
            'This setting is not added to Start Studio. Controls and all outcomes are retained.\n'
            'See research/FOCUSED_EYE_FREEU_RESULTS_V1.md for metrics and decision.\n',encoding='utf-8')
    (ROOT / 'research/FOCUSED_EYE_FREEU_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(decision=decision,comparison_to_cfg3=comparisons['control3']['overall']),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-pass',action='store_true')
    main(parser.parse_args().visual_pass)
