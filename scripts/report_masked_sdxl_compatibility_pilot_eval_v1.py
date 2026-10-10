"""Disclose every fixed pilot outcome and gate against both preserved controls.

Contact sheets are evaluation artifacts kept locally. This report never changes
the studio, publishes photographs, selects a person/seed, or launches replication.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/masked_sdxl_compatibility_pilot_eval_v1'
IDS = ['263', '6681', '10149']
ARMS = {'control3': 'Current CFG3', 'local': 'Local appearance control', 'trained': 'Trained compatibility pilot'}
KEYS = ['hole_psnr', 'hole_ssim', 'hole_lpips', 'eye_lpips', 'facenet_cosine', 'arcface_conditioning_cosine']
LOWER = {'hole_lpips', 'eye_lpips'}
IDENTITY_KEYS = ['facenet_cosine', 'arcface_conditioning_cosine']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def number(value, signed=False):
    return 'NA' if value is None else format(value, '+.6f' if signed else '.6f')


def valid(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def comparisons(lookup):
    result = {}
    for control in ['control3', 'local']:
        stats = {}
        for key in KEYS:
            pairs = [(lookup.get((ident, control), {}).get('metrics', {}).get(key),
                      lookup.get((ident, 'trained'), {}).get('metrics', {}).get(key)) for ident in IDS]
            pairs = [(a, b) for a, b in pairs if valid(a) and valid(b)]
            before = sum(p[0] for p in pairs) / len(pairs) if pairs else None
            after = sum(p[1] for p in pairs) / len(pairs) if pairs else None
            delta = after - before if pairs else None
            passed = len(pairs) == 3 and (delta <= 0 if key in LOWER else delta >= 0)
            if key == 'hole_ssim':
                passed = len(pairs) == 3 and delta > 0
            stats[key] = dict(control=before, candidate=after, delta=delta, n=len(pairs), expected=3, passed=passed)
        people = {}
        for ident in IDS:
            guards = {}
            for key in IDENTITY_KEYS:
                before = lookup.get((ident, control), {}).get('metrics', {}).get(key)
                after = lookup.get((ident, 'trained'), {}).get('metrics', {}).get(key)
                delta = after - before if valid(before) and valid(after) else None
                guards[key] = dict(control=before, candidate=after, delta=delta,
                                   minimum_delta=-.005, passed=delta is not None and delta >= -.005)
            people[ident] = dict(metrics=guards, passed=all(g['passed'] for g in guards.values()))
        result[control] = dict(metrics=stats, mean_gate=all(v['passed'] for v in stats.values()),
                               per_person_identity=people, identity_guard=all(p['passed'] for p in people.values()))
    return result


def metric_support(row):
    regions = row.get('regions', {})
    masked, eyes, visible = [regions.get(k, {}) for k in ['masked', 'eyes', 'visible']]
    return dict(identity=row['identity'], condition=row['condition'], status=row['status'],
        hole_pixels=masked.get('pixels'), hole_ssim_valid_centres=masked.get('ssim_valid_centres'),
        hole_ssim_status=masked.get('ssim_status'), hole_psnr_status=masked.get('psnr_status'),
        hole_lpips_bbox_xyxy=masked.get('bbox_xyxy'), hole_lpips_status=masked.get('bbox_lpips_status'),
        eye_pixels=eyes.get('pixels'), eye_lpips_bbox_xyxy=eyes.get('bbox_xyxy'), eye_lpips_status=eyes.get('bbox_lpips_status'),
        visible_pixels=visible.get('pixels'), visible_changed=visible.get('changed_pixels_vs_input'),
        target_detections=row.get('target_detections'), output_detections=row.get('output_detections'))


def render_sheets(output, cfg, lookup):
    import numpy as np
    from PIL import Image, ImageDraw
    cases_path = ROOT / 'outputs/masked_task_v1/cases.json'
    if sha(cases_path) != cfg['cases_sha256']:
        raise ValueError('Case manifest changed before visual report')
    cases = {c['identity']: c for c in read(cases_path) if c['location'] == 'eyes' and c['identity'] in IDS}
    sheets = output / 'sheets'
    sheets.mkdir(exist_ok=True)
    pins = []
    for ident in IDS:
        case = cases[ident]
        for key in ['target', 'input', 'mask']:
            if sha(case[key]) != case[key + '_sha256']:
                raise ValueError(f'Evaluation image changed: {key}/{ident}')
        with Image.open(case['mask']) as image:
            mask = np.asarray(image.convert('L')) >= 128
        ys, xs = np.nonzero(mask)
        box = (max(0, int(xs.min()) - 10), max(0, int(ys.min()) - 10),
               min(512, int(xs.max()) + 11), min(512, int(ys.max()) + 11))
        for raw in [False, True]:
            with Image.open(case['target']) as image:
                original = image.convert('RGB').copy()
            with Image.open(case['input']) as image:
                damaged = image.convert('RGB').copy()
            images = [('Original - evaluation only', original), ('Erased input', damaged)]
            for arm, label in ARMS.items():
                row = lookup.get((ident, arm), {})
                if row.get('status') == 'complete':
                    path = output / row['file']
                    if raw:
                        path = path.with_name(path.stem + '_raw.png')
                    if sha(path) != row['raw_sha256' if raw else 'sha256']:
                        raise ValueError('Generated image changed before visual report')
                    with Image.open(path) as image:
                        rgb = image.convert('RGB').copy()
                    images.append((label + (' RAW' if raw else ''), rgb))
                else:
                    images.append((label + ' ' + row.get('status', 'missing').upper(), Image.new('RGB', (512, 512), 'gray')))
            sheet = Image.new('RGB', (320 * len(images), 560), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, (label, image) in enumerate(images):
                x = 320 * index
                sheet.paste(image.resize((320, 320)), (x, 30))
                sheet.paste(image.crop(box).resize((320, 180)), (x, 375))
                draw.text((x + 5, 8), label, fill='black')
                draw.text((x + 5, 356), 'Same supplied hole bbox + context', fill='black')
            path = sheets / f"{ident}_{'raw' if raw else 'finished'}_comparison.png"
            sheet.save(path)
            pins.append(dict(path=str(path), sha256=sha(path), identity=ident,
                             kind='raw' if raw else 'poisson_finished', evaluation_only=True))
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--no-sheets', action='store_true', help='Generate the numeric report only')
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = read(output / 'config.json')
    source = read(output / 'summary.json')
    generation = read(output / 'generation_receipt.json')
    scoring = read(output / 'scoring_receipt.json')
    if sha(__file__) != cfg['report_wrapper_sha256']:
        raise ValueError('Report wrapper changed after generation was frozen')
    if source['config_sha256'] != sha(output / 'config.json') or scoring['summary_sha256'] != sha(output / 'summary.json'):
        raise ValueError('Scored summary/configuration pin changed')
    rows = source['rows']
    lookup = {(r['identity'], r['condition']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate evaluation records')
    paired = comparisons(lookup)
    neutral_path = output / 'neutral_replay.json'
    neutral = read(neutral_path) if neutral_path.exists() else None
    neutral_pass = bool(neutral and neutral['passed'] and all(v['byte_exact'] and v['pixel_exact'] for v in neutral['checks'].values()))
    complete = len(rows) == cfg['expected'] == 9 and all(r['status'] == 'complete' for r in rows)
    visible_pass = complete and all(r.get('metrics', {}).get('visible_changed') == 0 for r in rows)
    supports_pass = source['matched_region_supports']['all_matched']
    numerical = complete and visible_pass and neutral_pass and supports_pass and generation['status'] == 'complete' and all(
        p['mean_gate'] and p['identity_guard'] for p in paired.values())
    visual_path = output / 'visual_review.json'
    visual = read(visual_path) if visual_path.exists() else None
    visual_pass = bool(visual and visual.get('approved') is True and visual.get('reviewed_ids') == IDS
                       and visual.get('summary_sha256') == sha(output / 'summary.json')
                       and all(isinstance(visual.get('notes', {}).get(ident), str) and visual['notes'][ident].strip() for ident in IDS))
    failed_gates = []
    for name, passed in [('all_nine_records_complete', complete), ('neutral_raw_and_final_byte_exact', neutral_pass),
                         ('visible_rgb_preserved', visible_pass), ('matched_region_supports', supports_pass),
                         ('generation_complete', generation['status'] == 'complete')]:
        if not passed:
            failed_gates.append(name)
    for arm, comparison in paired.items():
        failed_gates += [f'{arm}/{key}/mean' for key, value in comparison['metrics'].items() if not value['passed']]
        failed_gates += [f'{arm}/{ident}/{key}/identity_guard' for ident, person in comparison['per_person_identity'].items()
                         for key, value in person['metrics'].items() if not value['passed']]
    supports = [metric_support(r) for r in rows]
    result = dict(summary_sha256=sha(output / 'summary.json'), config_sha256=sha(output / 'config.json'),
        generation_receipt_sha256=sha(output / 'generation_receipt.json'), training_updates=64,
        training_resolution=512, evaluation_resolution=1024, export_resolution=512, trainable_parameters=385,
        comparisons=paired, numerical_gate=numerical, non_worsening_tolerance=0.,
        identity_guard_minimum_delta=-.005, neutral_replay_passed=neutral_pass, complete=complete,
        visible_preservation_passed=visible_pass, failed_gates=failed_gates, metric_supports=supports,
        matched_region_supports=source['matched_region_supports'],
        evaluation_component_sources=source['evaluation_component_sources'],
        coverage=source['coverage'], visual_review_pending=visual is None,
        visual_review=visual, visual_review_passed=visual_pass,
        eligible_to_propose_replication=numerical and visual_pass,
        replication_authorized=False, website_promotion=False)
    if not args.no_sheets:
        result['contact_sheets'] = render_sheets(output, cfg, lookup)
    save(output / 'paired_results.json', result)
    lines = ['# Fixed masked SDXL compatibility training pilot', '',
        'One 385-parameter scorer trained for 64 fixed updates at 512 on 16 existing training identities. '
        'Evaluation uses people 263, 6681 and 10149, four existing clean references, the preserved damaged eye inputs and masks, '
        'inference seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, strength 1, model offload, 1024 inference / 512 output and Poisson finishing. '
        'Current CFG3 and local appearance control images and scores are hash-verified reuses. '
        'The initial scorer is loaded and disabled for one separate local-appearance 263 neutral replay before three trained candidates.', '',
        f"Numerical gate: {'PASS' if numerical else 'FAIL'}. Completed {sum(r['status'] == 'complete' for r in rows)}/9 records. "
        f"Candidate generation attempts: {generation['candidate_attempts']}/3. Neutral replay: {'PASS' if neutral_pass else 'FAIL'}. "
        f"Visual review: {'PASS' if visual_pass else 'PENDING' if visual is None else 'FAIL'}.", '',
        '## Mean values and metric coverage', '', '| Metric | Current CFG3 | Local appearance control | Trained pilot |',
        '|---|---:|---:|---:|']
    for key in KEYS:
        cells = []
        for arm in ARMS:
            values = [r.get('metrics', {}).get(key) for r in rows if r['condition'] == arm]
            values = [v for v in values if valid(v)]
            cells.append(f'{number(sum(values) / len(values))} ({len(values)}/3)' if values else 'NA (0/3)')
        lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Every person and outcome', '',
        '| Person | Arm | Status | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace conditioning family |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for ident in IDS:
        for arm, label in ARMS.items():
            row = lookup.get((ident, arm), {})
            metrics = row.get('metrics', {})
            lines.append(f"| {ident} | {label} | {row.get('status', 'missing')} | " + ' | '.join(number(metrics.get(k)) for k in KEYS) + ' |')
    lines += ['', '## Matched gates against both controls', '',
        'Hole SSIM must strictly improve on each control mean. All other five metric means must be no worse, using zero tolerance. '
        'Every person must retain FaceNet and ArcFace cosine at least control minus 0.005 against each control. All six metrics require 3/3 valid pairs.', '',
        '| Control | Metric | Control mean | Pilot mean | Delta | Valid pairs | Gate |', '|---|---|---:|---:|---:|---:|---|']
    for arm, comparison in paired.items():
        for key, value in comparison['metrics'].items():
            lines.append(f"| {ARMS[arm]} | {key} | {number(value['control'])} | {number(value['candidate'])} | "
                         f"{number(value['delta'], True)} | {value['n']}/3 | {'PASS' if value['passed'] else 'FAIL'} |")
    lines += ['', '| Control | Person | Identity metric | Delta | Minimum allowed delta | Gate |', '|---|---|---|---:|---:|---|']
    for arm, comparison in paired.items():
        for ident, person in comparison['per_person_identity'].items():
            for key, value in person['metrics'].items():
                lines.append(f"| {ARMS[arm]} | {ident} | {key} | {number(value['delta'], True)} | -0.005000 | {'PASS' if value['passed'] else 'FAIL'} |")
    lines += ['', '## Failures, detections and visible pixels', '', '| Arm | Complete | Failed | Blocked | Reused | Visible exact |', '|---|---:|---:|---:|---:|---:|']
    for arm, label in ARMS.items():
        value = source['coverage'][arm]
        lines.append(f"| {label} | {value['complete']}/3 | {value['failed']} | {value['blocked']} | {value['reused']} | {value['visible_preservation']['exact']}/3 |")
    for row in rows:
        if row['status'] != 'complete':
            lines += ['', f"- {row['identity']} / {row['condition']} / {row['status']}: {row.get('error', 'No error text')}."]
    lines += ['', '## Metric supports for every person', '',
        '| Person | Arm | Hole pixels | Valid 11x11 SSIM centres | Eye pixels | Visible pixels | Visible pixels changed |',
        '|---|---|---:|---:|---:|---:|---:|']
    for support in supports:
        lines.append(f"| {support['identity']} | {ARMS[support['condition']]} | " + ' | '.join(
            'NA' if support[k] is None else str(support[k]) for k in ['hole_pixels', 'hole_ssim_valid_centres', 'eye_pixels', 'visible_pixels', 'visible_changed']) + ' |')
    lines += ['', 'LPIPS support is the fixed hole or eye bounding box plus 10 pixels of context, with no resizing. '
              'The exact crop boxes, metric status strings and full target/output face detections are recorded for every person in paired_results.json.', '',
              '| Person | Arm | FaceNet target/output detection | ArcFace target/output detection |', '|---|---|---|---|']
    for row in rows:
        cells = []
        for family in ['facenet', 'arcface_conditioning']:
            values = [row.get(field, {}).get(family, {}) for field in ['target_detections', 'output_detections']]
            cells.append(' / '.join(f"{v.get('status', 'unmeasured')} ({v.get('face_count', 'NA')})" for v in values))
        lines.append(f"| {row['identity']} | {ARMS[row['condition']]} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Decision and limits', '',
        'Every completed, failed and blocked attempt is retained. Negative mean differences and per-person regressions remain in the tables. '
        'A failed numerical gate stops this recipe. Even a numerical pass requires completed visual review of every person before proposing replication. '
        'This script grants no replication or website promotion.', '',
        'FaceNet is separate from generation conditioning; ArcFace is in the conditioning family. '
        'Cosines and SSIM are measurements, not accuracy percentages. Three development people and one inference seed do not estimate unknown-client accuracy. '
        'These learned reference tokens do not establish anatomical eye correspondence or verify recovery of the true hidden face. '
        'Training at 512 and evaluation at 1024 is an explicit pilot limitation. Final visible RGB is preserved by compositing; raw outputs are retained and shown separately.', '',
        'Generation did not open evaluation originals. The CPU evaluator and contact-sheet report use originals only for evaluation. '
        'Photographs and image artifacts remain local. Existing models, results, metrics and source files are preserved.', '',
        f"Failed gate count: {len(failed_gates)}."]
    lines += [f'- {name}' for name in failed_gates]
    if visual is not None:
        lines += ['', '## Saved visual review', '', *[f"- Person {ident}: {visual.get('notes', {}).get(ident, 'Missing review')}" for ident in IDS]]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(numerical_gate=numerical, visual_review_passed=visual_pass,
                         failed_gates=failed_gates, replication_authorized=False, website_promotion=False), indent=2))


if __name__ == '__main__':
    main()
