"""Copy the completed SSIM32 comparison into three fresh, existing local test sets.

This script is deliberately separate from generation, scoring, and research
exports. It copies unchanged, hash-verified PNG bytes only after all evaluation
records and a complete visual review exist. A failed scientific gate is preserved
as a failed research comparison, never presented as a selected app method.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/masked_sdxl_ssim_continuation_eval_v1'
CASES = ROOT / 'outputs/masked_task_v1/cases.json'
DEST = Path('C:/Users/rehaa/Downloads/All Face Datasets/Ready to test sets/'
            'Reference reconstruction - 3 people - 2026-10-09')
SUBFOLDER = 'Experimental - Masked SSIM 32 updates'
IDS = ['263', '6681', '10149']
NAMES = {'control3': 'Current CFG3', 'local': 'Neutral local appearance',
         'prior': 'Prior 64-update scorer', 'epsilon': 'Epsilon continuation',
         'ssim': 'SSIM continuation'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dig(value, key):
    for part in key.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def number(value):
    return f'{value:.6f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'NA'


def passed(value):
    return 'PASS' if value is True else 'FAIL' if value is False else 'UNAVAILABLE'


def safe_text(value):
    return str(value).replace('|', '\\|').replace('\r', '').replace('\n', ' ')


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_evidence(path, pins):
    raw = Path(path).read_bytes()
    pins[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def verify_hash(path, expected):
    require(bool(expected) and Path(path).is_file() and sha(path) == expected,
            f'Source missing or source hash changed: {path}')


def source_path(relative):
    path = (SOURCE / relative).resolve()
    require(path.is_relative_to(SOURCE.resolve()), 'Evaluation image escaped its source directory')
    return path


def image_item(path, filename, expected, role):
    path = Path(path).resolve()
    require(path.suffix.lower() == '.png' and Path(filename).name == filename,
            'Only flat PNG copies are supported')
    verify_hash(path, expected)
    return dict(source=str(path), file=filename, sha256=expected, role=role,
                bytes=path.stat().st_size)


def readme(ident, rows, paired, cfg):
    rejected = paired.get('numerical_gate') is not True or paired.get('visual_review_passed') is not True
    decision = ('The SSIM recipe failed at least one fixed quality or visual gate. This folder is a research '
                'comparison, not a selected app method. Do not treat the SSIM candidate as an improvement.' if rejected else
                'The saved fixed numerical and visual gates passed. This is still a three-person, one-seed '
                'research comparison requiring later replication; it is not a selected app method.')
    lines = [f'# Person {ident} — masked SSIM 32-update research comparison', '', decision, '',
        f"Global numerical gate: **{passed(paired.get('numerical_gate'))}**. "
        f"Global visual gate: **{passed(paired.get('visual_review_passed'))}**. "
        f"Eligible to propose replication: **{paired.get('eligible_to_propose_replication')}**. "
        'This export grants no replication or app promotion.', '',
        '## What is in this folder', '',
        '- `Original - EVALUATION ONLY.png` is the held-out original. Use it only for comparison; it is not a reconstruction reference.',
        '- `Erased input.png` and `Supplied mask.png` are the unchanged evaluation input and hole mask.',
        '- `Reference 01.png` through `Reference 04.png` are the four original clean references used by this evaluation, in config order.',
        '- Each of the five arms has an unchanged `Raw.png` and `Finished.png` file. The finished output includes Poisson finishing and final visible-pixel preservation.',
        '- `Raw comparison.png` and `Finished comparison.png` are the saved, reviewed comparison sheets for this person.',
        '- `copy_receipt.json` records every copied file, its source, and verified SHA-256 hash.', '',
        '## Measured finished-output values', '',
        'All values below are copied from the saved frozen evaluator. Raw outputs are supplied for visual '
        'inspection and are not assigned these finished-output scores. PSNR is in dB. SSIM and identity cosines '
        'are measurements, not accuracy percentages. `NA` means no saved finite value; the receipt retains '
        'the complete original region and detection statuses.', '']
    groups = [
        ('Whole image', [('PSNR', 'metrics.psnr'), ('SSIM', 'metrics.ssim'), ('LPIPS', 'metrics.lpips')]),
        ('Original supplied hole', [('PSNR', 'metrics.hole_psnr'), ('SSIM', 'metrics.hole_ssim'), ('Crop LPIPS', 'metrics.hole_lpips')]),
        ('Eye annotation intersected with the hole', [('PSNR', 'regions.eyes.psnr'), ('SSIM', 'regions.eyes.ssim'), ('Crop LPIPS', 'metrics.eye_lpips')]),
        ('Identity and visible preservation', [('FaceNet cosine', 'metrics.facenet_cosine'),
             ('ArcFace conditioning cosine', 'metrics.arcface_conditioning_cosine'),
             ('Visible pixels changed', 'metrics.visible_changed')]),
    ]
    for heading, columns in groups:
        lines += ['### ' + heading, '', '| Arm | ' + ' | '.join(label for label, _ in columns) + ' |',
                  '|---|' + '---:|' * len(columns)]
        for arm, label in NAMES.items():
            lines.append('| ' + label + ' | ' + ' | '.join(number(dig(rows[arm], key)) for _, key in columns) + ' |')
        lines += ['']
    lines += ['Hole/eye LPIPS uses the fixed region bounding box plus 10 pixels of context, with no resizing; '
        'it is not pixel-masked LPIPS. Regional SSIM uses fully contained 11 × 11 windows. Whole-image scores '
        'include the preserved visible area and do not substitute for hole/eye scores. FaceNet is separate '
        'from generation conditioning; ArcFace belongs to the conditioning family.', '',
        '## Saved decision and review', '',
        'The fixed candidate gate requires strictly higher mean hole SSIM against each of the four controls, '
        'non-worsening mean hole PSNR, hole/eye LPIPS, FaceNet and ArcFace, all three valid pairs, and each '
        'person’s identity cosine within 0.005 of every control. Exact neutral replay, matched supports, '
        'complete records and zero final visible changed pixels are also required.', '',
        '| Control | Mean metrics gate | Every-person identity guard |', '|---|---|---|']
    for arm, comparison in paired['comparisons'].items():
        lines.append(f"| {NAMES[arm]} | {passed(comparison.get('mean_gate'))} | {passed(comparison.get('identity_guard'))} |")
    lines += ['', f"Saved visual note for person {ident}: " + safe_text(paired['visual_review']['notes'][ident]), '',
              f"Global failed gate count: {len(paired.get('failed_gates', []))}.", '']
    lines += ['- ' + safe_text(item) for item in paired.get('failed_gates', [])]
    lines += ['', '## Fixed experiment scope', '',
        'Both new arms continue the same completed 64-update scorer for 32 updates on eight other training '
        'identities, with reset optimizers and the same recorded identity/noise/timestep schedule. The SSIM arm '
        'adds 0.1 fully-contained masked structural loss to epsilon loss. The five arms are current CFG3, '
        'neutral local appearance, prior 64-update scorer, matched epsilon continuation, and SSIM continuation.', '',
        f"Recorded inference seeds: {cfg.get('seeds')}; steps: {cfg.get('steps')}; CFG: {cfg.get('guidance')}; "
        f"FaceID scale: {cfg.get('scale')}; appearance scale: {cfg.get('appearance_scale')}; strength: {cfg.get('strength')}; "
        f"inference/export resolution: {cfg.get('evaluation_resolution')}/{cfg.get('export_resolution')}.", '',
        'Three development people and one inference seed do not establish generalization or recovery of the true '
        'hidden face. This folder preserves all five comparisons regardless of outcome. All photographs stay '
        'in this local test collection. No existing app method, prior test folder or source image is changed.']
    return '\n'.join(lines) + '\n'


def main():
    pins = {}
    cfg = read_evidence(SOURCE / 'config.json', pins)
    summary = read_evidence(SOURCE / 'summary.json', pins)
    paired = read_evidence(SOURCE / 'paired_results.json', pins)
    generation = read_evidence(SOURCE / 'generation_receipt.json', pins)
    scoring = read_evidence(SOURCE / 'scoring_receipt.json', pins)
    visual = read_evidence(SOURCE / 'visual_review.json', pins)
    cases_list = read_evidence(CASES, pins)
    require(cfg.get('conditions') == list(NAMES) and cfg.get('selected_identities') == IDS and
            cfg.get('training_updates') == 32 and cfg.get('expected') == 15,
            'The fixed SSIM32 five-arm/three-person experiment is required')
    require(generation.get('status') == 'complete' and scoring.get('status') == 'complete' and
            paired.get('complete') is True, 'Generation, scoring and reporting must be complete')
    for path, expected in [
        (CASES, cfg['cases_sha256']),
        (SOURCE / 'config.json', summary['config_sha256']),
        (SOURCE / 'config.json', paired['config_sha256']),
        (SOURCE / 'config.json', scoring['config_sha256']),
        (SOURCE / 'config.json', generation['config_sha256']),
        (SOURCE / 'summary.json', paired['summary_sha256']),
        (SOURCE / 'summary.json', scoring['summary_sha256']),
        (SOURCE / 'summary.json', visual['summary_sha256']),
        (SOURCE / 'generation_receipt.json', paired['generation_receipt_sha256']),
        (ROOT / 'scripts/report_masked_sdxl_ssim_continuation_eval_v1.py', cfg['report_wrapper_sha256']),
        (ROOT / 'scripts/score_masked_sdxl_ssim_continuation_eval_v1.py', cfg['scorer_wrapper_sha256']),
        (ROOT / 'scripts/score_masked_sdxl_v1.py', cfg['scorer_sha256']),
    ]:
        verify_hash(path, expected)
    require(visual.get('reviewed_ids') == IDS and isinstance(visual.get('approved'), bool) and
            all(isinstance(visual.get('notes', {}).get(i), str) and visual['notes'][i].strip() for i in IDS),
            'A completed saved review of all three people is required, whether approved or rejected')
    require(paired.get('visual_review') == visual and paired.get('visual_review_pending') is False,
            'Rerun the unchanged report after saving the complete visual review')
    lookup = {(r['identity'], r['condition']): r for r in summary['rows']}
    require(len(lookup) == len(summary['rows']) == 15 and
            set(lookup) == {(i, a) for i in IDS for a in NAMES} and
            all(r.get('status') == 'complete' for r in lookup.values()), 'Exactly fifteen complete outcomes are required')
    selected_cases = [c for c in cases_list if c['identity'] in IDS and c['location'] == 'eyes']
    require(len(selected_cases) == 3, 'Exactly one preserved eye case per person is required')
    cases = {c['identity']: c for c in selected_cases}
    require(set(cases) == set(IDS), 'Eye case identities do not match')
    sheets = {(p['identity'], p['kind']): p for p in paired.get('contact_sheets', [])}
    require(len(sheets) == 6, 'Run the unchanged report with all six pinned contact sheets')
    base = DEST.resolve(strict=True)
    require(base.is_dir() and not base.is_relative_to(ROOT.resolve()), 'Local test-set root is absent or inside the repository')
    plans = []
    # Preflight every destination and every image before making any new folder.
    for position, ident in enumerate(IDS, 1):
        person = (base / f'Set {position:02d} - Person {ident}').resolve(strict=True)
        require(person.is_dir() and person.parent == base, 'Expected existing person folder is absent or redirected')
        target = person / SUBFOLDER
        require(not target.exists(), f'Refusing overwrite of local comparison set: {target}')
        case = cases[ident]
        images = [image_item(case[key], name, case[key + '_sha256'], role) for key, name, role in [
            ('target', 'Original - EVALUATION ONLY.png', 'evaluation_original'),
            ('input', 'Erased input.png', 'generation_input'), ('mask', 'Supplied mask.png', 'supplied_hole_mask')]]
        references = cfg['references'].get(ident + '/clean', [])
        require(len(references) == 4, f'Four original clean reference pins are required for {ident}')
        for index, reference in enumerate(references, 1):
            require(reference['sha256'] != case['target_sha256'], 'Evaluation original cannot be a reconstruction reference')
            images.append(image_item(reference['path'], f'Reference {index:02d}.png', reference['sha256'], 'clean_reference'))
        rows = {arm: lookup[(ident, arm)] for arm in NAMES}
        for arm, name in NAMES.items():
            row = rows[arm]
            final = source_path(row['file'])
            images.append(image_item(final, name + ' - Finished.png', row['sha256'], arm + '/finished'))
            images.append(image_item(final.with_name(final.stem + '_raw.png'), name + ' - Raw.png', row['raw_sha256'], arm + '/raw'))
        for kind, filename in [('raw', 'Raw comparison.png'), ('poisson_finished', 'Finished comparison.png')]:
            pin = sheets[(ident, kind)]
            require(Path(pin['path']).resolve().is_relative_to(SOURCE.resolve()), 'Comparison sheet escaped evaluation source')
            images.append(image_item(pin['path'], filename, pin['sha256'], 'comparison_sheet/' + kind))
        require(len(images) == 19 and len({i['file'] for i in images}) == 19, 'Unexpected or duplicate copy plan')
        plans.append(dict(identity=ident, target=target, images=images, rows=rows,
                          readme=readme(ident, rows, paired, cfg)))
    for path, expected in pins.items():
        verify_hash(path, expected)
    for plan in plans:
        target = plan['target']
        target.mkdir(exist_ok=False)
        receipt = dict(status='copying', identity=plan['identity'], created_at_utc=datetime.now(timezone.utc).isoformat(),
            local_only=True, image_bytes_modified=False, source_evidence_sha256=pins,
            exporter_sha256=sha(__file__), global_numerical_gate=paired['numerical_gate'],
            global_visual_review_passed=paired['visual_review_passed'], failed_gates=paired.get('failed_gates', []),
            selected_app_method=False, replication_authorized=False, website_promotion=False,
            finished_score_records=plan['rows'], visual_review=visual, copied_files=[])
        try:
            for item in plan['images']:
                src = Path(item['source'])
                verify_hash(src, item['sha256'])
                dst = target / item['file']
                with src.open('rb') as incoming, dst.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                shutil.copystat(src, dst)
                verify_hash(dst, item['sha256'])
                receipt['copied_files'].append(item)
            with (target / 'README.md').open('x', encoding='utf-8') as handle:
                handle.write(plan['readme'])
            receipt['readme_sha256'] = sha(target / 'README.md')
            for path, expected in pins.items():
                verify_hash(path, expected)
            receipt.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat())
            write_new_json(target / 'copy_receipt.json', receipt)
        except BaseException as exc:
            receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_new_json(target / 'copy_failure_receipt.json', receipt)
            raise
        print(json.dumps(dict(identity=plan['identity'], folder=str(target), png_files=len(plan['images']),
                              status=receipt['status'], selected_app_method=False)))


if __name__ == '__main__':
    main()
