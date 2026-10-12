"""Export the fixed SSIM continuation's saved numeric evidence, without images.

Run after the unchanged scoring and reporting wrappers. This standard-library
exporter neither opens photographs nor computes image metrics, trains, generates,
changes gates, or modifies any source evaluation artifact.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'outputs/masked_sdxl_ssim_continuation_eval_v1'
TRAIN = ROOT / 'outputs/masked_sdxl_ssim_continuation_v1'
PROBE = ROOT / 'outputs/masked_sdxl_ssim_loss_probe_v1/receipt.json'
IDS = ['263', '6681', '10149']
ARMS = dict(control3='Current CFG3', local='Neutral local appearance',
            prior='Prior 64-update scorer', epsilon='Epsilon continuation',
            ssim='SSIM continuation')
METRICS = {
    'whole_psnr': ('metrics.psnr', 'regions.whole.psnr'),
    'whole_ssim': ('metrics.ssim', 'regions.whole.ssim'),
    'whole_lpips': ('metrics.lpips',),
    'hole_psnr': ('metrics.hole_psnr', 'regions.masked.psnr'),
    'hole_ssim': ('metrics.hole_ssim', 'regions.masked.ssim'),
    'hole_lpips': ('metrics.hole_lpips', 'regions.masked.bbox_lpips'),
    'eye_psnr': ('regions.eyes.psnr',),
    'eye_ssim': ('regions.eyes.ssim',),
    'eye_lpips': ('metrics.eye_lpips', 'regions.eyes.bbox_lpips'),
    'facenet_cosine': ('metrics.facenet_cosine',),
    'arcface_conditioning_cosine': ('metrics.arcface_conditioning_cosine',),
}
LOWER = {'whole_lpips', 'hole_lpips', 'eye_lpips'}
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dig(value, key):
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def public(value):
    """Keep numbers/hashes and text; omit image paths and machine user prefixes."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k != 'contact_sheets'}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str):
        if value.lower().endswith(IMAGE_SUFFIXES):
            return '[local image path omitted]'
        value = value.replace(str(ROOT), '.').replace(ROOT.as_posix(), '.')
        return re.sub(r'C:[/\\]Users[/\\][^/\\\s]+', '<USER_HOME>', value, flags=re.I)
    return value


class Sources:
    def __init__(self):
        self.pins = {}

    def record(self, path):
        path = Path(path).resolve()
        pin = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        if str(path) in self.pins and self.pins[str(path)]['sha256'] != pin['sha256']:
            raise ValueError(f'Evidence changed during export: {path}')
        self.pins[str(path)] = pin
        return pin

    def read(self, path, required=True):
        path = Path(path).resolve()
        if not path.is_file():
            if required:
                raise ValueError(f'Required saved evidence is absent: {path}')
            return None
        raw = path.read_bytes()
        pin = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if str(path) in self.pins and self.pins[str(path)]['sha256'] != pin['sha256']:
            raise ValueError(f'Evidence changed during export: {path}')
        self.pins[str(path)] = pin
        return json.loads(raw)

    def require_hash(self, path, expected):
        actual = self.record(path)['sha256']
        if not expected or actual != expected:
            raise ValueError(f'Evidence/source pin mismatch: {path}')

    def unchanged(self):
        for path, pin in self.pins.items():
            if sha(path) != pin['sha256']:
                raise ValueError(f'Evidence changed during export: {path}')


def normalize(row):
    values, origins = {}, {}
    for key, paths in METRICS.items():
        available = [(path, dig(row, path)) for path in paths if finite(dig(row, path))]
        if available and any(value != available[0][1] for _, value in available):
            raise ValueError(f'Conflicting saved metric aliases: {row.get("identity")}/{row.get("condition")}/{key}')
        values[key] = available[0][1] if available else None
        prefix = 'whole' if key.startswith('whole_') else 'masked' if key.startswith('hole_') else 'eyes'
        status_key = 'bbox_lpips_status' if key.endswith('lpips') else 'psnr_status' if key.endswith('psnr') else 'ssim_status'
        saved_status = dig(row, f'regions.{prefix}.{status_key}') if not key.endswith('cosine') else None
        origins[key] = dict(source=available[0][0] if available else None,
                            available=bool(available), saved_status=saved_status,
                            unavailable_reason=None if available else saved_status or 'not_available_in_saved_score')
    return dict(identity=row['identity'], condition=row['condition'], status=row.get('status', 'missing'),
                metrics=values, metric_provenance=origins,
                visible_changed=dig(row, 'metrics.visible_changed'), source_record=row)


def means(rows):
    return {arm: {key: dict(mean=sum(vals) / len(vals) if vals else None, n=len(vals), expected=len(IDS))
                  for key in METRICS
                  for vals in [[r['metrics'][key] for r in rows if r['condition'] == arm and finite(r['metrics'][key])]]}
            for arm in ARMS}


def all_comparisons(rows):
    lookup = {(r['identity'], r['condition']): r for r in rows}
    result = {}
    for arm in ['control3', 'local', 'prior', 'epsilon']:
        stats = {}
        for key in METRICS:
            pairs = [(lookup[(i, arm)]['metrics'][key], lookup[(i, 'ssim')]['metrics'][key]) for i in IDS]
            pairs = [(a, b) for a, b in pairs if finite(a) and finite(b)]
            before = sum(a for a, _ in pairs) / len(pairs) if pairs else None
            after = sum(b for _, b in pairs) / len(pairs) if pairs else None
            stats[key] = dict(control_mean=before, candidate_mean=after,
                              delta=after - before if pairs else None, n=len(pairs), expected=len(IDS),
                              preferred_direction='lower' if key in LOWER else 'higher',
                              gate_added_by_exporter=False)
        result[arm] = stats
    return result


def training_evidence(sources, training):
    result = {}
    for arm in ['epsilon', 'ssim']:
        folder = training / arm / 'training'
        cfg = sources.read(folder / 'config.json', required=False)
        receipt = sources.read(folder / 'receipt.json', required=False)
        trace = sources.read(folder / 'trace.json', required=False)
        result[arm] = dict(config=cfg, receipt=receipt, trace=trace,
                           availability='available' if receipt else 'unavailable')
        if receipt and receipt.get('config_sha256'):
            sources.require_hash(folder / 'config.json', receipt['config_sha256'])
        if receipt and receipt.get('trace_sha256'):
            sources.require_hash(folder / 'trace.json', receipt['trace_sha256'])
        for pin in (receipt or {}).get('checkpoints', {}).values():
            sources.require_hash(pin['path'], pin['sha256'])
        for pin in (cfg or {}).get('source_pins', []):
            sources.require_hash(pin['path'], pin['sha256'])
        for key in ['starting_checkpoint', 'probe_receipt', 'cache_receipt', 'prior_training_receipt']:
            pin = (cfg or {}).get(key)
            if pin:
                sources.require_hash(pin['path'], pin['sha256'])
    schedules = {arm: [{k: row.get(k) for k in ['update', 'identity', 'timestep', 'noise_seed']}
                       for row in (result[arm]['trace'] or [])] for arm in result}
    result['schedule_comparison'] = dict(expected_updates_per_arm=32,
        recorded_updates={arm: len(schedule) for arm, schedule in schedules.items()},
        all_32_schedules_match=all(len(s) == 32 for s in schedules.values()) and schedules['epsilon'] == schedules['ssim'])
    return result


def supervisor_evidence(sources, training):
    """Preserve stale/failed original records and supplemental recovery separately."""
    paths = sorted({*training.glob('*supervis*/*.json'), *training.glob('*supervis*/*.json.partial')})
    probe_supervisor = ROOT / 'outputs/masked_sdxl_ssim_loss_supervisor_v1/receipt.json'
    if probe_supervisor.is_file():
        paths.append(probe_supervisor)
    return [dict(path=str(p), record=sources.read(p)) for p in paths]


def number(value, signed=False):
    return format(value, '+.6f' if signed else '.6f') if finite(value) else 'NA'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ') if value is not None else 'NA'


def status(value):
    return 'PASS' if value is True else 'FAIL' if value is False else 'UNAVAILABLE'


def render(evidence):
    gates = evidence['official_gate_report']
    generation = evidence['generation']
    lines = ['# Masked SSIM continuation results v1', '',
        'Two fixed 32-update continuations of the same 64-update scorer compare epsilon loss with epsilon plus '
        '0.1 fully-contained masked structural loss. This export copies saved evaluator values and recomputes only '
        'arithmetic means and paired differences. It does not read photographs or recalculate image metrics.', '',
        f"Numerical gate: **{status(gates.get('numerical_gate'))}**. "
        f"Visual review: **{'PENDING' if gates.get('visual_review_pending') else status(gates.get('visual_review_passed'))}**. "
        f"Generation: {cell(generation.get('status'))}; attempts {generation.get('candidate_attempts', 'NA')}/6. "
        'No replication or website promotion is authorized by this export.', '',
        'Three fixed development people (263, 6681, 10149), seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, '
        'strength 1, 1024 inference, 512 export, and Poisson finishing. Controls are the preserved CFG3, local '
        'appearance, prior 64-update scorer, and the matched fresh epsilon continuation.', '',
        '## Metric definitions and availability', '',
        'PSNR is in dB; SSIM and identity cosines are measurements, not accuracy percentages. Hole metrics use '
        'the original supplied mask. Eye PSNR and SSIM are copied from `regions.eyes`, the annotated eye region '
        'intersected with the hole. Regional SSIM averages fully contained 11 × 11 windows. Hole and eye LPIPS '
        'are fixed bounding-box crops with 10 pixels of context and no resizing. Whole-image values include '
        'visible pixels preserved by compositing and cannot replace the hole results. `NA` denotes unavailable '
        'finite values; exact original status strings and metric source fields are retained in the JSON.', '',
        'FaceNet is separate from generation conditioning. ArcFace belongs to the conditioning family.', '',
        '## All metric means', '',
        '| Metric | ' + ' | '.join(ARMS.values()) + ' |', '|---|' + '---:|' * len(ARMS)]
    for key in METRICS:
        lines.append('| ' + key + ' | ' + ' | '.join(
            f"{number(evidence['means'][a][key]['mean'])} ({evidence['means'][a][key]['n']}/3)" for a in ARMS) + ' |')
    for title, keys in [
        ('Whole-image outcomes', ['whole_psnr', 'whole_ssim', 'whole_lpips']),
        ('Hole and eye outcomes', ['hole_psnr', 'hole_ssim', 'hole_lpips', 'eye_psnr', 'eye_ssim', 'eye_lpips']),
        ('Identity and visible-pixel outcomes', ['facenet_cosine', 'arcface_conditioning_cosine'])]:
        lines += ['', '## ' + title, '', '| Person | Arm | Status | ' + ' | '.join(keys) +
                  (' | Visible pixels changed' if title.startswith('Identity') else '') + ' |',
                  '|---|---|---|' + '---:|' * (len(keys) + int(title.startswith('Identity')))]
        for row in evidence['rows']:
            vals = [number(row['metrics'][key]) for key in keys]
            if title.startswith('Identity'):
                vals.append(cell(row['visible_changed']))
            lines.append(f"| {row['identity']} | {ARMS[row['condition']]} | {row['status']} | " + ' | '.join(vals) + ' |')
    lines += ['', '## Official fixed gates', '',
        'Hole SSIM must strictly improve against every control. Hole PSNR, hole/eye LPIPS and mean FaceNet/ArcFace '
        'must be no worse (zero tolerance), with 3/3 valid pairs. Each person must retain both identity cosines '
        'within 0.005 of each control. Complete records, exact neutral replay, matched supports and zero final '
        'visible changed pixels are required. Whole-image and eye PSNR/SSIM are supplementary diagnostics; '
        'the exporter adds no new selection gate.', '',
        '| Control | Metric | Control mean | SSIM mean | Delta | Pairs | Gate |', '|---|---|---:|---:|---:|---:|---|']
    for arm, comparison in gates['comparisons'].items():
        for key, item in comparison['metrics'].items():
            lines.append(f"| {ARMS[arm]} | {key} | {number(item['control'])} | {number(item['candidate'])} | "
                         f"{number(item['delta'], True)} | {item['n']}/3 | {status(item['passed'])} |")
    lines += ['', '| Control | Person | Identity metric | Delta | Minimum | Gate |', '|---|---|---|---:|---:|---|']
    for arm, comparison in gates['comparisons'].items():
        for ident, person in comparison['per_person_identity'].items():
            for key, item in person['metrics'].items():
                lines.append(f"| {ARMS[arm]} | {ident} | {key} | {number(item['delta'], True)} | -0.005000 | {status(item['passed'])} |")
    lines += ['', 'Failed official gates: ' + str(len(gates.get('failed_gates', []))) + '.', '']
    lines += ['- ' + item for item in gates.get('failed_gates', [])]
    lines += ['', '## Coverage, support and failures', '',
              '| Arm | Complete | Failed | Blocked | Missing | Reused |', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [r for r in evidence['rows'] if r['condition'] == arm]
        counts = Counter(r['status'] for r in rows)
        lines.append(f"| {ARMS[arm]} | {counts['complete']}/3 | {counts['failed']} | {counts['blocked']} | {counts['missing']} | "
                     f"{sum(bool(r['source_record'].get('reused')) for r in rows)} |")
    lines += ['', '| Person | Arm | Hole pixels | Hole SSIM centres | Eye pixels | Eye SSIM centres | FaceNet target/output | ArcFace target/output |',
              '|---|---|---:|---:|---:|---:|---|---|']
    for row in evidence['rows']:
        raw = row['source_record']
        geometry = [cell(dig(raw, key)) for key in ['regions.masked.pixels', 'regions.masked.ssim_valid_centres',
                                                   'regions.eyes.pixels', 'regions.eyes.ssim_valid_centres']]
        detections = [' / '.join(f"{cell(dig(raw, f'{side}.{family}.status'))} ({cell(dig(raw, f'{side}.{family}.face_count'))})"
                                for side in ['target_detections', 'output_detections'])
                      for family in ['facenet', 'arcface_conditioning']]
        lines.append(f"| {row['identity']} | {ARMS[row['condition']]} | " + ' | '.join(geometry + detections) + ' |')
    for row in evidence['rows']:
        if row['status'] != 'complete':
            lines += ['', f"{row['identity']}/{row['condition']}: {cell(row['source_record'].get('error', 'No saved error text'))}"]
    lines += ['', '## Training, probe and resources', '',
        '| Stage | Status | Completed updates | Seconds | Peak GPU allocated bytes | Peak GPU reserved bytes | Peak working-set bytes |',
        '|---|---|---:|---:|---:|---:|---:|']
    stages = [('Disposable structural-loss probe', evidence['probe'])] + [
        (arm, evidence['training'][arm]['receipt']) for arm in ['epsilon', 'ssim']]
    for label, receipt in stages:
        receipt = receipt or {}
        lines.append('| ' + label + ' | ' + ' | '.join(cell(v) for v in [receipt.get('status'),
            receipt.get('completed_updates'), receipt.get('elapsed_seconds'), receipt.get('peak_allocated_bytes'),
            receipt.get('peak_reserved_bytes'), dig(receipt, 'process_memory.peak_working_set_bytes')]) + ' |')
    probe = evidence['probe'] or {}
    lines += ['', f"Probe gradient cosine: {number(probe.get('gradient_cosine'))}; structural/epsilon gradient norm ratio: "
              f"{number(probe.get('ssim_to_epsilon_gradient_norm_ratio'))}. These assess gradients, not generated-image quality.", '',
              f"All 32 recorded identity/timestep/noise schedules match: {status(evidence['training']['schedule_comparison']['all_32_schedules_match'])}.", '',
              'Both arms use 385 trainable scorer parameters and a reset AdamW optimizer. Training remains at 512; '
              'evaluation runs at 1024. The JSON retains every saved update, loss, gradient/resource record, config, '
              'checkpoint hash, and source pin.', '', '## Supervision and recovery', '']
    for item in evidence['supervision']:
        receipt = item['record']
        lines.append(f"- `{cell(item['path'])}`: {cell(receipt.get('status'))}. " +
                     cell(receipt.get('reason') or receipt.get('stop_reason') or receipt.get('error') or receipt.get('scope', 'Saved supervision record.')))
    lines += ['', 'Original supervisor records and supplemental recovery records remain separate and retain their '
              'recorded statuses. A stale original status is not rewritten as successful supervision.', '',
              '## Visual review and limits', '']
    visual = gates.get('visual_review')
    if visual:
        lines += [f"- Person {ident}: {cell(visual.get('notes', {}).get(ident, 'Missing saved review'))}" for ident in IDS]
    else:
        lines += ['No completed visual review is saved. The exporter does not infer visual approval from scores.']
    lines += ['', 'All outcomes are retained, including regressions, failures, blocked records and unavailable metrics. '
              'Three development identities and one inference seed do not estimate unknown-client accuracy. Learned '
              'reference tokens do not establish anatomical correspondence or recovery of the true hidden face. '
              'A failed fixed gate stops this recipe; even a pass requires visual review before a later replication '
              'proposal. Photographs, generated images and contact sheets remain local and are not included in these exports.', '',
              '## Provenance', '',
              f"Exported at {evidence['exported_at_utc']}. Source evaluation: `{evidence['evaluation_directory']}`. "
              'The companion JSON records hashes of every consumed evidence file, the pinned report/scorer/source '
              'files and the continuation checkpoints. Saved source files were checked unchanged at export. '
              'The exporter does not reopen private images or rehash large model assets; their original recorded '
              'pins are retained as provenance, not represented as newly verified.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=EVAL)
    parser.add_argument('--training', type=Path, default=TRAIN)
    parser.add_argument('--probe', type=Path, default=PROBE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research')
    args = parser.parse_args()
    sources = Sources()
    output = args.evaluation.resolve()
    cfg = sources.read(output / 'config.json')
    summary = sources.read(output / 'summary.json')
    scoring = sources.read(output / 'scoring_receipt.json')
    paired = sources.read(output / 'paired_results.json')
    generation = sources.read(output / 'generation_receipt.json')
    neutral = sources.read(output / 'neutral_replay.json', required=False)
    visual = sources.read(output / 'visual_review.json', required=False)
    if cfg.get('conditions') != list(ARMS) or cfg.get('selected_identities') != IDS or cfg.get('training_updates') != 32:
        raise ValueError('This exporter requires the fixed five-arm, three-person SSIM32 experiment')
    if scoring.get('status') != 'complete':
        raise ValueError('Run the frozen scoring wrapper before exporting research results')
    for path, expected in [
        (output / 'config.json', summary['config_sha256']),
        (output / 'config.json', scoring['config_sha256']),
        (output / 'config.json', generation['config_sha256']),
        (output / 'config.json', paired['config_sha256']),
        (output / 'summary.json', scoring['summary_sha256']),
        (output / 'summary.json', paired['summary_sha256']),
        (output / 'generation_receipt.json', paired['generation_receipt_sha256']),
        (output / 'summary_frozen_scorer.json', summary['frozen_scorer_output_sha256']),
        (ROOT / 'scripts/score_masked_sdxl_v1.py', cfg['scorer_sha256']),
        (ROOT / 'scripts/score_masked_sdxl_ssim_continuation_eval_v1.py', cfg['scorer_wrapper_sha256']),
        (ROOT / 'scripts/report_masked_sdxl_ssim_continuation_eval_v1.py', cfg['report_wrapper_sha256']),
        (Path(cfg['protocol_path']), cfg['protocol_sha256']),
    ]:
        sources.require_hash(path, expected)
    if paired.get('visual_review') != visual:
        raise ValueError('Visual review changed; rerun the unchanged report wrapper before exporting')
    for pin in cfg.get('metric_source_pins', []):
        sources.require_hash(pin['path'], pin['sha256'])
    lookup = {(r['identity'], r['condition']): r for r in summary['rows']}
    expected = {(i, a) for i in IDS for a in ARMS}
    if len(lookup) != len(summary['rows']) or not set(lookup).issubset(expected):
        raise ValueError('Duplicate or unexpected evaluation rows')
    spec = importlib.util.spec_from_file_location('pinned_ssim_report', ROOT / 'scripts/report_masked_sdxl_ssim_continuation_eval_v1.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    if report.comparisons(lookup) != paired['comparisons']:
        raise ValueError('Saved gates differ from the unchanged pinned report implementation')
    rows = [normalize(lookup.get((i, a), dict(identity=i, condition=a, status='missing')))
            for i in IDS for a in ARMS]
    training = training_evidence(sources, args.training.resolve())
    probe = sources.read(args.probe.resolve(), required=False)
    supervision = supervisor_evidence(sources, args.training.resolve())
    for arm in ['epsilon', 'ssim']:
        arm_pin = cfg.get('training_arms', {}).get(arm, {})
        for name in ['config', 'receipt']:
            if arm_pin.get(name + '_sha256'):
                sources.require_hash(arm_pin[name + '_path'], arm_pin[name + '_sha256'])
    sources.record(__file__)
    result = dict(schema_version=1, exported_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Complete saved SSIM32 development outcomes; no accuracy percentage or generalization claim',
        evaluation_directory=str(output), ids=IDS, arms=ARMS, metric_definitions={
            'values':'Copied unchanged from saved scorer output; no image metric recomputation',
            'eyes':'Annotation eye components intersected with the supplied hole, not the full eye bbox',
            'regional_ssim':'Fully contained 11x11 Gaussian SSIM centres from unchanged evaluator',
            'regional_lpips':'Fixed bounding box plus 10px context, no resizing; not pixel-masked LPIPS',
            'averaging':'Unweighted arithmetic mean of available finite per-person values; coverage retained',
            'unavailable':'null means no saved finite value; original status retained (including perfect_infinite)'},
        rows=rows, means=means(rows), supplementary_paired_comparisons=all_comparisons(rows),
        official_gate_report=paired, scoring=scoring, generation=generation, neutral_replay=neutral,
        evaluation_config=cfg, summary_metadata={k: v for k, v in summary.items() if k != 'rows'},
        training=training, probe=probe, supervision=supervision,
        provenance=dict(verified_files=list(sources.pins.values()),
                        private_images_opened_by_exporter=False, image_artifacts_exported=False,
                        image_metrics_recalculated=False, large_model_assets_rehashed=False),
        replication_authorized=False, website_promotion=False)
    result = public(result)
    sources.unchanged()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / 'masked_ssim_continuation_v1_evidence.json'
    md_path = destination / 'MASKED_SSIM_CONTINUATION_RESULTS_V1.md'
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    md_path.write_text(render(result), encoding='utf-8')
    print(json.dumps(dict(evidence=str(json_path), report=str(md_path), records=len(rows),
                          numerical_gate=paired.get('numerical_gate'),
                          failed_gate_count=len(paired.get('failed_gates', [])),
                          image_artifacts_exported=False), indent=2))


if __name__ == '__main__':
    main()
