"""Evaluate one frozen 64-update compatibility pilot after an exact neutral replay.

Generation opens damaged inputs, supplied masks and four clean references only.
Original evaluation photographs are opened only by the separate CPU scorer and
report process. Existing results, weights, metrics and source files are read-only.
The default output is a fresh directory; failed or blocked rows are retained.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/masked_sdxl_compatibility_pilot_eval_v1'
TRAIN = ROOT / 'outputs/masked_sdxl_compatibility_pilot_v1/training'
PROTOCOL = ROOT / 'research/MASKED_SDXL_COMPATIBILITY_PILOT_PREFLIGHT_V1.md'
IDS = ['263', '6681', '10149']
TRAIN_IDS = ['8570', '6792', '4410', '2250', '3930', '8173', '2083', '7506',
             '4625', '38', '4164', '674', '8660', '476', '1281', '8748']
HOLDOUT_IDS = ['263', '6681', '7400', '10149']
ARMS = ['control3', 'local', 'trained']
EXPECTED = 9
TRAIN_UPDATES = 64
TRAIN_RESOLUTION = 512
EVAL_RESOLUTION = 1024
EXPORT_RESOLUTION = 512
PARAMETERS = 385


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(value, message):
    if not value:
        raise ValueError(message)


def check_pin(record):
    path = Path(record['path'])
    require(path.is_file() and sha(path) == record['sha256'], f'Pinned file mismatch: {path}')


def preflight(training, protocol):
    """Validate protocol, completed training and all generation inputs before CUDA."""
    require(protocol.is_file(), 'Save the frozen pilot protocol before evaluation')
    cfg = read(training / 'config.json')
    receipt = read(training / 'receipt.json')
    require(receipt['status'] in ['complete', 'passed'], 'Training receipt does not record completed training')
    require(receipt['completed_updates'] == TRAIN_UPDATES, 'Checkpoint must come from exactly 64 updates')
    require(receipt['finite_gradient_updates'] == TRAIN_UPDATES, 'Every one of the 64 updates must have finite gradients')
    require(receipt['frozen_weights_have_gradients'] is False, 'Frozen backbone weights acquired gradients')
    require(math.isfinite(receipt['neutral_max_absolute_error']) and receipt['neutral_max_absolute_error'] <= 1e-3,
            'Training real-input neutrality did not pass')
    require(receipt['parameter_update'] is True, 'Training did not record a scorer parameter update')
    require(receipt['modified_processors'] == 70, 'Training processor count changed')
    require(receipt['backend_flags_restored'] is True, 'Training SDPA policy was not restored')
    require(cfg['updates'] == TRAIN_UPDATES, 'Training config update count changed')
    require(cfg['training_resolution'] == TRAIN_RESOLUTION, 'Training must use the frozen 512 resolution')
    require(cfg['trainable_parameters'] == receipt['trainable_parameters'] == PARAMETERS, 'Expected 385 trained parameters')
    require(list(map(str, cfg['training_ids'])) == TRAIN_IDS, 'Training identity order differs from the frozen 16-person manifest')
    require(not set(map(str, cfg['training_ids'])) & set(HOLDOUT_IDS), 'Training/adapter-holdout identity overlap')
    require(cfg.get('evaluation_ids', IDS) == IDS, 'Evaluation identities changed')
    require(cfg['protocol_sha256'] == receipt['protocol_sha256'] == sha(protocol), 'Frozen protocol hash changed')
    require(Path(cfg['protocol_path']).resolve() == protocol.resolve(), 'Protocol path differs from training config')
    require(receipt['config_sha256'] == sha(training / 'config.json'), 'Training config hash changed')
    require(cfg['source_pins'], 'Training must retain explicit source pins')
    for pin in cfg['source_pins']:
        check_pin(pin)
    required_sources = [ROOT / 'scripts' / name for name in [
        'reference_inpaint.py', 'score_masked_sdxl_v1.py', 'run_focused_eye_appearance_v1.py',
        'run_focused_eye_guidance_v1.py', 'local_appearance_processor_v1.py',
        'sdxl_token_compatibility_probe_v1.py', Path(__file__).name,
        'score_masked_sdxl_compatibility_pilot_eval_v1.py', 'report_masked_sdxl_compatibility_pilot_eval_v1.py']]
    required_sources.append(ROOT / 'research/osor_downloads.json')
    metric_sources = [ROOT / 'scripts' / name for name in
                      ['region_metrics_v1.py', 'extended_evaluation_metrics.py', 'score_target_compatibility_v1.py']]
    metric_sources += [ROOT / 'research' / name for name in
                       ['extended_evaluation_provenance_v1.json', 'identity_evaluator_provenance_v1.json']]
    required_sources += metric_sources
    pinned_sources = {Path(pin['path']).resolve() for pin in cfg['source_pins']}
    require(all(path.resolve() in pinned_sources for path in required_sources), 'Training source pins omit a consumed evaluation source or model manifest')
    if 'source_pins' in receipt:
        require(receipt['source_pins'] == cfg['source_pins'], 'Receipt/source pins differ')
    checkpoints = receipt.get('checkpoints', cfg.get('checkpoints'))
    require(isinstance(checkpoints, dict), 'Training checkpoint hash pins are missing')
    receipt = dict(receipt, checkpoints=checkpoints)
    for key, filename in [('initial', 'adapter_initial.pt'), ('last', 'adapter_last.pt')]:
        pin = receipt['checkpoints'][key]
        require(Path(pin['path']).resolve() == (training / filename).resolve(), 'Unexpected checkpoint path')
        check_pin(pin)

    baseline_path = ROOT / 'outputs/focused_eye_factorial_v1/config.json'
    baseline = read(baseline_path)
    local_path = ROOT / 'outputs/focused_eye_local_appearance_v1/config.json'
    local = read(local_path)
    for name, field in [('reference_inpaint.py', 'source_sha256'),
                        ('score_masked_sdxl_v1.py', 'scorer_sha256'),
                        ('run_focused_eye_appearance_v1.py', 'appearance_wrapper_sha256'),
                        ('run_focused_eye_guidance_v1.py', 'guidance_wrapper_sha256')]:
        require(sha(ROOT / 'scripts' / name) == baseline[field] == local[field], f'Frozen source changed: {name}')
    require(sha(ROOT / 'scripts/local_appearance_processor_v1.py') == local['helper_sha256'], 'Preserved routing helper changed')
    for key, expected in dict(steps=30, scale=.8, strength=1., guidance=3., resolution=1024,
                              blend='poisson', memory='model', appearance_scale=.2).items():
        require(baseline[key] == local[key] == expected, f'Frozen inference setting changed: {key}')
    require(baseline['references'] == local['references'], 'Preserved reference bank changed')
    require(baseline['appearance_features'] == local['appearance_features'], 'Preserved appearance bank changed')
    cases_path = ROOT / 'outputs/masked_task_v1/cases.json'
    require(sha(cases_path) == baseline['cases_sha256'] == local['cases_sha256'], 'Case manifest changed')
    cases = {c['identity']: c for c in read(cases_path) if c['location'] == 'eyes' and c['identity'] in IDS}
    require(set(cases) == set(IDS), 'Missing frozen eye case')
    for ident in IDS:
        c = cases[ident]
        # The evaluation-only original is never opened or hashed in this process.
        for key in ['input', 'mask']:
            check_pin(dict(path=c[key], sha256=c[key + '_sha256']))
        references = baseline['references'][ident + '/clean']
        require(len(references) == 4, 'Exactly four clean references are required')
        require(len({r['sha256'] for r in references}) == 4, 'Duplicate references')
        for pin in references:
            check_pin(pin)
            require(pin['sha256'] != c['target_sha256'], 'Evaluation target appears in reference bank')
        check_pin(baseline['appearance_features'][ident])
    for pin in baseline['appearance_assets']['files']:
        check_pin(pin)
    cache = Path(read(ROOT / 'configs/local.json')['cache'])
    # Keep the historical from_pretrained call: it consumes default aliases.
    # Each alias must equal its officially pinned fp16 download, rather than
    # silently changing variant selection and invalidating the neutral replay.
    model_pins = []
    for record in read(ROOT / 'research/osor_downloads.json'):
        if record['repository'] != 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1':
            continue
        require(record['revision'] == '115134f363124c53c7d878647567d04daf26e41e', 'SDXL model revision changed')
        name = record['filename'].replace('.fp16.safetensors', '.safetensors')
        pin = dict(path=str(cache / 'osor_models/sdxl' / name), sha256=record['sha256'],
                   official_filename=record['filename'], revision=record['revision'])
        check_pin(pin)
        model_pins.append(pin)
    require(len(model_pins) == 18, 'Pinned SDXL consumed-file manifest changed')
    baseline = dict(baseline, consumed_sdxl_file_pins=model_pins,
                    metric_source_pins=[dict(path=str(path), sha256=sha(path)) for path in metric_sources])
    require(sha(cache / 'reference_models/ip-adapter-faceid-portrait_sdxl.bin') == baseline['adapter_sha256'], 'FaceID weight pin changed')
    controls = []
    for arm, folder, source_arm in [('control3', 'focused_eye_guidance_v1', 'clean'),
                                     ('local', 'focused_eye_local_appearance_v1', 'local')]:
        origin = ROOT / 'outputs' / folder
        source = read(origin / 'summary.json')
        lookup = {r['identity']: r for r in source['rows'] if r['location'] == 'eyes'
                  and r['identity'] in IDS and r['condition'] == source_arm and r['seed'] == 17}
        require(set(lookup) == set(IDS), f'Missing preserved {arm} controls')
        for ident in IDS:
            row = lookup[ident]
            require(row['status'] == 'complete', f'Preserved {arm}/{ident} is incomplete')
            output = origin / row['file']
            raw = output.with_name(output.stem + '_raw.png')
            require(sha(output) == row['sha256'] and sha(raw) == row['raw_sha256'], 'Preserved control image hash changed')
            controls.append(dict(arm=arm, identity=ident, row=row, output=output, raw=raw,
                                 summary_path=origin / 'summary.json', summary_sha256=sha(origin / 'summary.json')))
    return cfg, receipt, baseline, local, cases, controls, cache


def load_state(torch, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict), 'Checkpoint must contain the scorer state dictionary')
    expected = {'net.0.weight': (32, 10), 'net.0.bias': (32,), 'net.2.weight': (1, 32), 'net.2.bias': (1,)}
    require(set(state) == set(expected), 'Unexpected checkpoint tensors')
    for key, shape in expected.items():
        require(tuple(state[key].shape) == shape and state[key].dtype == torch.float32,
                f'Expected float32 scorer tensor: {key}')
        require(bool(torch.isfinite(state[key]).all()), f'Nonfinite scorer tensor: {key}')
    return state


def score_and_report(cache, output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    python = cache / 'evaluation_env_v1/Scripts/python.exe'
    for name in ['score_masked_sdxl_compatibility_pilot_eval_v1.py',
                 'report_masked_sdxl_compatibility_pilot_eval_v1.py']:
        subprocess.run([str(python), str(ROOT / 'scripts' / name), '--output', str(output)],
                       env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=TRAIN)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--preflight-only', action='store_true', help='Validate CPU-side pins; do not create results or use CUDA')
    parser.add_argument('--skip-scoring', action='store_true', help='Run the two CPU wrappers separately after generation')
    args = parser.parse_args()
    training, output = args.training.resolve(), args.output.resolve()
    if not args.preflight_only:
        require(not output.exists(), 'Refusing to overwrite an existing evaluation output directory')
    tcfg, training_receipt, cfg, local, cases, controls, cache = preflight(training, args.protocol.resolve())
    # These imports do not initialize CUDA. All asset and training checks precede them.
    import numpy as np
    from PIL import Image
    from reference_inpaint import reconstruct, torch, StableDiffusionXLInpaintPipeline, DDIMScheduler
    import run_focused_eye_appearance_v1 as appearance
    from run_focused_eye_guidance_v1 import GuidanceProxy
    from run_focused_eye_freeu_v1 import idle_gpu
    from sdxl_token_compatibility_probe_v1 import CompatibilityScorer, TokenCompatibilityProcessor, install, FEATURES
    import diffusers.models.attention_processor as processor_source
    require(sha(processor_source.__file__) == local['attention_implementation_sha256'], 'Diffusers attention source changed; use the preserved reference environment')
    initial = load_state(torch, training / 'adapter_initial.pt')
    trained = load_state(torch, training / 'adapter_last.pt')
    require(bool((initial['net.2.weight'] == 0).all()) and bool((initial['net.2.bias'] == 0).all()), 'Initial scorer must have a zero last layer')
    require(any(not torch.equal(initial[k], trained[k]) for k in initial), 'Trained scorer is unchanged')
    scorer = CompatibilityScorer().float().cpu().eval()
    scorer.load_state_dict(initial, strict=True)
    require(sum(p.numel() for p in scorer.parameters()) == PARAMETERS, 'Scorer parameter count changed')
    masks, banks = {}, {}
    for ident in IDS:
        with Image.open(cases[ident]['mask']) as image:
            masks[ident] = np.asarray(image.convert('L')) >= 128
        require(masks[ident].shape == (EXPORT_RESOLUTION, EXPORT_RESOLUTION), 'Supplied mask must remain 512 square')
        require(bool(masks[ident].any()) and not bool(masks[ident].all()), 'Invalid supplied hole mask')
        banks[ident] = torch.load(cfg['appearance_features'][ident]['path'], map_location='cpu', weights_only=True)
        require(tuple(banks[ident].shape) == (2, 4, 257, 1280) and bool(torch.isfinite(banks[ident]).all()), 'Appearance feature bank changed')
    if args.preflight_only:
        print(json.dumps(dict(status='passed', cuda_used=False, training_updates=TRAIN_UPDATES,
                             training_resolution=TRAIN_RESOLUTION, evaluation_resolution=EVAL_RESOLUTION,
                             export_resolution=EXPORT_RESOLUTION, parameters=PARAMETERS,
                             evaluation_ids=IDS, protocol_sha256=sha(args.protocol), training_receipt_sha256=sha(training / 'receipt.json')), indent=2))
        return

    idle_gpu()
    output.mkdir(exist_ok=False)
    cfg = dict(cfg, conditions=ARMS, expected=EXPECTED, seeds=[17], selected_identities=IDS, locations=['eyes'],
               training=True, training_updates=TRAIN_UPDATES, training_resolution=TRAIN_RESOLUTION,
               evaluation_resolution=EVAL_RESOLUTION, export_resolution=EXPORT_RESOLUTION, trainable_parameters=PARAMETERS,
               training_ids=tcfg['training_ids'], protocol_path=str(args.protocol.resolve()), protocol_sha256=sha(args.protocol),
               runner_sha256=sha(__file__), scorer_wrapper_sha256=sha(ROOT / 'scripts/score_masked_sdxl_compatibility_pilot_eval_v1.py'),
               report_wrapper_sha256=sha(ROOT / 'scripts/report_masked_sdxl_compatibility_pilot_eval_v1.py'),
               compatibility_source_sha256=sha(ROOT / 'scripts/sdxl_token_compatibility_probe_v1.py'),
               training_config_path=str(training / 'config.json'), training_config_sha256=sha(training / 'config.json'),
               training_receipt_path=str(training / 'receipt.json'), training_receipt_sha256=sha(training / 'receipt.json'),
               checkpoints=training_receipt['checkpoints'], feature_order=FEATURES,
               single_change='385-parameter appearance attention compatibility bias trained for 64 fixed updates; frozen local routing and inference recipe',
               neutral_control='Initial checkpoint loaded, scorer disabled; same new processor, actual-timestep hook and supplied mask',
               neutral_expected='SHA256 equality of raw and Poisson-finished output against preserved local263 seed17',
               bias_bound=.5, query_chunk=64, mask_downsample='nearest binary to square query grid; no dilation or feather',
               precision='preserved fp16 inference weights and default SDPA backends; float32 scorer; no training autocast/backend context',
               generation_reads_evaluation_original=False, frozen_at_utc=datetime.now(timezone.utc).isoformat())
    cfg['inference_versions'] = {name: version(name) for name in ['torch', 'diffusers', 'transformers', 'safetensors', 'Pillow']}
    cfg['inference_executable'] = sys.executable
    cfg['preserved_control_versions'] = local.get('environment')
    cfg['training_versions'] = training_receipt.get('versions', tcfg.get('versions'))
    save(output / 'config.json', cfg)
    rows = []
    for control in controls:
        dest = output / f"{control['identity']}_eyes_{control['arm']}_17.png"
        shutil.copy2(control['output'], dest)
        shutil.copy2(control['raw'], dest.with_name(dest.stem + '_raw.png'))
        row = dict(control['row'], condition=control['arm'], reference_condition='clean', file=dest.name,
                   reused=True, source=str(control['output']), source_summary=str(control['summary_path']),
                   source_summary_sha256=control['summary_sha256'])
        rows.append(row)
        score = output / 'scores' / (dest.name + '.json')
        score.parent.mkdir(exist_ok=True)
        save(score, row)
    save(output / 'rows.json', rows)
    receipt = dict(status='running', expected=EXPECTED, reused_controls=6, new_candidates=3, neutral_calls=1,
                   protocol_sha256=cfg['protocol_sha256'], config_sha256=sha(output / 'config.json'),
                   training_receipt_sha256=cfg['training_receipt_sha256'], candidate_attempts=0,
                   neutral_replay_passed=False, evaluation_original_read=False)
    save(output / 'generation_receipt.json', receipt)
    pipe = runtime = proxy = None
    hook_handle = None
    start = time.monotonic()
    try:
        torch.set_num_threads(4)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(cache / 'osor_models/sdxl'),
            torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter([str(cache / 'reference_models'), str(appearance.ASSETS / 'sdxl_models')],
            subfolder=['', ''], weight_name=['ip-adapter-faceid-portrait_sdxl.bin', 'ip-adapter-plus-face_sdxl_vit-h.safetensors'],
            image_encoder_folder=None, local_files_only=True)
        projections = pipe.unet.encoder_hid_proj.image_projection_layers
        actual_tokens = [int(projections[0].num_tokens), int(projections[1].latents.shape[1])]
        require(actual_tokens == [16, 16], 'Original adapter projection token layout changed')
        scorer.enabled = False
        cfg['wrapped_processors'] = install(pipe.unet, scorer, masks['263'], chunk_size=64)
        require(cfg['wrapped_processors'] == local['wrapped_processors'], 'Processor count differs from local control')
        timestep_log = []

        def actual_timestep(module, positional, keyword):
            timestep = keyword.get('timestep', positional[1] if len(positional) > 1 else None)
            require(timestep is not None, 'UNet timestep was not supplied')
            values = torch.as_tensor(timestep).detach().float().cpu().reshape(-1)
            require(bool(torch.isfinite(values).all()) and bool((values == values[0]).all()), 'Invalid or mixed UNet timesteps')
            value = float(values[0])
            require(0 <= value <= 999, 'Actual UNet timestep is outside the frozen schedule')
            scorer.timestep_fraction = value / 999.
            timestep_log.append(value)

        hook_handle = pipe.unet.register_forward_pre_hook(actual_timestep, with_kwargs=True)
        # Install the scorer before offload hooks, preserving its float32 weights.
        for module in [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
            module.requires_grad_(False)
        pipe.enable_model_cpu_offload()
        require(all(p.dtype == torch.float32 for p in scorer.parameters()), 'Scorer precision changed')
        cfg['scheduler'] = dict(pipe.scheduler.config)
        cfg['actual_projection_tokens_per_reference'] = actual_tokens
        cfg['attention_implementation_path'] = processor_source.__file__
        cfg['attention_implementation_sha256'] = sha(processor_source.__file__)
        cfg['default_sdpa_backends'] = {k: getattr(torch.backends.cuda, k + '_sdp_enabled')() for k in ['flash', 'mem_efficient', 'math', 'cudnn']}
        save(output / 'config.json', cfg)
        appearance.OUT = output
        proxy = appearance.AppearanceProxy(GuidanceProxy(pipe, 3.))
        proxy.appearance_scale = .2
        runtime = {'pipeline': proxy, 'memory': 'model'}

        def run(ident, destination):
            mask = masks[ident]
            scorer.visible_fraction = float((~mask).mean())
            for processor in pipe.unet.attn_processors.values():
                if isinstance(processor, TokenCompatibilityProcessor):
                    processor.target_mask.copy_(torch.as_tensor(mask, device=processor.target_mask.device, dtype=torch.bool))
            proxy.appearance = banks[ident]
            timestep_log.clear()
            c = cases[ident]
            result = reconstruct(Path(c['input']), Path(c['mask']),
                [Path(r['path']) for r in cfg['references'][ident + '/clean']], destination,
                seed=17, steps=30, scale=.8, strength=1., memory='model', blend='poisson',
                runtime=runtime, model_resolution=EVAL_RESOLUTION)
            require(len(timestep_log) == 30, 'Actual-timestep hook did not observe all 30 UNet forwards')
            require(all(p.dtype == torch.float32 for p in scorer.parameters()), 'Scorer precision changed during inference')
            return result

        replay = output / 'replay/263_initial_disabled.png'
        run('263', replay)
        reference = output / '263_eyes_local_17.png'
        checks = {}
        for suffix in ['.png', '_raw.png']:
            new = replay.with_name(replay.stem + suffix)
            old = reference.with_name(reference.stem + suffix)
            checks[suffix] = dict(byte_exact=sha(new) == sha(old),
                pixel_exact=np.array_equal(np.asarray(Image.open(new)), np.asarray(Image.open(old))),
                new_sha256=sha(new), control_sha256=sha(old))
        neutral = dict(identity='263', seed=17, initial_checkpoint_sha256=sha(training / 'adapter_initial.pt'),
                       scorer_enabled=False, timesteps=list(timestep_log), visible_fraction=scorer.visible_fraction,
                       checks=checks, passed=all(x['byte_exact'] and x['pixel_exact'] for x in checks.values()))
        save(output / 'neutral_replay.json', neutral)
        receipt['neutral_replay_passed'] = neutral['passed']
        save(output / 'generation_receipt.json', receipt)
        require(neutral['passed'], 'Neutral raw/finished replay was not byte-exact; candidates are blocked')
        scorer.load_state_dict(trained, strict=True)
        scorer.enabled = True
        for ident in IDS:
            idle_gpu()
            destination = output / f'{ident}_eyes_trained_17.png'
            row = dict(identity=ident, location='eyes', condition='trained', reference_condition='clean', seed=17,
                       file=destination.name, reused=False, attempted_generation=True, guidance=3., appearance_scale=.2,
                       checkpoint_sha256=sha(training / 'adapter_last.pt'))
            case_start = time.monotonic()
            receipt['candidate_attempts'] += 1
            try:
                meta = run(ident, destination)
                row.update(status='complete', sha256=sha(destination), raw_sha256=meta['raw_output_sha256'],
                           peak_allocated_bytes=meta['peak_allocated_bytes'], timesteps=list(timestep_log),
                           visible_target_fraction=scorer.visible_fraction)
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}', timesteps=list(timestep_log))
            row['seconds'] = time.monotonic() - case_start
            rows.append(row)
            save(output / 'rows.json', rows)
            save(output / 'generation_receipt.json', receipt)
            print('TRAINED PILOT', ident, row['status'], flush=True)
        receipt['status'] = 'complete' if all(r['status'] == 'complete' for r in rows) else 'failed'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        existing = {r['identity'] for r in rows if r['condition'] == 'trained'}
        for ident in IDS:
            if ident not in existing:
                rows.append(dict(identity=ident, location='eyes', condition='trained', reference_condition='clean', seed=17,
                    file=f'{ident}_eyes_trained_17.png', reused=False, attempted_generation=False,
                    status='blocked', error=receipt['error']))
        save(output / 'rows.json', rows)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        receipt.update(elapsed_seconds=time.monotonic() - start, config_sha256=sha(output / 'config.json'))
        save(output / 'generation_receipt.json', receipt)
        del runtime, proxy, pipe, scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not args.skip_scoring:
        score_and_report(cache, output)
    require(receipt['status'] == 'complete', 'Evaluation generation failed; all completed, failed or blocked outcomes are retained')


if __name__ == '__main__':
    main()
