"""Isolated decoder reweighting on the preserved three-person reference protocol.

Inference opens damaged inputs, masks and references only. Clean originals belong
to the separate frozen evaluator. Existing controls/checkpoints are never edited.
"""
import argparse
import gc
import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from reference_inpaint import (
    ROOT, CACHE, sha, reconstruct, torch,
    StableDiffusionXLInpaintPipeline, DDIMScheduler,
)
from run_focused_eye_guidance_v1 import GuidanceProxy

OUT = ROOT / 'outputs/focused_eye_freeu_v1'
IDS = ['263', '6681', '10149']
FREEU = dict(b1=1.3, b2=1.4, s1=.9, s2=.2)
PROTOCOL = ROOT / 'research/FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def idle_gpu():
    # Do not start a second GPU workload while a studio reconstruction is active.
    for port in [8765, 8766]:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/session', timeout=2) as response:
                session = json.load(response)
            if session.get('busy'):
                raise RuntimeError(f'Studio on port {port} is busy; wait before research inference.')
        except (urllib.error.URLError, TimeoutError):
            pass


def read_cases(cfg):
    path = ROOT / 'outputs/masked_task_v1/cases.json'
    assert sha(path) == cfg['cases_sha256']
    cases = {c['identity']: c for c in json.loads(path.read_text()) if c['location'] == 'eyes'}
    for ident in IDS:
        c = cases[ident]
        for key in ['input', 'mask']:
            assert sha(c[key]) == c[key + '_sha256']
        refs = cfg['references'][ident + '/clean']
        assert len(refs) == 4
        for r in refs:
            assert sha(r['path']) == r['sha256']
            assert r['sha256'] != c['target_sha256'], 'Target cannot be a reference'
    return cases


def copy_controls(cfg, seeds, rows):
    source_dir = ROOT / 'outputs/focused_eye_guidance_replication_v2'
    original = json.loads((source_dir / 'summary.json').read_text())
    assert original['all_completed']
    lookup = {(r['identity'], r['seed'], r['condition']): r for r in original['rows']}
    for ident in IDS:
        for seed in seeds:
            for old_arm, arm in [('control', 'control5'), ('candidate', 'control3')]:
                r = lookup[ident, seed, old_arm]
                source = source_dir / r['file']
                raw = source.with_name(source.stem + '_raw.png')
                assert sha(source) == r['sha256'] and sha(raw) == r['raw_sha256']
                dest = OUT / f'{ident}_eyes_{arm}_{seed}.png'
                if dest.exists():
                    assert sha(dest) == r['sha256']
                    assert sha(dest.with_name(dest.stem + '_raw.png')) == r['raw_sha256']
                else:
                    shutil.copy2(source, dest)
                    shutil.copy2(raw, dest.with_name(dest.stem + '_raw.png'))
                copied = dict(r, condition=arm, file=dest.name, reused=True, source=str(source),
                              source_summary_sha256=sha(source_dir / 'summary.json'))
                rows.append(copied)
                # Keep measurements explicit; scoring is not claimed to be fresh.
                score = OUT / 'scores' / (dest.name + '.json')
                score.parent.mkdir(exist_ok=True)
                save(score, copied)
    save(OUT / 'rows.json', rows)


def main(replicate=False):
    idle_gpu()
    if replicate:
        cfg = json.loads((OUT / 'config.json').read_text())
        decision = json.loads((OUT / 'screen_decision.json').read_text())
        assert decision['replicate'] is True, 'Screen and visual gate must pass first'
        assert sha(PROTOCOL) == cfg['protocol_sha256']
        assert sha(__file__) == cfg['runner_sha256']
        rows = json.loads((OUT / 'rows.json').read_text())
        assert len(rows) == 9 and all(r['status'] == 'complete' for r in rows)
        seeds = [29, 43]
        cfg.update(expected=27, seeds=[17, 29, 43], stage='three_seed_replication')
    else:
        OUT.mkdir(exist_ok=False)
        cfg = json.loads((ROOT / 'outputs/focused_eye_guidance_replication_v2/config.json').read_text())
        assert sha(ROOT / 'scripts/reference_inpaint.py') == cfg['generator_sha256']
        assert sha(ROOT / 'scripts/score_masked_sdxl_v1.py') == cfg['scorer_sha256']
        cfg.update(expected=9, conditions=['control5', 'control3', 'freeu'], selected_identities=IDS,
                   seeds=[17], guidance=3., freeu=FREEU, stage='three_person_screen', training=False,
                   single_change='Installed Diffusers FreeU decoder reweighting on unchanged CFG3',
                   runner_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                   generator_sha256=sha(ROOT / 'scripts/reference_inpaint.py'),
                   scorer_sha256=sha(ROOT / 'scripts/score_masked_sdxl_v1.py'),
                   frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                   trainable_parameters=0, target_pixels_used_for_generation=False)
        cfg['environment'] = {name: version(name) for name in ['torch', 'diffusers', 'transformers', 'insightface']}
        import diffusers.utils.torch_utils as torch_utils
        import diffusers.models.unets.unet_2d_condition as unet_source
        import diffusers.models.unets.unet_2d_blocks as block_source
        cfg['implementation'] = {str(Path(m.__file__)): sha(m.__file__) for m in [torch_utils, unet_source, block_source]}
        rows = []
        seeds = [17]
    cases = read_cases(cfg)
    for path, digest in cfg['implementation'].items():
        assert sha(path) == digest
    assert sha(ROOT / 'scripts/reference_inpaint.py') == cfg['generator_sha256']
    assert sha(ROOT / 'scripts/score_masked_sdxl_v1.py') == cfg['scorer_sha256']
    assert sha(CACHE / 'reference_models/ip-adapter-faceid-portrait_sdxl.bin') == cfg['adapter_sha256']
    copy_controls(cfg, seeds, rows)
    save(OUT / 'config.json', cfg)
    torch.set_num_threads(4)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(CACHE / 'osor_models/sdxl'), torch_dtype=torch.float16,
        local_files_only=True, add_watermarker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(str(CACHE / 'reference_models'), subfolder='',
                         weight_name='ip-adapter-faceid-portrait_sdxl.bin',
                         image_encoder_folder=None, local_files_only=True)
    pipe.enable_model_cpu_offload()
    cfg['scheduler'] = dict(pipe.scheduler.config)
    for module in [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
        module.requires_grad_(False)
    cfg['trainable_parameters'] = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    assert cfg['trainable_parameters'] == 0
    save(OUT / 'config.json', cfg)
    proxy = GuidanceProxy(pipe, 3.)
    runtime = dict(pipeline=proxy, memory='model')

    def run(ident, seed, output):
        c = cases[ident]
        return reconstruct(Path(c['input']), Path(c['mask']),
            [Path(r['path']) for r in cfg['references'][ident + '/clean']], output,
            seed=seed, steps=30, scale=.8, strength=1., memory='model', blend='poisson',
            runtime=runtime, model_resolution=1024)

    if not replicate:
        # Enabling and disabling must leave the previous raw/final control intact.
        pipe.enable_freeu(**FREEU)
        pipe.disable_freeu()
        replay = OUT / 'replay/263_eyes_control3_17.png'
        run('263', 17, replay)
        original = OUT / '263_eyes_control3_17.png'
        checks = {}
        for suffix in ['.png', '_raw.png']:
            new = replay.with_name(replay.stem + suffix)
            old = original.with_name(original.stem + suffix)
            checks[suffix] = dict(pixel_exact=np.array_equal(np.asarray(Image.open(new)), np.asarray(Image.open(old))),
                                  new_sha256=sha(new), control_sha256=sha(old))
        save(OUT / 'replay.json', checks)
        assert all(v['pixel_exact'] for v in checks.values()), 'Disabled FreeU replay mismatch'
    pipe.enable_freeu(**FREEU)
    cfg['upblock_settings'] = [dict(resolution_idx=b.resolution_idx, applies=b.resolution_idx in [0, 1],
        **{k: getattr(b, k) for k in FREEU}) for b in pipe.unet.up_blocks]
    save(OUT / 'config.json', cfg)
    for seed in seeds:
        for ident in IDS:
            idle_gpu()
            output = OUT / f'{ident}_eyes_freeu_{seed}.png'
            start = time.monotonic()
            row = dict(identity=ident, location='eyes', condition='freeu', seed=seed,
                       reference_condition='clean', file=output.name, guidance=3., freeu=FREEU, reused=False)
            try:
                meta = run(ident, seed, output)
                row.update(status='complete', sha256=sha(output), raw_sha256=meta['raw_output_sha256'],
                           seconds=time.monotonic() - start, peak_allocated_bytes=meta['peak_allocated_bytes'])
            except Exception as exc:
                row.update(status='failed', error=repr(exc), seconds=time.monotonic() - start)
            rows.append(row)
            save(OUT / 'rows.json', rows)
            print('FREEU', ident, seed, row['status'], flush=True)
    pipe.disable_freeu()
    del runtime, proxy, pipe
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([str(CACHE / 'evaluation_env_v1/Scripts/python.exe'),
                    str(ROOT / 'scripts/score_focused_eye_freeu_v1.py')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate', action='store_true')
    main(parser.parse_args().replicate)
