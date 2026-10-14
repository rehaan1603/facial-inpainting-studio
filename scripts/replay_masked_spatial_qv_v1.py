"""Pre-training enabled-zero replay. Never opens the evaluation original."""
import gc
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from spatial_qv_common_v1 import ROOT, PROTOCOL, START_SHA, read, save, sha, pin, verify, require, verify_control, verify_cpu_tests
import run_masked_sdxl_compatibility_pilot_eval_v2 as old

OUT = ROOT / 'outputs/masked_spatial_qv_replay_v1'


def main():
    require(not OUT.exists(), 'Refuse replay overwrite')
    probe_path = ROOT / 'outputs/masked_spatial_qv_probe_v1/receipt.json'
    probe = read(probe_path)
    require(probe['status'] == 'passed' and probe['enabled_zero_prediction_exact']
            and probe['neutral_scorer_gradient']['passed'], 'Full GPU probe must pass first')
    for record in probe['source_pins']:
        verify(record)
    verify_control(); verify_cpu_tests()
    _, prior, cfg, local, cases, _, cache = old.preflight(old.TRAIN, old.PROTOCOL)
    require(prior['checkpoints']['last']['sha256'] == START_SHA, 'Starting scorer changed')
    control_dir = ROOT / 'outputs/masked_sdxl_compatibility_pilot_eval_v2'
    control = next(r for r in read(control_dir / 'summary.json')['rows']
                   if r['identity'] == '263' and r['condition'] == 'trained')
    original = control_dir / control['file']
    verify(dict(path=str(original), sha256=control['sha256']))
    verify(dict(path=str(original.with_name(original.stem + '_raw.png')), sha256=control['raw_sha256']))
    import numpy as np
    from PIL import Image
    from reference_inpaint import reconstruct, torch, StableDiffusionXLInpaintPipeline, DDIMScheduler
    import run_focused_eye_appearance_v1 as appearance
    from run_focused_eye_guidance_v1 import GuidanceProxy
    from run_focused_eye_freeu_v1 import idle_gpu
    from sdxl_token_compatibility_probe_v1 import CompatibilityScorer, install
    from masked_spatial_qv_v1 import SpatialQVAdapter, install as install_spatial
    import diffusers.models.attention_processor as attention_source
    require(sha(attention_source.__file__) == local['attention_implementation_sha256'], 'Attention changed')
    with Image.open(cases['263']['mask']) as im:
        mask = np.asarray(im.convert('L')) >= 128
    bank = torch.load(cfg['appearance_features']['263']['path'], map_location='cpu', weights_only=True)
    idle_gpu(); OUT.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', identity='263', seed=17, evaluation_original_read=False,
                   quality_scores_created=0, checkpoints_created=0, probe_receipt=pin(probe_path),
                   source_pins=[pin(__file__), pin(PROTOCOL), pin(ROOT/'scripts/masked_spatial_qv_v1.py'),
                                pin(ROOT/'scripts/spatial_qv_common_v1.py')],
                   starting_checkpoint=prior['checkpoints']['last'], inference_resolution=1024, export_resolution=512)
    save(OUT/'receipt.json', receipt)
    pipe = proxy = runtime = hook = None
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(cache/'osor_models/sdxl'),
                    torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter([str(cache/'reference_models'), str(appearance.ASSETS/'sdxl_models')],
            subfolder=['', ''], weight_name=['ip-adapter-faceid-portrait_sdxl.bin',
            'ip-adapter-plus-face_sdxl_vit-h.safetensors'], image_encoder_folder=None, local_files_only=True)
        scorer = CompatibilityScorer().float().cpu().eval()
        scorer.load_state_dict(old.load_state(torch, prior['checkpoints']['last']['path']), strict=True)
        scorer.enabled = True
        require(install(pipe.unet, scorer, mask, 64) == 70, 'Scorer processor count changed')
        # Freeze native maps before spatial installation, retaining new trainable
        # parameters until the install validation has completed.
        for module in [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
            module.requires_grad_(False)
        adapter = SpatialQVAdapter().float().cpu().eval()
        install_spatial(pipe.unet, adapter, mask)
        pipe.unet.requires_grad_(False)
        times = []
        def capture(module, positional, keyword):
            t = keyword.get('timestep', positional[1] if len(positional)>1 else None)
            values = torch.as_tensor(t).detach().float().cpu().reshape(-1)
            require(torch.isfinite(values).all() and (values == values[0]).all(), 'Invalid timestep')
            scorer.timestep_fraction = float(values[0])/999.
            times.append(float(values[0]))
        hook = pipe.unet.register_forward_pre_hook(capture, with_kwargs=True)
        scorer.visible_fraction = float((~mask).mean())
        pipe.enable_model_cpu_offload()
        appearance.OUT = OUT
        proxy = appearance.AppearanceProxy(GuidanceProxy(pipe, 3.))
        proxy.appearance_scale = .2; proxy.appearance = bank
        runtime = {'pipeline': proxy, 'memory': 'model'}
        result = OUT/'263_enabled_zero.png'
        c = cases['263']
        reconstruct(Path(c['input']), Path(c['mask']), [Path(r['path']) for r in cfg['references']['263/clean']],
            result, seed=17, steps=30, scale=.8, strength=1., memory='model', blend='poisson',
            runtime=runtime, model_resolution=1024)
        require(len(times) == 30 and all(p.dtype == torch.float32 for p in adapter.parameters()), 'Inference contract changed')
        checks = {}
        for suffix in ['.png', '_raw.png']:
            new = result.with_name(result.stem+suffix); ref = original.with_name(original.stem+suffix)
            with Image.open(new) as a, Image.open(ref) as b:
                exact = np.array_equal(np.asarray(a), np.asarray(b))
            checks[suffix] = dict(pixel_exact=exact, byte_exact=sha(new)==sha(ref),
                                  output=pin(new), reference=pin(ref))
        receipt.update(checks=checks, timesteps=times,
                       passed=all(x['pixel_exact'] and x['byte_exact'] for x in checks.values()))
        require(receipt['passed'], 'Enabled-zero image replay mismatch; training blocked')
        for record in receipt['source_pins']:
            verify(record)
        receipt['status'] = 'passed'
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if hook is not None:
            hook.remove()
        receipt['seconds'] = time.monotonic()-started
        save(OUT/'receipt.json', receipt)
        del runtime, proxy, pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(receipt['status'], flush=True)


if __name__ == '__main__':
    main()
