"""Factorial scaffold/reference/context experiment. No clean-target/gallery access."""
import json, sys, time, subprocess, hashlib, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.reference_context_v2 import VisibleContextCorrector, sample_explicit

BASE = ROOT / 'outputs/scaffold_conditioning_v1'
PROTOCOL = ROOT / 'research/protocols/scaffold_conditioning_v1.json'
MANIFEST = ROOT / 'outputs/distortion_aware_v1/inference_manifest.json'


def main():
    cfg = json.loads(PROTOCOL.read_text())
    cases = {c['case_id']: c for c in json.loads(MANIFEST.read_text())['cases']}
    previous = ROOT / 'outputs/context_support_v1'
    prior_context = {(r['case_id'], r['seed']): r for r in json.loads((ROOT / 'outputs/reference_context_v2/comparison.json').read_text())['rows'] if r['mode']=='reference_context'}
    scaffolds = {(r['case_id'], r['seed']): r for r in json.loads((previous / 'comparison.json').read_text())['rows'] if r['mode']=='fixed_r0'}
    assert json.loads((previous / 'signature.json').read_text())['signature']['inference_manifest_sha256'] == sha(MANIFEST)
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    source, weights = cache / 'refldm_source_v1', cache / 'refldm_weights_v1'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'af6690c19fdc6421802fd7996510bcfef259bfd1'
    assert subprocess.check_output(['git', '-C', str(source), 'diff'], text=True) == (ROOT / 'research/refldm_author_compatibility.patch').read_text()
    weight_records = json.loads((ROOT / 'research/refldm_downloads_v1.json').read_text())['files']
    for f in weight_records: assert sha(weights / f['name']) == f['sha256']
    for cid in cfg['cases']:
        c = cases[cid]
        assert c['identity'] not in cfg['reserved_identities']
        for p, h in [(c['observed'], c['observed_sha256']), (c['mask'], c['mask_sha256'])]+[(r['path'], r['sha256']) for r in c['references']]:
            assert sha(p) == h
        for seed in cfg['seeds']:
            s = scaffolds[(cid, seed)]
            assert s['status']=='complete' and sha(s['output']) == s['output_sha256']
    BASE.mkdir(exist_ok=False)
    write_new(BASE / 'signature.json', {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(PROTOCOL), 'runner_sha256': sha(__file__),
        'mechanism_sha256': sha(ROOT / 'src/preservation/reference_context_v2.py'),
        'manifest_sha256': sha(MANIFEST), 'prior_reference_context_comparison_sha256': sha(ROOT / 'outputs/reference_context_v2/comparison.json'), 'scaffold_comparison_sha256': sha(previous / 'comparison.json'),
        'source_commit': commit, 'patch_sha256': sha(ROOT / 'research/refldm_author_compatibility.patch'),
        'weight_sha256': {f['name']: f['sha256'] for f in weight_records}, 'torch': torch.__version__,
        'inference_scope': 'Observed pixels, masks, supplied references and prior scaffold only'})
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm import cache_kv
    from inference import read_and_normalize_image
    from torchvision.transforms.functional import to_pil_image
    torch.set_num_threads(4)
    config = OmegaConf.load(source / 'configs/refldm.yaml')
    config.model.params.first_stage_config.params.ckpt_path = str(weights / 'vqgan.ckpt')
    for k in ['ckpt_path', 'perceptual_loss_config']: config.model.params.pop(k, None)
    model = instantiate_from_config(config.model)
    keys = model.load_state_dict(torch.load(weights / 'refldm.ckpt', map_location='cpu', weights_only=True), strict=False)
    assert not keys.missing_keys and not keys.unexpected_keys
    model.cuda().eval(); torch.set_grad_enabled(False)
    shape = [model.model.diffusion_model.out_channels, model.image_size, model.image_size]
    sampler = DDIMSampler(model, print_tqdm=False, schedule='uniform_trailing')
    rows, golden_done = [], False
    for cid in cfg['cases']:
        c = cases[cid]
        observed = np.asarray(Image.open(c['observed']).convert('RGB'))
        mask = np.asarray(Image.open(c['mask']).convert('L')) >= 128
        obs_tensor = read_and_normalize_image(c['observed']).unsqueeze(0).cuda()
        mask_tensor = torch.from_numpy(mask.copy())[None, None].cuda()
        refs = torch.cat([read_and_normalize_image(r['path']) for r in c['references']], dim=-1).unsqueeze(0).cuda()
        for seed in cfg['seeds']:
            scaffold = scaffolds[(cid, seed)]
            condition = model.get_learned_conditioning({'lq_image': read_and_normalize_image(scaffold['output']).unsqueeze(0).cuda(), 'ref_image': refs})
            tensor_sha = lambda t: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for arm in cfg['arms']:
                key = f'{cid}_s{seed}_{arm}'
                receipt = {'key': key, 'case_id': cid, 'identity': c['identity'], 'seed': seed, 'mode': arm, 'status': 'failed',
                           'scaffold_sha256': scaffold['output_sha256']}
                if arm == 'reference_context':
                    old = prior_context[(cid, seed)]
                    assert sha(old['output']) == old['output_sha256']
                    receipt.update(status='complete', output=old['output'], output_sha256=old['output_sha256'], reused_reference_context=True)
                elif arm == 'scaffold':
                    receipt.update(status='complete', output=scaffold['output'], output_sha256=scaffold['output_sha256'], reused_scaffold=True)
                else:
                    start = time.monotonic()
                    try:
                        cond = {k: v.detach().clone() for k, v in condition.items()}
                        latent_mask = torch.nn.functional.adaptive_max_pool2d(mask_tensor.float(), cond['lq_image'].shape[-2:])
                        if arm.startswith('masked_lq'): cond['lq_image'] *= (1-latent_mask)
                        elif arm.startswith('global_lq'): cond['lq_image'] *= (1-latent_mask.mean())
                        receipt['latent_missing_fraction'] = float(latent_mask.mean())
                        if 'no_reference' in arm: cond['ref_image'].zero_()
                        null = {k: v.detach().clone() for k, v in cond.items()}; null['ref_image'].zero_()
                        if not golden_done:
                            # Check the explicit route without changing the first actual arm's conditions.
                            golden_condition = {k: v.detach().clone() for k, v in condition.items()}
                            golden_null = {k: v.detach().clone() for k, v in golden_condition.items()}; golden_null['ref_image'].zero_()
                            cache_kv.clear_cache(); torch.manual_seed(seed)
                            noise = torch.randn([1, *shape], device='cuda')
                            with model.ema_scope():
                                author, _ = sampler.sample(S=50, batch_size=1, shape=shape, conditioning=golden_condition,
                                    unconditional_conditioning=golden_null, unconditional_guidance_scale=1.5, x_T=noise, verbose=False)
                            cache_kv.clear_cache(); torch.manual_seed(seed)
                            noise = torch.randn([1, *shape], device='cuda')
                            disabled = VisibleContextCorrector(obs_tensor, mask_tensor, gain=0)
                            with model.ema_scope():
                                explicit, _ = sample_explicit(sampler, golden_condition, golden_null, shape, noise, disabled)
                            golden = {'latent_bit_exact': bool(torch.equal(author, explicit)),
                                      'latent_max_abs_difference': float((author-explicit).abs().max()), 'hook_calls': disabled.calls,
                                      'source_scope': 'Same scaffold/reference/noise; no clean target read'}
                            write_new(BASE / 'gain_zero_equivalence.json', golden)
                            if not golden['latent_bit_exact'] or disabled.calls != 50:
                                raise RuntimeError('Explicit sampler equivalence failed; stop experiment')
                            golden_done = True
                        cache_kv.clear_cache(); torch.manual_seed(seed)
                        noise = torch.randn([1, *shape], device='cuda')
                        gain = cfg['correction']['gain'] if arm.endswith('_context') else 0
                        corrector = VisibleContextCorrector(obs_tensor, mask_tensor, gain=gain, steps=cfg['correction']['steps'])
                        torch.cuda.reset_peak_memory_stats(); started_generation = time.monotonic()
                        with model.ema_scope():
                            latent, _ = sample_explicit(sampler, cond, null, shape, noise, corrector)
                        decoded = ((model.decode_first_stage(latent)+1)/2).clamp(0, 1)
                        if not torch.isfinite(decoded).all(): raise ValueError('Nonfinite decoded result')
                        if corrector.calls != 50 or (gain and [a['step_index'] for a in corrector.applied] != cfg['correction']['steps']):
                            raise RuntimeError('Correction hook did not execute as specified')
                        folder = BASE / 'generations'; folder.mkdir(exist_ok=True)
                        raw = folder / (key+'_raw.png'); to_pil_image(decoded.squeeze(0).cpu()).save(raw)
                        rgb = np.asarray(Image.open(raw).convert('RGB'))
                        output = np.where(mask[..., None], rgb, observed)
                        path = folder / (key+'.png'); Image.fromarray(output).save(path)
                        receipt.update(status='complete', output=str(path), output_sha256=sha(path), raw_sha256=sha(raw),
                            reference_condition_sha256=tensor_sha(cond['ref_image']), reference_condition_abs_sum=float(cond['ref_image'].abs().sum()),
                            low_quality_condition_sha256=tensor_sha(cond['lq_image']), correction_gain=gain,
                            correction_applied=corrector.applied, hook_calls=corrector.calls,
                            generation_seconds=time.monotonic()-started_generation,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(), known_pixels_unchanged=bool(np.array_equal(output[~mask], observed[~mask])))
                    except Exception as error:
                        receipt['error'] = f'{type(error).__name__}: {error}'; receipt['traceback'] = traceback.format_exc()
                    receipt['seconds_including_setup'] = time.monotonic()-start
                rows.append(receipt); write_new(BASE / 'receipts' / (key+'.json'), receipt)
                print('REFERENCE CONTEXT', len(rows), '/40', key, receipt['status'], flush=True)
                if arm not in ['scaffold', 'reference_context'] and not golden_done: raise RuntimeError('Failed equivalence guard; ledger retained')
    write_new(BASE / 'comparison.json', {'rows': rows, 'signature_sha256': sha(BASE / 'signature.json'), 'final_test_used': False})


if __name__ == '__main__': main()
