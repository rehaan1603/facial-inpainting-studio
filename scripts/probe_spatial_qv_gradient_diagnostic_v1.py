"""Training-only masked spatial Q/V neutrality and gradient probe; no saved checkpoint."""
import argparse
import ctypes
import gc
import inspect
import math
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from masked_ssim_training_loss_v1 import masked_ssim_loss
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from safetensors.torch import load_file

from train_masked_sdxl_compatibility_pilot_v2 import (
    ROOT, IDS, sha, read, require, save, pin, verify, check_tensor, backend_flags)
from sdxl_token_compatibility_probe_v1 import CompatibilityScorer, install, ensure_studio_idle
from masked_spatial_qv_v1 import SpatialQVAdapter, install as install_spatial
from spatial_qv_common_v1 import save, verify_control, verify_cpu_tests, CONTROL

OUT = ROOT/'outputs/spatial_qv_gradient_diagnostic_v1'
PROTOCOL = ROOT/'research/SPATIAL_QV_GRADIENT_DIAGNOSTIC_PLAN_V1.md'
PILOT = ROOT/'outputs/masked_sdxl_compatibility_pilot_v2'
CHECKPOINT_SHA = '0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba'
SSIM_WEIGHT = .1
NOISE_SEED = 20261011
TIMESTEP = 300


def process_memory():
    """Windows peak includes earlier loading peaks; no other process is touched."""
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
            *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
              'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
              'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    api = ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    api.restype = ctypes.c_int
    require(api(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb), 'RAM measurement failed')
    return dict(peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                working_set_bytes=int(counters.WorkingSetSize),
                peak_pagefile_bytes=int(counters.PeakPagefileUsage))


def preflight():
    require(PROTOCOL.is_file() and not OUT.exists(), 'Missing plan or existing probe; refuse overwrite')
    verify_control(); verify_cpu_tests()
    config = read(PILOT/'training/config.json'); receipt = read(PILOT/'training/receipt.json')
    cache = read(PILOT/'cache/receipt.json')
    require(receipt['status']=='complete' and receipt['completed_updates']==64, 'Completed prior pilot required')
    require(sha(PILOT/'training/config.json')==receipt['config_sha256'], 'Prior config changed')
    require(cache['status']=='complete' and cache['train_identities']==IDS, 'Cache/data changed')
    require(sha(PILOT/'cache/receipt.json')==config['cache_receipt']['sha256'], 'Cache receipt changed')
    require(cache['manifest_sha256']=='b90a1ccddad5cd7c6efafe39209567765e918aabbf0a68022955e3baea2a875f', 'Manifest changed')
    for item in config['source_pins']: verify(item)
    for item in cache['sources'].values(): verify(item)
    for item in cache['implementations'].values(): verify(item)
    for item in cache['asset_provenance']['assets']: verify(item)
    for item in receipt['checkpoints'].values(): verify(item)
    verify(cache['common_condition'])
    for item in cache['records']: verify(item)
    require({k:version(k) for k in config['versions']}==config['versions'], 'Audited runtime versions changed')
    case = cache['records'][0]
    require(case['identity']=='6792' and case['role']=='train' and case['target']['hq_id']=='28120', 'Fixed training case changed')
    require(not set(IDS)&{'263','6681','7400','10149'}, 'Identity leakage')
    require(receipt['checkpoints']['last']['sha256']==CHECKPOINT_SHA, 'Starting checkpoint changed')
    verify(dict(path=case['target']['image_path'],sha256=case['target']['source_sha256']))
    return config, receipt, cache, case


def gradient_vector(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=False)
    require(all(torch.isfinite(g).all() for g in gradients), 'Nonfinite gradient component')
    vector = torch.cat([g.detach().float().reshape(-1) for g in gradients])
    norm = float(vector.norm())
    require(math.isfinite(norm) and norm>0, 'Loss failed to deliver finite nonzero scorer gradient')
    return gradients, vector, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true'); args = parser.parse_args()
    if not args.probe:
        print('No probe executed. Freeze/review the preflight before --probe.', flush=True); return
    cfg, old, cache, record = preflight()
    require(torch.cuda.is_available(), 'CUDA required'); ensure_studio_idle()
    external = Path(read(ROOT/'configs/local.json')['cache']); sd = external/'osor_models/sdxl'
    import diffusers.models.attention_processor as attention_source
    import diffusers.models.autoencoders.vae as vae_source
    import diffusers.utils.accelerate_utils as hook_source
    import diffusers.models.modeling_utils as modeling_source
    import torch.utils.checkpoint as checkpoint_source
    OUT.mkdir(parents=True, exist_ok=False)
    evidence = dict(status='running', scope='One training-photo gradient/resource probe only; no quality evaluation',
        started_at_utc=datetime.now(timezone.utc).isoformat(), identity='6792', target_hq_id='28120',
        trainable_parameters=10625, added_parameters=10240, resolution=512, timestep=TIMESTEP, noise_seed=NOISE_SEED,
        ssim_weight=SSIM_WEIGHT, decoded_rgb_policy='affine /2 + .5, unclamped',
        objective='0.25 full epsilon MSE + 0.75 hole epsilon MSE + 0.1 fully-contained masked structural loss',
        starting_checkpoint=pin(old['checkpoints']['last']['path']),
        cache_receipt=pin(PILOT/'cache/receipt.json'), prior_training_receipt=pin(PILOT/'training/receipt.json'),
        source_pins=[pin(__file__),pin(PROTOCOL),pin(ROOT/'scripts/masked_ssim_training_loss_v1.py'),pin(ROOT/'scripts/run_spatial_qv_gradient_diagnostic_bounded_v1.py'),pin(ROOT/'scripts/masked_spatial_qv_v1.py'),pin(ROOT/'scripts/spatial_qv_common_v1.py'),
            *[pin(ROOT/'scripts'/n) for n in ('train_masked_sdxl_compatibility_pilot_v2.py',
                'sdxl_token_compatibility_probe_v1.py','local_appearance_processor_v1.py')],
            *[pin(p) for p in (inspect.getfile(AutoencoderKL),inspect.getfile(UNet2DConditionModel),
                inspect.getfile(DDPMScheduler),attention_source.__file__,vae_source.__file__,
                hook_source.__file__,modeling_source.__file__,checkpoint_source.__file__)]],
        cpu_tests=verify_cpu_tests(),same_loss_control_receipt=pin(CONTROL/'receipt.json'),
        versions=cfg['versions'], backend_flags_before=backend_flags(), stages={}, checkpoints_created=0,
        quality_scores_created=0, old_checkpoint_changed=False)
    start=time.monotonic(); stage_start=start
    def progress(stage):
        nonlocal stage_start
        now=time.monotonic()
        if 'current_stage' in evidence: evidence['stages'][evidence['current_stage']]=now-stage_start
        stage_start=now; evidence.update(current_stage=stage,elapsed_seconds=now-start)
        evidence['process_memory']=process_memory(); save(OUT/'receipt.json',evidence)
        print(stage,flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(20261010)
        progress('load_models')
        unet=UNet2DConditionModel.from_pretrained(str(sd),subfolder='unet',variant='fp16',
            torch_dtype=torch.float16,local_files_only=True)
        portrait=torch.load(external/'reference_models/ip-adapter-faceid-portrait_sdxl.bin',map_location='cpu',weights_only=True)
        tensors=load_file(str(external/'eye_appearance_models_v1/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors'),device='cpu')
        appearance={prefix:{k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+'.')}
                    for prefix in ('image_proj','ip_adapter')}
        unet._load_ip_adapter_weights([portrait,appearance])
        require(not unet._load_ip_adapter_loras([portrait,appearance]), 'Unexpected embedded LoRA')
        del portrait,appearance,tensors
        unet.requires_grad_(False).train()
        require(unet.config.in_channels==9 and sum(p.numel() for p in unet.parameters())==3366130756, 'Frozen architecture changed')
        require(all(m.p==0 for m in unet.modules() if isinstance(m,torch.nn.Dropout)), 'Dropout changed')
        for p in unet.attn_processors.values():
            if isinstance(p,IPAdapterAttnProcessor2_0):p.num_tokens=[16,16];p.scale=[.8,.2]
        unet.enable_gradient_checkpointing();unet.cuda()
        vae=AutoencoderKL.from_pretrained(str(sd),subfolder='vae',variant='fp16',torch_dtype=torch.float32,
            local_files_only=True,use_safetensors=True).float().cpu().eval()
        vae.requires_grad_(False);vae.disable_tiling();vae.disable_slicing();vae.enable_gradient_checkpointing()
        require(not any(hasattr(m,'_hf_hook') for m in vae.modules()), 'VAE offload hook would alter audited device/gradient path')
        require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in vae.parameters()), 'VAE precision/device changed')
        require(vae.decoder.gradient_checkpointing and float(vae.config.scaling_factor)==.13025, 'VAE protocol changed')
        evidence['frozen_vae_parameters']=sum(p.numel() for p in vae.parameters())
        stored=torch.load(record['tensor_path'],map_location='cpu',weights_only=True)
        for k,s in [('target_latent',(1,4,64,64)),('masked_latent',(1,4,64,64)),('mask',(1,1,64,64)),
                    ('identity',(2,4,512)),('appearance',(2,4,257,1280))]:check_tensor(stored[k],s,k)
        common=torch.load(cache['common_condition']['path'],map_location='cpu',weights_only=True)
        for k,s in [('prompt_embeds',(1,77,2048)),('pooled_prompt_embeds',(1,1280)),('add_time_ids',(1,6))]:check_tensor(common[k],s,k)
        common={k:v.cuda().half() for k,v in common.items()}
        x1,y1,x2,y2=record['mask']['bbox_xyxy_exclusive']
        binary=np.zeros((512,512),dtype=bool);binary[y1:y2,x1:x2]=True
        import hashlib
        require(hashlib.sha256(binary.tobytes()).hexdigest()==record['mask']['binary_sha256'],'Full-resolution mask bytes changed')
        rgb_mask=torch.from_numpy(binary.astype(np.float32))[None,None]
        require(torch.equal(F.interpolate(rgb_mask,(64,64),mode='nearest'),stored['mask']), 'Latent mask mismatch')
        verify(dict(path=record['target']['image_path'],sha256=record['target']['source_sha256']))
        with Image.open(record['target']['image_path']) as im:
            pixels=np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)).copy()
        target_rgb=torch.from_numpy(pixels).permute(2,0,1)[None].float()/255.
        del pixels
        scorer=CompatibilityScorer().cuda().float()
        scorer.load_state_dict(torch.load(old['checkpoints']['last']['path'],map_location='cpu',weights_only=True),strict=True)
        scorer.timestep_fraction=TIMESTEP/999.;scorer.visible_fraction=record['visible_fraction']
        hole=stored['mask'].cuda().float()
        evidence['modified_processors']=install(unet,scorer,hole[0,0].bool(),64)
        require(evidence['modified_processors']==70 and sum(p.numel() for p in unet.parameters() if p.requires_grad)==385, 'Trainable parameter count changed')
        scorer_params=list(scorer.parameters())
        scheduler=DDPMScheduler.from_config(read(sd/'scheduler/scheduler_config.json'))
        require(scheduler.config.prediction_type=='epsilon','Only epsilon scheduler was audited')
        clean=stored['target_latent'].cuda().half();masked=stored['masked_latent'].cuda().half()
        noise=torch.randn(clean.shape,device='cuda',dtype=torch.float16,
            generator=torch.Generator(device='cuda').manual_seed(NOISE_SEED))
        timestep=torch.tensor([TIMESTEP],device='cuda',dtype=torch.long)
        noisy=scheduler.add_noise(clean,noise,timestep)
        alpha=scheduler.alphas_cumprod[TIMESTEP].cuda().float()
        oracle=(noisy.float()-(1-alpha).sqrt()*noise.float())/alpha.sqrt()
        residual=oracle-clean.float()
        evidence['alpha_bar']=float(alpha);evidence['oracle_x0_inverse_residual']=dict(
            max_absolute=float(residual.abs().max()),rms=float(residual.square().mean().sqrt()),
            reason='Inherited scheduler.add_noise fp16 coefficient/arithmetic rounding; not exact inverse')
        require(torch.isfinite(residual).all() and residual.abs().max()<=.02, 'Unexpected scheduler inverse residual')
        sample=torch.cat([noisy,hole.to(noisy.dtype),masked],dim=1)
        added=dict(text_embeds=common['pooled_prompt_embeds'],time_ids=common['add_time_ids'],
            image_embeds=[stored['identity'][1:2].cuda().half(),stored['appearance'][1:2].cuda().half()])
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            evidence['backend_flags_inside']=backend_flags();progress('original_forward')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                original=unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],
                    added_cond_kwargs=added,return_dict=False)[0]
            neutral_prediction=original.detach().clone()
            base_sq=(original.float()-noise.float()).square()
            base_loss=.25*base_sq.mean()+.75*(base_sq*hole).sum()/(4*hole.sum())
            _,neutral_gradient,_=gradient_vector(base_loss,scorer_params,False)
            neutral_gradient=neutral_gradient.detach().clone()
            evidence['neutral_epsilon_objective']=float(base_loss.detach())
            del original,base_sq,base_loss
            gc.collect();torch.cuda.empty_cache()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                repeat=unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],
                    added_cond_kwargs=added,return_dict=False)[0]
            rs=(repeat.float()-noise.float()).square()
            rl=.25*rs.mean()+.75*(rs*hole).sum()/(4*hole.sum())
            _,repeat_vector,_=gradient_vector(rl,scorer_params,False)
            def comparison(a,b):
                return dict(maximum_absolute_difference=float((a-b).abs().max()),
                    passed=torch.allclose(a,b,atol=1e-7,rtol=1e-4),exact=torch.equal(a,b))
            evidence['native_repeat']=comparison(neutral_gradient,repeat_vector)
            evidence['native_repeat_prediction_exact']=torch.equal(repeat.detach(),neutral_prediction)
            evidence['native_gradient_vector']=neutral_gradient.cpu().tolist()
            evidence['repeat_gradient_vector']=repeat_vector.cpu().tolist()
            del repeat,rs,rl,repeat_vector
            gc.collect();torch.cuda.empty_cache()
            adapter=SpatialQVAdapter().cuda().float()
            processor=install_spatial(unet,adapter,hole[0,0].bool())
            require(sum(p.numel() for p in unet.parameters() if p.requires_grad)==10625,'New trainable count changed')
            params=scorer_params+list(adapter.parameters())
            require(len({id(p) for p in params})==len(params) and
                    {id(p) for p in params}=={id(p) for p in unet.parameters() if p.requires_grad},
                    'Optimizer parameter set does not exactly match trainable UNet parameters')
            initial=[p.detach().clone() for p in params]
            optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            progress('enabled_zero_forward')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],
                    added_cond_kwargs=added,return_dict=False)[0]
            require(torch.isfinite(prediction).all() and prediction.requires_grad,'Invalid/no-gradient UNet prediction')
            evidence['enabled_zero_prediction_exact']=torch.equal(prediction.detach(),neutral_prediction)
            evidence['enabled_zero_prediction_max_difference']=float((prediction.detach().float()-neutral_prediction.float()).abs().max())
            require(evidence['enabled_zero_prediction_exact'],'Enabled-zero full-model prediction differs')
            del neutral_prediction
            squared=(prediction.float()-noise.float()).square()
            full=squared.mean();regional=(squared*hole).sum()/(4*hole.sum());epsilon=.25*full+.75*regional
            require(torch.isfinite(epsilon),'Nonfinite epsilon loss')
            progress('epsilon_backward')
            epsilon_grad,epsilon_vector,epsilon_norm=gradient_vector(epsilon,params,True)
            evidence['epsilon_gradient_norm']=epsilon_norm
            scorer_vector=torch.cat([g.detach().float().reshape(-1) for g in epsilon_grad[:len(scorer_params)]])
            evidence['neutral_scorer_gradient']=dict(atol=1e-7,rtol=1e-4,
                maximum_absolute_difference=float((scorer_vector-neutral_gradient).abs().max()),
                exact=torch.equal(scorer_vector,neutral_gradient),
                passed=torch.allclose(scorer_vector,neutral_gradient,atol=1e-7,rtol=1e-4))
            _,only_vector,_=gradient_vector(epsilon,scorer_params,False)
            evidence['scorer_only_vs_native']=comparison(only_vector,neutral_gradient)
            evidence['scorer_only_vs_all']=comparison(only_vector,scorer_vector)
            evidence['all_request_scorer_vector']=scorer_vector.cpu().tolist()
            evidence['scorer_only_vector']=only_vector.cpu().tolist()
            require(torch.cuda.max_memory_allocated()<=7_800_000_000,'Diagnostic memory bound exceeded')
            evidence['optimizer_steps']=0
        evidence['backend_flags_after']=backend_flags()
        evidence['backend_flags_restored']=evidence['backend_flags_after']==evidence['backend_flags_before']
        require(evidence['backend_flags_restored'],'Backend flags were not restored')
        verify(old['checkpoints']['last']);verify(old['checkpoints']['initial'])
        for item in evidence['source_pins']:verify(item)
        verify_control();verify_cpu_tests()
        evidence.update(status='passed',diagnostic_only=True,gpu=torch.cuda.get_device_name(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        progress('complete')
    except BaseException as exc:
        evidence.update(status='failed',error=f'{type(exc).__name__}: {exc}',out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),
            backend_flags_after=backend_flags(),finished_at_utc=datetime.now(timezone.utc).isoformat())
        evidence['backend_flags_restored']=evidence['backend_flags_after']==evidence['backend_flags_before']
        evidence['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        evidence['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        progress('failed');raise
    print({k:evidence.get(k) for k in ('status','elapsed_seconds','epsilon_gradient_norm','ssim_gradient_norm',
        'gradient_cosine','peak_allocated_bytes','backend_flags_restored')},flush=True)


if __name__=='__main__':main()
