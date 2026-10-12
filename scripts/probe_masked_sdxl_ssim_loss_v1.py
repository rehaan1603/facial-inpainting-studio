"""One disposable decoded-hole SSIM gradient probe; no checkpoint or quality claim."""
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

OUT = ROOT/'outputs/masked_sdxl_ssim_loss_probe_v1'
PROTOCOL = ROOT/'research/MASKED_SSIM_CONTINUATION_PREFLIGHT_V1.md'
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
        trainable_parameters=385, resolution=512, timestep=TIMESTEP, noise_seed=NOISE_SEED,
        ssim_weight=SSIM_WEIGHT, decoded_rgb_policy='affine /2 + .5, unclamped',
        objective='0.25 full epsilon MSE + 0.75 hole epsilon MSE + 0.1 fully-contained masked structural loss',
        starting_checkpoint=pin(old['checkpoints']['last']['path']),
        cache_receipt=pin(PILOT/'cache/receipt.json'), prior_training_receipt=pin(PILOT/'training/receipt.json'),
        source_pins=[pin(__file__),pin(PROTOCOL),pin(ROOT/'scripts/masked_ssim_training_loss_v1.py'),pin(ROOT/'scripts/run_masked_sdxl_ssim_loss_bounded_v1.py'),
            *[pin(ROOT/'scripts'/n) for n in ('train_masked_sdxl_compatibility_pilot_v2.py',
                'sdxl_token_compatibility_probe_v1.py','local_appearance_processor_v1.py')],
            *[pin(p) for p in (inspect.getfile(AutoencoderKL),inspect.getfile(UNet2DConditionModel),
                inspect.getfile(DDPMScheduler),attention_source.__file__,vae_source.__file__,
                hook_source.__file__,modeling_source.__file__,checkpoint_source.__file__)]],
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
        params=list(scorer.parameters()); initial=[p.detach().clone() for p in params]
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
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            evidence['backend_flags_inside']=backend_flags();progress('unet_forward')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],
                    added_cond_kwargs=added,return_dict=False)[0]
            require(torch.isfinite(prediction).all() and prediction.requires_grad,'Invalid/no-gradient UNet prediction')
            squared=(prediction.float()-noise.float()).square()
            full=squared.mean();regional=(squared*hole).sum()/(4*hole.sum());epsilon=.25*full+.75*regional
            require(torch.isfinite(epsilon),'Nonfinite epsilon loss')
            progress('epsilon_backward')
            epsilon_grad,epsilon_vector,epsilon_norm=gradient_vector(epsilon,params,True)
            evidence['epsilon_gradient_norm']=epsilon_norm
            progress('cpu_vae_decode')
            x0=(noisy.float()-(1-alpha).sqrt()*prediction.float())/alpha.sqrt()
            decoded=vae.decode(x0.to(device='cpu',dtype=torch.float32)/float(vae.config.scaling_factor),return_dict=False)[0]
            require(tuple(decoded.shape)==(1,3,512,512) and torch.isfinite(decoded).all() and decoded.requires_grad,'Invalid/no-gradient decoded target')
            rgb=decoded/2+.5
            ssim_loss=masked_ssim_loss(rgb,target_rgb,rgb_mask)
            require(torch.isfinite(ssim_loss),'Nonfinite decoded-hole loss')
            evidence['decoded_rgb_range']=dict(minimum=float(rgb.detach().min()),maximum=float(rgb.detach().max()),
                outside_unit_interval_fraction=float(((rgb.detach()<0)|(rgb.detach()>1)).float().mean()))
            evidence['mask']=dict(pixels=int(rgb_mask.sum()),binary_sha256=record['mask']['binary_sha256'],
                                 bbox_xyxy_exclusive=record['mask']['bbox_xyxy_exclusive'])
            evidence['losses']=dict(full_epsilon_mse=float(full.detach()),hole_epsilon_mse=float(regional.detach()),
                epsilon_objective=float(epsilon.detach()),masked_ssim_loss=float(ssim_loss.detach()),
                combined=float(epsilon.detach())+SSIM_WEIGHT*float(ssim_loss.detach()))
            progress('ssim_backward')
            ssim_grad,ssim_vector,ssim_norm=gradient_vector(ssim_loss,params,False)
            eg,rg=epsilon_vector.double(),ssim_vector.double()
            cosine=float(torch.dot(eg,rg)/(eg.norm()*rg.norm()))
            require(math.isfinite(cosine) and -1.00000001<=cosine<=1.00000001,'Invalid gradient cosine')
            evidence.update(ssim_gradient_norm=ssim_norm,weighted_ssim_gradient_norm=SSIM_WEIGHT*ssim_norm,
                ssim_to_epsilon_gradient_norm_ratio=ssim_norm/epsilon_norm,gradient_cosine=cosine)
            require(torch.cuda.max_memory_allocated()<=7_800_000_000,'Fixed GPU memory budget exceeded')
            progress('disposable_update')
            for p,ge,gr in zip(params,epsilon_grad,ssim_grad):p.grad=ge.detach()+SSIM_WEIGHT*gr.detach()
            combined_norm=float(torch.cat([p.grad.float().flatten() for p in params]).norm())
            require(math.isfinite(combined_norm) and combined_norm>0,'Invalid combined gradient')
            evidence['combined_gradient_norm']=combined_norm
            require(not any(p.grad is not None for p in unet.parameters() if not p.requires_grad),'Frozen UNet acquired gradients')
            require(not any(p.grad is not None for p in vae.parameters()),'Frozen VAE acquired gradients')
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            require(all(torch.isfinite(p).all() for p in params),'Nonfinite updated scorer')
            evidence['parameter_update']=any(not torch.equal(a,p) for a,p in zip(initial,params))
            require(evidence['parameter_update'],'No in-memory update')
            evidence['frozen_unet_gradients']=False;evidence['frozen_vae_gradients']=False
        evidence['backend_flags_after']=backend_flags()
        evidence['backend_flags_restored']=evidence['backend_flags_after']==evidence['backend_flags_before']
        require(evidence['backend_flags_restored'],'Backend flags were not restored')
        verify(old['checkpoints']['last']);verify(old['checkpoints']['initial'])
        evidence.update(status='passed',gpu=torch.cuda.get_device_name(),
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
