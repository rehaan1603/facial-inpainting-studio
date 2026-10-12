"""One fixed, fresh matched continuation arm; preserve every earlier experiment."""
import argparse
import gc
import hashlib
import inspect
import math
import os
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
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
    ROOT, IDS, sha, read, require, save, pin, verify, check_tensor, backend_flags,
    state_save)
from probe_masked_sdxl_ssim_loss_v1 import process_memory
from sdxl_token_compatibility_probe_v1 import (
    CompatibilityScorer, TokenCompatibilityProcessor, install, ensure_studio_idle)

BASE = ROOT/'outputs/masked_sdxl_ssim_continuation_v1'
PILOT = ROOT/'outputs/masked_sdxl_compatibility_pilot_v2'
PROBE = ROOT/'outputs/masked_sdxl_ssim_loss_probe_v1/receipt.json'
PROTOCOL = ROOT/'research/MASKED_SSIM_CONTINUATION_PREFLIGHT_V1.md'
CHECKPOINT_SHA = '0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba'
SEED = 20261010
UPDATES = 32
TIMESTEPS = [100, 300]
EVAL_FILES = ['run_masked_sdxl_ssim_continuation_eval_v1.py',
              'score_masked_sdxl_ssim_continuation_eval_v1.py',
              'report_masked_sdxl_ssim_continuation_eval_v1.py']


def preflight(arm):
    out = BASE/arm/'training'
    require(PROTOCOL.is_file() and not out.exists(), 'Missing reviewed protocol or existing arm; refuse resume/overwrite')
    cfg = read(PILOT/'training/config.json')
    old = read(PILOT/'training/receipt.json')
    cache = read(PILOT/'cache/receipt.json')
    proof = read(PROBE)
    require(old['status']=='complete' and old['completed_updates']==64 and old['finite_gradient_updates']==64,
            'Completed original 64-update pilot required')
    require(sha(PILOT/'training/config.json')==old['config_sha256'], 'Previous config changed')
    require(sha(PILOT/'training/trace.json')==old['trace_sha256'], 'Previous trace changed')
    require(cache['status']=='complete' and cache['train_identities']==IDS
            and [r['identity'] for r in cache['records']]==IDS, 'Eight-case cache order changed')
    require(not set(IDS)&{'263','6681','7400','10149'}
            and {'263','6681','7400','10149'}<=set(cache['excluded_identities']), 'Protected identity boundary changed')
    require(sha(PILOT/'cache/receipt.json')==cfg['cache_receipt']['sha256'], 'Cache receipt changed')
    require(cache['manifest_sha256']=='b90a1ccddad5cd7c6efafe39209567765e918aabbf0a68022955e3baea2a875f',
            'Training-only recovered manifest changed')
    require(proof['status']=='passed' and proof['ssim_gradient_norm']>0 and proof['epsilon_gradient_norm']>0
            and proof['backend_flags_restored'] and proof['parameter_update'], 'Successful decoded-gradient probe required')
    require(proof['frozen_unet_gradients'] is False and proof['frozen_vae_gradients'] is False,
            'Probe frozen-weight guards failed')
    require(proof['starting_checkpoint']['sha256']==CHECKPOINT_SHA
            and old['checkpoints']['last']['sha256']==CHECKPOINT_SHA, 'Starting checkpoint changed')
    actual_versions = {k:version(k) for k in cfg['versions']}
    require(actual_versions==cfg['versions']==proof['versions'], 'Audited runtime version changed')
    for item in cfg['source_pins']+proof['source_pins']: verify(item)
    for item in cache['sources'].values(): verify(item)
    for item in cache['implementations'].values(): verify(item)
    for item in cache['asset_provenance']['assets']: verify(item)
    verify(cache['asset_provenance']['appearance_receipt'])
    verify(cache['common_condition'])
    for item in cache['records']:
        verify(item)
        require(Path(item['tensor_path']).resolve()==Path(item['path']).resolve(), 'Cache tensor path mismatch')
        require(item['role']=='train' and item['target']['split']=='train'
                and item['target']['identity']==item['identity'], 'Invalid training target role')
        require(len(item['references'])==4 and all(r['identity']==item['identity'] and r['split']=='train'
                for r in item['references']), 'Reference role/bank changed')
    for item in old['checkpoints'].values(): verify(item)
    verify(proof['cache_receipt']); verify(proof['prior_training_receipt'])
    require(all((ROOT/'scripts'/name).is_file() for name in EVAL_FILES), 'Freeze matched evaluator/scorer/report before training')
    return cfg, old, cache, proof, out, actual_versions


def copy_initial(source, destination):
    """Keep exact source-checkpoint bytes, including the serialized archive names."""
    require(not destination.exists(), 'Refusing initial checkpoint overwrite')
    verify(source)
    with Path(source['path']).open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    result = pin(destination)
    require(result['sha256']==CHECKPOINT_SHA, 'Initial checkpoint is not an exact original-last copy')
    return result


def load_rgb_targets(cache):
    """Decode only the eight existing training targets, once each, for SSIM arm."""
    targets = {}
    for record in cache['records']:
        verify(dict(path=record['target']['image_path'], sha256=record['target']['source_sha256']))
        x1,y1,x2,y2 = record['mask']['bbox_xyxy_exclusive']
        require(0<=x1<x2<=512 and 0<=y1<y2<=512, 'Invalid full-resolution hole bbox')
        binary = np.zeros((512,512), dtype=bool)
        binary[y1:y2,x1:x2] = True
        require(hashlib.sha256(binary.tobytes()).hexdigest()==record['mask']['binary_sha256'], 'Training hole bytes changed')
        mask = torch.from_numpy(binary.astype(np.float32))[None,None]
        stored = torch.load(record['tensor_path'], map_location='cpu', weights_only=True)
        check_tensor(stored['mask'], (1,1,64,64), 'cached latent mask')
        require(torch.equal(F.interpolate(mask,(64,64),mode='nearest'),stored['mask']), 'Latent mask does not match 512 hole')
        with Image.open(record['target']['image_path']) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize(
                (512,512),Image.Resampling.LANCZOS)).copy()
        rgb = torch.from_numpy(pixels).permute(2,0,1)[None].float()/255.
        check_tensor(rgb,(1,3,512,512),'RGB training target')
        require(mask.sum()>0 and float(mask.mean())==record['mask']['coverage'], 'Hole coverage changed')
        targets[record['identity']] = (rgb,mask)
        del stored,pixels,binary
    require(list(targets)==IDS, 'RGB target order changed')
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['epsilon','ssim'],required=True)
    parser.add_argument('--train',action='store_true')
    args = parser.parse_args()
    if not args.train:
        print('No training executed. Review/freeze the paired protocol before --train.',flush=True)
        return
    cfg,old,cache,proof,out,versions = preflight(args.arm)
    require(torch.cuda.is_available(), 'CUDA required'); ensure_studio_idle()
    external = Path(read(ROOT/'configs/local.json')['cache']); sd = external/'osor_models/sdxl'
    import diffusers.models.attention_processor as attention_source
    import diffusers.models.autoencoders.vae as vae_source
    import diffusers.models.modeling_utils as modeling_source
    import diffusers.utils.accelerate_utils as hook_source
    import torch.utils.checkpoint as checkpoint_source
    out.mkdir(parents=True,exist_ok=False)
    weight = .1 if args.arm=='ssim' else 0.
    config = dict(arm=args.arm,updates=UPDATES,training_resolution=512,training_ids=IDS,
        seed=SEED,timesteps=TIMESTEPS,trainable_parameters=385,batch=1,query_chunk=64,ssim_weight=weight,
        starting_checkpoint=old['checkpoints']['last'],cache_receipt=pin(PILOT/'cache/receipt.json'),
        probe_receipt=pin(PROBE),prior_training_receipt=pin(PILOT/'training/receipt.json'),
        protocol_sha256=sha(PROTOCOL),protocol_path=str(PROTOCOL.resolve()),
        objective='0.25 full epsilon MSE + 0.75 normalized hole epsilon MSE'+(' + 0.1 fully-contained masked structural loss' if weight else ''),
        decoded_rgb_policy='affine /2 + .5, unclamped' if weight else None,
        timestep_schedule='first sixteen updates t100, next sixteen t300; same ordered eight identities in each block',
        noise_seed_rule='20261010 + zero-based update index + 1',
        optimizer=dict(name='AdamW',reset=True,lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.),
        precision='fp16 frozen UNet; bf16 CUDA autocast; fp32 scorer/loss; SSIM-only frozen CPU fp32 VAE',
        backend_policy=['FLASH_ATTENTION','MATH'],versions=versions,checkpoints={},
        scope='Matched eight-identity development continuations; training loss does not imply generated-image improvement',
        source_pins=[pin(__file__),pin(PROTOCOL),pin(ROOT/'scripts/masked_ssim_training_loss_v1.py'),pin(ROOT/'scripts/run_masked_sdxl_ssim_continuation_bounded_v1.py'),
            *[pin(ROOT/'scripts'/name) for name in EVAL_FILES],
            *[pin(ROOT/'scripts'/name) for name in ('train_masked_sdxl_compatibility_pilot_v2.py',
                'probe_masked_sdxl_ssim_loss_v1.py','sdxl_token_compatibility_probe_v1.py',
                'local_appearance_processor_v1.py','reference_inpaint.py','score_masked_sdxl_v1.py',
                'region_metrics_v1.py','extended_evaluation_metrics.py')],
            *[pin(path) for path in (inspect.getfile(UNet2DConditionModel),inspect.getfile(DDPMScheduler),
                inspect.getfile(AutoencoderKL),attention_source.__file__,vae_source.__file__,modeling_source.__file__,
                hook_source.__file__,checkpoint_source.__file__)]],
        model_assets=cache['asset_provenance']['assets'])
    save(out/'config.json',config)
    receipt = dict(status='running',arm=args.arm,started_at_utc=datetime.now(timezone.utc).isoformat(),
        trainable_parameters=385,completed_updates=0,finite_gradient_updates=0,starting_checkpoint=config['starting_checkpoint'],
        protocol_sha256=config['protocol_sha256'],checkpoints={},stages={},
        frozen_weights_have_gradients=False,frozen_vae_have_gradients=False if weight else None,
        decoded_training_photographs=0,unique_decoded_training_targets=0,
        evaluation_photographs_decoded=0,evaluation_annotations_decoded=0,
        backend_flags_before=backend_flags())
    start=time.monotonic(); stage_start=start; trace=[]
    def progress(stage):
        nonlocal stage_start
        now=time.monotonic()
        if 'current_stage' in receipt: receipt['stages'][receipt['current_stage']]=now-stage_start
        stage_start=now;receipt.update(current_stage=stage,elapsed_seconds=now-start)
        receipt['process_memory']=process_memory();save(out/'receipt.json',receipt)
        print(stage,flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(SEED)
        progress('load_models')
        unet=UNet2DConditionModel.from_pretrained(str(sd),subfolder='unet',variant='fp16',
            torch_dtype=torch.float16,local_files_only=True)
        portrait=torch.load(external/'reference_models/ip-adapter-faceid-portrait_sdxl.bin',map_location='cpu',weights_only=True)
        tensors=load_file(str(external/'eye_appearance_models_v1/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors'),device='cpu')
        appearance={prefix:{k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+'.')}
                    for prefix in ('image_proj','ip_adapter')}
        unet._load_ip_adapter_weights([portrait,appearance])
        require(not unet._load_ip_adapter_loras([portrait,appearance]),'Unexpected embedded LoRA')
        del portrait,appearance,tensors
        unet.requires_grad_(False).train()
        require(unet.config.in_channels==9 and sum(p.numel() for p in unet.parameters())==3366130756,'Frozen UNet architecture changed')
        require(all(m.p==0 for m in unet.modules() if isinstance(m,torch.nn.Dropout)),'UNet dropout changed')
        for processor in unet.attn_processors.values():
            if isinstance(processor,IPAdapterAttnProcessor2_0):processor.num_tokens=[16,16];processor.scale=[.8,.2]
        projections=unet.encoder_hid_proj.image_projection_layers
        require([int(projections[0].num_tokens),int(projections[1].latents.shape[1])]==[16,16],'Reference token count changed')
        unet.enable_gradient_checkpointing();unet.cuda()
        vae=None;rgb_targets={}
        if weight:
            vae=AutoencoderKL.from_pretrained(str(sd),subfolder='vae',variant='fp16',torch_dtype=torch.float32,
                local_files_only=True,use_safetensors=True).float().cpu().eval()
            vae.requires_grad_(False);vae.disable_tiling();vae.disable_slicing();vae.enable_gradient_checkpointing()
            require(not any(hasattr(m,'_hf_hook') for m in vae.modules()),'Unexpected VAE offload hook')
            require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in vae.parameters()),'VAE device/precision changed')
            require(vae.decoder.gradient_checkpointing and float(vae.config.scaling_factor)==.13025,'VAE protocol changed')
            require(sum(p.numel() for p in vae.parameters())==83653863,'Frozen VAE parameter count changed')
            receipt['frozen_vae_parameters']=83653863
            progress('load_training_rgb_targets')
            rgb_targets=load_rgb_targets(cache)
            receipt['decoded_training_photographs']=len(rgb_targets)
            receipt['unique_decoded_training_targets']=len(rgb_targets)
        common=torch.load(cache['common_condition']['path'],map_location='cpu',weights_only=True)
        for key,shape in [('prompt_embeds',(1,77,2048)),('pooled_prompt_embeds',(1,1280)),('add_time_ids',(1,6))]:
            check_tensor(common[key],shape,key)
        common={k:v.cuda().half() for k,v in common.items()}
        scorer=CompatibilityScorer().cuda().float()
        starting_state=torch.load(old['checkpoints']['last']['path'],map_location='cpu',weights_only=True)
        scorer.load_state_dict(starting_state,strict=True)
        config['checkpoints']['initial']=copy_initial(old['checkpoints']['last'],out/'adapter_initial.pt')
        receipt['checkpoints']=dict(config['checkpoints'])
        initial=[p.detach().clone() for p in scorer.parameters()]
        params=list(scorer.parameters())
        mask=torch.load(cache['records'][0]['tensor_path'],map_location='cpu',weights_only=True)['mask']
        receipt['modified_processors']=install(unet,scorer,mask[0,0].bool(),64)
        require(receipt['modified_processors']==70 and sum(p.numel() for p in unet.parameters() if p.requires_grad)==385,
                'Shared trainable parameter count changed')
        scheduler=DDPMScheduler.from_config(read(sd/'scheduler/scheduler_config.json'))
        require(scheduler.config.prediction_type=='epsilon' and scheduler.config.num_train_timesteps==1000,'Noise prediction scheduler changed')
        config['scheduler']=dict(scheduler.config)
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        save(out/'config.json',config);save(out/'receipt.json',receipt)
        del mask,starting_state
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            receipt['backend_flags_inside']=backend_flags()
            for index in range(UPDATES):
                ensure_studio_idle()
                record=cache['records'][index%len(IDS)];t=TIMESTEPS[index//16]
                progress(f'update_{index+1:02d}_batch')
                stored=torch.load(record['tensor_path'],map_location='cpu',weights_only=True)
                for key,shape in [('target_latent',(1,4,64,64)),('masked_latent',(1,4,64,64)),('mask',(1,1,64,64)),
                                  ('identity',(2,4,512)),('appearance',(2,4,257,1280))]:check_tensor(stored[key],shape,key)
                require(((stored['mask']==0)|(stored['mask']==1)).all() and stored['mask'].sum()>0,'Invalid hole mask')
                clean=stored['target_latent'].cuda().half();masked=stored['masked_latent'].cuda().half()
                hole=stored['mask'].cuda().float()
                timestep=torch.tensor([t],device='cuda',dtype=torch.long)
                noise=torch.randn(clean.shape,device='cuda',dtype=torch.float16,
                    generator=torch.Generator(device='cuda').manual_seed(SEED+index+1))
                noisy=scheduler.add_noise(clean,noise,timestep)
                sample=torch.cat([noisy,hole.to(noisy.dtype),masked],dim=1)
                added=dict(text_embeds=common['pooled_prompt_embeds'],time_ids=common['add_time_ids'],
                    image_embeds=[stored['identity'][1:2].cuda().half(),stored['appearance'][1:2].cuda().half()])
                scorer.timestep_fraction=t/999.;scorer.visible_fraction=record['visible_fraction']
                for processor in unet.attn_processors.values():
                    if isinstance(processor,TokenCompatibilityProcessor):processor.target_mask=hole[0,0].bool()
                optimizer.zero_grad(set_to_none=True);step_start=time.monotonic()
                progress(f'update_{index+1:02d}_forward')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],
                        added_cond_kwargs=added,return_dict=False)[0]
                require(torch.isfinite(prediction).all() and prediction.requires_grad,'Invalid/no-gradient prediction')
                squared=(prediction.float()-noise.float()).square()
                full=squared.mean();regional=(squared*hole).sum()/(4*hole.sum());epsilon=.25*full+.75*regional
                ssim_loss=None;rgb_range=None;oracle_residual=None
                if weight:
                    progress(f'update_{index+1:02d}_decode')
                    alpha=scheduler.alphas_cumprod[t].cuda().float()
                    oracle=(noisy.float()-(1-alpha).sqrt()*noise.float())/alpha.sqrt()
                    residual=oracle-clean.float()
                    require(torch.isfinite(residual).all() and residual.abs().max()<=.02,'Unexpected fp16 noise inversion residual')
                    oracle_residual=dict(max_absolute=float(residual.abs().max()),rms=float(residual.square().mean().sqrt()))
                    x0=(noisy.float()-(1-alpha).sqrt()*prediction.float())/alpha.sqrt()
                    decoded=vae.decode(x0.to(device='cpu',dtype=torch.float32)/.13025,return_dict=False)[0]
                    require(tuple(decoded.shape)==(1,3,512,512) and torch.isfinite(decoded).all()
                            and decoded.requires_grad,'Invalid/no-gradient decoded target')
                    rgb=decoded/2+.5;target_rgb,rgb_mask=rgb_targets[record['identity']]
                    ssim_loss=masked_ssim_loss(rgb,target_rgb,rgb_mask)
                    rgb_range=dict(minimum=float(rgb.detach().min()),maximum=float(rgb.detach().max()),
                        outside_unit_interval_fraction=float(((rgb.detach()<0)|(rgb.detach()>1)).float().mean()))
                    loss=epsilon+weight*ssim_loss.to(device='cuda')
                else:loss=epsilon
                require(torch.isfinite(loss),'Nonfinite training objective')
                progress(f'update_{index+1:02d}_backward')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'Missing/nonfinite scorer gradient')
                norm=float(torch.sqrt(sum(p.grad.float().square().sum() for p in params)))
                require(math.isfinite(norm) and norm>0,'Zero/nonfinite aggregate scorer gradient')
                frozen_grad=any(p.grad is not None for p in unet.parameters() if not p.requires_grad)
                require(not frozen_grad,'Frozen UNet acquired gradients')
                if vae is not None:require(not any(p.grad is not None for p in vae.parameters()),'Frozen VAE acquired gradients')
                progress(f'update_{index+1:02d}_optimizer')
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
                require(torch.cuda.max_memory_allocated()<=7_800_000_000,'Fixed GPU memory budget exceeded')
                require(all(torch.isfinite(p).all() for p in params),'Nonfinite scorer after update')
                trace.append(dict(update=index+1,identity=record['identity'],timestep=t,noise_seed=SEED+index+1,
                    full_epsilon_mse=float(full.detach()),hole_epsilon_mse=float(regional.detach()),
                    epsilon_objective=float(epsilon.detach()),masked_ssim_loss=float(ssim_loss.detach()) if ssim_loss is not None else None,
                    ssim_weight=weight,loss=float(loss.detach()),gradient_norm=norm,
                    decoded_rgb_range=rgb_range,oracle_x0_inverse_residual=oracle_residual,
                    step_seconds=time.monotonic()-step_start))
                receipt['completed_updates']=index+1;receipt['finite_gradient_updates']=index+1
                save(out/'trace.json',trace)
                del stored,clean,masked,hole,timestep,noise,noisy,sample,added,prediction,squared,full,regional,epsilon,loss
                if weight:del alpha,oracle,residual,x0,decoded,rgb,ssim_loss,target_rgb,rgb_mask
            require(any(not torch.equal(a,p) for a,p in zip(initial,params)),'Scorer did not change')
            config['checkpoints']['last']=state_save(scorer,out/'adapter_last.pt')
        receipt['backend_flags_after']=backend_flags()
        receipt['backend_flags_restored']=receipt['backend_flags_after']==receipt['backend_flags_before']
        require(receipt['backend_flags_restored'],'Backend flags not restored')
        for item in old['checkpoints'].values():verify(item)
        for item in config['source_pins']:verify(item)
        verify(config['cache_receipt']);verify(config['probe_receipt'])
        config['training_elapsed_seconds']=time.monotonic()-start
        save(out/'config.json',config)
        receipt.update(status='complete',config_sha256=sha(out/'config.json'),trace_sha256=sha(out/'trace.json'),
            checkpoints=config['checkpoints'],gpu=torch.cuda.get_device_name(),parameter_update=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        progress('complete')
    except BaseException as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),
            backend_flags_after=backend_flags(),finished_at_utc=datetime.now(timezone.utc).isoformat())
        receipt['backend_flags_restored']=receipt['backend_flags_after']==receipt['backend_flags_before']
        receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        progress('failed');raise
    print({k:receipt[k] for k in ('status','arm','completed_updates','finite_gradient_updates',
        'elapsed_seconds','peak_allocated_bytes','backend_flags_restored')},flush=True)


if __name__=='__main__':main()
