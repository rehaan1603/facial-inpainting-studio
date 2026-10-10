"""Train exactly one private 385-parameter masked SDXL compatibility pilot.

The frozen protocol, complete CPU cache and 512 backward receipt are mandatory.
No evaluation image is decoded and no prior result/checkpoint is overwritten.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from local_appearance_processor_v1 import LocalAppearanceProcessor
from sdxl_token_compatibility_probe_v1 import (CompatibilityScorer,
    TokenCompatibilityProcessor, FEATURES, install, ensure_studio_idle)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache'
OUT = ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/training'
PROTOCOL = ROOT/'research/MASKED_SDXL_COMPATIBILITY_PILOT_PREFLIGHT_V2.md'
IDS = ['6792', '2250', '2083', '4164', '674', '476', '1281', '8748']
SEED = 20261010
TIMESTEPS = [100,300,500,700]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def require(condition, message):
    if not condition: raise ValueError(message)


def save(path, value):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def pin(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))


def backend_flags():
    return {name: getattr(torch.backends.cuda,name+'_sdp_enabled')()
            for name in ['flash','mem_efficient','math','cudnn']}


def verify(record):
    require(Path(record['path']).is_file() and sha(record['path'])==record['sha256'],
            f"Pinned file changed: {record['path']}")


def cache_preflight():
    require(PROTOCOL.is_file(), 'Save and review protocol before training')
    require(not OUT.exists(), 'Refusing overwrite or resume of training attempt')
    cache=read(CACHE_DIR/'receipt.json')
    require(cache['status']=='complete','Training cache is not complete')
    require(cache['train_identities']==IDS and [r['identity'] for r in cache['records']]==IDS,
            'Training identity order differs from fixed eight cases')
    require(not set(IDS)&set(cache['excluded_identities']), 'Holdout identity leakage')
    require({'263','6681','10149','7400'}<=set(cache['excluded_identities']), 'Holdout exclusions missing')
    require(cache['manifest_sha256']=='b90a1ccddad5cd7c6efafe39209567765e918aabbf0a68022955e3baea2a875f', 'Manifest changed')
    require(cache['protocol']['resolution']==512, 'Cache resolution changed')
    for record in list(cache['sources'].values())+list(cache['implementations'].values()): verify(record)
    for record in cache['asset_provenance']['assets']: verify(record)
    verify(cache['asset_provenance']['appearance_receipt'])
    verify(cache['common_condition'])
    for record in cache['records']:
        verify(record)
        require(Path(record['tensor_path']).resolve()==(CACHE_DIR/record['file']).resolve(), 'Cache tensor path changed')
    probe=read(ROOT/'outputs/masked_sdxl_compatibility_probe_v3/512.json')
    require(probe['status']=='passed' and probe['resolution']==512 and probe['actual_trainable_parameters']==385,
            'Successful matching 512 gradient probe is required')
    require(sha(ROOT/'scripts/sdxl_token_compatibility_probe_v1.py')==probe['script_sha256'], 'Probed architecture changed')
    require(sha(ROOT/'scripts/local_appearance_processor_v1.py')==probe['local_routing_source_sha256'], 'Neutral helper changed')
    import diffusers.models.attention_processor as attention_source
    require(sha(attention_source.__file__)==probe['attention_source_sha256'], 'Probed attention runtime changed')
    require(sha(inspect.getfile(UNet2DConditionModel))==probe['unet_source_sha256'], 'Probed UNet runtime changed')
    require({x:version(x) for x in probe['versions']}==probe['versions'], 'Probed environment changed')
    return cache,probe


def state_save(scorer, path):
    require(not path.exists(), 'Refusing checkpoint overwrite')
    state={k:v.detach().cpu().float().clone() for k,v in scorer.state_dict().items()}
    require(sum(v.numel() for v in state.values())==385 and all(torch.isfinite(v).all() for v in state.values()), 'Invalid checkpoint')
    torch.save(state,path)
    return pin(path)


def check_tensor(t, shape, label):
    require(tuple(t.shape)==tuple(shape) and torch.isfinite(t).all(), f'Invalid {label}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    args=parser.parse_args()
    if not args.train:
        print('No training executed. Review frozen pilot preflight then use --train.',flush=True)
        return
    cache,probe=cache_preflight()
    require(torch.cuda.is_available(), 'CUDA required for reviewed pilot')
    ensure_studio_idle()
    local=read(ROOT/'configs/local.json'); external=Path(local['cache'])
    import diffusers.models.attention_processor as attention_source
    sd=external/'osor_models/sdxl'
    # Reserve output before loading GPU weights; preserve any failed attempt.
    OUT.mkdir(parents=True,exist_ok=False)
    config=dict(updates=64,training_resolution=512,trainable_parameters=385,training_ids=IDS,
        evaluation_ids=['263','6681','10149'],seed=SEED,timesteps=TIMESTEPS,feature_order=FEATURES,
        protocol_path=str(PROTOCOL.resolve()),protocol_sha256=sha(PROTOCOL),cache_receipt=pin(CACHE_DIR/'receipt.json'),
        scope='Fixed real-data development pilot; no evaluation results inferred from training loss',
        objective='0.25 full epsilon MSE + 0.75 normalized hole epsilon MSE',
        optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.),
        noise_seed_rule='20261010 + zero-based update index + 1',
        precision='fp16 frozen weights; bf16 CUDA autocast; fp32 scorer/loss',
        backend_policy=['FLASH_ATTENTION','MATH'],batch=1,query_chunk=64,
        source_pins=[pin(__file__),pin(PROTOCOL),pin(ROOT/'scripts/sdxl_token_compatibility_probe_v1.py'),
                     pin(ROOT/'scripts/local_appearance_processor_v1.py'),pin(inspect.getfile(UNet2DConditionModel)),
                     pin(inspect.getfile(DDPMScheduler)),pin(attention_source.__file__),
                     *[pin(ROOT/'scripts'/name) for name in ['run_focused_eye_guidance_v1.py',
                        'score_masked_sdxl_v1.py','run_masked_sdxl_compatibility_pilot_eval_v2.py',
                        'score_masked_sdxl_compatibility_pilot_eval_v2.py','report_masked_sdxl_compatibility_pilot_eval_v2.py',
                        'region_metrics_v1.py','extended_evaluation_metrics.py','score_target_compatibility_v1.py']],
                     pin(ROOT/'research/extended_evaluation_provenance_v1.json'),pin(ROOT/'research/identity_evaluator_provenance_v1.json'),
                     *list(cache['sources'].values()), *list(cache['implementations'].values())],
        probe_receipt=pin(ROOT/'outputs/masked_sdxl_compatibility_probe_v3/512.json'),
        versions={x:version(x) for x in ['torch','diffusers','transformers','safetensors']},
        cache_assets=cache['asset_provenance']['assets'],checkpoints={})
    save(OUT/'config.json',config)
    receipt=dict(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=config['protocol_sha256'],completed_updates=0,finite_gradient_updates=0,
        trainable_parameters=385,frozen_weights_have_gradients=False,current_stage='load',checkpoints={})
    save(OUT/'receipt.json',receipt)
    start=time.monotonic(); trace=[]
    def progress(stage):
        receipt['current_stage']=stage;receipt['elapsed_seconds']=time.monotonic()-start
        save(OUT/'receipt.json',receipt);print(stage,flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(SEED)
        # Ensure UNet/adapter pins match the exact model that passed the gradient probe.
        for record in probe['assets']: verify(record)
        unet=UNet2DConditionModel.from_pretrained(str(sd),subfolder='unet',variant='fp16',torch_dtype=torch.float16,local_files_only=True)
        portrait=torch.load(external/'reference_models/ip-adapter-faceid-portrait_sdxl.bin',map_location='cpu',weights_only=True)
        tensors=load_file(str(external/'eye_appearance_models_v1/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors'),device='cpu')
        appearance={prefix:{k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+'.')} for prefix in ['image_proj','ip_adapter']}
        unet._load_ip_adapter_weights([portrait,appearance])
        require(not unet._load_ip_adapter_loras([portrait,appearance]),'Unexpected embedded LoRA')
        del portrait,appearance,tensors
        unet.requires_grad_(False).train()
        require(unet.config.in_channels==9 and all(m.p==0. for m in unet.modules() if isinstance(m,torch.nn.Dropout)),
                'UNet input/dropout contract changed')
        require(sum(p.numel() for p in unet.parameters())==3366130756, 'Loaded frozen parameter count changed')
        for processor in unet.attn_processors.values():
            if isinstance(processor,IPAdapterAttnProcessor2_0): processor.num_tokens=[16,16];processor.scale=[.8,.2]
        projections=unet.encoder_hid_proj.image_projection_layers
        require([int(projections[0].num_tokens),int(projections[1].latents.shape[1])]==[16,16], 'Reference token count changed')
        unet.enable_gradient_checkpointing();unet.cuda()
        common=torch.load(cache['common_condition']['path'],map_location='cpu',weights_only=True)
        for key,shape in [('prompt_embeds',(1,77,2048)),('pooled_prompt_embeds',(1,1280)),('add_time_ids',(1,6))]: check_tensor(common[key],shape,key)
        common={k:v.cuda().half() for k,v in common.items()}
        scheduler=DDPMScheduler.from_config(read(sd/'scheduler/scheduler_config.json'))
        require(scheduler.config.prediction_type=='epsilon' and scheduler.config.num_train_timesteps==1000,'Denoising target changed')
        config['scheduler']=dict(scheduler.config)
        torch.manual_seed(SEED)
        scorer=CompatibilityScorer().cuda().float()
        original=dict(unet.attn_processors)
        params=list(scorer.parameters())
        config['checkpoints']['initial']=state_save(scorer,OUT/'adapter_initial.pt')
        receipt['checkpoints']=dict(config['checkpoints'])
        save(OUT/'config.json',config);save(OUT/'receipt.json',receipt)
        initial=[p.detach().clone() for p in params]
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        def batch(record,index,t):
            stored=torch.load(record['tensor_path'],map_location='cpu',weights_only=True)
            for key,shape in [('target_latent',(1,4,64,64)),('masked_latent',(1,4,64,64)),('mask',(1,1,64,64)),('identity',(2,4,512)),('appearance',(2,4,257,1280))]:
                check_tensor(stored[key],shape,key)
            require(((stored['mask']==0)|(stored['mask']==1)).all() and stored['mask'].sum()>0, 'Invalid hole mask')
            clean=stored['target_latent'].cuda().half()
            masked=stored['masked_latent'].cuda().half()
            hole=stored['mask'].cuda().float()
            timestep=torch.tensor([t],device='cuda',dtype=torch.long)
            noise=torch.randn(clean.shape,device='cuda',dtype=torch.float16,generator=torch.Generator(device='cuda').manual_seed(SEED+index+1))
            noisy=scheduler.add_noise(clean,noise,timestep)
            sample=torch.cat([noisy,hole.to(noisy.dtype),masked],dim=1)
            added=dict(text_embeds=common['pooled_prompt_embeds'],time_ids=common['add_time_ids'],
                       image_embeds=[stored['identity'][1:2].cuda().half(),stored['appearance'][1:2].cuda().half()])
            return sample,hole,noise,timestep,added
        def forward(sample,timestep,added):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return unet(sample,timestep,encoder_hidden_states=common['prompt_embeds'],added_cond_kwargs=added,return_dict=False)[0]
        # Real-input neutrality precedes any training. Same noise for both paths.
        receipt['backend_flags_before']=backend_flags()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            receipt['backend_flags_inside']=backend_flags()
            progress('real_input_neutral_check')
            record=cache['records'][0];sample,hole,noise,timestep,added=batch(record,0,TIMESTEPS[0])
            mask=hole[0,0].bool();scorer.timestep_fraction=TIMESTEPS[0]/999.;scorer.visible_fraction=record['visible_fraction']
            unet.set_attn_processor({name:LocalAppearanceProcessor(p,mask) if isinstance(p,IPAdapterAttnProcessor2_0) else p for name,p in original.items()})
            with torch.no_grad(): neutral_control=forward(sample,timestep,added).detach()
            unet.set_attn_processor(original.copy())
            receipt['modified_processors']=install(unet,scorer,mask,64)
            require(receipt['modified_processors']==70 and sum(p.numel() for p in unet.parameters() if p.requires_grad)==385,'Trainable module count mismatch')
            with torch.no_grad(): neutral_prediction=forward(sample,timestep,added).detach()
            error=float((neutral_prediction.float()-neutral_control.float()).abs().max())
            require(math.isfinite(error) and error<=1e-3,'Real-input neutral equivalence failed')
            receipt['neutral_max_absolute_error']=error
            del sample,hole,noise,timestep,added,neutral_prediction,neutral_control,mask
            save(OUT/'config.json',config)
            progress('train_start')
            for index in range(64):
                ensure_studio_idle()
                t=TIMESTEPS[index//16];record=cache['records'][index%len(IDS)]
                sample,hole,noise,timestep,added=batch(record,index,t)
                scorer.timestep_fraction=t/999.;scorer.visible_fraction=float(record['visible_fraction'])
                # No mask/timestep/coverage mutation until backward finishes.
                for processor in unet.attn_processors.values():
                    if isinstance(processor,TokenCompatibilityProcessor): processor.target_mask=hole[0,0].bool()
                optimizer.zero_grad(set_to_none=True)
                step_start=time.monotonic()
                prediction=forward(sample,timestep,added)
                squared=(prediction.float()-noise.float()).square()
                full=squared.mean();regional=(squared*hole).sum()/(4*hole.sum())
                loss=.25*full+.75*regional
                require(torch.isfinite(loss),'Nonfinite training loss')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'Missing/nonfinite scorer gradients')
                norm=float(torch.sqrt(sum(p.grad.float().square().sum() for p in params)))
                require(math.isfinite(norm) and norm>0,'Zero/nonfinite aggregate scorer gradient')
                frozen_grad=any(p.grad is not None for p in unet.parameters() if not p.requires_grad)
                require(not frozen_grad,'Frozen weights acquired gradients')
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                require(all(torch.isfinite(p).all() for p in params),'Nonfinite scorer after update')
                trace.append(dict(update=index+1,identity=record['identity'],timestep=t,noise_seed=SEED+index+1,
                    full_epsilon_mse=float(full.detach()),hole_epsilon_mse=float(regional.detach()),loss=float(loss.detach()),
                    gradient_norm=norm,step_seconds=time.monotonic()-step_start))
                receipt['completed_updates']=index+1;receipt['finite_gradient_updates']=index+1
                save(OUT/'trace.json',trace)
                del sample,hole,noise,timestep,added,prediction,squared,full,regional,loss
                progress(f'update_{index+1}_of_64')
            require(any(not torch.equal(a,p) for a,p in zip(initial,params)),'Scorer did not change')
            config['checkpoints']['last']=state_save(scorer,OUT/'adapter_last.pt')
        # Context exits before completion: preserve default inference backend flags.
        receipt['backend_flags_after']=backend_flags()
        receipt['backend_flags_restored']=receipt['backend_flags_after']==receipt['backend_flags_before']
        require(receipt['backend_flags_restored'],'Training backend flags were not restored')
        config['training_elapsed_seconds']=time.monotonic()-start
        save(OUT/'config.json',config)
        receipt.update(status='complete',config_sha256=sha(OUT/'config.json'),checkpoints=config['checkpoints'],
            trace_sha256=sha(OUT/'trace.json'),gpu=torch.cuda.get_device_name(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            parameter_update=True,finished_at_utc=datetime.now(timezone.utc).isoformat())
        progress('complete')
    except BaseException as exc:
        receipt['backend_flags_after']=backend_flags()
        if 'backend_flags_before' in receipt:
            receipt['backend_flags_restored']=receipt['backend_flags_after']==receipt['backend_flags_before']
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',
            out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),finished_at_utc=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_available(): receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        progress('failed')
        raise
    print(json.dumps(receipt,indent=2,allow_nan=False),flush=True)


if __name__=='__main__': main()
