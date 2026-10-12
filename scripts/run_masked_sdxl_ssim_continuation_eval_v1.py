"""Six matched continuation outputs after an exact enabled starting-state replay.

The original generation/evaluation sources are preserved. Generation does not
open evaluation originals; the unchanged CPU scorer alone measures quality.
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import run_masked_sdxl_compatibility_pilot_eval_v2 as old
from run_masked_sdxl_compatibility_pilot_eval_v2 import ROOT, sha, read, save, require, check_pin, load_state

BASE=ROOT/'outputs/masked_sdxl_ssim_continuation_v1'
OUT=ROOT/'outputs/masked_sdxl_ssim_continuation_eval_v1'
PROTOCOL=ROOT/'research/MASKED_SSIM_CONTINUATION_PREFLIGHT_V1.md'
IDS=['263','6681','10149']
ARMS=['control3','local','prior','epsilon','ssim']
START_SHA='0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba'
WRAPPERS=['score_masked_sdxl_ssim_continuation_eval_v1.py','report_masked_sdxl_ssim_continuation_eval_v1.py']


def preflight():
    _,prior_receipt,cfg,local,cases,controls,cache=old.preflight(old.TRAIN,old.PROTOCOL)
    require(PROTOCOL.is_file(),'Frozen continuation plan missing')
    trainings={}
    for arm in ['epsilon','ssim']:
        folder=BASE/arm/'training';t=read(folder/'config.json');r=read(folder/'receipt.json')
        require(r['status']=='complete' and r['completed_updates']==r['finite_gradient_updates']==t['updates']==32,'Incomplete continuation')
        require(t['arm']==r['arm']==arm and t['training_ids']==old.TRAIN_IDS,'Arm/data mismatch')
        require(t['training_resolution']==512 and t['trainable_parameters']==r['trainable_parameters']==385,'Architecture/resolution changed')
        require(t['optimizer']['reset'] and t['starting_checkpoint']['sha256']==START_SHA,'Initialization or optimizer changed')
        require(t['ssim_weight']==(.1 if arm=='ssim' else 0.) and t['timesteps']==[100,300] and t['seed']==20261010,'Recipe changed')
        require(r['frozen_weights_have_gradients'] is False and r['backend_flags_restored'] and r['parameter_update'],'Training guards failed')
        if arm=='ssim':require(r['frozen_vae_have_gradients'] is False,'Frozen decoder guard failed')
        require(t['protocol_sha256']==r['protocol_sha256']==sha(PROTOCOL),'Frozen protocol changed')
        require(sha(folder/'config.json')==r['config_sha256'] and sha(folder/'trace.json')==r['trace_sha256'],'Trace/config changed')
        for p in t['source_pins']:check_pin(p)
        for p in r['checkpoints'].values():check_pin(p)
        require(r['checkpoints']['initial']['sha256']==START_SHA,'Initial checkpoint bytes differ')
        trace=read(folder/'trace.json')
        require(len(trace)==32 and all(x['update']==i+1 and x['identity']==old.TRAIN_IDS[i%8]
            and x['timestep']==[100,300][i//16] and x['noise_seed']==20261010+i+1 for i,x in enumerate(trace)), 'Fixed schedule changed')
        trainings[arm]=dict(config_path=str(folder/'config.json'),config_sha256=sha(folder/'config.json'),
            receipt_path=str(folder/'receipt.json'),receipt_sha256=sha(folder/'receipt.json'),checkpoints=r['checkpoints'],
            config=t,receipt=r)
    left,right=[trainings[a]['config'] for a in ['epsilon','ssim']]
    for field in ['training_ids','seed','timesteps','optimizer','training_resolution','trainable_parameters','cache_receipt','starting_checkpoint']:
        require(left[field]==right[field],f'Matched arms differ: {field}')
    prior_dir=ROOT/'outputs/masked_sdxl_compatibility_pilot_eval_v2'
    summary=read(prior_dir/'summary.json')
    prior_rows=[x for x in summary['rows'] if x['condition']=='trained']
    require(len(prior_rows)==3 and {x['identity'] for x in prior_rows}==set(IDS),'Missing prior controls')
    for row in prior_rows:
        require(row['status']=='complete' and row['seed']==17,'Prior control not matched')
        output=prior_dir/row['file'];raw=output.with_name(output.stem+'_raw.png')
        require(sha(output)==row['sha256'] and sha(raw)==row['raw_sha256'],'Prior output changed')
        controls.append(dict(arm='prior',identity=row['identity'],row=row,output=output,raw=raw,
            summary_path=prior_dir/'summary.json',summary_sha256=sha(prior_dir/'summary.json')))
    require(prior_receipt['checkpoints']['last']['sha256']==START_SHA,'Original last checkpoint changed')
    return trainings,cfg,local,cases,controls,cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--skip-scoring',action='store_true');args=parser.parse_args()
    trainings,cfg,local,cases,controls,cache=preflight()
    import numpy as np
    from PIL import Image
    from reference_inpaint import reconstruct,torch,StableDiffusionXLInpaintPipeline,DDIMScheduler
    import run_focused_eye_appearance_v1 as appearance
    from run_focused_eye_guidance_v1 import GuidanceProxy
    from run_focused_eye_freeu_v1 import idle_gpu
    from sdxl_token_compatibility_probe_v1 import CompatibilityScorer,TokenCompatibilityProcessor,install
    import diffusers.models.attention_processor as processor_source
    require(sha(processor_source.__file__)==local['attention_implementation_sha256'],'Inference attention changed')
    states={a:load_state(torch,trainings[a]['checkpoints']['last']['path']) for a in ['epsilon','ssim']}
    initial=load_state(torch,trainings['epsilon']['checkpoints']['initial']['path'])
    other=load_state(torch,trainings['ssim']['checkpoints']['initial']['path'])
    require(all(torch.equal(initial[k],other[k]) for k in initial),'Initial parameter states differ')
    scorer=CompatibilityScorer().float().cpu().eval();scorer.load_state_dict(initial,strict=True)
    masks={};banks={}
    for ident in IDS:
        with Image.open(cases[ident]['mask']) as image:masks[ident]=np.asarray(image.convert('L'))>=128
        require(masks[ident].shape==(512,512) and masks[ident].any() and not masks[ident].all(),'Supplied mask changed')
        banks[ident]=torch.load(cfg['appearance_features'][ident]['path'],map_location='cpu',weights_only=True)
        require(tuple(banks[ident].shape)==(2,4,257,1280) and torch.isfinite(banks[ident]).all(),'Reference bank changed')
    if args.preflight_only:
        print(json.dumps(dict(status='passed',cuda_used=False,updates_per_arm=32,parameters=385,expected=15),indent=2));return
    require(not OUT.exists(),'Refusing prior evaluation overwrite');idle_gpu();OUT.mkdir(exist_ok=False)
    cfg=dict(cfg,conditions=ARMS,expected=15,seeds=[17],selected_identities=IDS,locations=['eyes'],training=True,
        training_updates=32,base_training_updates=64,training_resolution=512,trainable_parameters=385,
        evaluation_resolution=1024,export_resolution=512,training_ids=old.TRAIN_IDS,
        training_arms={a:{k:v for k,v in t.items() if k not in ['config','receipt']} for a,t in trainings.items()},
        protocol_path=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),runner_sha256=sha(__file__),
        scorer_wrapper_sha256=sha(ROOT/'scripts'/WRAPPERS[0]),report_wrapper_sha256=sha(ROOT/'scripts'/WRAPPERS[1]),
        compatibility_source_sha256=sha(ROOT/'scripts/sdxl_token_compatibility_probe_v1.py'),
        single_change='Added 0.1 fully-contained masked structural loss against an equally trained epsilon-only continuation',
        generation_reads_evaluation_original=False,inference_versions={k:version(k) for k in ['torch','diffusers','transformers','safetensors']},
        frozen_at_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'config.json',cfg);rows=[]
    for c in controls:
        dest=OUT/f"{c['identity']}_eyes_{c['arm']}_17.png"
        shutil.copy2(c['output'],dest);shutil.copy2(c['raw'],dest.with_name(dest.stem+'_raw.png'))
        row=dict(c['row'],condition=c['arm'],file=dest.name,reused=True,attempted_generation=False,reference_condition='clean',
            source_summary=str(c['summary_path']),source_summary_sha256=c['summary_sha256'])
        rows.append(row);scores=OUT/'scores';scores.mkdir(exist_ok=True);save(scores/(dest.name+'.json'),row)
    save(OUT/'rows.json',rows)
    receipt=dict(status='running',expected=15,reused_controls=9,new_candidates=6,neutral_calls=1,candidate_attempts=0,
        attempts_by_arm={'epsilon':0,'ssim':0},neutral_replay_passed=False,evaluation_original_read=False,
        protocol_sha256=cfg['protocol_sha256'],config_sha256=sha(OUT/'config.json'))
    save(OUT/'generation_receipt.json',receipt)
    pipe=runtime=proxy=None;hook=None;start=time.monotonic()
    try:
        torch.set_num_threads(4)
        pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(cache/'osor_models/sdxl'),
            torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter([str(cache/'reference_models'),str(appearance.ASSETS/'sdxl_models')],
            subfolder=['',''],weight_name=['ip-adapter-faceid-portrait_sdxl.bin','ip-adapter-plus-face_sdxl_vit-h.safetensors'],
            image_encoder_folder=None,local_files_only=True)
        projections=pipe.unet.encoder_hid_proj.image_projection_layers
        require([int(projections[0].num_tokens),int(projections[1].latents.shape[1])]==[16,16],'Projection tokens changed')
        scorer.enabled=True;cfg['wrapped_processors']=install(pipe.unet,scorer,masks['263'],64)
        require(cfg['wrapped_processors']==70,'Processor count changed')
        timestep_log=[]
        def actual_timestep(module,positional,keyword):
            t=keyword.get('timestep',positional[1] if len(positional)>1 else None)
            require(t is not None,'UNet timestep missing')
            values=torch.as_tensor(t).detach().float().cpu().reshape(-1)
            require(torch.isfinite(values).all() and (values==values[0]).all(),'Mixed/nonfinite timestep')
            value=float(values[0]);require(0<=value<=999,'Schedule changed')
            scorer.timestep_fraction=value/999.;timestep_log.append(value)
        hook=pipe.unet.register_forward_pre_hook(actual_timestep,with_kwargs=True)
        for module in [pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2]:module.requires_grad_(False)
        pipe.enable_model_cpu_offload()
        cfg['scheduler']=dict(pipe.scheduler.config);save(OUT/'config.json',cfg)
        appearance.OUT=OUT;proxy=appearance.AppearanceProxy(GuidanceProxy(pipe,3.));proxy.appearance_scale=.2
        runtime={'pipeline':proxy,'memory':'model'}
        def run(ident,destination):
            scorer.visible_fraction=float((~masks[ident]).mean())
            for processor in pipe.unet.attn_processors.values():
                if isinstance(processor,TokenCompatibilityProcessor):
                    processor.target_mask.copy_(torch.as_tensor(masks[ident],device=processor.target_mask.device,dtype=torch.bool))
            proxy.appearance=banks[ident];timestep_log.clear();c=cases[ident]
            meta=reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],
                destination,seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
            require(len(timestep_log)==30 and all(p.dtype==torch.float32 for p in scorer.parameters()),'Inference contract changed')
            return meta
        replay=OUT/'replay/263_initial_enabled.png';run('263',replay)
        reference=OUT/'263_eyes_prior_17.png';checks={}
        for suffix in ['.png','_raw.png']:
            new=replay.with_name(replay.stem+suffix);control=reference.with_name(reference.stem+suffix)
            with Image.open(new) as a,Image.open(control) as b:pixel_exact=np.array_equal(np.asarray(a),np.asarray(b))
            checks[suffix]=dict(byte_exact=sha(new)==sha(control),pixel_exact=pixel_exact,new_sha256=sha(new),control_sha256=sha(control))
        neutral=dict(passed=all(x['byte_exact'] and x['pixel_exact'] for x in checks.values()),checks=checks,
            identity='263',seed=17,scorer_enabled=True,initial_checkpoint_sha256=START_SHA,timesteps=list(timestep_log))
        save(OUT/'neutral_replay.json',neutral);receipt['neutral_replay_passed']=neutral['passed']
        save(OUT/'generation_receipt.json',receipt);require(neutral['passed'],'Initial enabled replay failed; candidate generation blocked')
        for arm in ['epsilon','ssim']:
            scorer.load_state_dict(states[arm],strict=True)
            for ident in IDS:
                idle_gpu();destination=OUT/f'{ident}_eyes_{arm}_17.png';case_start=time.monotonic()
                row=dict(identity=ident,location='eyes',condition=arm,reference_condition='clean',seed=17,file=destination.name,
                    reused=False,attempted_generation=True,checkpoint_sha256=trainings[arm]['checkpoints']['last']['sha256'])
                receipt['candidate_attempts']+=1;receipt['attempts_by_arm'][arm]+=1
                try:
                    meta=run(ident,destination)
                    row.update(status='complete',sha256=sha(destination),raw_sha256=meta['raw_output_sha256'],
                        peak_allocated_bytes=meta['peak_allocated_bytes'],timesteps=list(timestep_log))
                except Exception as exc:row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                row['seconds']=time.monotonic()-case_start;rows.append(row)
                save(OUT/'rows.json',rows);save(OUT/'generation_receipt.json',receipt)
                print(arm,ident,row['status'],flush=True)
        receipt['status']='complete' if all(r['status']=='complete' for r in rows) else 'failed'
    except Exception as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        present={(r['identity'],r['condition']) for r in rows}
        for arm in ['epsilon','ssim']:
            for ident in IDS:
                if (ident,arm) not in present:
                    rows.append(dict(identity=ident,location='eyes',condition=arm,seed=17,file=f'{ident}_eyes_{arm}_17.png',
                        status='blocked',attempted_generation=False,reused=False,error=receipt['error']))
        save(OUT/'rows.json',rows)
    finally:
        if hook is not None:hook.remove()
        receipt.update(elapsed_seconds=time.monotonic()-start,config_sha256=sha(OUT/'config.json'))
        save(OUT/'generation_receipt.json',receipt);del runtime,proxy,pipe,scorer;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not args.skip_scoring:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
        for name in WRAPPERS:
            subprocess.run([str(cache/'evaluation_env_v1/Scripts/python.exe'),str(ROOT/'scripts'/name)],env=env,check=True)
    require(receipt['status']=='complete','Generation failures retained; no candidate promotion')


if __name__=='__main__':main()
