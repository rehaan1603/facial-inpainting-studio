"""Replay enabled initial compatibility processor before trained candidates."""
import gc
import json
import time
from pathlib import Path
import run_masked_sdxl_compatibility_pilot_eval_v2 as runner

ROOT=runner.ROOT
OUT=ROOT/'outputs/masked_sdxl_zero_enabled_replay_v1'
PROTOCOL=ROOT/'research/MASKED_SDXL_ZERO_STATE_REPLAY_PREFLIGHT_V1.md'

def main():
    runner.require(not OUT.exists(),'Refusing replay overwrite')
    tcfg,tr,cfg,local,cases,controls,cache=runner.preflight(runner.TRAIN,runner.PROTOCOL)
    import numpy as np
    from PIL import Image
    from reference_inpaint import reconstruct,torch,StableDiffusionXLInpaintPipeline,DDIMScheduler
    from run_focused_eye_guidance_v1 import GuidanceProxy
    from run_focused_eye_freeu_v1 import idle_gpu
    import run_focused_eye_appearance_v1 as appearance
    from sdxl_token_compatibility_probe_v1 import CompatibilityScorer,TokenCompatibilityProcessor,install
    idle_gpu();OUT.mkdir()
    receipt=dict(status='running',protocol_sha256=runner.sha(PROTOCOL),script_sha256=runner.sha(__file__),
        training_config_sha256=runner.sha(runner.TRAIN/'config.json'),initial_sha256=tr['checkpoints']['initial']['sha256'],
        trained_checkpoint_used=False,scorer_enabled=True,generation_original_read=False,identity='263',seed=17,
        source_pins=tcfg['source_pins'],checks={})
    runner.save(OUT/'receipt.json',receipt);start=time.monotonic()
    pipe=runtime=proxy=None;hook=None
    try:
        state=runner.load_state(torch,runner.TRAIN/'adapter_initial.pt')
        runner.require(bool((state['net.2.weight']==0).all()) and bool((state['net.2.bias']==0).all()),'Initial state is not neutral')
        scorer=CompatibilityScorer().float().cpu().eval();scorer.load_state_dict(state);scorer.enabled=True
        mask=np.asarray(Image.open(cases['263']['mask']).convert('L'))>=128
        scorer.visible_fraction=float((~mask).mean())
        torch.set_num_threads(4)
        pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(cache/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter([str(cache/'reference_models'),str(appearance.ASSETS/'sdxl_models')],subfolder=['',''],
            weight_name=['ip-adapter-faceid-portrait_sdxl.bin','ip-adapter-plus-face_sdxl_vit-h.safetensors'],image_encoder_folder=None,local_files_only=True)
        runner.require(install(pipe.unet,scorer,mask,64)==70,'Processor count changed')
        timesteps=[]
        def actual_timestep(module,positional,keyword):
            ts=keyword.get('timestep',positional[1] if len(positional)>1 else None)
            values=torch.as_tensor(ts).detach().float().cpu().reshape(-1)
            runner.require(bool(torch.isfinite(values).all()) and bool((values==values[0]).all()),'Invalid timestep')
            value=float(values[0]);runner.require(0<=value<=999,'Invalid timestep')
            scorer.timestep_fraction=value/999.;timesteps.append(value)
        hook=pipe.unet.register_forward_pre_hook(actual_timestep,with_kwargs=True)
        for module in [pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2]:module.requires_grad_(False)
        pipe.enable_model_cpu_offload()
        appearance.OUT=OUT
        proxy=appearance.AppearanceProxy(GuidanceProxy(pipe,3.));proxy.appearance_scale=.2
        proxy.appearance=torch.load(cfg['appearance_features']['263']['path'],map_location='cpu',weights_only=True)
        runtime=dict(pipeline=proxy,memory='model')
        c=cases['263'];dest=OUT/'263_initial_enabled.png'
        reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references']['263/clean']],dest,
            seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
        runner.require(len(timesteps)==30 and all(p.dtype==torch.float32 for p in scorer.parameters()),'Inference contract changed')
        reference=next(r['output'] for r in controls if r['arm']=='local' and r['identity']=='263')
        for suffix in ['.png','_raw.png']:
            new=dest.with_name(dest.stem+suffix);old=reference.with_name(reference.stem+suffix)
            receipt['checks'][suffix]=dict(byte_exact=runner.sha(new)==runner.sha(old),
                pixel_exact=np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),new_sha256=runner.sha(new),control_sha256=runner.sha(old))
        receipt.update(status='passed' if all(v['byte_exact'] and v['pixel_exact'] for v in receipt['checks'].values()) else 'failed',
            timesteps=timesteps,visible_fraction=scorer.visible_fraction)
    except Exception as exc:receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        if hook is not None:hook.remove()
        receipt['elapsed_seconds']=time.monotonic()-start;runner.save(OUT/'receipt.json',receipt)
        del runtime,proxy,pipe;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print(json.dumps({k:receipt[k] for k in ['status','checks','elapsed_seconds']},indent=2),flush=True)
    runner.require(receipt['status']=='passed','Enabled zero-state raw/final replay failed; trained candidates blocked')

if __name__=='__main__':main()
