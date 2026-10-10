"""Fixed denoising-budget check; no change to references, models or metrics."""
import gc,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from reference_inpaint import ROOT,CACHE,sha,reconstruct,torch,StableDiffusionXLInpaintPipeline,DDIMScheduler
from run_focused_eye_guidance_v1 import GuidanceProxy
from run_focused_eye_freeu_v1 import idle_gpu,copy_controls
import run_focused_eye_freeu_v1 as controls

OUT=ROOT/'outputs/focused_eye_steps_v1';IDS=['263','6681','10149']


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    idle_gpu()
    cfg=json.loads((ROOT/'outputs/focused_eye_guidance_replication_v2/config.json').read_text())
    assert sha(ROOT/'scripts/reference_inpaint.py')==cfg['generator_sha256']
    assert sha(ROOT/'scripts/score_masked_sdxl_v1.py')==cfg['scorer_sha256']
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    for ident in IDS:
        for k in ['input','mask']:assert sha(cases[ident][k])==cases[ident][k+'_sha256']
        for r in cfg['references'][ident+'/clean']:
            assert sha(r['path'])==r['sha256'] and r['sha256']!=cases[ident]['target_sha256']
    cfg.update(expected=9,conditions=['control5','control3','steps50'],seeds=[17],steps=50,guidance=3.,training=False,
        single_change='DDIM30 to50 at otherwise unchanged CFG3; no appearance or FreeU',
        runner_sha256=sha(__file__),control_copy_helper_sha256=sha(ROOT/'scripts/run_focused_eye_freeu_v1.py'),
        protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_STEPS_PREFLIGHT_V1.md'),trainable_parameters=0)
    OUT.mkdir(exist_ok=False);rows=[];controls.OUT=OUT;copy_controls(cfg,[17],rows);save(OUT/'config.json',cfg)
    torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload();cfg['scheduler']=dict(pipe.scheduler.config);save(OUT/'config.json',cfg)
    for m in [pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2]:m.requires_grad_(False)
    proxy=GuidanceProxy(pipe,3.);runtime={'pipeline':proxy,'memory':'model'}
    def run(ident,output,steps):
        c=cases[ident]
        return reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],output,
            seed=17,steps=steps,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
    p=OUT/'replay/263_control3.png';run('263',p,30);replay={}
    for suffix in ['.png','_raw.png']:
        new=p.with_name(p.stem+suffix);old=OUT/f'263_eyes_control3_17{suffix}'
        replay[suffix]=dict(pixel_exact=np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),new_sha256=sha(new),control_sha256=sha(old))
    save(OUT/'replay.json',replay);assert all(v['pixel_exact'] for v in replay.values())
    for ident in IDS:
        idle_gpu();p=OUT/f'{ident}_eyes_steps50_17.png';start=time.monotonic()
        row=dict(identity=ident,location='eyes',condition='steps50',seed=17,file=p.name,reused=False)
        try:
            meta=run(ident,p,50)
            row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
        rows.append(row);save(OUT/'rows.json',rows);print('STEPS50',ident,row['status'],flush=True)
    del runtime,proxy,pipe;gc.collect();torch.cuda.empty_cache()
    subprocess.run([str(CACHE/'evaluation_env_v1/Scripts/python.exe'),str(ROOT/'scripts/score_focused_eye_steps_v1.py')],check=True)


if __name__=='__main__':main()
