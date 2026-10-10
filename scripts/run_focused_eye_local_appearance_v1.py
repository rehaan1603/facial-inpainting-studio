"""Isolated hole-local appearance routing; preserved joint-reference attention."""
import gc
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import run_focused_eye_appearance_v1 as appearance
from reference_inpaint import ROOT, CACHE, sha, reconstruct, torch, StableDiffusionXLInpaintPipeline, DDIMScheduler
from run_focused_eye_guidance_v1 import GuidanceProxy
from run_focused_eye_freeu_v1 import idle_gpu
from local_appearance_processor_v1 import install, set_mask, LocalAppearanceProcessor

OUT = ROOT / 'outputs/focused_eye_local_appearance_v1'
IDS = ['263','6681','10149']


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    idle_gpu()
    cfg=json.loads((ROOT/'outputs/focused_eye_factorial_v1/config.json').read_text())
    assert sha(ROOT/'scripts/reference_inpaint.py')==cfg['source_sha256']
    assert sha(ROOT/'scripts/score_masked_sdxl_v1.py')==cfg['scorer_sha256']
    assert sha(ROOT/'scripts/run_focused_eye_appearance_v1.py')==cfg['appearance_wrapper_sha256']
    assert sha(ROOT/'scripts/run_focused_eye_guidance_v1.py')==cfg['guidance_wrapper_sha256']
    import diffusers.models.attention_processor as processor_source
    cfg.update(expected=12,conditions=['control3','control5','global','local'],seeds=[17],selected_identities=IDS,
        runner_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/local_appearance_processor_v1.py'),
        protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_LOCAL_APPEARANCE_PREFLIGHT_V1.md'),
        attention_implementation_sha256=sha(processor_source.__file__), attention_implementation_path=processor_source.__file__,
        single_change='Route existing aggregate appearance attention by supplied hole query mask; joint-reference softmax unchanged',
        trainable_parameters=0,training=False,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        mask_downsample='nearest binary to square query grid; no dilation or feather',guidance=3.,appearance_scale=.2)
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    banks={}
    for ident in IDS:
        c=cases[ident]
        for key in ['input','mask']:assert sha(c[key])==c[key+'_sha256']
        for r in cfg['references'][ident+'/clean']:
            assert sha(r['path'])==r['sha256'] and r['sha256']!=c['target_sha256']
        bank=cfg['appearance_features'][ident]
        assert sha(bank['path'])==bank['sha256']
        banks[ident]=torch.load(bank['path'],map_location='cpu',weights_only=True)
        assert tuple(banks[ident].shape)==(2,4,257,1280) and torch.isfinite(banks[ident]).all()
    # Verify the existing official appearance assets, without downloading or updating.
    for asset in cfg['appearance_assets']['files']:
        assert sha(asset['path'])==asset['sha256']
    assert sha(CACHE/'reference_models/ip-adapter-faceid-portrait_sdxl.bin')==cfg['adapter_sha256']
    OUT.mkdir(exist_ok=False)
    save(OUT/'config.json',cfg)
    rows=[]
    for arm,folder in [('control3','focused_eye_guidance_v1'),('control5','focused_reference_finish_v1'),('global','focused_eye_factorial_v1')]:
        origin=ROOT/'outputs'/folder
        lookup={r['identity']:r for r in json.loads((origin/'summary.json').read_text())['rows'] if r['location']=='eyes' and r['identity'] in IDS}
        for ident in IDS:
            r=lookup[ident];assert r['status']=='complete' and r['seed']==17
            source=origin/r['file'];raw=source.with_name(source.stem+'_raw.png')
            assert sha(source)==r['sha256'] and sha(raw)==r['raw_sha256']
            dest=OUT/f'{ident}_eyes_{arm}_17.png'
            shutil.copy2(source,dest);shutil.copy2(raw,dest.with_name(dest.stem+'_raw.png'))
            copied=dict(r,condition=arm,file=dest.name,reused=True,source=str(source))
            rows.append(copied)
            score=OUT/'scores'/(dest.name+'.json');score.parent.mkdir(exist_ok=True);save(score,copied)
    save(OUT/'rows.json',rows)
    torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter([str(CACHE/'reference_models'),str(appearance.ASSETS/'sdxl_models')],subfolder=['',''],
        weight_name=['ip-adapter-faceid-portrait_sdxl.bin','ip-adapter-plus-face_sdxl_vit-h.safetensors'],
        image_encoder_folder=None,local_files_only=True)
    cfg['wrapped_processors']=install(pipe,np.ones((512,512),bool))
    pipe.enable_model_cpu_offload()
    cfg['scheduler']=dict(pipe.scheduler.config)
    for module in [pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2]:module.requires_grad_(False)
    assert not any(p.requires_grad for p in pipe.unet.parameters())
    save(OUT/'config.json',cfg)
    appearance.OUT=OUT
    proxy=appearance.AppearanceProxy(GuidanceProxy(pipe,3.))
    runtime={'pipeline':proxy,'memory':'model'}

    def run(ident,output):
        c=cases[ident];proxy.appearance=banks[ident]
        return reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],output,
            seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)

    replays={}
    for name,scale,control in [('full_mask',.2,'global'),('appearance_off',0.,'control3')]:
        proxy.appearance_scale=scale
        set_mask(pipe,np.ones((512,512),bool) if scale else np.asarray(Image.open(cases['263']['mask']))>127)
        p=OUT/'replay'/f'263_{name}.png';run('263',p)
        reference=OUT/f'263_eyes_{control}_17.png'
        replays[name]={}
        for suffix in ['.png','_raw.png']:
            new=p.with_name(p.stem+suffix);old=reference.with_name(reference.stem+suffix)
            replays[name][suffix]=dict(pixel_exact=np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),
                new_sha256=sha(new),control_sha256=sha(old))
        save(OUT/'replays.json',replays)
        assert all(v['pixel_exact'] for v in replays[name].values()),'Neutral replay failed; stop before candidates'
    proxy.appearance_scale=.2
    for ident in IDS:
        idle_gpu();set_mask(pipe,np.asarray(Image.open(cases[ident]['mask']))>127)
        output=OUT/f'{ident}_eyes_local_17.png';start=time.monotonic()
        row=dict(identity=ident,location='eyes',condition='local',reference_condition='clean',seed=17,file=output.name,reused=False)
        try:
            meta=run(ident,output)
            row.update(status='complete',sha256=sha(output),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,
                peak_allocated_bytes=meta['peak_allocated_bytes'])
            grids={name:processor.query_grids for name,processor in pipe.unet.attn_processors.items() if isinstance(processor,LocalAppearanceProcessor)}
            save(OUT/f'{ident}_query_grids.json',grids)
        except Exception as exc:row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
        rows.append(row);save(OUT/'rows.json',rows);print('LOCAL APPEARANCE',ident,row['status'],flush=True)
    del runtime,proxy,pipe
    gc.collect();torch.cuda.empty_cache()
    subprocess.run([str(CACHE/'evaluation_env_v1/Scripts/python.exe'),str(ROOT/'scripts/score_focused_eye_local_appearance_v1.py')],check=True)


if __name__=='__main__':main()
