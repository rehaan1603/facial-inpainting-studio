"""Complete the fixed 2x2 interaction; no new encoder, scale or seed search."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import run_focused_eye_appearance_v1 as appearance
from reference_inpaint import ROOT,CACHE,sha,reconstruct,torch,StableDiffusionXLInpaintPipeline,DDIMScheduler
from run_focused_eye_guidance_v1 import GuidanceProxy

OUT=ROOT/'outputs/focused_eye_factorial_v1'
IDS=['263','6681','10149']

def main():
    cfg=json.loads((ROOT/'outputs/focused_eye_appearance_v1/config.json').read_text())
    assert cfg['guidance']==5. and cfg['appearance_scale']==.2
    cfg.update(guidance=3.,runner_sha256=sha(__file__),appearance_wrapper_sha256=sha(ROOT/'scripts/run_focused_eye_appearance_v1.py'),
               guidance_wrapper_sha256=sha(ROOT/'scripts/run_focused_eye_guidance_v1.py'),
               source_sha256=sha(ROOT/'scripts/reference_inpaint.py'),
               protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md'),
               single_change='Add fixed appearance0.2 to the existing CFG3 control; complete fixed 2x2 interaction',
               expected=3,training=False)
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    banks={}
    for ident in IDS:
        for k in ['input','mask']:assert sha(cases[ident][k])==cases[ident][k+'_sha256']
        for r in cfg['references'][ident+'/clean']:assert sha(r['path'])==r['sha256']
        r=cfg['appearance_features'][ident];assert sha(r['path'])==r['sha256']
        banks[ident]=torch.load(r['path'],map_location='cpu',weights_only=True)
    OUT.mkdir(exist_ok=False);(OUT/'config.json').write_text(json.dumps(cfg,indent=2));torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter([str(CACHE/'reference_models'),str(appearance.ASSETS/'sdxl_models')],subfolder=['',''],
                         weight_name=['ip-adapter-faceid-portrait_sdxl.bin','ip-adapter-plus-face_sdxl_vit-h.safetensors'],
                         image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload()
    appearance.OUT=OUT  # Route wrapper receipts to this fresh experiment, never the old one.
    proxy=appearance.AppearanceProxy(GuidanceProxy(pipe,3.));runtime={'pipeline':proxy,'memory':'model'}
    def run(ident,p):
        proxy.appearance=banks[ident];c=cases[ident]
        return reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],p,
                           seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
    p=OUT/'replay/263_eyes_clean_17.png';run('263',p);checks={}
    for suffix in ['.png','_raw.png']:
        new=p.with_name(p.stem+suffix);old=ROOT/'outputs/focused_eye_guidance_v1'/new.name
        checks[suffix]=dict(pixel_exact=np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),new_sha256=sha(new),control_sha256=sha(old))
    (OUT/'replay.json').write_text(json.dumps(checks,indent=2));assert all(v['pixel_exact'] for v in checks.values()),'Neutral CFG3 replay mismatch'
    proxy.appearance_scale=.2;rows=[]
    for ident in IDS:
        p=OUT/f'{ident}_eyes_clean_17.png';start=time.monotonic();row=dict(identity=ident,location='eyes',condition='clean',seed=17,file=p.name,guidance=3.,appearance_scale=.2)
        try:
            meta=run(ident,p);row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
        rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2));print('INTERACTION',ident,row['status'],flush=True)

if __name__=='__main__':main()
