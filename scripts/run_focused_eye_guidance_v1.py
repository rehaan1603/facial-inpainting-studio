"""Isolated CFG3 diagnostic using the unchanged reference generator and CFG5 replay."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
from reference_inpaint import (ROOT,CACHE,sha,reconstruct,torch,
                              StableDiffusionXLInpaintPipeline,DDIMScheduler)

OUT=ROOT/'outputs/focused_eye_guidance_v1'

class GuidanceProxy:
    """Delegate the existing pipeline unchanged except the explicit CFG argument."""
    def __init__(self,pipeline,guidance):
        self.pipeline=pipeline
        self.guidance=guidance
    def __getattr__(self,name):
        return getattr(self.pipeline,name)
    def __call__(self,*args,**kwargs):
        assert kwargs['guidance_scale']==5.0
        kwargs['guidance_scale']=self.guidance
        return self.pipeline(*args,**kwargs)

def main():
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'outputs/masked_sdxl_resolution_v1/config.json').read_text())
    cfg.update(expected=3,selected_identities=['263','6681','10149'],locations=['eyes'],
               guidance=3.,scale=.8,blend='poisson',
               source_sha256=sha(ROOT/'scripts/reference_inpaint.py'),
               runner_sha256=sha(__file__),
               single_change='CFG5 to3 versus1024 Poisson scale0.8; no other inference change',
               baseline='outputs/focused_reference_finish_v1',training=False)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    for ident in cfg['selected_identities']:
        c=cases[ident]
        for k in ['input','mask']:assert sha(c[k])==c[k+'_sha256']
        for r in cfg['references'][ident+'/clean']:assert sha(r['path'])==r['sha256']
    torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload()
    proxy=GuidanceProxy(pipe,5.)
    runtime={'pipeline':proxy,'memory':'model'}
    def run(ident,output):
        c=cases[ident]
        return reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],output,
            seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
    replay=OUT/'replay/263_eyes_clean_17.png';run('263',replay)
    checks={}
    for suffix,folder in [('.png','focused_reference_finish_v1'),('_raw.png','masked_sdxl_resolution_v1')]:
        new=replay.with_name(replay.stem+suffix)
        old=ROOT/'outputs'/folder/new.name
        checks[suffix]={'pixel_exact':np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),
                        'new_sha256':sha(new),'control_sha256':sha(old)}
    (OUT/'replay.json').write_text(json.dumps(checks,indent=2))
    assert all(v['pixel_exact'] for v in checks.values()),'Neutral replay mismatch; stop before candidates'
    proxy.guidance=3.;rows=[]
    for ident in cfg['selected_identities']:
        p=OUT/f'{ident}_eyes_clean_17.png';start=time.monotonic()
        row=dict(identity=ident,location='eyes',condition='clean',seed=17,file=p.name,guidance=3.)
        try:
            meta=run(ident,p)
            row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],
                       seconds=time.monotonic()-start,peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:
            row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
            rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2));raise
        rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',ident,flush=True)

if __name__=='__main__':main()
