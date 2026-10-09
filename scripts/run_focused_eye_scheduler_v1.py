"""One sampler change: cached native Euler versus the fixed DDIM5 control."""
import json,time
from pathlib import Path
from reference_inpaint import ROOT,CACHE,sha,reconstruct,torch,StableDiffusionXLInpaintPipeline

def main():
    out=ROOT/'outputs/focused_eye_scheduler_v1';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'outputs/masked_sdxl_resolution_v1/config.json').read_text())
    scheduler_path=CACHE/'osor_models/sdxl/scheduler/scheduler_config.json'
    cfg.update(expected=3,selected_identities=['263','6681','10149'],locations=['eyes'],blend='poisson',
        source_sha256=sha(ROOT/'scripts/reference_inpaint.py'),runner_sha256=sha(__file__),
        scheduler_config=json.loads(scheduler_path.read_text()),scheduler_config_sha256=sha(scheduler_path),
        single_change='Original checkpoint Euler scheduler instead of DDIM; guidance remains5',training=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    for ident in cfg['selected_identities']:
        c=cases[ident]
        for k in ['input','mask']:assert sha(c[k])==c[k+'_sha256']
        for r in cfg['references'][ident+'/clean']:assert sha(r['path'])==r['sha256']
    torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    assert type(pipe.scheduler).__name__=='EulerDiscreteScheduler'
    pipe.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload();runtime={'pipeline':pipe,'memory':'model'};rows=[]
    for ident in cfg['selected_identities']:
        c=cases[ident];p=out/f'{ident}_eyes_clean_17.png';start=time.monotonic()
        row=dict(identity=ident,location='eyes',condition='clean',seed=17,file=p.name,scheduler='EulerDiscreteScheduler')
        try:
            meta=reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],p,
                seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
            row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:
            row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start);rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2));raise
        rows.append(row);(out/'rows.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',ident,flush=True)

if __name__=='__main__':main()
