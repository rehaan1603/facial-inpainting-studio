"""Frozen CFG-only replication; retain every result and use no evaluation pixels."""
import json, shutil, subprocess, time
from pathlib import Path
from reference_inpaint import ROOT, CACHE, sha, reconstruct, torch, StableDiffusionXLInpaintPipeline, DDIMScheduler
from run_focused_eye_guidance_v1 import GuidanceProxy

OUT = ROOT / 'outputs/focused_eye_guidance_replication_v2'
IDS = ['263', '6681', '10149']

def main():
    OUT.mkdir(exist_ok=False)
    cfg = json.loads((ROOT / 'outputs/focused_eye_guidance_v1/config.json').read_text())
    cfg.update(expected=18, conditions=['control', 'candidate'], reference_condition='clean',
               selected_identities=IDS, seeds=[17, 29, 43], new_seeds=[29, 43],
               arms={'control': 5., 'candidate': 3.},
               single_change='Guidance 5 versus 3; no per-person or seed selection',
               runner_sha256=sha(__file__), generator_sha256=sha(ROOT/'scripts/reference_inpaint.py'),
               proxy_sha256=sha(ROOT/'scripts/run_focused_eye_guidance_v1.py'),
               protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_GUIDANCE_REPLICATION_PREFLIGHT_V2.md'),
               scorer_sha256=sha(ROOT/'scripts/score_masked_sdxl_v1.py'), training=False)
    cases = {c['identity']: c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json') == cfg['cases_sha256']
    for ident in IDS:
        for k in ['input','mask']: assert sha(cases[ident][k]) == cases[ident][k+'_sha256']
        for r in cfg['references'][ident+'/clean']: assert sha(r['path']) == r['sha256']
    (OUT/'config.json').write_text(json.dumps(cfg, indent=2))
    rows=[]
    for arm, folder in [('control','focused_reference_finish_v1'),('candidate','focused_eye_guidance_v1')]:
        old = {r['identity']: r for r in json.loads((ROOT/'outputs'/folder/'summary.json').read_text())['rows'] if r['location']=='eyes' and r['identity'] in IDS}
        for ident in IDS:
            r=old[ident]; assert r['status']=='complete' and r['seed']==17
            source=ROOT/'outputs'/folder/r['file']; assert sha(source)==r['sha256']
            raw=source.with_name(source.stem+'_raw.png'); assert sha(raw)==r['raw_sha256']
            dest=OUT/f'{ident}_eyes_{arm}_17.png'
            shutil.copy2(source,dest); shutil.copy2(raw,dest.with_name(dest.stem+'_raw.png'))
            rows.append(dict(identity=ident,location='eyes',condition=arm,reference_condition='clean',seed=17,file=dest.name,
                             guidance=cfg['arms'][arm],status='complete',sha256=sha(dest),raw_sha256=sha(raw),
                             reused=True,source=str(source)))
    (OUT/'rows.json').write_text(json.dumps(rows,indent=2))
    torch.set_num_threads(4)
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload()
    cfg['scheduler']=dict(pipe.scheduler.config)
    cfg['adapter_sha256']=sha(CACHE/'reference_models/ip-adapter-faceid-portrait_sdxl.bin')
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    proxy=GuidanceProxy(pipe,5.); runtime={'pipeline':proxy,'memory':'model'}
    for seed in cfg['new_seeds']:
      for ident in IDS:
        for arm,guidance in cfg['arms'].items():
            proxy.guidance=guidance; c=cases[ident]; p=OUT/f'{ident}_eyes_{arm}_{seed}.png'; start=time.monotonic()
            row=dict(identity=ident,location='eyes',condition=arm,reference_condition='clean',seed=seed,file=p.name,guidance=guidance,reused=False)
            try:
                meta=reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],p,
                                 seed=seed,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
                row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,
                           peak_allocated_bytes=meta['peak_allocated_bytes'])
            except Exception as exc:
                row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
            rows.append(row); (OUT/'rows.json').write_text(json.dumps(rows,indent=2))
            print('REPLICATION',len(rows),'/18',ident,seed,arm,row['status'],flush=True)
    # Free inference memory before scoring with the existing independent environment.
    del runtime, proxy, pipe
    torch.cuda.empty_cache()
    subprocess.run([str(CACHE/'evaluation_env_v1/Scripts/python.exe'),str(ROOT/'scripts/score_focused_eye_guidance_replication_v2.py')],check=True)

if __name__=='__main__': main()
