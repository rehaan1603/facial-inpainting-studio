"""Oracle-removal feasibility and null-reference controls; no learned policy."""
import json
import time
import shutil
import torch
from PIL import Image
from reference_intervention_core import ROOT, load_model, sha
from train_target_compatibility_v1 import load_bundles

OUT = ROOT/'outputs/reference_rejection_v1'
IDENTITIES = ['263','6681','7400','10149']
SEEDS = [17,29,43]


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    dest=OUT/'evaluation'
    dest.mkdir()
    original=ROOT/'outputs/identity_compatibility_v2/evaluation'
    previous=json.loads((original/'rows.json').read_text())
    controls=[r for r in previous if r['arm']=='baseline']
    assert len(controls)==24
    rows=[]
    for control in controls:
        assert control['status']=='complete' and sha(original/control['file'])==control['sha256']
        shutil.copyfile(original/control['file'],dest/control['file'])
        rows.append(dict(control,reused_from=str((original/control['file']).relative_to(ROOT))))
    config=dict(identities=IDENTITIES,seeds=SEEDS,steps=50,guidance=1.5,precision='bfloat16',
        arms=['drop_first','zero_references'],conditions=['clean','wrong_patch'],
        scope='Oracle-removal feasibility only; no learned corruption detector',
        script_sha256=sha(__file__),preflight_sha256=sha(ROOT/'research/REFERENCE_REJECTION_PROBE_V1_PREFLIGHT.md'),
        controls_sha256=sha(original/'rows.json'),bundle_loader_sha256=sha(ROOT/'scripts/train_target_compatibility_v1.py'))
    write(OUT/'config.json',config)
    write(dest/'rows.json',rows)
    torch.set_num_threads(4)
    model=load_model().cuda()
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm import cache_kv
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    replays=[]
    for bundle in load_bundles():
        identity=bundle['identity']
        if identity not in IDENTITIES:continue
        lq=bundle['old']['lq'].cuda()
        jobs=[('replay',17,c) for c in ['clean','wrong_patch']]+[(a,s,'independent_of_first_reference') for s in SEEDS for a in config['arms']]
        for arm,seed,condition in jobs:
            cache_kv.clear_cache()
            cache_kv.mode='save'
            refs=bundle['old']['references'].clone().cuda()
            if arm=='replay':refs[:1]=bundle['new']['variants'][condition].cuda()
            elif arm=='drop_first':refs=refs[1:]
            else:refs.zero_()
            cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1))
            null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
            torch.manual_seed(seed)
            noise=torch.randn(1,8,64,64,device='cuda')
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            row=dict(identity=identity,arm=arm,seed=seed,condition=condition)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,
                        unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
                    rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                assert torch.isfinite(rgb).all()
                file=dest/f'{identity}_{arm}_{seed}_{condition}.png'
                array=(rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')
                Image.fromarray(array).save(file)
                row.update(status='complete',file=file.name,sha256=sha(file),seconds=time.monotonic()-start,
                           peak_mib=torch.cuda.max_memory_allocated()/2**20)
                if arm=='replay':
                    expected=next(r for r in controls if r['identity']==identity and r['seed']==seed and r['condition']==condition)
                    row['replay_identical']=row['sha256']==expected['sha256']
                    assert row['replay_identical'],'Baseline replay mismatch'
                    replays.append(row)
                    write(OUT/'replays.json',replays)
                else:
                    rows.append(row)
                    write(dest/'rows.json',rows)
                print('COMPLETE',identity,arm,seed,condition,flush=True)
            except Exception as error:
                row.update(status='failed',error=repr(error))
                write(OUT/'failure.json',row)
                raise
    assert len(rows)==48 and len(replays)==8 and all(r['replay_identical'] for r in replays)
    write(OUT/'completion.json',dict(status='complete',new_intervention_images=24,reused_controls=24,replay_calls=8,
        rows_sha256=sha(dest/'rows.json'),config_sha256=sha(OUT/'config.json')))
    print('ALL REJECTION PROBE GENERATIONS COMPLETE',flush=True)


if __name__=='__main__':main()
