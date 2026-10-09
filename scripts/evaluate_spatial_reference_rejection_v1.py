"""Fixed oracle-region and equal-area corner controls; no training or tuning."""
import json
import shutil
import time
import torch
from PIL import Image
from reference_intervention_core import ROOT, load_model, sha
from train_target_compatibility_v1 import load_bundles
from spatial_reference_rejection_v1 import SpatialRejection, install, region_box

OUT = ROOT/'outputs/spatial_reference_rejection_v1'
IDS = ['263','6681','7400','10149']
SEEDS = [17,29,43]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    dest = OUT/'evaluation'; dest.mkdir()
    original = ROOT/'outputs/identity_compatibility_v2/evaluation'
    controls = [r for r in json.loads((original/'rows.json').read_text()) if r['arm']=='baseline']
    assert len(controls)==24
    rows=[]
    for r in controls:
        assert r['status']=='complete' and sha(original/r['file'])==r['sha256']
        shutil.copyfile(original/r['file'], dest/r['file'])
        rows.append(dict(r, reused_from=str((original/r['file']).relative_to(ROOT))))
    cfg=dict(identities=IDS, seeds=SEEDS, conditions=['clean','wrong_patch'], arms=['oracle_region','random_corner'],
        boxes={i:{a:region_box(a,i) for a in ['oracle_region','random_corner']} for i in IDS},
        steps=50,guidance=1.5,precision='bfloat16',reference_encoding='separate latents',
        script_sha256=sha(__file__),mechanism_sha256=sha(ROOT/'scripts/spatial_reference_rejection_v1.py'),
        preflight_sha256=sha(ROOT/'research/SPATIAL_REFERENCE_REJECTION_PREFLIGHT_V1.md'))
    write(OUT/'config.json',cfg);write(dest/'rows.json',rows)
    torch.set_num_threads(4)
    model=load_model().cuda();intervention=SpatialRejection();layers=install(model,intervention)
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm import cache_kv
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing');replays=[]
    for bundle in load_bundles():
        identity=bundle['identity']
        if identity not in IDS:continue
        lq=bundle['old']['lq'].cuda()
        jobs=[('neutral',17,c) for c in cfg['conditions']]+[(a,s,c) for s in SEEDS for c in cfg['conditions'] for a in cfg['arms']]
        for arm,seed,condition in jobs:
            cache_kv.clear_cache();cache_kv.mode='save'
            intervention.box=region_box(arm,identity);intervention.calls=0;intervention.token_counts=set()
            refs=bundle['old']['references'].clone().cuda();refs[:1]=bundle['new']['variants'][condition].cuda()
            cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1))
            null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
            torch.manual_seed(seed);noise=torch.randn(1,8,64,64,device='cuda')
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            row=dict(identity=identity,arm=arm,seed=seed,condition=condition,box=intervention.box)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,
                        unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
                    rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                assert torch.isfinite(rgb).all()
                if arm!='neutral':assert intervention.calls>0
                file=dest/f'{identity}_{arm}_{seed}_{condition}.png'
                Image.fromarray((rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')).save(file)
                row.update(status='complete',file=file.name,sha256=sha(file),seconds=time.monotonic()-start,
                    peak_mib=torch.cuda.max_memory_allocated()/2**20,attention_calls=intervention.calls,
                    token_counts=sorted(intervention.token_counts))
                if arm=='neutral':
                    expected=next(r for r in controls if r['identity']==identity and r['seed']==seed and r['condition']==condition)
                    row['replay_identical']=row['sha256']==expected['sha256'];assert row['replay_identical'],'Neutral replay mismatch'
                    replays.append(row);write(OUT/'replays.json',replays)
                else:rows.append(row);write(dest/'rows.json',rows)
                print('COMPLETE',identity,arm,seed,condition,flush=True)
            except Exception as error:
                row.update(status='failed',error=repr(error));write(OUT/'failure.json',row);raise
    assert len(rows)==72 and len(replays)==8 and all(r['replay_identical'] for r in replays)
    write(OUT/'completion.json',dict(status='complete',new_images=48,reused_controls=24,replay_calls=8,
        layers=layers,rows_sha256=sha(dest/'rows.json'),config_sha256=sha(OUT/'config.json')))


if __name__=='__main__':main()
