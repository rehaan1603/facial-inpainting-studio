"""Three fixed seeds and all four observed identities; no best-seed selection."""
import json,time
import numpy as np
from PIL import Image
import torch
from reference_intervention_core import ROOT,load_model,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles
from train_identity_compatibility_v2 import OUT

def main():
    cfg=json.loads((OUT/'config.json').read_text());tr=json.loads((OUT/'training.json').read_text())
    assert len(tr['arms'])==2 and tr['config_sha256']==sha(OUT/'config.json')
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    (dest/'provenance.json').write_text(json.dumps(dict(config_sha256=sha(OUT/'config.json'),script_sha256=sha(__file__),training_sha256=sha(OUT/'training.json')),indent=2))
    torch.set_num_threads(4);model=load_model().cuda();adapter=Compatibility().cuda();install(model,adapter)
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing');rows=[]
    old=ROOT/'outputs/target_compatibility_v1'
    oldtraining=json.loads((old/'training/receipt.json').read_text())
    legacy=json.loads((old/'evaluation/rows.json').read_text())
    for r in load_bundles():
        if r['identity'] not in cfg['evaluation']['identities']:continue
        lq=r['old']['lq'].cuda()
        for arm in ['baseline','regional_v1','diffusion','identity']:
            adapter.enabled=arm!='baseline';adapter.mode='full'
            if adapter.enabled:
                path=old/'training/full.pt' if arm=='regional_v1' else OUT/(arm+'.pt')
                expected=next(a['sha256'] for a in oldtraining['arms'] if a['arm']=='full') if arm=='regional_v1' else next(a['checkpoint_sha256'] for a in tr['arms'] if a['arm']==arm)
                assert sha(path)==expected;adapter.load_state_dict(torch.load(path,weights_only=True))
            for seed in cfg['evaluation']['seeds']:
                for condition in cfg['evaluation']['conditions']:
                    refs=r['old']['references'].clone().cuda();refs[:1]=r['new']['variants'][condition].cuda()
                    adapter.context=features(lq,refs,torch.tensor(r['diagnostics']['conditions'][condition],device='cuda'))
                    cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1));null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
                    torch.manual_seed(seed);noise=torch.randn(1,8,64,64,device='cuda');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    row=dict(identity=r['identity'],arm=arm,seed=seed,condition=condition)
                    try:
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
                            rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                        assert torch.isfinite(rgb).all()
                        array=(rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')
                        path=dest/f'{r["identity"]}_{arm}_{seed}_{condition}.png';Image.fromarray(array).save(path)
                        row.update(status='complete',file=path.name,sha256=sha(path),seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)
                        if seed==17 and arm in ['baseline','regional_v1']:
                            prior=next(a for a in legacy if a['identity']==r['identity'] and a['arm']==('full' if arm=='regional_v1' else arm) and a['condition']==condition)
                            row['replay_exact']=row['sha256']==prior['sha256'];assert row['replay_exact']
                    except Exception as error:row.update(status='failed',error=repr(error))
                    rows.append(row);(dest/'rows.json').write_text(json.dumps(rows,indent=2))
                    print('GENERATED',len(rows),row['identity'],arm,seed,condition,row['status'],flush=True)
    assert len(rows)==96

if __name__=='__main__':main()
