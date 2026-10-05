"""Fresh crop candidate plus hash-verified matched prior controls, all fixed seeds."""
import json,time,shutil
import numpy as np
from PIL import Image
import torch
from reference_intervention_core import ROOT,load_model,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles
from train_identity_crop_v3 import OUT

def main():
    cfg=json.loads((OUT/'config.json').read_text());tr=json.loads((OUT/'training.json').read_text())
    assert tr['config_sha256']==sha(OUT/'config.json') and tr['arms'][0]['steps']==64
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False);rows=[]
    prior=ROOT/'outputs/identity_compatibility_v2';oldcfg=json.loads((prior/'config.json').read_text())
    assert oldcfg['evaluation']==cfg['evaluation']
    old=json.loads((prior/'evaluation/rows.json').read_text())
    for record in old:
        if record['arm'] not in ['baseline','regional_v1','identity']:continue
        r=dict(record);r['arm']='identity_full' if r['arm']=='identity' else r['arm']
        r['reused_from']='outputs/identity_compatibility_v2/evaluation/'+record['file']
        source=prior/'evaluation'/record['file'];assert sha(source)==record['sha256']
        r['file']=f'{r["identity"]}_{r["arm"]}_{r["seed"]}_{r["condition"]}.png'
        shutil.copyfile(source,dest/r['file']);rows.append(r)
    assert len(rows)==72
    (dest/'rows.json').write_text(json.dumps(rows,indent=2))
    (dest/'provenance.json').write_text(json.dumps(dict(script_sha256=sha(__file__),config_sha256=sha(OUT/'config.json'),
        source_rows_sha256=sha(prior/'evaluation/rows.json'),reused=72,fresh_expected=24),indent=2))
    torch.set_num_threads(4);model=load_model().cuda();adapter=Compatibility().cuda();install(model,adapter)
    p=OUT/'identity_crop.pt';assert sha(p)==tr['arms'][0]['checkpoint_sha256'];adapter.load_state_dict(torch.load(p,weights_only=True))
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    for r in load_bundles():
        if r['identity'] not in cfg['evaluation']['identities']:continue
        lq=r['old']['lq'].cuda()
        for seed in cfg['evaluation']['seeds']:
            for condition in cfg['evaluation']['conditions']:
                refs=r['old']['references'].clone().cuda();refs[:1]=r['new']['variants'][condition].cuda()
                adapter.context=features(lq,refs,torch.tensor(r['diagnostics']['conditions'][condition],device='cuda'))
                cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1));null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
                torch.manual_seed(seed);noise=torch.randn(1,8,64,64,device='cuda');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                row=dict(identity=r['identity'],arm='identity_crop',seed=seed,condition=condition)
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
                        rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                    assert torch.isfinite(rgb).all()
                    array=(rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')
                    p=dest/f'{r["identity"]}_identity_crop_{seed}_{condition}.png';Image.fromarray(array).save(p)
                    row.update(status='complete',file=p.name,sha256=sha(p),seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)
                except Exception as error:row.update(status='failed',error=repr(error))
                rows.append(row);(dest/'rows.json').write_text(json.dumps(rows,indent=2))
                print('GENERATED',len(rows)-72,row['identity'],seed,condition,row['status'],flush=True)
    assert len(rows)==96

if __name__=='__main__':main()
