"""Fixed post-training diagnostic, no target supplied to the sampler."""
import json
import time
import numpy as np
from PIL import Image
import torch
from reference_intervention_core import load_model,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles
from prepare_target_compatibility_v1 import OUT,BASE


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());training=json.loads((OUT/'training/receipt.json').read_text())
    assert len(training['arms'])==len(protocol['training_arms'])
    assert training['protocol_sha256']==sha(OUT/'protocol.json')
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    settings=dict(seed=17,steps=50,guidance=1.5,schedule='uniform_trailing',precision='bfloat16',
        protocol_sha256=sha(OUT/'protocol.json'),training_sha256=sha(OUT/'training/receipt.json'),script_sha256=sha(__file__),
        expected_outputs=96,final_test_used=False,new_dataset_used=False)
    (dest/'settings.json').write_text(json.dumps(settings,indent=2))
    torch.set_num_threads(4);model=load_model().cuda();adapter=Compatibility().cuda();install(model,adapter)
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    legacy=json.loads((BASE/'evaluation/rows.json').read_text());rows=[]
    for record in load_bundles():
        if record['role']!='adapter_holdout':continue
        identity=record['identity'];lq=record['old']['lq'].cuda()
        for arm in ['baseline',*protocol['training_arms']]:
            adapter.enabled=arm!='baseline';adapter.mode=arm
            if adapter.enabled:
                path=OUT/'training'/(arm+'.pt');assert sha(path)==next(r['sha256'] for r in training['arms'] if r['arm']==arm)
                adapter.load_state_dict(torch.load(path,weights_only=True))
            for condition in protocol['conditions']:
                refs=record['old']['references'].clone().cuda();refs[:1]=record['new']['variants'][condition].cuda()
                adapter.context=features(lq,refs,torch.tensor(record['diagnostics']['conditions'][condition],device='cuda'))
                cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1))
                null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
                torch.manual_seed(17);noise=torch.randn(1,8,64,64,device='cuda');start=time.monotonic()
                row=dict(identity=identity,arm=arm,condition=condition)
                try:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,
                            unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
                        rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                    assert torch.isfinite(rgb).all()
                    array=(rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')
                    path=dest/f'{identity}_{arm}_{condition}.png';Image.fromarray(array).save(path)
                    row.update(status='complete',file=path.name,sha256=sha(path),seconds=time.monotonic()-start)
                    if arm=='baseline' and condition in ['clean','wrong_patch']:
                        prior=next(r for r in legacy if r['identity']==identity and r['arm']=='baseline' and r['condition']==('clean' if condition=='clean' else 'corrupt'))
                        row['baseline_replay_exact']=sha(path)==prior['sha256']
                        assert row['baseline_replay_exact'],'Baseline replay mismatch'
                    if adapter.enabled:row['reference_weights']=adapter.priors().detach().softmax(-1).cpu().tolist()
                except Exception as error:
                    row.update(status='failed',error=str(error))
                rows.append(row);(dest/'rows.json').write_text(json.dumps(rows,indent=2))
                print('OUTPUT',identity,arm,condition,row['status'],flush=True)
    assert len(rows)==96
    print('EVALUATION GENERATION COMPLETE',flush=True)


if __name__=='__main__':main()
