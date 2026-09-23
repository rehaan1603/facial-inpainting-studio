"""Frozen, development-only author ReF-LDM baseline. Never reads clean targets."""
import json, sys, time, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new

def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    source=cache/'refldm_source_v1'; weights=cache/'refldm_weights_v1'
    protocol=ROOT/'research/protocols/refldm_development_v1.json'
    p=json.loads(protocol.read_text()); manifest=ROOT/'outputs/distortion_aware_v1/inference_manifest.json'
    assert sha(manifest)==p['input_manifest_sha256']
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==p['source_commit']
    for f in json.loads((ROOT/'research/refldm_downloads_v1.json').read_text())['files']:
        assert sha(weights/f['name'])==f['sha256']
    cases=json.loads(manifest.read_text())['cases']
    assert len(cases)==24 and {c['identity'] for c in cases}=={'1306','2790','1043','787'}
    for c in cases:
        for path,digest in [(c['observed'],c['observed_sha256']),(c['mask'],c['mask_sha256'])]+[(r['path'],r['sha256']) for r in c['references']]:
            assert sha(path)==digest
    out=ROOT/'outputs/refldm_development_v1';out.mkdir(exist_ok=False)
    patch=subprocess.check_output(['git','-C',str(source),'diff'],text=True)
    (out/'author_compatibility.patch').write_text(patch)
    write_new(out/'signature.json',{'protocol_sha256':sha(protocol),'runner_sha256':sha(__file__),
        'manifest_sha256':sha(manifest),'source_commit':p['source_commit'],'patch_sha256':sha(out/'author_compatibility.patch'),
        'torch':torch.__version__,'precision':'author default float32','checkpoint_load_device':'cpu'})
    sys.path.insert(0,str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from inference import read_and_normalize_image
    from torchvision.transforms.functional import to_pil_image
    config=OmegaConf.load(source/'configs/refldm.yaml')
    config.model.params.first_stage_config.params.ckpt_path=str(weights/'vqgan.ckpt')
    for k in ['ckpt_path','perceptual_loss_config']:config.model.params.pop(k,None)
    model=instantiate_from_config(config.model)
    keys=model.load_state_dict(torch.load(weights/'refldm.ckpt',map_location='cpu',weights_only=True),strict=False)
    write_new(out/'checkpoint_keys.json',{'missing':list(keys.missing_keys),'unexpected':list(keys.unexpected_keys)})
    if keys.missing_keys or keys.unexpected_keys:raise RuntimeError('Checkpoint mismatch; inspect keys before sampling')
    model.to('cuda').eval();torch.set_grad_enabled(False)
    sampler=DDIMSampler(model,print_tqdm=False,schedule=p['schedule'])
    shape=[model.model.diffusion_model.out_channels,model.image_size,model.image_size];rows=[]
    for c in cases:
        folder=out/c['case_id'];folder.mkdir();start=time.monotonic()
        common={k:c[k] for k in ['case_id','identity','kind']}
        try:
            torch.manual_seed(p['seed']);noise=torch.randn([1,*shape],device='cuda')
            condition={'lq_image':read_and_normalize_image(c['observed']).unsqueeze(0).cuda(),
                'ref_image':torch.cat([read_and_normalize_image(r['path']) for r in c['references']],dim=-1).unsqueeze(0).cuda()}
            condition=model.get_learned_conditioning(condition)
            null={k:v.detach().clone() for k,v in condition.items()};null['ref_image']*=0
            with model.ema_scope():
                latent,_=sampler.sample(S=p['steps'],unconditional_guidance_scale=p['guidance'],conditioning=condition,
                    unconditional_conditioning=null,shape=shape,x_T=noise,batch_size=1,verbose=False)
            decoded=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
            if not torch.isfinite(decoded).all():raise ValueError('Nonfinite output')
            to_pil_image(decoded.squeeze(0).cpu()).save(folder/'native.png')
            raw=np.asarray(Image.open(folder/'native.png').convert('RGB'))
            observed=np.asarray(Image.open(c['observed']).convert('RGB'));mask=np.asarray(Image.open(c['mask']).convert('L'))>=128
            Image.fromarray(np.where(mask[...,None],raw,observed)).save(folder/'composited.png')
            for arm in ['native','composited']:
                path=folder/(arm+'.png')
                rows.append(dict(common,key=c['case_id']+'_'+arm,mode=arm,status='complete',output=str(path),output_sha256=sha(path),seconds=time.monotonic()-start))
        except Exception as e:
            for arm in ['native','composited']:rows.append(dict(common,key=c['case_id']+'_'+arm,mode=arm,status='failed',error=f'{type(e).__name__}: {e}'))
        write_new(folder/'rows.json',rows[-2:]);print(c['case_id'],rows[-1]['status'],flush=True)
    write_new(out/'comparison.json',{'rows':rows,'final_test_used':False})

if __name__=='__main__':main()
