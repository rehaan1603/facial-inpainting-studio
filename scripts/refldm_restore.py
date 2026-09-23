"""Local experimental partial-damage restoration using verified author weights."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 3<=len(args.references)<=4:raise ValueError('Provide 3–4 same-person reference photos.')
    if args.output.exists():raise ValueError('Output already exists.')
    source_image=np.asarray(Image.open(args.image).convert('RGB'));mask=np.asarray(Image.open(args.mask).convert('L'))>=128
    if source_image.shape!=(512,512,3) or mask.shape!=(512,512) or not mask.any():raise ValueError('Expected a 512-pixel image and nonempty matching mask.')
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);source=cache/'refldm_source_v1';weights=cache/'refldm_weights_v1'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!='af6690c19fdc6421802fd7996510bcfef259bfd1':raise ValueError('Unexpected model source revision.')
    for f in json.loads((ROOT/'research/refldm_downloads_v1.json').read_text())['files']:
        if sha(weights/f['name'])!=f['sha256']:raise ValueError('Model weight checksum mismatch.')
    sys.path.insert(0,str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from inference import read_and_normalize_image
    from torchvision.transforms.functional import to_pil_image
    start=time.monotonic();config=OmegaConf.load(source/'configs/refldm.yaml')
    config.model.params.first_stage_config.params.ckpt_path=str(weights/'vqgan.ckpt')
    for k in ['ckpt_path','perceptual_loss_config']:config.model.params.pop(k,None)
    model=instantiate_from_config(config.model)
    keys=model.load_state_dict(torch.load(weights/'refldm.ckpt',map_location='cpu',weights_only=True),strict=False)
    if keys.missing_keys or keys.unexpected_keys:raise ValueError('Unexpected model parameter mismatch.')
    model.to('cuda').eval();torch.set_grad_enabled(False);torch.manual_seed(17)
    shape=[model.model.diffusion_model.out_channels,model.image_size,model.image_size]
    noise=torch.randn([1,*shape],device='cuda')
    condition={'lq_image':read_and_normalize_image(args.image).unsqueeze(0).cuda(),
        'ref_image':torch.cat([read_and_normalize_image(r) for r in args.references],dim=-1).unsqueeze(0).cuda()}
    condition=model.get_learned_conditioning(condition);null={k:v.detach().clone() for k,v in condition.items()};null['ref_image']*=0
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    with model.ema_scope():
        latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=condition,unconditional_conditioning=null,
            shape=shape,x_T=noise,batch_size=1,verbose=False)
    decoded=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
    if not torch.isfinite(decoded).all():raise ValueError('Model produced nonfinite pixels.')
    raw=args.output.with_name(args.output.stem+'_raw.png');to_pil_image(decoded.squeeze(0).cpu()).save(raw)
    rgb=np.asarray(Image.open(raw).convert('RGB'));Image.fromarray(np.where(mask[...,None],rgb,source_image)).save(args.output)
    metadata={'backbone':'refldm','seed':17,'steps':50,'guidance':1.5,'reference_count':len(args.references),
        'input_sha256':sha(args.image),'mask_sha256':sha(args.mask),'reference_sha256':[sha(r) for r in args.references],
        'raw_sha256':sha(raw),'result_sha256':sha(args.output),'inference_seconds_including_offload':time.monotonic()-start,
        'scope':'Experimental partial-damage restoration. Not missing-region inpainting or verified true-face recovery.',
        'known_pixels_unchanged':True,'model_weight_sha256':sha(weights/'refldm.ckpt')}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
