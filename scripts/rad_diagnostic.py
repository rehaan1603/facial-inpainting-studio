import sys,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
import os
os.chdir(ROOT)
cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);sys.path.insert(0,str(cache/'rad_source_v1/src'))
from diffusers.models.unets.unet_2d_local import LocalUNet2DModel
from diffusers.schedulers.scheduling_localddpm import LocalDDPMScheduler
from diffusers.pipelines.localddpm.inpainting_pipeline_localddpm import InPaintLocalDDPMPipeline
from diffusers.utils.set_random_steps import set_random_steps
from peft import LoraConfig
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--image',type=Path,required=True);parser.add_argument('--mask',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--steps',type=int,choices=[100,1000],default=100);options=parser.parse_args()
out=options.output_dir;out.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(17)
config=json.loads(Path('research/rad_base_config_v1.json').read_text());model=LocalUNet2DModel.from_config(config)
model.add_adapter(LoraConfig(r=16,lora_alpha=16,init_lora_weights='gaussian',target_modules=['to_k','to_q','to_v','to_out.0','conv_in','conv_out','conv1','conv2']))
model.load_state_dict(load_file(str(cache/'rad_weights_v1/model.safetensors')),strict=True);model.cuda().eval();print('Strict checkpoint load passed',flush=True)
args=SimpleNamespace(ddpm_num_steps=2000,ddpm_mask_num_steps=1000)
scheduler=LocalDDPMScheduler(num_train_timesteps=2000,num_mask_timesteps=1000,variance_type='learned_range');scheduler.blur_sigma=0;scheduler.random_steps,scheduler.m_steps,scheduler.om_steps=set_random_steps(args)
pipe=InPaintLocalDDPMPipeline(unet=model,scheduler=scheduler).to('cuda')
a=Image.open(options.image).convert('RGB');m=Image.open(options.mask).convert('L');
assert a.size==(512,512) and m.size==a.size,'Diagnostic expects matching 512px inputs'
aa=np.array(a);mm=np.array(m)>127
safe=np.where(mm[...,None],128,aa).astype('uint8');x=torch.from_numpy(np.array(Image.fromarray(safe).resize((256,256),Image.Resampling.BILINEAR)).copy()).permute(2,0,1)[None].float().cuda()/127.5-1
mask=torch.nn.functional.adaptive_max_pool2d(torch.from_numpy(mm.astype('float32'))[None,None],(256,256)).cuda()
start=time.monotonic()
with torch.inference_mode():r=pipe(clean_image=x,mask=mask,num_inference_steps=options.steps,output_type='np',args=args).images[0]
assert np.isfinite(r).all();raw=Image.fromarray(np.rint(r*255).astype('uint8'));raw.save(out/'native.png');restored=np.array(raw.resize((512,512),Image.Resampling.BICUBIC));restored[~mm]=aa[~mm];Image.fromarray(restored).save(out/'result.png');(out/'receipt.json').write_text(json.dumps({'seconds':time.monotonic()-start,'strict_load':True,'steps':options.steps,'seed':17,'scope':'New author model smoke test, target not supplied during inference'}));print('Completed',flush=True)
