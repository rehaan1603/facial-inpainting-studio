"""Author-style float32 baseline smoke on an already-observed development identity."""
import json,zipfile,time,hashlib,sys
from pathlib import Path
import torch
from PIL import Image
from reference_intervention_core import ROOT,load_model,sha
CACHE=Path('C:/Users/rehaa/.cache/facial-inpainting/refldm_official_benchmark_v1');OUT=ROOT/'outputs/refldm_separate_encoding_smoke_v1'
def main():
 OUT.mkdir(exist_ok=False)
 rows=json.loads((CACHE/'overlap_audit.json').read_text())['records']
 row=next(r for r in rows if r['target']=='11491.png')
 assert row['identity']=='6681' and set(row['reference_identities'])=={'6681'}
 assert row['flags']==['pilot_development_overlap']
 cfg=json.loads((ROOT/'configs/local.json').read_text());refs=row['references'][:5]
 selected={f'CelebA-Test-Ref/celeba_test_images/{kind}/11491.png':OUT/f'{kind}.png' for kind in ['hq','lq']}
 with zipfile.ZipFile(CACHE/'CelebA-Test-Ref.zip') as z:
  for member,p in selected.items():p.write_bytes(z.read(member))
 sys.path.insert(0,str(Path(cfg['cache'])/'refldm_source_v1'))
 from inference import read_and_normalize_image
 from torchvision.transforms.functional import to_pil_image
 from ldm.models.diffusion.ddim import DDIMSampler
 reference_paths=[Path(cfg['hq'])/'CelebA-HQ-img'/f'{int(Path(x).stem)}.jpg' for x in refs]
 config={'identity':'6681','target':'11491.png','source':'official benchmark archive','scope':'Observed-identity smoke only, not paper benchmark reproduction','steps':100,'seed':17,'precision':'float32','reference_selection':'First five in official CSV order; not asserted equal to author benchmark selection','references':[dict(path=str(p),sha256=sha(p)) for p in reference_paths],'lq_sha256':sha(OUT/'lq.png'),'hq_sha256':sha(OUT/'hq.png'),'script_sha256':sha(__file__),'guidance':1.5,'encoding':'Separate reference encoding; only change against author-style smoke','output_conversion':'Author torchvision to_pil_image'}
 (OUT/'config.json').write_text(json.dumps(config,indent=2));start=time.monotonic()
 try:
  torch.set_num_threads(4);torch.set_grad_enabled(False);model=load_model().cuda();sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
  torch.manual_seed(17);noise=torch.randn(1,8,64,64,device='cuda');torch.cuda.reset_peak_memory_stats()
  image=read_and_normalize_image(OUT/'lq.png');ref=torch.cat([read_and_normalize_image(p) for p in reference_paths],dim=-1)
  cond=model.get_learned_conditioning({'lq_image':image.unsqueeze(0).cuda()})
  cond['ref_image']=torch.cat([model.get_learned_conditioning({'ref_image':read_and_normalize_image(p).unsqueeze(0).cuda()})['ref_image'] for p in reference_paths],dim=-1)
  null={k:v.detach().clone() for k,v in cond.items()};null['ref_image']*=0
  latent,_=sampler.sample(S=100,unconditional_guidance_scale=1.5,conditioning=cond,unconditional_conditioning=null,shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
  result=((model.decode_first_stage(latent)+1)/2).clamp(0,1).squeeze(0).cpu();assert torch.isfinite(result).all()
  to_pil_image(result).save(OUT/'result.png')
  receipt={'status':'complete','seconds':time.monotonic()-start,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'result_sha256':sha(OUT/'result.png'),'config_sha256':sha(OUT/'config.json')}
 except Exception as e:
  receipt={'status':'failed','error':repr(e),'seconds':time.monotonic()-start,'config_sha256':sha(OUT/'config.json')}
  (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));raise
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
