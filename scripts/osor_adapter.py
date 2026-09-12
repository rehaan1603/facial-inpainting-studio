"""Pinned author OSOR inference with sequential CPU offload for the 8 GB laptop.

The author's infer() is inherited unchanged. Only loading, validation and memory
placement are adapted. Verified fp16 base variants are cast to author bf16 dtype.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from accelerate import cpu_offload
from diffusers import AutoencoderKL
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
REPO=CACHE/'OSOR/osor-sdxlinpainting'
sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(REPO))
from inference_enhance import SDXLInferenceEnhance
from src.models.generator_enhance import DiffGANGeneratorEnhance

class OSORLocal(SDXLInferenceEnhance):
    def __init__(self):
        self.load_audit={}
        super().__init__(str(CACHE/'osor_models/sdxl'),str(CACHE/'osor_models/osor/osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth'),str(CACHE/'osor_models/fixed_prompt.pt'),['to_k','to_q','to_v','to_out.0','conv','conv1','conv2','conv_shortcut','proj_in','proj_out','ff.net.2','ff.net.0.proj'],256,400,device='cuda',weight_dtype=torch.bfloat16)
    def _init_vae(self):
        self.vae=AutoencoderKL.from_pretrained(self.base_model_path,subfolder='vae',torch_dtype=self.weight_dtype,local_files_only=True).eval().requires_grad_(False)
        cpu_offload(self.vae,execution_device=torch.device('cuda'))
    def _init_generator(self):
        print('Loading OSOR generator on CPU, with sequential GPU offload for inference.',flush=True)
        self.G=DiffGANGeneratorEnhance(self.base_model_path,self.lora_rank,self.lora_modules,self.weight_dtype).to(dtype=self.weight_dtype)
        raw=torch.load(self.weight_path,map_location='cpu',weights_only=True,mmap=True)
        state={k.replace('module.','').replace('_orig_mod.',''):v for k,v in raw.items()}
        trainable={name for name,p in self.G.named_parameters() if p.requires_grad}
        keys=self.G.load_state_dict(state,strict=False)
        assert not keys.unexpected_keys,keys.unexpected_keys
        assert not trainable.intersection(keys.missing_keys),'Checkpoint is missing trained parameters'
        self.load_audit={'checkpoint_keys':len(state),'trainable_keys':len(trainable),'missing_frozen_keys':len(keys.missing_keys),'missing_trainable_keys':0,'unexpected_keys':0,'memory_mode':'accelerate sequential CPU offload','inference_dtype':'bfloat16','base_variant':'fp16','source_infer_sha256':hashlib.sha256((REPO/'scripts/inference_enhance.py').read_bytes()).hexdigest()}
        del state,raw
        self.G.eval().requires_grad_(False);cpu_offload(self.G,execution_device=torch.device('cuda'))
        print('All trained OSOR parameters loaded and validated.',flush=True)
    @torch.inference_mode()
    def __call__(self,observed,mask,seed=17):
        assert observed.shape==(256,256,3) and mask.shape==(256,256)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        x=torch.from_numpy(observed.transpose(2,0,1).copy()).cuda()[None]
        m=torch.from_numpy(mask.astype('float32')).cuda()[None,None]
        result,alpha=self.infer(x,m,return_type='pt')
        pred=result[0].permute(1,2,0).cpu().numpy()
        if not np.isfinite(pred).all():raise RuntimeError('OSOR returned non-finite output')
        return pred,np.asarray(alpha[0])
