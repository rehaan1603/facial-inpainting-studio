"""Thin inference wrapper over an unmodified, pinned official ResShift checkout."""
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
COMMIT='bb03b7d21614cace01787e097c8a6ab6b945227d'


class ResShiftFace:
    def __init__(self):
        local=json.loads((ROOT/'configs/local.json').read_text())
        cache=Path(local['cache']); repo=cache/'ResShift'
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        assert head==COMMIT, 'Unexpected ResShift source revision'
        assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip(), 'Tracked ResShift source modified'
        sys.path.insert(0,str(repo))
        cfg=OmegaConf.load(repo/'configs/inpaint_lama256_face.yaml')
        def instantiate(node):
            module,name=node.target.rsplit('.',1)
            cls=getattr(importlib.import_module(module),name)
            return cls(**OmegaConf.to_container(node.params,resolve=True))
        self.diffusion=instantiate(cfg.diffusion)
        self.model=instantiate(cfg.model).cuda().eval()
        self.vae=instantiate(cfg.autoencoder).cuda().eval()
        files={'model':'resshift_inpainting_face_s4.pth','autoencoder':'celeba256_vq_f4_dim3_face.pth'}
        hashes={}
        for key,filename in files.items():
            path=cache/'models'/filename
            state=torch.load(path,map_location='cpu',weights_only=True)
            state=state.get('state_dict',state)
            state={k.removeprefix('module.').removeprefix('_orig_mod.'):v for k,v in state.items()}
            (self.model if key=='model' else self.vae).load_state_dict(state,strict=True)
            hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.model.requires_grad_(False); self.vae.requires_grad_(False)
        report={'repository':'https://github.com/zsyOAOA/ResShift','commit':head,'weights_sha256':hashes,
            'strict_state_dict_load':True,'diffusion_steps':4,'input_range':'RGB [-1,1]; supplied mask 1 -> unknown, -1 -> known',
            'preprocessing':'Erase supplied-mask pixels to black before normalization; retain original observed RGB for final compositing.',
            'source_configuration':'configs/inpaint_lama256_face.yaml','precision':'float32 modules with CUDA float16 autocast',
            'license':'S-Lab License 1.0 (non-commercial); original license retained in cached checkout.',
            'limits':'Official code and weights with a local inference wrapper, not full paper reproduction. Source config references FFHQ training; VAE checkpoint is named celeba256. Pretraining exposure to evaluation images remains unresolved.'}
        (ROOT/'research/resshift_provenance.json').write_text(json.dumps(report,indent=2))

    @torch.inference_mode()
    def __call__(self,observed,mask,seed=12345):
        assert observed.shape==(256,256,3) and mask.shape==(256,256)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        x=torch.from_numpy(observed.transpose(2,0,1).copy()).unsqueeze(0).cuda()
        m=torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        y=x*(1-m)*2-1
        with torch.autocast('cuda',dtype=torch.float16):
            result=self.diffusion.p_sample_loop(y=y,model=self.model,first_stage_model=self.vae,
                noise=None,noise_repeat=False,clip_denoised=False,denoised_fn=None,
                model_kwargs={'lq':y,'mask':m*2-1},progress=False)
        raw=(result.clamp(-1,1)+1)/2
        composed=raw*m+x*(1-m)
        array=composed[0].permute(1,2,0).float().cpu().numpy()
        assert np.isfinite(array).all()
        return array
