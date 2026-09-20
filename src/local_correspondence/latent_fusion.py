"""Experimental aligned native-VAE feature conditioning; no reference pixel paste."""
import numpy as np


def region_weights(quality,compatibility,reliability,policy):
    quality,compatibility,reliability=map(np.asarray,(quality,compatibility,reliability))
    if quality.ndim!=2 or compatibility.shape!=quality.shape or reliability.shape!=(quality.shape[1],):
        raise ValueError('Expected references x regions and region reliability')
    if not all(np.isfinite(v).all() for v in [quality,compatibility,reliability]):raise ValueError('Nonfinite features')
    if policy=='equal':return np.full(quality.shape,1/quality.shape[0])
    score=quality.copy()
    if policy=='damage':score=quality+.25*compatibility*reliability[None]
    elif policy not in ['single','quality']:raise ValueError('Unknown fusion policy')
    if policy=='single':
        out=np.zeros_like(score);out[np.argmax(score,axis=0),np.arange(score.shape[1])]=1;return out
    score=(score-score.max(axis=0,keepdims=True))/.15
    out=np.exp(score);return out/out.sum(axis=0,keepdims=True)


def fuse_latents(reference_latents,weights,region_basis):
    references,weights,basis=map(np.asarray,(reference_latents,weights,region_basis))
    if references.ndim!=4 or references.shape[1:]!=(4,64,64):raise ValueError('Expected native 4x64x64 reference features')
    if weights.shape!=(len(references),len(basis)) or basis.shape[1:]!=(64,64):raise ValueError('Map geometry mismatch')
    if not np.allclose(weights.sum(0),1) or np.any(weights<0) or np.any(basis<0):raise ValueError('Invalid weights')
    spatial=np.einsum('nr,rhw->nhw',weights,basis)
    total=spatial.sum(0,keepdims=True)
    spatial=np.divide(spatial,total,out=np.full_like(spatial,1/len(references)),where=total>1e-8)
    return np.einsum('nchw,nhw->chw',references,spatial).astype('float32'),spatial.astype('float32')


class LocalLatentPipeline:
    """Inject after the first step to retain baseline random-number consumption."""
    def __init__(self,pipeline,features,gate):
        self.pipeline,self.features,self.gate=pipeline,features,gate
    def __getattr__(self,name):return getattr(self.pipeline,name)
    def __call__(self,*args,**kwargs):
        if not np.any(self.gate):return self.pipeline(*args,**kwargs)
        import torch
        original=kwargs.get('callback_on_step_end')
        kwargs['callback_on_step_end_tensor_inputs']=['latents','masked_image_latents']
        def callback(pipe,step,timestep,tensors):
            if step==0:
                current=tensors['masked_image_latents']
                feature=torch.as_tensor(self.features,device=current.device,dtype=current.dtype)[None]
                gate=torch.as_tensor(self.gate,device=current.device,dtype=current.dtype)[None,None]
                if feature.shape[1:]!=current.shape[1:]:raise ValueError('Native latent geometry mismatch')
                tensors['masked_image_latents']=current*(1-gate)+feature*gate
            return original(pipe,step,timestep,tensors) if original else tensors
        kwargs['callback_on_step_end']=callback
        return self.pipeline(*args,**kwargs)
