"""Experimental observed-target regional log-prior; not an established novel method."""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F

# Canonical aligned-face approximation, explicitly not a pose estimator/segmentation.
BOXES=((.23,.30,.48,.51),(.52,.30,.77,.51),(.40,.43,.60,.67),(.32,.62,.68,.83),(0.,0.,1.,1.))


def crop(x,box):
    h,w=x.shape[-2:];x0,y0,x1,y1=box
    return x[...,int(y0*h):max(int(y0*h)+1,int(y1*h)),int(x0*w):max(int(x0*w)+1,int(x1*w))]


def descriptors(x):
    return torch.stack([F.adaptive_avg_pool2d(crop(x,b),(4,4)).flatten(1) for b in BOXES],1)


def features(lq, refs, diagnostics):
    """No target latent, clean target pixels, identity evaluator or corruption labels."""
    with torch.no_grad():
        target=descriptors(lq.float())[0]  # R,C
        reference=descriptors(refs.float()).transpose(0,1)  # R,N,C
        local=F.cosine_similarity(target[:,None],reference,dim=-1)
        global_cos=local[-1:].expand_as(local)
        # Cross-reference consensus is fallible; it is a feature, never an oracle.
        consensus=F.cosine_similarity(reference,reference.median(1,keepdim=True).values,dim=-1)
        magnitude=(reference.norm(dim=-1)-target.norm(dim=-1)[:,None]).abs().clamp(max=20)/20
        # quality diagnostics: pose proxy discrepancy, sharpness, noise/edge proxy,
        # brightness difference, detection confidence, observed evidence coverage.
        side=diagnostics.to(local).unsqueeze(0).expand(5,-1,-1)
        return torch.cat([local[...,None],global_cos[...,None],consensus[...,None],magnitude[...,None],side],-1).detach()


class Compatibility(nn.Module):
    def __init__(self, mode='full', temperature=1.):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(10,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.enabled=True;self.mode=mode;self.temperature=temperature;self.context=None

    def priors(self):
        x=self.context.clone()
        if self.mode=='no_target':x[:,:,[0,1,3,4,7,9]]=0
        elif self.mode=='target_only':x[:,:,2]=0;x[:,:,5:7]=0;x[:,:,8]=0
        elif self.mode=='reliability_only':x[:,:,[0,1,3,4,7,9]]=0
        if self.mode in ['global','target_only','reliability_only']:
            x=x[-1:].expand_as(x)
        logits=self.net(x.float()).squeeze(-1).clamp(-4,4)/self.temperature
        return F.log_softmax(logits,-1)+math.log(x.shape[1])


def labels(length,device):
    side=math.isqrt(length)
    assert side*side==length
    result=torch.full((side,side),4,dtype=torch.long,device=device)
    for i,(x0,y0,x1,y1) in enumerate(BOXES[:-1]):
        result[int(y0*side):max(int(y0*side)+1,int(y1*side)),int(x0*side):max(int(x0*side)+1,int(x1*side))]=i
    return result.flatten()


def install(model,adapter):
    from ldm import cache_kv
    from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy
    count=0
    for module in model.model.diffusion_model.modules():
        if not isinstance(module,QKVAttentionLegacy):continue
        original=module.forward
        def forward(self,qkv,original=original):
            if cache_kv.mode!='use' or not adapter.enabled:return original(qkv)
            batch,width,length=qkv.shape;channels=width//(3*self.n_heads)
            q,k,v=qkv.reshape(batch*self.n_heads,3*channels,length).split(channels,1)
            keys,values=cache_kv.k[id(self)],cache_kv.v[id(self)]
            prior=adapter.priors()[labels(length,q.device)]
            assert len(keys)==prior.shape[1]
            bias=torch.cat([torch.zeros(length,length,device=q.device)]+[
                prior[:,i:i+1].expand(-1,key.shape[-1]) for i,key in enumerate(keys)],-1)
            # Author CFG batches null first and conditional second. Keep null exact.
            if batch==2:
                bias=torch.cat([torch.zeros_like(bias)[None].expand(self.n_heads,-1,-1),
                                bias[None].expand(self.n_heads,-1,-1)],0)
            elif batch!=1:raise ValueError('Only one target with optional two-way CFG is supported')
            k=torch.cat([k]+keys,-1);v=torch.cat([v]+values,-1)
            scale=1/math.sqrt(math.sqrt(channels))
            logits=torch.einsum('bct,bcs->bts',q*scale,k*scale)
            weights=torch.softmax(logits.float()+bias,dim=-1).to(q.dtype)
            return torch.einsum('bts,bcs->bct',weights,v).reshape(batch,-1,length)
        module.forward=types.MethodType(forward,module);count+=1
    assert count
    return count
