"""Isolated spatial-reference mechanism. Not installed in the app or trained.

Reference hidden states must come from the SAME frozen self-attention layer,
with actual registered source coordinates. FaceID's compressed tokens cannot
be substituted. Geometry/cache provenance checks belong to a future gated runner.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

HOOK = 'mid_block.attentions.0.transformer_blocks.0.attn1.processor'


def grid_centres(side, device=None):
    c=(torch.arange(side,device=device,dtype=torch.float32)+.5)/side
    y,x=torch.meshgrid(c,c,indexing='ij')
    return torch.stack([x,y],-1).reshape(-1,2)


def coordinate_bias(query_xy, source_xy, reliability, sigma=.08):
    if sigma<=0 or not math.isfinite(sigma):raise ValueError('Invalid spatial bandwidth')
    if not torch.isfinite(query_xy).all() or not torch.isfinite(source_xy).all():raise ValueError('Nonfinite coordinates')
    if not torch.isfinite(reliability).all() or not ((reliability>0)&(reliability<=1)).all():raise ValueError('Invalid reliability')
    distance=(query_xy[:,None,:]-source_xy[None,:,:]).square().sum(-1)
    return -distance/(2*sigma**2)+reliability.log()[None,:]


class SpatialReferenceAttention(nn.Module):
    """20 head gates, independent neutral residual, same frozen Q/K/V/output maps."""
    def __init__(self,delegate,target_mask,reference_hidden=None,coordinates=None,reliability=None,heads=20):
        super().__init__()
        if type(delegate) is not AttnProcessor2_0:raise TypeError('Requires unchanged Torch2 self-attention delegate')
        if heads!=20:raise ValueError('The audited layer has exactly20 heads')
        self.delegate=delegate
        self.head_gate=nn.Parameter(torch.zeros(heads,dtype=torch.float32))
        mask=torch.as_tensor(target_mask,dtype=torch.bool).clone()
        if mask.ndim!=2 or mask.shape[0]!=mask.shape[1]:raise ValueError('Square mask required')
        self.register_buffer('target_mask',mask,persistent=False)
        self.enabled=True
        if reference_hidden is None:
            if coordinates is not None or reliability is not None:raise ValueError('Incomplete bank')
            self.register_buffer('reference_hidden',None,persistent=False)
            self.register_buffer('coordinates',None,persistent=False)
            self.register_buffer('reliability',None,persistent=False)
        else:
            h=torch.as_tensor(reference_hidden).detach().clone()
            c=torch.as_tensor(coordinates,dtype=torch.float32).detach().clone()
            r=torch.as_tensor(reliability,dtype=torch.float32).detach().clone()
            if h.ndim!=2 or not 1<=len(h)<=512 or c.shape!=(len(h),2) or r.shape!=(len(h),):raise ValueError('Invalid bank shapes/cap')
            if not torch.isfinite(h).all():raise ValueError('Nonfinite reference features')
            coordinate_bias(c[:1],c,r)
            self.register_buffer('reference_hidden',h,persistent=False)
            self.register_buffer('coordinates',c,persistent=False)
            self.register_buffer('reliability',r,persistent=False)

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,**kwargs):
        if encoder_hidden_states is not None:raise ValueError('Only audited self-attention may be replaced')
        base=self.delegate(attn,hidden_states,encoder_hidden_states=None,attention_mask=attention_mask,temb=temb,**kwargs)
        if not self.enabled or self.reference_hidden is None or not self.target_mask.any():return base
        if hidden_states.ndim!=3 or attn.heads!=20:raise ValueError('Unexpected target layout')
        if attn.spatial_norm is not None or attn.group_norm is not None:raise ValueError('Unaudited attention normalization')
        if attn.to_out[1].p!=0:raise ValueError('Nonzero attention dropout needs separate protocol')
        batch,length,channels=hidden_states.shape;side=math.isqrt(length)
        if side*side!=length or self.reference_hidden.shape[1]!=channels:raise ValueError('Reference layer/grid mismatch')
        coords=grid_centres(side,hidden_states.device)
        yx=(coords[:,[1,0]]*self.target_mask.shape[0]).long()
        mask=self.target_mask.to(hidden_states.device)[yx[:,0],yx[:,1]]
        indices=mask.nonzero().flatten()
        if not len(indices):return base
        q=attn.to_q(hidden_states[:,indices]);bank=self.reference_hidden.to(hidden_states)
        k=attn.to_k(bank)[None].expand(batch,-1,-1);v=attn.to_v(bank)[None].expand(batch,-1,-1)
        dim=q.shape[-1]//attn.heads
        reshape=lambda x:x.reshape(batch,-1,attn.heads,dim).transpose(1,2)
        q,k,v=reshape(q),reshape(k),reshape(v)
        if attn.norm_q is not None:q=attn.norm_q(q)
        if attn.norm_k is not None:k=attn.norm_k(k)
        bias=coordinate_bias(coords[indices],self.coordinates.to(coords),self.reliability.to(coords))
        extra=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype)[None,None],dropout_p=0.,is_causal=False)
        extra=extra*(.2*self.head_gate.tanh()).to(extra)[None,:,None,None]
        extra=extra.transpose(1,2).reshape(batch,len(indices),-1)
        # Bias already appears in the delegate. The extra branch is linear only.
        extra=F.linear(extra,attn.to_out[0].weight,bias=None)/attn.rescale_output_factor
        delta=torch.zeros_like(base).index_copy(1,indices,extra.to(base))
        return base+delta


def install(unet,processor):
    if HOOK not in unet.attn_processors:raise ValueError('Audited hook missing')
    if any(p.requires_grad for p in unet.parameters()):raise ValueError('Freeze pretrained weights before installation')
    if sum(p.numel() for p in processor.parameters() if p.requires_grad)!=20:raise ValueError('Trainable count changed')
    attention=unet.get_submodule(HOOK.rsplit('.processor',1)[0])
    if attention.heads!=20 or attention.to_q.in_features!=1280:raise ValueError('Audited model architecture changed')
    processors=dict(unet.attn_processors);processors[HOOK]=processor
    unet.set_attn_processor(processors)
    if sum(p.numel() for p in unet.parameters() if p.requires_grad)!=20:raise ValueError('Unexpected trainable model parameters')
