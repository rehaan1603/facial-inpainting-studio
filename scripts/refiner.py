"""Compact mask-refinement control; receives observed RGB and supplied mask only."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())


class MaskRefiner(nn.Module):
    def __init__(self,width=16):
        super().__init__();self.width=width
        self.down1=Block(4,width);self.down2=Block(width,width*2);self.mid=Block(width*2,width*4)
        self.up2=Block(width*6,width*2);self.up1=Block(width*3,width);self.out=nn.Conv2d(width,1,1)

    def forward(self,observed,supplied):
        a=self.down1(torch.cat([observed*2-1,supplied],dim=1));b=self.down2(F.avg_pool2d(a,2));c=self.mid(F.avg_pool2d(b,2))
        d=self.up2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],dim=1))
        return self.out(self.up1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],dim=1)))


def refinement_loss(logits,true,negative_weight):
    # Region-balanced BCE makes the cost independent of the random occlusion area.
    positive=F.softplus(-logits)*true;negative=F.softplus(logits)*(1-true)
    dims=(1,2,3)
    return (positive.sum(dims)/true.sum(dims).clamp_min(1)+negative_weight*negative.sum(dims)/(1-true).sum(dims).clamp_min(1)).mean()/(1+negative_weight)
