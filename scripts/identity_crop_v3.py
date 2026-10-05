"""Differentiable analogue of the evaluator's zero-margin face crop."""
import torch.nn.functional as F

def crop_tensor(rgb,box):
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=rgb.shape[-1] and 0<=y0<y1<=rgb.shape[-2]):
        raise ValueError('Invalid frozen face crop')
    return F.interpolate(rgb[:,:,y0:y1,x0:x1],size=(160,160),mode='bilinear',align_corners=False,antialias=True)

def standardize(rgb):
    return (rgb*255-127.5)/128
