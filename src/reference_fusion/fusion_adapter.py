"""Proxy injecting stock spatial reference masks without changing frozen inference."""
import torch

class RoutedPipeline:
    def __init__(self,pipeline,masks):
        self.pipeline=pipeline
        self.masks=torch.from_numpy(masks.copy())[None]
    def __getattr__(self,name):return getattr(self.pipeline,name)
    def __call__(self,*args,**kwargs):
        if 'cross_attention_kwargs' in kwargs:raise ValueError('Unexpected external attention override')
        kwargs['cross_attention_kwargs']={'ip_adapter_masks':[self.masks]}
        return self.pipeline(*args,**kwargs)
