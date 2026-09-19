import numpy as np
from src.degradation.face_region_masks import region_masks

def mask_demand(mask, bbox=None, landmarks=None):
    regions=region_masks(mask.shape,bbox,landmarks)
    demand={name:float((mask & region).sum()) for name,region in regions.items()}
    total=sum(demand.values())
    if total==0:return {name:1/len(regions) for name in regions},'mask_outside_coarse_regions_uniform_fallback'
    return {name:value/total for name,value in demand.items()},'coarse_overlap'
