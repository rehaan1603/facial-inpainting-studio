"""Partition-of-unity reference masks, using no clean-target information."""
import numpy as np
from scipy.ndimage import gaussian_filter
from src.degradation.face_region_masks import region_masks

POLICIES=('concat','equal','identity','quality','mask_aware','regional')

def softmax(values,temperature=.15):
    values=np.asarray(values,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or temperature<=0:raise ValueError('Invalid weighting inputs')
    values=(values-values.max())/temperature
    result=np.exp(values);return result/result.sum()

def build_masks(analysis,binary,geometry,policy='regional',temperature=.15,sigma=8.):
    if policy not in POLICIES:raise ValueError('Unknown routing policy')
    valid=[r for r in analysis['references'] if r['valid']]
    if not valid:raise ValueError('No usable reference photos; check the saved diagnostics')
    binary=np.asarray(binary,dtype=bool)
    if binary.ndim!=2 or not binary.any():raise ValueError('A nonempty damage mask is required')
    n=len(valid);equal=np.full(n,1/n);h,w=binary.shape
    global_scores={'identity':[(r['identity_compatibility']+1)/2 for r in valid],
                   'quality':[r['global_quality'] for r in valid],
                   'mask_aware':[r['mask_aware_score'] for r in valid]}
    global_weights=softmax(global_scores.get(policy,global_scores['mask_aware']),temperature)
    if policy in ('equal','concat'):global_weights=equal
    maps=np.broadcast_to(global_weights[:,None,None],(n,h,w)).copy()
    region_weights={}
    if policy=='regional':
        regions=region_masks(binary.shape,geometry.get('bbox'),geometry.get('landmarks'))
        numerator=np.zeros_like(maps);denominator=np.zeros((h,w),dtype=float)
        for name,area in regions.items():
            scores=[]
            for r in valid:
                q=r['regions'][name]
                scores.append(.45*q['quality']+.20*q['visibility_proxy']+.10*r['global_quality']+
                              .10*r['pose_similarity']+.10*(r['identity_compatibility']+1)/2+.05*q['exposure'])
            weights=softmax(scores,temperature);region_weights[name]=weights.tolist()
            basis=gaussian_filter(area.astype(float),sigma=sigma,mode='nearest')
            numerator+=weights[:,None,None]*basis;denominator+=basis
        covered=denominator>1e-6
        maps[:,covered]=numerator[:,covered]/denominator[covered]
    maps[:,~binary]=equal[:,None]
    maps=np.clip(maps,0,1);maps/=maps.sum(0,keepdims=True)
    result=maps.astype('float32')
    if not np.isfinite(result).all() or not np.allclose(result.sum(0),1,atol=1e-6):raise ValueError('Invalid routing maps')
    return result,{'policy':policy,'valid_reference_indices':[r['index'] for r in valid],
                   'global_weights':global_weights.tolist(),'region_weights':region_weights,
                   'outside_damage_weights':equal.tolist(),'temperature':temperature,'sigma':sigma,
                   'weight_sum_max_error':float(np.abs(result.sum(0)-1).max()),
                   'conditioning':'spatial routing of global FaceID descriptors; no reference image patch transfer',
                   'trainable_parameters':0}
