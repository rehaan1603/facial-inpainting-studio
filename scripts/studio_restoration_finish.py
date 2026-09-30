"""Studio-only patch lighting correction; no target or reference access."""
import cv2
import numpy as np

def finish_restoration(observed, generated, mask):
    a=np.asarray(observed);g=np.asarray(generated);m=np.asarray(mask,dtype=bool)
    if a.dtype!=np.uint8 or g.dtype!=np.uint8 or a.shape!=g.shape or a.ndim!=3 or a.shape[2]!=3 or m.shape!=a.shape[:2]:
        raise ValueError('Expected matching uint8 RGB images and mask.')
    hard=np.where(m[...,None],g,a)
    if not m.any() or m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any():
        return hard, 'mask_composite_border_or_empty'
    x,y,w,h=cv2.boundingRect(m.astype(np.uint8))
    if min(w,h)<4:return hard, 'mask_composite_small_region'
    result=cv2.seamlessClone(g,a,m.astype(np.uint8)*255,(x+w//2,y+h//2),cv2.NORMAL_CLONE)
    result[~m]=a[~m]
    return result, 'gradient_domain_normal_clone'
