"""Coarse image-left/right regions, not anatomical segmentation."""
import numpy as np
from PIL import Image, ImageDraw

REGIONS = ('left_eye', 'right_eye', 'nose', 'mouth', 'left_cheek', 'right_cheek', 'forehead', 'chin')

def region_masks(shape, bbox=None, landmarks=None):
    h, w = shape
    x0, y0, x1, y1 = bbox if bbox is not None else (w*.15,h*.1,w*.85,h*.95)
    bw, bh = max(x1-x0,1), max(y1-y0,1)
    points = np.asarray(landmarks) if landmarks is not None else np.array([
        [x0+.32*bw,y0+.36*bh], [x0+.68*bw,y0+.36*bh],
        [x0+.5*bw,y0+.56*bh], [x0+.38*bw,y0+.76*bh], [x0+.62*bw,y0+.76*bh]])
    eyes = sorted(points[:2], key=lambda p:p[0])
    centers = [eyes[0],eyes[1],points[2],points[3:5].mean(0),
               (x0+.24*bw,y0+.60*bh),(x0+.76*bw,y0+.60*bh),
               (x0+.50*bw,y0+.18*bh),(x0+.50*bw,y0+.9*bh)]
    sizes = [(.24,.18),(.24,.18),(.24,.25),(.44,.21),(.28,.25),(.28,.25),(.7,.25),(.5,.2)]
    result = {}
    for name, (cx,cy), (rx,ry) in zip(REGIONS, centers, sizes):
        mask = Image.new('L',(w,h)); ImageDraw.Draw(mask).ellipse((cx-rx*bw/2,cy-ry*bh/2,cx+rx*bw/2,cy+ry*bh/2),fill=255)
        result[name] = np.asarray(mask)>0
    return result

def damage_mask(shape, kind, severity='medium', seed=17, bbox=None, landmarks=None):
    from .severity import SEVERITIES
    h,w=shape; regions=region_masks(shape,bbox,landmarks)
    region_kind={'eyes':['left_eye','right_eye'],'nose':['nose'],'mouth':['mouth'],
                 'central_face':['left_eye','right_eye','nose','mouth'],
                 'half_face':['left_eye','left_cheek','nose','mouth']}
    if kind in region_kind:
        mask=np.logical_or.reduce([regions[r] for r in region_kind[kind]])
        from scipy.ndimage import binary_dilation, binary_erosion
        size=SEVERITIES[severity]['size']
        if size>1:mask=binary_dilation(mask,iterations=max(1,round(min(h,w)*.018)))
        if size<1:mask=binary_erosion(mask,iterations=max(1,round(min(h,w)*.008)))
        return mask
    rng=np.random.default_rng(seed); canvas=Image.new('L',(w,h));draw=ImageDraw.Draw(canvas)
    factor=SEVERITIES[severity]['size']
    if kind=='rectangle':
        cx,cy=rng.uniform(.35,.65,2);dx,dy=.20*factor,.15*factor
        draw.rectangle(((cx-dx)*w,(cy-dy)*h,(cx+dx)*w,(cy+dy)*h),fill=255)
    elif kind=='brush':
        points=[tuple(p) for p in rng.uniform(.2,.8,(5,2))*[w,h]]
        draw.line(points,fill=255,width=max(1,round(min(h,w)*.07*factor)),joint='curve')
    else:raise ValueError(f'Unknown mask kind: {kind}')
    return np.asarray(canvas)>0
