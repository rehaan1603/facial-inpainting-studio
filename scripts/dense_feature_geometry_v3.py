"""Pure frozen-feature matching/geometry; no image truth or detector API."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def grid_selection(mask=None):
    y,x=np.meshgrid((np.arange(16)+.5)*32,(np.arange(16)+.5)*32,indexing='ij')
    coords=np.stack([x,y],-1).reshape(-1,2)
    keep=((coords>=48)&(coords<=464)).all(1)
    if mask is not None:
        distance=distance_transform_edt(~mask)
        keep &= distance[coords[:,1].astype(int),coords[:,0].astype(int)]>32
    return coords,keep


def mutual_matches(target,reference):
    if min(len(target),len(reference))<2:return []
    def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    if not np.isfinite(target).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite features')
    similarity=norm(target)@norm(reference).T
    j=similarity.argmax(1);i=similarity.argmax(0)
    second_t=np.partition(similarity,-2,axis=1)[:,-2]
    second_r=np.partition(similarity,-2,axis=0)[-2,:]
    return [(a,int(b)) for a,b in enumerate(j) if i[b]==a and similarity[a,b]>=.6 and similarity[a,b]-second_t[a]>=.02 and similarity[a,b]-second_r[b]>=.02]


def fit_pairs(xy,uv):
    result=dict(eligible=False,mutual_matches=len(xy))
    if len(xy)<12:return dict(result,reason='Fewer than 12 mutual matches')
    order=np.lexsort((xy[:,1],xy[:,0]));xy=xy[order];uv=uv[order]
    validation=np.arange(len(xy))%3==0
    cv2.setRNGSeed(20261011)
    transform,inliers=cv2.estimateAffinePartial2D(uv[~validation],xy[~validation],method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=2000,confidence=.99)
    if transform is None:return dict(result,reason='Similarity fit failed')
    inliers=inliers.ravel().astype(bool)
    rms=float(np.sqrt(np.square(uv[validation]@transform[:,:2].T+transform[:,2]-xy[validation]).sum(1).mean()))
    span=np.ptp(xy[~validation][inliers],axis=0) if inliers.any() else np.zeros(2)
    scale=float(np.linalg.norm(transform[:,0]))
    passed=inliers.sum()>=8 and inliers.mean()>=.6 and rms<=12 and min(span)>=100 and np.linalg.det(transform[:,:2])>0 and .5<=scale<=2
    return dict(result,eligible=bool(passed),reason=None if passed else 'Fixed held-out/coverage/fit gate failed',held_out_rms_pixels=rms,fit_inliers=int(inliers.sum()),fit_inlier_fraction=float(inliers.mean()),fit_span_xy=span.tolist(),scale=scale,transform=transform.tolist())


def compare(target,reference,mask):
    if target.shape!=(256,1280) or reference.shape!=(256,1280):raise ValueError('Expected actual 16x16 CLIP grid')
    coords,tk=grid_selection(mask);_,rk=grid_selection()
    pairs=mutual_matches(target[tk],reference[rk]);xy=coords[tk];uv=coords[rk]
    return fit_pairs(np.array([xy[a] for a,b in pairs]),np.array([uv[b] for a,b in pairs]))
