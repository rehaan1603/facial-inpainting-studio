"""Detector-independent, visible-only SIFT alignment diagnostic; not app code."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from spatial_reference_geometry_v1 import sanitize


def features(rgb, hole=None):
    if hole is not None:
        rgb = sanitize(rgb, hole)
    height, width = rgb.shape[:2]
    allowed = np.zeros((height, width), np.uint8)
    allowed[height//10:height-height//10, width//10:width-width//10] = 255
    keypoints, descriptors = cv2.SIFT_create(nfeatures=1500, contrastThreshold=.02).detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), allowed)
    if descriptors is None:
        return np.empty((0, 2)), np.empty((0, 128), np.float32)
    distance = distance_transform_edt(~hole) if hole is not None else None
    keep = [i for i,k in enumerate(keypoints) if distance is None or distance[int(round(k.pt[1])),int(round(k.pt[0]))] > max(8,6*k.size)]
    return np.array([keypoints[i].pt for i in keep]).reshape(-1,2), descriptors[keep]


def match_geometry(observed, hole, reference):
    target, td = features(observed, hole)
    source, sd = features(reference)
    result = dict(eligible=False, target_features=len(target), reference_features=len(source))
    if min(len(td),len(sd)) < 2:
        return dict(result, reason='Insufficient descriptors')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2) if len(pair)==2 for m,n in [pair] if m.distance < .75*n.distance}
    forward, backward = ratio(td,sd),ratio(sd,td)
    pairs = sorted([(t,s) for t,s in forward.items() if backward.get(s)==t],key=lambda p:tuple(target[p[0]]))
    result['mutual_matches']=len(pairs)
    if len(pairs)<12:
        return dict(result,reason='Fewer than 12 mutual matches')
    xy=np.array([target[t] for t,s in pairs]); uv=np.array([source[s] for t,s in pairs])
    validation=np.arange(len(pairs))%3==0
    cv2.setRNGSeed(20261011)
    transform,inliers=cv2.estimateAffinePartial2D(uv[~validation],xy[~validation],method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=2000,confidence=.99)
    if transform is None:
        return dict(result,reason='Similarity fit failed')
    inliers=inliers.ravel().astype(bool)
    predicted=uv[validation]@transform[:,:2].T+transform[:,2]
    rms=float(np.sqrt(np.square(predicted-xy[validation]).sum(1).mean()))
    spans=np.ptp(xy[~validation][inliers],axis=0) if inliers.any() else np.zeros(2)
    determinant=float(np.linalg.det(transform[:,:2])); scale=float(np.linalg.norm(transform[:,0]))
    passed=int(inliers.sum())>=8 and inliers.mean()>=.6 and rms<=12 and min(spans)>=100 and determinant>0 and .5<=scale<=2
    return dict(result,eligible=bool(passed),reason=None if passed else 'Fixed held-out/coverage/fit gate failed',held_out_rms_pixels=rms,fit_inliers=int(inliers.sum()),fit_inlier_fraction=float(inliers.mean()),fit_span_xy=spans.tolist(),scale=scale,transform=transform.tolist())
