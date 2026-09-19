import numpy as np

def pose_proxy(landmarks):
    points=np.asarray(landmarks,dtype=float);left,right=sorted(points[:2],key=lambda p:p[0])
    axis=right-left;distance=max(float(np.linalg.norm(axis)),1.)
    return {'yaw_proxy':float(np.clip(np.dot(points[2]-(left+right)/2,axis/distance)/distance,-1,1)),
            'roll_radians':float(np.arctan2(axis[1],axis[0]))}

def similarity(a,b):
    if a is None or b is None:return .5
    return float(np.exp(-2*abs(a['yaw_proxy']-b['yaw_proxy'])-abs(a['roll_radians']-b['roll_radians'])))
