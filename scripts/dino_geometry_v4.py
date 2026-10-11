"""Native37x37 dense geometry, reusing unchanged v3 matcher and fit gates."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from dense_feature_geometry_v3 import mutual_matches,fit_pairs

def grid(mask=None,padding=0):
    c=(np.arange(37)+.5)*512/37
    y,x=np.meshgrid(c,c,indexing='ij');coords=np.stack([x,y],-1).reshape(-1,2)
    keep=((coords>=48)&(coords<=464)).all(1)
    if mask is not None:
        if mask.shape!=(512,512) or mask.dtype!=bool:raise ValueError('Expected512x512 boolean mask')
        pixels=np.floor(coords+.5).astype(int)
        keep &= distance_transform_edt(~mask)[pixels[:,1],pixels[:,0]]>32
    if padding:keep &= coords[:,0]>=padding+32
    return coords,keep

def compare(a,b,mask,reference_mask=None,shift=0):
    if a.shape!=(1369,384) or b.shape!=(1369,384):raise ValueError('Expected native1369x384 features')
    coords,tk=grid(mask);_,rk=grid(reference_mask,padding=shift)
    pairs=mutual_matches(a[tk],b[rk]);xy=np.array([coords[tk][i] for i,j in pairs]);uv=np.array([coords[rk][j] for i,j in pairs])
    result=fit_pairs(xy,uv)
    if reference_mask is not None and 'transform' in result:
        m=np.asarray(result['transform']);source=coords[rk]
        result['known_transform_rms_pixels']=float(np.sqrt(((source@m[:,:2].T+m[:,2]-(source-[shift,0]))**2).sum(1).mean()))
    return result
