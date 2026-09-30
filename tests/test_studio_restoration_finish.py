import sys,numpy as np
sys.path.insert(0,'scripts')
from studio_restoration_finish import finish_restoration
x=np.full((32,32,3),100,np.uint8);g=np.full_like(x,180);m=np.zeros((32,32),bool);m[8:24,8:24]=1
r,mode=finish_restoration(x,g,m);assert np.array_equal(r[~m],x[~m]);assert np.abs(r[m].astype(float)-100).mean()<2
for mask in [np.ones((32,32),bool),np.zeros((32,32),bool)]:
 r,_=finish_restoration(x,g,mask);assert np.array_equal(r,np.where(mask[...,None],g,x))
try:finish_restoration(x,g[:3],m)
except ValueError:pass
else:raise AssertionError('geometry mismatch accepted')
print('Finishing checks passed: lighting, outside preservation, full/empty masks, invalid geometry.')
