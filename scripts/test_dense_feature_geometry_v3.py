import unittest
import numpy as np
from dense_feature_geometry_v3 import mutual_matches,fit_pairs,grid_selection

class DenseTests(unittest.TestCase):
    def test_known_permutation(self):
        x=np.eye(20,dtype=np.float32);p=np.random.default_rng(4).permutation(20)
        self.assertEqual(mutual_matches(x,x[p]),[(i,int(np.where(p==i)[0][0])) for i in range(20)])
    def test_ambiguous_and_opposite_reject(self):
        self.assertEqual(mutual_matches(np.ones((20,4)),np.ones((20,4))),[])
        self.assertEqual(mutual_matches(np.eye(20),-np.eye(20)),[])
    def test_known_transform_and_order(self):
        xy,_=grid_selection();uv=(xy-np.array([12.,8.]))/1.1
        result=fit_pairs(xy,uv);self.assertTrue(result['eligible']);self.assertLess(result['held_out_rms_pixels'],.001)
        self.assertEqual(result,fit_pairs(xy[::-1],uv[::-1]))
    def test_incoherent_geometry(self):
        xy,_=grid_selection();uv=xy[np.random.default_rng(92).permutation(256)]
        self.assertFalse(fit_pairs(xy,uv)['eligible'])
    def test_mask_grid(self):
        mask=np.zeros((512,512),bool);mask[180:290,130:390]=True
        coords,keep=grid_selection(mask)
        self.assertFalse(((coords[keep,1]>180)&(coords[keep,1]<290)&(coords[keep,0]>130)&(coords[keep,0]<390)).any())

if __name__=='__main__':unittest.main()
