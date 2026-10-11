import unittest
import numpy as np
from dino_geometry_v4 import grid,compare
from dense_feature_geometry_v3 import fit_pairs,mutual_matches

class GridTests(unittest.TestCase):
    def test_coordinates(self):
        xy,keep=grid();self.assertEqual(xy.shape,(1369,2));np.testing.assert_allclose(xy[0],[256/37,256/37]);self.assertTrue(keep.any())
    def test_mask(self):
        mask=np.zeros((512,512),bool);mask[180:290,130:390]=True
        xy,k=grid(mask);self.assertFalse(((xy[k,0]>130)&(xy[k,0]<390)&(xy[k,1]>180)&(xy[k,1]<290)).any())
        with self.assertRaises(ValueError):grid(np.zeros((64,64),bool))
        with self.assertRaises(ValueError):grid(mask.astype(float))
    def test_geometry(self):
        xy,_=grid();got=fit_pairs(xy,xy+[8,0]);self.assertTrue(got['eligible']);self.assertLess(got['held_out_rms_pixels'],.001)
    def test_native_identity(self):
        a=np.random.default_rng(14).normal(size=(1369,384)).astype(np.float32);mask=np.zeros((512,512),bool);mask[200:260,150:380]=True
        got=compare(a,a,mask,mask);self.assertTrue(got['eligible']);self.assertLess(got['known_transform_rms_pixels'],1e-4)

if __name__=='__main__':unittest.main()
