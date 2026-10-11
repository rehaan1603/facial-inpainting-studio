import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from visible_feature_geometry_v2 import features,match_geometry

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.random.default_rng(72).integers(0,256,(512,512,3),dtype=np.uint8)
        self.mask=np.zeros((512,512),bool);self.mask[200:260,150:380]=True
    def test_hidden_pixels_invariant(self):
        other=self.rgb.copy();other[self.mask]=0
        a,ad=features(self.rgb,self.mask);b,bd=features(other,self.mask)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ad,bd)
    def test_identical_visible_registration(self):
        result=match_geometry(self.rgb,self.mask,self.rgb)
        self.assertTrue(result['eligible'],result)
        self.assertLess(result['held_out_rms_pixels'],.01)
    def test_blank_fails(self):
        self.assertFalse(match_geometry(np.zeros_like(self.rgb),self.mask,np.zeros_like(self.rgb))['eligible'])
    def test_repeatable_and_immutable(self):
        original=self.rgb.copy()
        self.assertEqual(match_geometry(self.rgb,self.mask,self.rgb),match_geometry(self.rgb,self.mask,self.rgb))
        np.testing.assert_array_equal(self.rgb,original)

if __name__=='__main__':unittest.main()
