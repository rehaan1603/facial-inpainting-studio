import unittest
import numpy as np
from src.preservation.proxy_calibration import box,fit,predict,mix

class ProxyCalibrationTests(unittest.TestCase):
    def test_box_matches_explicit_local_mean(self):
        x=np.arange(48).reshape(6,8);p=np.pad(x,2,mode='reflect')
        expected=np.array([[p[y:y+5,x:x+5].mean() for x in range(8)] for y in range(6)])
        np.testing.assert_allclose(box(x,2),expected)
    def test_calibration_recovers_useful_blend_and_preserves_outside(self):
        rng=np.random.default_rng(7);o=rng.integers(20,120,(32,32,3),dtype=np.uint8);g=o+80;t=o+20
        mask=np.zeros((32,32),bool);mask[4:28,4:28]=True
        model=fit([(o,g,t,mask)]);out,w=predict(o,g,mask,model)
        self.assertAlmostEqual(model['global_weight'],.25,places=6)
        np.testing.assert_array_equal(out[mask],t[mask]);np.testing.assert_array_equal(out[~mask],o[~mask])
    def test_zero_residual_and_invalid_weights(self):
        o=np.full((16,16,3),80,np.uint8);mask=np.ones((16,16),bool)
        model=fit([(o,o,o,mask)]);out,_=predict(o,o,mask,model);np.testing.assert_array_equal(out,o)
        with self.assertRaises(ValueError):mix(o,o,mask,float('nan'))
