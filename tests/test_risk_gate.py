import unittest
import numpy as np
from scripts.reference_risk_diagnostic import gate

class RiskGateTests(unittest.TestCase):
    def test_equal_coverage_and_known_pixel_preservation(self):
        observed=np.zeros((8,8,3),dtype=np.uint8);generated=np.full_like(observed,200)
        mask=np.zeros((8,8),bool);mask[1:7,1:7]=True
        a,ai=gate(observed,generated,mask,np.arange(64).reshape(8,8))
        b,bi=gate(observed,generated,mask,np.zeros((8,8)))
        self.assertEqual(len(ai),9);self.assertEqual(len(bi),9)
        self.assertTrue(np.array_equal(bi,np.flatnonzero(mask)[:9]))
        self.assertTrue((a[~mask]==0).all());self.assertTrue((b[~mask]==0).all())
        self.assertEqual(np.count_nonzero(a[...,0]),27)
        self.assertEqual(np.count_nonzero(b[...,0]),27)
