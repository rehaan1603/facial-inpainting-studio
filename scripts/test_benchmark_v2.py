import unittest
import numpy as np
from benchmark_v2 import brush,case

class BenchmarkTests(unittest.TestCase):
    def test_independent_corruption(self):
        true=brush(43,256)
        a=np.zeros((256,256,3),np.float32); b=np.ones_like(a)
        ya,ma,pa=case(a,true,51); yb,mb,pb=case(b,true,51)
        np.testing.assert_array_equal(ya[true],yb[true])
        np.testing.assert_array_equal(ya[~true],a[~true])
        for name in ma:
            np.testing.assert_array_equal(ma[name],mb[name])
            self.assertEqual(ma[name].dtype,np.dtype(bool))
        self.assertEqual(pa,pb)
    def test_conditions_and_translation(self):
        true=brush(1,256)
        _,m,_=case(np.zeros((256,256,3),np.float32),true,10)
        np.testing.assert_array_equal(m['accurate'],true)
        self.assertFalse(np.any(m['under'] & ~true))
        self.assertFalse(np.any(true & ~m['over']))
        self.assertFalse(np.array_equal(m['mixed_boundary'],true))

if __name__=='__main__': unittest.main()
