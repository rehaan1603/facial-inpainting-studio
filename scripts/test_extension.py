import unittest
import numpy as np
from evaluate_extension import probability_blend
class SoftCompositing(unittest.TestCase):
    def test_endpoints_and_outside_preservation(self):
        observed=np.full((8,8,3),.2,np.float32);raw=np.full_like(observed,.8);mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        none=probability_blend(raw,observed,np.zeros((8,8),np.float32),mask)
        full=probability_blend(raw,observed,np.ones((8,8),np.float32),mask)
        self.assertTrue(np.array_equal(none,observed));self.assertTrue(np.array_equal(full[mask],raw[mask]));self.assertTrue(np.array_equal(full[~mask],observed[~mask]))
        half=probability_blend(raw,observed,np.full((8,8),.5,np.float32),mask)
        self.assertTrue(np.allclose(half[mask],.5));self.assertTrue(np.array_equal(half[~mask],observed[~mask]))
if __name__=='__main__':unittest.main()
