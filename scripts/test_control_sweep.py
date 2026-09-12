import unittest
import numpy as np
from control_sweep import feather,choose


class Controls(unittest.TestCase):
    def test_feather_preserves_outside_and_limits(self):
        mask=np.zeros((16,16),bool);mask[2:14,2:14]=True
        alpha=feather(mask,4)
        self.assertTrue((alpha[~mask]==0).all())
        self.assertEqual(alpha[2,2],.25)
        self.assertEqual(alpha[8,8],1)
        self.assertTrue(np.array_equal(feather(mask,0),mask))
        self.assertTrue((feather(np.zeros_like(mask),4)==0).all())

    def test_selection_ignores_assessment(self):
        rows=[{'partition':'tuning','radius':r,'feather':0,'full_face_lpips':v} for r,v in [(0,.2),(8,.1)]]
        rows.append({'partition':'assessment','radius':0,'feather':0,'full_face_lpips':-999})
        self.assertEqual(choose(rows)[0],(8,0))


if __name__=='__main__':unittest.main()
