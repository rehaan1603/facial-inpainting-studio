import unittest
import numpy as np
from src.local_correspondence.features import CANONICAL,similarity_transform


class CorrespondenceTests(unittest.TestCase):
    def test_known_rotation_scale_translation(self):
        angle=.21;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        source=CANONICAL@r*1.4+[13,-8]
        matrix,error=similarity_transform(source)
        self.assertTrue(np.allclose(source@matrix[:,:2].T+matrix[:,2],CANONICAL,atol=1e-8))
        self.assertLess(error,1e-10)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):similarity_transform(np.zeros((5,2)))
        with self.assertRaises(ValueError):similarity_transform(np.full((5,2),np.nan))


if __name__=='__main__':unittest.main()
