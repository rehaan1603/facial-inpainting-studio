"""Geometry prerequisite tests with synthetic coordinates; no model or faces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from spatial_reference_geometry_v1 import sanitize, safe_anchors, fit_geometry, estimate_geometry


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((128,128),bool)
        self.mask[20:45,20:108] = True
        self.shape = np.random.default_rng(14).uniform([15,60],[113,115],(68,2))
        self.shape[17:27] = np.array([60,27])
        self.shape[36:48] = np.array([60,34])

    def test_hidden_pixels_never_reach_extractor(self):
        seen=[]
        class Detector:
            def shape(self,rgb):
                seen.append(rgb.copy())
                raise ValueError('no face')
        a=np.zeros((128,128,3),np.uint8); b=a.copy();b[self.mask]=255
        refs=[a]*4
        x=estimate_geometry(a,self.mask,refs,Detector())
        y=estimate_geometry(b,self.mask,refs,Detector())
        self.assertEqual(x,y);np.testing.assert_array_equal(seen[0],seen[1])
        self.assertTrue((seen[0][self.mask]==128).all())
        self.assertEqual(x['status'],'fallback')

    def test_target_eye_brow_predictions_do_not_affect_fit(self):
        refs=[self.shape.copy() for _ in range(4)]
        a=fit_geometry(self.shape,[5,5,123,123],refs,self.mask)
        other=self.shape.copy();other[17:27]=1000;other[36:48]=-1000
        b=fit_geometry(other,[5,5,123,123],refs,self.mask)
        self.assertEqual(a,b);self.assertEqual(a['status'],'usable')

    def test_similarity_transform_and_validation(self):
        ref=self.shape*.8+np.array([4,7])
        result=fit_geometry(self.shape,[5,5,123,123],[ref]*4,self.mask)
        self.assertEqual(result['status'],'usable')
        self.assertLess(result['references'][0]['held_out_normalized_rms'],1e-10)
        matrix=np.array(result['references'][0]['transform'])
        self.assertGreater(np.linalg.det(matrix[:,:2]),0)
        self.assertFalse(set(result['fit_anchors'])&set(result['validation_anchors']))

    def test_missing_or_misleading_references_fallback(self):
        a=fit_geometry(self.shape,[5,5,123,123],[self.shape,None,None,None],self.mask)
        self.assertEqual(a['status'],'fallback')
        wrong=self.shape.copy();wrong[36:48,0]+=80
        b=fit_geometry(self.shape,[5,5,123,123],[self.shape,wrong,self.shape,self.shape],self.mask)
        self.assertEqual(b['status'],'fallback')
        self.assertIn('disagree',b['reason'])

    def test_no_unsafe_anchors(self):
        self.shape[0]=[21,21];self.shape[1]=[21,46];self.shape[2]=[-1,80]
        indices=safe_anchors(self.shape,self.mask)
        self.assertFalse(set([0,1,2])&set(indices))
        self.assertFalse(set(range(17,27))&set(indices))
        self.assertFalse(set(range(36,48))&set(indices))

    def test_inputs_unchanged_and_invalid_masks_rejected(self):
        a=np.zeros((128,128,3),np.uint8);b=a.copy()
        sanitize(a,self.mask);np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):sanitize(a,np.ones((128,128),bool))


if __name__=='__main__':unittest.main()
