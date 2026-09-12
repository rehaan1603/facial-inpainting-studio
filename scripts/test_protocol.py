"""Analytic tests for leakage and regional metrics, not model quality."""
import unittest
import numpy as np
from pilot import make_case, metrics, morph
from audit_annotations import check_mask
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

class ProtocolTests(unittest.TestCase):
    def test_no_hidden_pixel_leakage(self):
        a=np.zeros((256,256,3),np.float32); b=np.ones_like(a)
        ya,o,ma=make_case(a,12); yb,ob,mb=make_case(b,12)
        self.assertTrue(np.array_equal(o,ob))
        self.assertTrue(np.array_equal(ya[o],yb[o]))
        self.assertTrue(np.array_equal(ya[~o],a[~o]))
        self.assertTrue(np.array_equal(ma['under_4px'],mb['under_4px']))
    def test_metrics_separate_regions(self):
        x=np.zeros((8,8,3),np.float32); o=np.zeros((8,8),bool); o[2:6,2:6]=True
        p=x.copy(); p[o]=1
        r=metrics(p,x,x,o,o)
        self.assertEqual(r['hole_mae'],1); self.assertEqual(r['visible_mae'],0)
        self.assertIsNone(r['missed_mae']); self.assertIsNone(r['overcovered_mae'])
        p=x.copy(); p[~o]=1
        r=metrics(p,x,x,o,o)
        self.assertEqual(r['hole_mae'],0); self.assertEqual(r['visible_mae'],1)
    def test_morphology_polarity(self):
        m=np.zeros((32,32),bool); m[8:24,8:24]=True
        self.assertGreater(morph(m,2).sum(),m.sum())
        self.assertLess(morph(m,-2).sum(),m.sum())
    def test_binary_rgb_annotation(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'mask.png'
            a=np.zeros((512,512,3),np.uint8); a[20:40]=255
            Image.fromarray(a).save(p)
            self.assertIsNone(check_mask((p,'hq')))
            a[0,0]=[255,0,0]; Image.fromarray(a).save(p)
            self.assertIsNotNone(check_mask((p,'hq')))

if __name__=='__main__': unittest.main()
