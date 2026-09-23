import unittest
import numpy as np
from PIL import Image
from webapp.geometry import fit_evidence


class UploadGeometryTests(unittest.TestCase):
    def test_portrait_landscape_maps_and_padding(self):
        for width, height in [(300, 600), (600, 300), (301, 701)]:
            source=Image.new('RGB',(width,height),(10,20,30))
            mask=Image.new('L',source.size,255)
            confidence=Image.new('L',source.size,100)
            image,mask,confidence,g=fit_evidence(source,mask,confidence)
            a,m,c=map(np.asarray,(image,mask,confidence))
            x,y=g['offset'];w,h=g['fitted_size']
            self.assertLessEqual(abs(w/h-width/height), .005)
            self.assertTrue((a[y:y+h,x:x+w]==[10,20,30]).all())
            self.assertTrue((m[y:y+h,x:x+w]==255).all())
            self.assertTrue((c[y:y+h,x:x+w]==100).all())
            outside=np.ones(m.shape,dtype=bool);outside[y:y+h,x:x+w]=False
            self.assertTrue((m[outside]==0).all())
            self.assertTrue((c[outside]==255).all())

    def test_square_unchanged(self):
        rng=np.random.default_rng(17)
        a=rng.integers(0,256,(512,512,3),dtype=np.uint8)
        image,_,_,_=fit_evidence(Image.fromarray(a),Image.new('L',(512,512)),Image.new('L',(512,512),255))
        self.assertTrue(np.array_equal(a,np.asarray(image)))

    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            fit_evidence(Image.new('RGB',(30,60)),Image.new('L',(60,30)),Image.new('L',(30,60)))
