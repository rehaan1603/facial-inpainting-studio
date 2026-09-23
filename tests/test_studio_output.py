import unittest
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from webapp.output import compose_prediction
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_reference_inpaint import neutralize


class StudioOutputTests(unittest.TestCase):
    def test_occluder_colour_cannot_enter_conditioning(self):
        a=np.full((32,32,3),80,dtype=np.uint8);b=a.copy()
        mask=np.zeros((32,32),dtype=bool);mask[8:24,8:24]=True
        a[mask]=0;b[mask]=255
        self.assertTrue(np.array_equal(neutralize(a,mask),neutralize(b,mask)))
        self.assertTrue(np.array_equal(neutralize(a,mask)[~mask],a[~mask]))

    def test_low_resolution_prediction_preserves_visible_full_resolution_pixels(self):
        rgb = np.random.default_rng(7).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[101:203, 99:211] = 255
        result = np.asarray(compose_prediction(Image.fromarray(rgb), Image.fromarray(mask), np.ones((256, 256, 3))))
        self.assertTrue(np.array_equal(result[mask == 0], rgb[mask == 0]))
        self.assertTrue((result[mask > 0] == 255).all())

    def test_invalid_prediction_rejected(self):
        with self.assertRaises(ValueError):
            compose_prediction(Image.new('RGB', (32, 32)), Image.new('L', (32, 32)), np.full((16, 16, 3), np.nan))
