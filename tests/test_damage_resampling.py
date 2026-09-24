"""Synthetic client-image regressions; no model, dataset or GPU is needed."""
import unittest
import numpy as np
from PIL import Image
from webapp.geometry import fit_evidence, resize_binary_mask
from webapp.output import compose_prediction


class DamageResamplingTests(unittest.TestCase):
    def test_thin_component_survives_beside_large_hole(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 200] = 255
        mask[20:60, 20:60] = 255
        reduced = resize_binary_mask(Image.fromarray(mask), (256, 256))
        restored_support = np.asarray(reduced.resize((512, 512), Image.Resampling.NEAREST))
        self.assertTrue((restored_support[mask != 0] == 255).all())
        self.assertEqual(int((np.asarray(reduced) != 0).sum()), 550)

    def test_isolated_pixels_survive_fractional_downsampling(self):
        mask = np.zeros((701, 301), dtype=np.uint8)
        points = [(0, 0), (700, 300), (173, 121), (499, 230)]
        for y, x in points:
            mask[y, x] = 255
        reduced = np.asarray(resize_binary_mask(Image.fromarray(mask), (219, 512)))
        for y, x in points:
            self.assertEqual(reduced[y * 512 // 701, x * 219 // 301], 255)

    def test_model_mask_expansion_does_not_expand_final_edit(self):
        rgb = np.random.default_rng(11).integers(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, 200] = 255
        model_mask = np.asarray(resize_binary_mask(Image.fromarray(mask), (256, 256))) != 0
        prediction = np.zeros((256, 256, 3), dtype=float)
        prediction[model_mask] = 1
        result = np.asarray(compose_prediction(Image.fromarray(rgb), Image.fromarray(mask), prediction))
        self.assertTrue(np.array_equal(result[mask == 0], rgb[mask == 0]))
        self.assertTrue((result[mask != 0] > 0).all())

    def test_same_size_threshold_and_enlargement_are_unchanged(self):
        mask = np.array([[0, 127, 128], [255, 25, 254]], dtype=np.uint8)
        expected = (mask >= 128).astype('uint8') * 255
        self.assertTrue(np.array_equal(np.asarray(resize_binary_mask(Image.fromarray(mask), (3, 2))), expected))
        enlarged = resize_binary_mask(Image.fromarray(mask), (11, 7))
        self.assertTrue(np.array_equal(np.asarray(enlarged), np.asarray(Image.fromarray(expected).resize((11, 7), Image.Resampling.NEAREST))))

    def test_fitted_evidence_keeps_missing_partial_lines_and_padding(self):
        for width, height in [(1200, 701), (701, 1200)]:
            source = Image.new('RGB', (width, height), (90, 100, 110))
            mask = np.zeros((height, width), dtype=np.uint8)
            confidence = np.full((height, width), 255, dtype=np.uint8)
            mask[100:height-100, 200] = 255
            confidence[100:height-100, 200] = 0
            mask[100:height-100, 400] = 255
            confidence[100:height-100, 400] = 128
            _, fitted_mask, fitted_confidence, geometry = fit_evidence(source, Image.fromarray(mask), Image.fromarray(confidence))
            m, c = np.asarray(fitted_mask), np.asarray(fitted_confidence)
            self.assertTrue((c == 0).any())
            self.assertTrue((c == 128).any())
            self.assertTrue((m[c < 255] == 255).all())
            self.assertTrue((c[m == 0] == 255).all())
            x, y = geometry['offset']
            w, h = geometry['fitted_size']
            self.assertLess(abs(w / h - width / height), .005)
            outside = np.ones(m.shape, dtype=bool)
            outside[y:y+h, x:x+w] = False
            self.assertTrue((m[outside] == 0).all())
            self.assertTrue((c[outside] == 255).all())

    def test_square_same_size_preserves_all_three_arrays(self):
        rng = np.random.default_rng(17)
        rgb = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        confidence = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        result = fit_evidence(*map(Image.fromarray, (rgb, mask, confidence)))
        for actual, expected in zip(result[:3], (rgb, mask, confidence)):
            self.assertTrue(np.array_equal(np.asarray(actual), expected))

    def test_invalid_resize_dimensions_rejected(self):
        for size in [(0, 2), (2, -1), (True, 2), (2.5, 2), (2,)]:
            with self.assertRaises(ValueError):
                resize_binary_mask(Image.new('L', (4, 4)), size)


if __name__ == '__main__':
    unittest.main()
