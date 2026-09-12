"""Run directly in reference_env; Poisson tests skip clearly without OpenCV."""
import unittest
from unittest.mock import patch

import numpy as np

from reference_blending import harmonize_reference

try:
    import cv2
except ImportError:
    cv2 = None


class ContractAndFallbacks(unittest.TestCase):
    def setUp(self):
        self.observed = np.full((48, 48, 3), (65, 85, 105), np.uint8)
        self.generated = np.full_like(self.observed, 195)
        self.mask = np.zeros((48, 48), dtype=bool)
        self.mask[12:36, 12:36] = True

    def test_hard_composition_and_inputs_unchanged(self):
        originals = [array.copy() for array in (self.generated, self.observed, self.mask)]
        result, meta = harmonize_reference(self.generated, self.observed, self.mask, "hard")
        self.assertTrue(np.array_equal(result[self.mask], self.generated[self.mask]))
        self.assertTrue(np.array_equal(result[~self.mask], self.observed[~self.mask]))
        self.assertEqual(meta["mode"], "hard")
        self.assertFalse(meta["fallback"])
        for array, original in zip((self.generated, self.observed, self.mask), originals):
            self.assertTrue(np.array_equal(array, original))

    def test_invalid_contract_is_rejected(self):
        with self.assertRaises(TypeError):
            harmonize_reference(self.generated.astype(float), self.observed, self.mask)
        with self.assertRaises(TypeError):
            harmonize_reference(self.generated, self.observed, self.mask.astype(np.uint8))
        with self.assertRaises(ValueError):
            harmonize_reference(self.generated[:-1], self.observed, self.mask)
        with self.assertRaises(ValueError):
            harmonize_reference(self.generated, self.observed, self.mask[:-1])
        with self.assertRaises(ValueError):
            harmonize_reference(self.generated, self.observed, self.mask, "unknown")

    def test_empty_mask_returns_exact_observed_and_explicit_reason(self):
        result, meta = harmonize_reference(self.generated, self.observed, np.zeros_like(self.mask))
        self.assertTrue(np.array_equal(result, self.observed))
        self.assertEqual(meta["mode"], "unchanged")
        self.assertEqual(meta["fallback_reason"], "empty_mask")

    def test_tiny_or_thin_masks_report_hard_fallback(self):
        for region in ((slice(20, 21), slice(20, 21)), (slice(5, 40), slice(20, 22))):
            with self.subTest(region=region):
                mask = np.zeros_like(self.mask)
                mask[region] = True
                result, meta = harmonize_reference(self.generated, self.observed, mask)
                self.assertEqual(meta["fallback_reason"], "mask_has_no_3x3_interior")
                self.assertTrue(meta["fallback"])
                self.assertTrue(np.array_equal(result[mask], self.generated[mask]))
                self.assertTrue(np.array_equal(result[~mask], self.observed[~mask]))

    def test_each_image_border_has_explicit_fallback(self):
        for region in ((slice(0, 8), slice(15, 25)), (slice(-8, None), slice(15, 25)), (slice(15, 25), slice(0, 8)), (slice(15, 25), slice(-8, None))):
            mask = np.zeros_like(self.mask)
            mask[region] = True
            result, meta = harmonize_reference(self.generated, self.observed, mask)
            self.assertEqual(meta["fallback_reason"], "mask_touches_image_border")
            self.assertTrue(np.array_equal(result[~mask], self.observed[~mask]))

    def test_near_border_reports_missing_computational_collar(self):
        mask = np.zeros_like(self.mask)
        mask[1:9, 15:25] = True
        result, meta = harmonize_reference(self.generated, self.observed, mask)
        self.assertEqual(meta["fallback_reason"], "insufficient_known_boundary_collar")
        self.assertTrue(np.array_equal(result[mask], self.generated[mask]))
        self.assertTrue(np.array_equal(result[~mask], self.observed[~mask]))


@unittest.skipIf(cv2 is None, "OpenCV is absent; run in reference_env to verify actual Poisson blending.")
class PoissonBehaviour(unittest.TestCase):
    def test_synthetic_colour_seam_reduces_without_target_or_outside_changes(self):
        observed = np.full((80, 96, 3), (65, 85, 105), np.uint8)
        generated = np.full_like(observed, (175, 165, 155))
        mask = np.zeros(observed.shape[:2], dtype=bool)
        mask[15:65, 18:78] = True
        # A contrasting shape verifies that generated interior gradients survive.
        generated[32:48, 38:58] = (205, 205, 205)
        observed[mask] = (15, 20, 25)  # Submitted occlusion, not target RGB.
        before = [array.copy() for array in (generated, observed, mask)]
        result, meta = harmonize_reference(generated, observed, mask)
        self.assertEqual(meta["mode"], "poisson")
        self.assertFalse(meta["fallback"])
        self.assertTrue(np.array_equal(result[~mask], observed[~mask]))
        known = observed[14, 18:78].astype(float)
        hard_seam = np.abs(generated[15, 18:78].astype(float) - known).mean()
        blended_seam = np.abs(result[15, 18:78].astype(float) - known).mean()
        self.assertLess(blended_seam, hard_seam * 0.2)
        self.assertGreater(result[35:45, 42:54].astype(float).mean() - result[22:28, 42:54].astype(float).mean(), 15)
        for array, original in zip((generated, observed, mask), before):
            self.assertTrue(np.array_equal(array, original))

    def test_disconnected_mask_and_known_internal_hole_are_exact_outside(self):
        rng = np.random.default_rng(7)
        observed = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, observed.shape, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=bool)
        mask[10:40, 10:40] = True
        mask[19:29, 19:29] = False
        mask[50:66, 52:68] = True
        result, meta = harmonize_reference(generated, observed, mask)
        self.assertEqual(meta["mode"], "poisson")
        self.assertTrue(np.array_equal(result[~mask], observed[~mask]))

    def test_opencv_failure_is_not_silently_replaced(self):
        image = np.zeros((32, 32, 3), np.uint8)
        mask = np.zeros((32, 32), bool)
        mask[8:24, 8:24] = True
        with patch.object(cv2, "seamlessClone", side_effect=cv2.error("synthetic solver failure")):
            with self.assertRaisesRegex(RuntimeError, "Poisson harmonization failed"):
                harmonize_reference(image, image, mask)


if __name__ == "__main__":
    unittest.main()
