"""Synthetic safeguards for the new inference input boundary."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('structure_runner', ROOT/'scripts/run_structure_transport.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class InputScopeTests(unittest.TestCase):
    def test_missing_rgb_payload_neutralized_without_altering_visible_input(self):
        original = np.arange(12*12*3, dtype=np.uint16).reshape(12, 12, 3).astype(np.uint8)
        old = original.copy()
        mask = np.zeros((12,12), dtype=bool)
        mask[2:8,4:9] = True
        other = original.copy()
        other[mask] = 201
        left = runner.sanitized_observation(original, mask)
        right = runner.sanitized_observation(other, mask)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left[~mask], original[~mask])
        self.assertTrue(np.all(left[mask] == 128))
        np.testing.assert_array_equal(original, old)

    def test_ambiguous_mask_and_geometry_rejected(self):
        image = np.zeros((12,12,3), np.uint8)
        for mask in [np.zeros((12,12), np.uint8), np.zeros((11,12), bool)]:
            with self.assertRaises(ValueError):
                runner.sanitized_observation(image, mask)


if __name__ == '__main__':
    unittest.main()
