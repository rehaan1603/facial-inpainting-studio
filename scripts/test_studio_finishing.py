"""Check the actual optional finishing process and exact raw-output escape hatch."""
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webapp.finishing import match_edges


class FinishingTests(unittest.TestCase):
    def fixture(self, folder):
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True
        observed = np.full((64, 64, 3), 120, dtype='uint8')
        observed[mask] = 0
        raw = observed.copy()
        raw[mask] = 210
        for name, pixels in [('input', observed), ('effective_mask', mask.astype('uint8')*255), ('result_raw', raw)]:
            Image.fromarray(pixels).save(folder / (name + '.png'))
        return observed, mask

    def test_hard_preserves_raw_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            raw = (folder / 'result_raw.png').read_bytes()
            metadata, warnings = match_edges(folder, 'hard')
            self.assertEqual(raw, (folder / 'result.png').read_bytes())
            self.assertEqual(metadata['mode'], 'hard')
            self.assertEqual(warnings, [])

    def test_actual_worker_fixes_offset_without_leaking_black_occlusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            observed, mask = self.fixture(folder)
            raw_hash = hashlib.sha256((folder / 'result_raw.png').read_bytes()).hexdigest()
            metadata, warnings = match_edges(folder, 'poisson')
            self.assertEqual(metadata['mode'], 'poisson', 'Real finishing worker must run, not fallback.')
            self.assertEqual(warnings, [])
            result = np.asarray(Image.open(folder / 'result.png'))
            self.assertTrue(np.array_equal(result[~mask], observed[~mask]))
            self.assertLess(np.abs(result[mask].astype(float) - 120).mean(), 3)
            self.assertEqual(raw_hash, hashlib.sha256((folder / 'result_raw.png').read_bytes()).hexdigest())

    def test_invalid_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                match_edges(Path(tmp), 'unknown')
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_worker_failure_preserves_raw_result_and_reports_fallback(self):
        for error in [subprocess.CalledProcessError(1, ['worker']),
                      subprocess.TimeoutExpired(['worker'], 60)]:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.fixture(folder)
                raw = (folder / 'result_raw.png').read_bytes()
                with patch('webapp.finishing.subprocess.run', side_effect=error):
                    metadata, warnings = match_edges(folder, 'poisson')
                self.assertEqual(raw, (folder / 'result.png').read_bytes())
                self.assertTrue(metadata['fallback'])
                self.assertEqual(metadata['failure_type'], type(error).__name__)
                self.assertEqual(metadata['requested_mode'], 'poisson')
                self.assertTrue(warnings)

    def test_missing_worker_receipt_preserves_raw_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            raw = (folder / 'result_raw.png').read_bytes()
            with patch('webapp.finishing.subprocess.run'):
                metadata, warnings = match_edges(folder, 'poisson')
            self.assertEqual(raw, (folder / 'result.png').read_bytes())
            self.assertEqual(metadata['fallback_reason'], 'edge_matching_worker_failed')
            self.assertTrue(warnings)


if __name__ == '__main__':
    unittest.main()
