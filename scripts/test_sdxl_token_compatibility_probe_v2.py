"""CPU failure-receipt tests; no model loading or GPU execution."""
import json
import tempfile
import unittest
from pathlib import Path
from sdxl_token_compatibility_probe_v2 import finish_receipt


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'receipt.json'

    def test_nonfinite_pass_is_rejected_and_readable(self):
        self.path.write_text(json.dumps({'status': 'passed', 'gradient_norm': float('nan')}))
        receipt = finish_receipt(self.path)
        self.assertEqual(receipt['status'], 'failed')
        self.assertIsNone(receipt['gradient_norm'])
        self.assertTrue(receipt['invalid_numeric_fields'])
        json.loads(self.path.read_text())

    def test_unfinished_receipt_retains_failure(self):
        self.path.write_text(json.dumps({'status': 'running', 'seed': 20261010}))
        receipt = finish_receipt(self.path, 'Nonfinite diagnostic serialization failed')
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(receipt['seed'], 20261010)
        self.assertIn('serialization', receipt['wrapper_exception'])

    def test_finite_pass_values_are_unchanged(self):
        saved = {'status': 'passed', 'gradient_norm': 0.1, 'neutral_max_absolute_error': 0.0}
        self.path.write_text(json.dumps(saved))
        self.assertEqual(finish_receipt(self.path), saved)


if __name__ == '__main__':
    unittest.main()
