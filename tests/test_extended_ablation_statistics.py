import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_extended_ablations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PairedStatisticsTests(unittest.TestCase):
    def test_exact_two_subject_sign_flip(self):
        result=module.paired([1.,1.])
        self.assertEqual(result['p_exact'],.5)
        self.assertEqual(result['ci95'],[1.,1.])

    def test_zero_difference(self):
        self.assertEqual(module.paired([0.,0.,0.])['p_exact'],1.)

    def test_missing_population_is_not_zero_score(self):
        self.assertIsNone(module.paired([])['mean'])

    def test_holm_preserves_original_order(self):
        self.assertEqual(module.holm([.04,.01,.03]),[.06,.03,.06])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            module.paired([float('nan')])

if __name__=='__main__':
    unittest.main()
