import unittest
import numpy as np
from src.preservation.context_support import support, probes, calibration_input, probe_loss, choose_radius, compose


class ContextSupportTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((128, 128), bool)
        self.mask[45:80, 40:85] = True

    def test_common_probes_are_reliable_disjoint_and_deterministic(self):
        hidden, receipt = probes(self.mask, [0, 2, 4, 8])
        self.assertTrue(receipt['eligible'])
        self.assertFalse((hidden & support(self.mask, 8)).any())
        self.assertEqual(int(hidden.sum()), 64 * receipt['count'])
        np.testing.assert_array_equal(hidden, probes(self.mask, [0, 2, 4, 8])[0])

    def test_hidden_values_cannot_reach_model(self):
        observed = np.random.default_rng(1).random((128, 128, 3))
        hidden, _ = probes(self.mask, [0, 2, 4, 8])
        changed = observed.copy()
        changed[hidden | self.mask] = 100
        for radius in [0, 2, 4, 8]:
            a, m = calibration_input(observed, self.mask, hidden, radius)
            b, _ = calibration_input(changed, self.mask, hidden, radius)
            np.testing.assert_array_equal(a, b)
            self.assertTrue((m[self.mask]).all())
            self.assertTrue((a[m] == 0).all())

    def test_selection_abstains_on_partial_failures_and_ties_are_stable(self):
        self.assertEqual(choose_radius({0: 1, 2: 1}, [2, 0]), (0, 'selected'))
        self.assertEqual(choose_radius({0: 1, 2: np.nan}, [0, 2]), (0, 'abstained'))
        self.assertEqual(choose_radius({0: 1}, [0, 2]), (0, 'abstained'))
        self.assertEqual(choose_radius({0: 1}, [0], eligible=False), (0, 'abstained'))

    def test_missing_context_abstains_and_preservation_is_exact(self):
        m = np.ones((128, 128), bool); m[0, 0] = False
        _, info = probes(m, [0, 2, 4, 8])
        self.assertFalse(info['eligible'])
        o = np.random.default_rng(3).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        p = np.full(o.shape, 120.5)
        result = compose(o, p, self.mask)
        np.testing.assert_array_equal(result[~self.mask], o[~self.mask])
        self.assertTrue((result[self.mask] == 120).all())

    def test_score_uses_only_shared_hidden_pixels(self):
        o = np.zeros((128, 128, 3))
        hidden, _ = probes(self.mask, [0, 2, 4, 8])
        p = o.copy(); p[hidden] = .2; p[~hidden] = 100
        self.assertAlmostEqual(probe_loss(p, o, hidden), .2)


if __name__ == '__main__':
    unittest.main()
