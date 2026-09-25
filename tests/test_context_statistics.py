import unittest
from src.preservation.context_statistics import identity_means, contrast, holm


def row(identity, seed, value):
    return {'identity': identity, 'seed': seed, 'evaluation': {'status': 'complete', 'metrics': {'score': value}}}


class ContextStatisticsTests(unittest.TestCase):
    def test_seeds_do_not_inflate_statistical_unit(self):
        left = [row(str(i), s, float(i + s)) for i in range(4) for s in [17, 29]]
        right = [row(str(i), s, float(i + s - 1)) for i in range(4) for s in [17, 29]]
        result = contrast(left, right, 'score', [17, 29])
        self.assertEqual(result['identities'], 4)
        self.assertEqual(result['mean_delta'], 1)
        self.assertEqual(result['p_exact'], .125)

    def test_failure_excludes_entire_identity_and_is_reported(self):
        rows = [row('a', 17, 1), row('a', 29, None), row('b', 17, 2)]
        means, excluded = identity_means(rows, 'score', [17, 29])
        self.assertEqual(means, {})
        self.assertEqual(excluded, ['a', 'b'])
        with self.assertRaises(ValueError):
            identity_means([row('a', 17, 1), row('a', 17, 2)], 'score', [17, 29])

    def test_holm_uses_all_prespecified_contrasts(self):
        tests = {'a': {'p_exact': .01}, 'b': {'p_exact': .03}, 'c': {'p_exact': None}}
        holm(tests)
        self.assertEqual(tests['a']['p_holm'], .03)
        self.assertEqual(tests['b']['p_holm'], .06)
        self.assertEqual(tests['c']['p_holm'], 1)


if __name__ == '__main__':
    unittest.main()
