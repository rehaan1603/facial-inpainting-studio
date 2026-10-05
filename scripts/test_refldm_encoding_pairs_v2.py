"""Prevent unmatched identity means and seed-level pseudo-replication."""
import unittest
from score_refldm_encoding_v2 import contrasts


class PairSummaryTests(unittest.TestCase):
    @staticmethod
    def row(target, identity, seed, arm, value):
        return dict(target=target, identity=identity, seed=seed, arm=arm,
                    metrics={'facenet_cosine': value})

    def test_missing_candidate_removes_matching_control_from_mean(self):
        rows = [self.row('a', '1', 17, 'joint_rgb', .8),
                self.row('a', '1', 17, 'separate_latents', .7),
                self.row('a', '1', 29, 'joint_rgb', None),
                self.row('a', '1', 29, 'separate_latents', .1)]
        result = contrasts(rows, ['a'])['facenet_cosine']
        self.assertEqual(result['n_pairs'], 1)
        self.assertEqual(result['n_identities'], 1)
        self.assertAlmostEqual(result['separate_mean'], .7)
        self.assertAlmostEqual(result['mean_paired_delta'], .1)

    def test_multiple_targets_and_seeds_remain_one_identity(self):
        rows = []
        for target, seeds, delta in [('a', [17,29,43], .1), ('b', [17], -.1)]:
            for seed in seeds:
                rows += [self.row(target, '1', seed, 'joint_rgb', .5+delta),
                         self.row(target, '1', seed, 'separate_latents', .5)]
        result = contrasts(rows, ['a','b'])['facenet_cosine']
        self.assertEqual(result['n_pairs'], 4)
        self.assertEqual(result['n_identities'], 1)
        self.assertAlmostEqual(result['identity_units'][0]['delta'], 0)
        self.assertEqual(result['improved'], 3)
        self.assertEqual(result['worsened'], 1)


if __name__=='__main__':
    unittest.main()
