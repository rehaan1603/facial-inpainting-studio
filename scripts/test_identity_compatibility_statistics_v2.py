import unittest
from score_identity_compatibility_v2 import paired

class PairedStatisticsTests(unittest.TestCase):
    def test_missing_measurement_keeps_incomplete_support_visible(self):
        rows=[]
        for seed in [17,29,43]:
            for arm in ['baseline','identity']:
                rows.append(dict(identity='a',seed=seed,condition='clean',arm=arm,
                    metrics={'lpips':None if seed==29 and arm=='identity' else .2 if arm=='baseline' else .1}))
        p=paired(rows,'identity','clean',None,'lpips')
        self.assertEqual(p['units'][0]['paired_seeds'],2)
        self.assertAlmostEqual(p['mean_delta'],-.1)

    def test_identity_weighting_does_not_treat_seeds_as_independent_people(self):
        rows=[]
        for person,seeds,gain in [('a',[17,29,43],.3),('b',[17],.1)]:
            for seed in seeds:
                for arm,value in [('baseline',.5),('identity',.5+gain)]:
                    rows.append(dict(identity=person,seed=seed,condition='clean',arm=arm,metrics={'psnr':value}))
        p=paired(rows,'identity','clean',None,'psnr')
        self.assertEqual(p['n_identities'],2)
        self.assertAlmostEqual(p['mean_delta'],.2)

if __name__=='__main__':unittest.main()
