"""Regression coverage for paired conditions, missing detections and identity units."""
import unittest
from score_spatial_reference_rejection_v1 import compare, IDS, KEYS


class PairTests(unittest.TestCase):
    def rows(self):
        return [dict(identity=i,seed=s,condition=c,arm=a,metrics={k:float(j*10+s+(100 if c=='wrong_patch' else 0)+(1 if a=='oracle_region' else 0)) for k in KEYS})
                for j,i in enumerate(IDS) for s in [17,29,43] for c in ['clean','wrong_patch'] for a in ['baseline','oracle_region']]

    def test_condition_and_seed_match(self):
        r=self.rows();v=compare(r,'oracle_region','baseline','wrong_patch')['psnr']
        self.assertEqual(v['n_pairs'],12);self.assertEqual(v['n_identities'],4)
        self.assertEqual(v['mean_delta'],1.0);self.assertEqual(v['improved'],12)
        self.assertEqual(v['identity_bootstrap_95_ci'],[1.0,1.0])
        self.assertEqual(compare(r,'oracle_region','baseline','clean',17)['lpips']['n_pairs'],4)

    def test_missing_detection_uses_matched_subset(self):
        r=self.rows()
        next(x for x in r if x['arm']=='oracle_region' and x['condition']=='wrong_patch' and x['identity']=='263' and x['seed']==17)['metrics']['facenet_cosine']=None
        v=compare(r,'oracle_region','baseline','wrong_patch')['facenet_cosine']
        self.assertEqual(v['n_pairs'],11);self.assertEqual(v['n_identities'],4)
        self.assertAlmostEqual(v['candidate_mean']-v['control_mean'],1.0)
        self.assertEqual(v['identity_units'][0]['n_seeds'],2)


if __name__=='__main__':unittest.main()
