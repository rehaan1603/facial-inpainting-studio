"""Summaries must serialize NumPy-derived pixel scores without changing values."""
import json
import unittest
import numpy as np
from score_reference_rejection_v1 import paired


class SummaryTests(unittest.TestCase):
    def test_numpy_metric_win_counts_are_json_serializable(self):
        rows=[]
        for identity in ['263','6681','7400','10149']:
            for seed in [17,29,43]:
                rows.extend([dict(identity=identity,seed=seed,arm=arm,condition='wrong_patch',
                    metrics=dict(psnr=np.float64(score))) for arm,score in [('drop_first',25),('baseline',24)]])
        result=paired(rows,'drop_first','baseline','wrong_patch')
        restored=json.loads(json.dumps(result,allow_nan=False))
        self.assertEqual(restored['psnr']['improved'],12)
        self.assertEqual(restored['psnr']['n_identities'],4)
        self.assertEqual(restored['psnr']['mean_delta'],1.)
        self.assertEqual(restored['facenet_cosine']['n_pairs'],0)


if __name__=='__main__':unittest.main()
