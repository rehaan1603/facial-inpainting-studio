import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_selection_comparison import paired_summary

class PairedTests(unittest.TestCase):
    def test_identity_not_row_is_statistical_unit(self):
        rows=[]
        for identity in ['a','b','c','d']:
            for seed in [17,29]:
                for condition in ['eyes','mouth','mixed']:
                    for policy,value in [('mask_aware',.6),('all',.5)]:
                        rows.append(dict(identity=identity,seed=seed,condition=condition,policy=policy,evaluation={'metrics':{'facenet_cosine':value}}))
        r=paired_summary(rows,'all','facenet_cosine')
        self.assertEqual(r['identities'],4);self.assertEqual(r['paired_rows'],24)
        self.assertAlmostEqual(r['mean_delta'],.1);self.assertEqual(r['p_exact'],.125)
        rows[0]['evaluation']['metrics']['facenet_cosine']=None
        self.assertEqual(paired_summary(rows,'all','facenet_cosine')['paired_rows'],23)

if __name__=='__main__':unittest.main()
