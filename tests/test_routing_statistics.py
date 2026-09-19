import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_regional_routing import paired

class RoutingStatistics(unittest.TestCase):
    def test_paired_identity_units_and_missing(self):
        rows=[]
        for identity in range(4):
            for condition in ['eyes','mouth','mixed']:
                for seed in [17,29]:
                    for policy,value in [('regional',.7),('equal',.6)]:
                        rows.append(dict(identity=str(identity),case_id=f'{identity}_{condition}',seed=seed,policy=policy,evaluation={'metrics':{'facenet_cosine':value}}))
        r=paired(rows,'equal');self.assertEqual(r['identities'],4);self.assertEqual(r['pairs'],24);self.assertAlmostEqual(r['mean_delta'],.1);self.assertEqual(r['p_exact'],.125)
        rows[0]['evaluation']['metrics']['facenet_cosine']=None
        self.assertEqual(paired(rows,'equal')['pairs'],23)

if __name__=='__main__':unittest.main()
