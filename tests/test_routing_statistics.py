import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_regional_routing import paired
from report_regional_routing_v2 import paired as paired_v2

class RoutingStatistics(unittest.TestCase):
    def test_paired_identity_units_and_missing(self):
        rows=[]
        for identity in range(4):
            for condition in ['eyes','mouth','mixed']:
                for seed in [17,29]:
                    for policy,value in [('regional',.7),('equal',.6)]:
                        rows.append(dict(identity=str(identity),case_id=f'{identity}_{condition}',seed=seed,policy=policy,evaluation={'metrics':{'facenet_cosine':value}}))
        self.assertEqual(paired(rows,'equal'),paired_v2(rows,'equal'))
        r=paired_v2(rows,'equal');self.assertEqual(r['identities'],4);self.assertEqual(r['pairs'],24);self.assertAlmostEqual(r['mean_delta'],.1);self.assertEqual(r['p_exact'],.125)
        rows[0]['evaluation']['metrics']['facenet_cosine']=None
        self.assertEqual(paired_v2(rows,'equal')['pairs'],23)
        for row in rows:row['evaluation']={'status':'failed'}
        missing=paired_v2(rows,'equal')
        self.assertEqual(missing['pairs'],0);self.assertIsNone(missing['p_exact']);self.assertIsNone(missing['mean_delta'])

if __name__=='__main__':unittest.main()
