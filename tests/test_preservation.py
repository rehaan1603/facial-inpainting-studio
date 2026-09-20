import unittest
import numpy as np
from src.preservation.confidence import evidence_map,preserve_observation


class PreservationTests(unittest.TestCase):
    def test_three_states_and_known_pixels(self):
        mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        obs=np.full((8,8,3),40,np.uint8);gen=np.full_like(obs,200)
        for missing,expected in [(True,200),(False,120)]:
            c=evidence_map(mask,missing,.5);result=preserve_observation(obs,gen,mask,c)
            self.assertTrue(np.all(result[mask]==expected));self.assertTrue(np.array_equal(result[~mask],obs[~mask]))
        self.assertTrue(np.array_equal(preserve_observation(obs,gen,mask,evidence_map(mask,False,1)),obs))

    def test_invalid_maps_rejected(self):
        mask=np.eye(8,dtype=bool);obs=np.zeros((8,8,3),np.uint8)
        for value in [float('nan'),-1,1.1]:
            with self.assertRaises(ValueError):preserve_observation(obs,obs,mask,np.full((8,8),value))
        with self.assertRaises(ValueError):evidence_map(np.ones((8,8),bool))


if __name__=='__main__':unittest.main()
