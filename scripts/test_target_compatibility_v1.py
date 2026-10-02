import unittest
import torch
from target_region_compatibility_v1 import Compatibility,features


class CompatibilityTests(unittest.TestCase):
    def test_neutral_and_reference_permutation(self):
        torch.manual_seed(9);a=Compatibility();a.context=torch.randn(5,4,10)
        self.assertTrue(torch.equal(a.priors(),torch.zeros(5,4)))
        torch.nn.init.normal_(a.net[-1].weight)
        expected=a.priors();order=[2,0,3,1];a.context=a.context[:,order]
        self.assertTrue(torch.allclose(a.priors(),expected[:,order],atol=1e-6))

    def test_observed_target_changes_features(self):
        torch.manual_seed(9);refs=torch.randn(4,8,64,64);lq=torch.randn(1,8,64,64);diag=torch.zeros(4,6)
        x=features(lq,refs,diag);y=features(-lq,refs,diag)
        self.assertFalse(torch.equal(x,y))
        self.assertTrue(torch.equal(x[:,:,2],y[:,:,2]))

    def test_no_target_ablation_removes_target_features(self):
        a=Compatibility('no_target');torch.nn.init.normal_(a.net[-1].weight)
        a.context=torch.randn(5,4,10);expected=a.priors()
        a.context[:,:,[0,1,3,4,7,9]]+=10
        self.assertTrue(torch.equal(expected,a.priors()))


if __name__=='__main__':unittest.main()
