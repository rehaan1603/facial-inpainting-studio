import json
import sys
import types
import unittest
import torch
from reference_intervention_core import ROOT
from spatial_reference_rejection_v1 import SpatialRejection, install, region_box, token_mask

sys.path.insert(0,json.loads((ROOT/'configs/local.json').read_text())['cache']+'/refldm_source_v1')
from ldm import cache_kv
from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        modules=torch.nn.Sequential(*[QKVAttentionLegacy(2) for _ in range(16)])
        self.a=modules[0];self.original=self.a.forward;self.control=SpatialRejection()
        install(types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=modules)),self.control)
        self.x=torch.randn(2,24,16)
        cache_kv.mode='use'
        cache_kv.k[id(self.a)]=[torch.randn(4,4,16) for _ in range(4)]
        cache_kv.v[id(self.a)]=[torch.randn(4,4,16) for _ in range(4)]

    def tearDown(self):
        cache_kv.clear_cache();cache_kv.mode=None

    def test_neutral_exact(self):
        self.assertTrue(torch.equal(self.original(self.x),self.a(self.x)))

    def test_conditional_mask_preserves_null(self):
        baseline=self.original(self.x);self.control.box=region_box('oracle_region','263')
        output=self.a(self.x)
        self.assertTrue(torch.equal(baseline[0],output[0]))
        self.assertFalse(torch.equal(baseline[1],output[1]))

    def test_masked_conditional_keys_values_have_no_influence(self):
        self.control.box=region_box('oracle_region','263');before=self.a(self.x)
        mask=token_mask(16,self.control.box,'cpu')
        for store in [cache_kv.k,cache_kv.v]:store[id(self.a)][0][2:,:,mask]+=1000
        self.assertTrue(torch.equal(before,self.a(self.x)))
        cache_kv.v[id(self.a)][1][2:]+=10
        self.assertFalse(torch.equal(before[1],self.a(self.x)[1]))

    def test_equal_area_all_scales_and_deterministic_controls(self):
        for identity in ['263','6681','7400','10149']:
            for n in [64,256,1024,4096]:
                a=token_mask(n,region_box('oracle_region',identity),'cpu')
                b=token_mask(n,region_box('random_corner',identity),'cpu')
                self.assertEqual(int(a.sum()),int(b.sum()))
                self.assertFalse(torch.equal(a,b))
                self.assertEqual(region_box('random_corner',identity),region_box('random_corner',identity))


if __name__=='__main__':unittest.main()
