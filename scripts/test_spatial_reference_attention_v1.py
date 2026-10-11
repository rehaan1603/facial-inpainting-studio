"""Real Diffusers attention unit tests, CPU-only synthetic tensors."""
import unittest
import torch
from diffusers.models.attention_processor import Attention,AttnProcessor2_0
from spatial_reference_attention_v1 import SpatialReferenceAttention,coordinate_bias,grid_centres

class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261011);torch.set_num_threads(2)
        self.attn=Attention(query_dim=80,heads=20,dim_head=4,dropout=0.,bias=False).eval().requires_grad_(False)
        self.hidden=torch.randn(2,16,80)
        self.mask=torch.zeros(8,8,dtype=torch.bool);self.mask[2:6,2:6]=True
        self.bank=torch.randn(12,80);self.coords=torch.rand(12,2);self.reliability=torch.ones(12)
    def processor(self,bank=None,coords=None):
        return SpatialReferenceAttention(AttnProcessor2_0(),self.mask,self.bank if bank is None else bank,self.coords if coords is None else coords,self.reliability)
    def test_zero_state_exact_enabled_and_disabled(self):
        p=self.processor();expected=AttnProcessor2_0()(self.attn,self.hidden)
        self.assertTrue(torch.equal(p(self.attn,self.hidden),expected))
        p.enabled=False;self.assertTrue(torch.equal(p(self.attn,self.hidden),expected))
    def test_only_masked_queries_change(self):
        p=self.processor();p.head_gate.data.fill_(.5)
        base=AttnProcessor2_0()(self.attn,self.hidden);got=p(self.attn,self.hidden)
        c=grid_centres(4);yx=(c[:,[1,0]]*8).long();m=self.mask[yx[:,0],yx[:,1]]
        self.assertTrue(torch.equal(base[:,~m],got[:,~m]));self.assertGreater((base[:,m]-got[:,m]).abs().max(),0)
    def test_finite_gradient_and_exact_parameter_count(self):
        p=self.processor();self.assertEqual(sum(x.numel() for x in p.parameters()),20)
        p(self.attn,self.hidden).square().mean().backward()
        self.assertTrue(torch.isfinite(p.head_gate.grad).all());self.assertGreater(p.head_gate.grad.norm(),0)
        self.assertTrue(all(x.grad is None for x in self.attn.parameters()))
    def test_coordinates_influence_nonzero_output(self):
        a=self.processor();b=self.processor(coords=1-self.coords)
        a.head_gate.data.fill_(.5);b.head_gate.data.fill_(.5)
        self.assertGreater((a(self.attn,self.hidden)-b(self.attn,self.hidden)).abs().max(),1e-4)
    def test_joint_token_permutation_invariance(self):
        a=self.processor();perm=torch.randperm(12)
        b=self.processor(bank=self.bank[perm],coords=self.coords[perm])
        a.head_gate.data.fill_(.5);b.head_gate.data.fill_(.5)
        torch.testing.assert_close(a(self.attn,self.hidden),b(self.attn,self.hidden),atol=1e-6,rtol=1e-6)
    def test_missing_bank_is_exact_fallback(self):
        p=SpatialReferenceAttention(AttnProcessor2_0(),self.mask)
        p.head_gate.data.fill_(.5)
        self.assertTrue(torch.equal(p(self.attn,self.hidden),AttnProcessor2_0()(self.attn,self.hidden)))
    def test_invalid_bank_and_reliability_rejected(self):
        with self.assertRaises(ValueError):self.processor(bank=torch.zeros(513,80),coords=torch.zeros(513,2))
        with self.assertRaises(ValueError):coordinate_bias(self.coords,self.coords,torch.zeros(12))
    def test_actual_channel_width_neutrality(self):
        attn=Attention(query_dim=1280,heads=20,dim_head=64,dropout=0.).eval().requires_grad_(False)
        x=torch.randn(1,16,1280);bank=torch.randn(12,1280)
        p=SpatialReferenceAttention(AttnProcessor2_0(),self.mask,bank,self.coords,self.reliability)
        self.assertTrue(torch.equal(p(attn,x),AttnProcessor2_0()(attn,x)))

if __name__=='__main__':unittest.main()
