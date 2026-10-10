"""Tiny CPU attention tests; no model loading or GPU work."""
import unittest
import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from local_appearance_processor_v1 import LocalAppearanceProcessor
from sdxl_token_compatibility_probe_v1 import CompatibilityScorer, TokenCompatibilityProcessor


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261010)
        self.attn=Attention(query_dim=32,cross_attention_dim=32,heads=4,dim_head=8).eval().requires_grad_(False)
        self.delegate=IPAdapterAttnProcessor2_0(hidden_size=32,cross_attention_dim=32,num_tokens=(2,2),scale=[.8,.2]).eval().requires_grad_(False)
        self.mask=torch.zeros(4,4,dtype=torch.bool);self.mask[:2]=True
        self.scorer=CompatibilityScorer()
        self.processor=TokenCompatibilityProcessor(self.delegate,self.scorer,self.mask,chunk_size=3)
        self.query=torch.randn(1,16,32)
        self.condition=(torch.randn(1,7,32),[torch.randn(1,4,2,32),torch.randn(1,4,2,32)])

    def call(self,processor): return processor(self.attn,self.query,encoder_hidden_states=self.condition)

    def test_parameter_count_and_neutral_local_equivalence(self):
        self.assertEqual(sum(p.numel() for p in self.scorer.parameters()),385)
        with torch.no_grad():
            expected=self.call(LocalAppearanceProcessor(self.delegate,self.mask))
            actual=self.call(self.processor)
        self.assertLessEqual(float((expected-actual).abs().max()),1e-5)

    def test_finite_nonzero_gradient_and_disposable_update(self):
        old=[p.detach().clone() for p in self.scorer.parameters()]
        loss=(self.call(self.processor)-torch.randn(1,16,32)).square().mean()
        loss.backward()
        gradients=[p.grad for p in self.scorer.parameters() if p.grad is not None]
        self.assertEqual(len(gradients),len(list(self.scorer.parameters())))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.square().sum()) for g in gradients),0)
        optimizer=torch.optim.AdamW(self.scorer.parameters(),lr=1e-4);optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,self.scorer.parameters())))
        self.assertTrue(all(p.grad is None for p in self.attn.parameters()))
        self.assertTrue(all(p.grad is None for p in self.delegate.parameters()))

    def test_changed_bias_is_query_local(self):
        self.scorer.enabled=False
        with torch.no_grad(): base=self.call(self.processor)
        self.scorer.enabled=True
        with torch.no_grad():
            self.scorer.net[-1].weight.fill_(.2)
            changed=self.call(self.processor)
        self.assertTrue(torch.equal(base[:,8:],changed[:,8:]))
        self.assertFalse(torch.equal(base[:,:8],changed[:,:8]))

    def test_chunking_preserves_joint_reference_attention(self):
        with torch.no_grad():
            self.scorer.net[-1].weight.fill_(.2)
            short=self.call(self.processor)
            self.processor.chunk_size=16
            long=self.call(self.processor)
        self.assertLessEqual(float((short-long).abs().max()),1e-5)

    def test_bias_bound_and_native_masks_rejected(self):
        q=torch.randn(1,4,16,8);k=torch.randn(1,4,8,8)
        with torch.no_grad():
            self.scorer.net[-1].weight.fill_(100)
            bias=self.scorer(q,k)
        self.assertLessEqual(float(bias.abs().max()),.5)
        with self.assertRaises(ValueError): self.processor(self.attn,self.query,encoder_hidden_states=self.condition,ip_adapter_masks=[None,None])


if __name__=='__main__': unittest.main()
