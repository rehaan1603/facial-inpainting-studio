"""Checks the actual attention intervention without image data or GPU allocation."""
import json
import sys
import types
import unittest
import torch
from reference_intervention_core import ROOT, ReferenceBias, install_adapter

cache = json.loads((ROOT/'configs/local.json').read_text())['cache']
sys.path.insert(0, cache+'/refldm_source_v1')
from ldm import cache_kv
from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.attention = QKVAttentionLegacy(2)
        self.original = self.attention.forward
        self.adapter = ReferenceBias()
        wrapper = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=torch.nn.Sequential(self.attention)))
        install_adapter(wrapper, self.adapter)
        self.input = torch.randn(1, 192, 5)
        cache_kv.mode = 'use'
        cache_kv.k[id(self.attention)] = [torch.randn(2,32,7)]
        cache_kv.v[id(self.attention)] = [torch.randn(2,32,7)]

    def tearDown(self):
        cache_kv.clear_cache()
        cache_kv.mode = None

    def test_neutral_exact_and_disabled_exact(self):
        baseline = self.original(self.input)
        self.assertTrue(torch.equal(baseline, self.attention(self.input)))
        self.adapter.enabled = False
        self.assertTrue(torch.equal(baseline, self.attention(self.input)))

    def test_removed_references_have_no_influence(self):
        self.adapter.drop_references = True
        before = self.attention(self.input)
        cache_kv.k[id(self.attention)][0].mul_(100)
        cache_kv.v[id(self.attention)][0].mul_(100)
        self.assertTrue(torch.equal(before, self.attention(self.input)))
        cache_kv.mode = None
        self.assertTrue(torch.equal(before, self.original(self.input)))

    def test_adapter_receives_nonzero_gradient(self):
        self.attention(self.input).square().mean().backward()
        self.assertGreater(self.adapter.net[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(self.adapter.net[-1].weight.grad).all())

    def test_nonzero_bias_changes_output(self):
        baseline = self.original(self.input)
        with torch.no_grad():
            self.adapter.net[-1].bias.fill_(0.5)
        self.assertFalse(torch.equal(baseline, self.attention(self.input)))


if __name__ == '__main__':
    unittest.main()
