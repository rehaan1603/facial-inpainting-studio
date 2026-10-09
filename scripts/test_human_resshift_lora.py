"""Meaningful adapter isolation and disable/reload regression checks; CPU only."""
import unittest
import torch
from torch import nn
from train_human_resshift_v1 import LowRankLinear, state, load_state


class AdapterTests(unittest.TestCase):
    def test_neutral_disable_and_frozen_weights(self):
        torch.manual_seed(4)
        base=nn.Linear(8,12).requires_grad_(False)
        original=base.weight.detach().clone()
        layer=LowRankLinear(base,4);x=torch.randn(2,5,8)
        expected=base(x)
        self.assertTrue(torch.equal(layer(x),expected))
        optimizer=torch.optim.SGD([p for p in layer.parameters() if p.requires_grad],lr=.1)
        loss=(layer(x)-1).square().mean();loss.backward();optimizer.step()
        self.assertIsNone(base.weight.grad)
        self.assertTrue(torch.equal(base.weight,original))
        self.assertFalse(torch.equal(layer(x),expected))
        layer.enabled=False
        self.assertTrue(torch.equal(layer(x),expected))
        layer.enabled=True
        saved=state(layer);output=layer(x).detach()
        with torch.no_grad():layer.up.weight.zero_()
        load_state(layer,saved)
        self.assertTrue(torch.equal(layer(x),output))


if __name__=='__main__':unittest.main()
