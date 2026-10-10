"""Real tiny attention comparisons verify neutral paths and spatial isolation."""
import unittest
import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers.models.attention import Attention
from local_appearance_processor_v1 import LocalAppearanceProcessor


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.attn = Attention(query_dim=32, cross_attention_dim=32, heads=4, dim_head=8).eval()
        self.processor = IPAdapterAttnProcessor2_0(hidden_size=32, cross_attention_dim=32,
            num_tokens=(2, 2), scale=[.8, .2]).eval()
        self.target = torch.randn(2, 16, 32)
        # Four independent references retain their original joint token bank.
        self.condition = (torch.randn(2, 7, 32), [torch.randn(2, 4, 2, 32), torch.randn(2, 4, 2, 32)])

    def call(self, processor):
        with torch.inference_mode():
            return processor(self.attn, self.target, encoder_hidden_states=self.condition)

    def test_full_mask_is_original_exact_and_reuses_weights(self):
        expected = self.call(self.processor)
        routed = LocalAppearanceProcessor(self.processor, torch.ones(8, 8)).eval()
        self.assertTrue(torch.equal(self.call(routed), expected))
        self.assertIs(routed.delegate.to_k_ip, self.processor.to_k_ip)

    def test_zero_appearance_is_original_exact(self):
        self.processor.scale = [.8, 0.]
        expected = self.call(self.processor)
        routed = LocalAppearanceProcessor(self.processor, torch.eye(8)).eval()
        self.assertTrue(torch.equal(self.call(routed), expected))

    def test_partial_mask_selects_full_inside_base_outside(self):
        full = self.call(self.processor)
        self.processor.scale = [.8, 0.]
        base = self.call(self.processor)
        self.processor.scale = [.8, .2]
        mask = torch.zeros(8, 8); mask[:4] = 1
        routed = self.call(LocalAppearanceProcessor(self.processor, mask).eval())
        self.assertTrue(torch.equal(routed[:, :8], full[:, :8]))
        self.assertTrue(torch.equal(routed[:, 8:], base[:, 8:]))
        self.assertEqual(self.processor.scale, [.8, .2])

    def test_empty_mask_is_base_exact(self):
        self.processor.scale = [.8, 0.]
        expected = self.call(self.processor)
        self.processor.scale = [.8, .2]
        self.assertTrue(torch.equal(self.call(LocalAppearanceProcessor(self.processor, torch.zeros(8,8))), expected))

    def test_native_masks_rejected(self):
        routed = LocalAppearanceProcessor(self.processor, torch.ones(8,8))
        with self.assertRaises(ValueError):
            routed(self.attn, self.target, encoder_hidden_states=self.condition, ip_adapter_masks=[None,None])


if __name__ == '__main__':
    unittest.main()
