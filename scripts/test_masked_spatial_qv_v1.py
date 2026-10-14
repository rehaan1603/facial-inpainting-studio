"""CPU-only synthetic checks of the audited SDXL Q/V residual and its graph."""

import copy
import io
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from masked_spatial_qv_v1 import (
    HEADS, HOOK, PARAMETER_COUNT, RANK, SEED, WIDTH,
    SpatialQVAdapter, SpatialQVProcessor, install,
)
from sdxl_token_compatibility_probe_v1 import CompatibilityScorer


def attention(dtype=torch.float32):
    return Attention(query_dim=WIDTH, heads=HEADS, dim_head=64,
                     dropout=0., bias=False, out_bias=True,
                     processor=AttnProcessor2_0()).eval().requires_grad_(False).to(dtype=dtype)


class SyntheticUNet(nn.Module):
    """One real attention, plus 70 extant shared-scorer processor placeholders."""

    def __init__(self):
        super().__init__()
        self.attention = attention()
        scorer = CompatibilityScorer()
        self.compatibility = nn.ModuleList()
        for _ in range(70):
            processor = nn.Module()
            processor.add_module('scorer', scorer)
            self.compatibility.append(processor)

    @property
    def attn_processors(self):
        return {HOOK: self.attention.processor,
                **{f'cross.{i}.attn2.processor': p for i, p in enumerate(self.compatibility)}}

    def get_submodule(self, target):
        if target == HOOK.removesuffix('.processor'):
            return self.attention
        return super().get_submodule(target)


class SpatialQVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(991)
        self.attn = attention()
        self.hidden = torch.randn(2, 16, WIDTH)
        self.mask = torch.zeros(8, 8, dtype=torch.bool)
        self.mask[2:6, 2:6] = True
        self.adapter = SpatialQVAdapter()
        self.processor = SpatialQVProcessor(self.attn.processor, self.adapter, self.mask)

    def test_exact_count_isolated_rng_and_linear_default_initialization(self):
        state = torch.random.get_rng_state().clone()
        adapter = SpatialQVAdapter()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), PARAMETER_COUNT)
        self.assertEqual(PARAMETER_COUNT, 10240)
        self.assertEqual(set(adapter.state_dict()), {'q_a', 'q_b', 'v_a', 'v_b'})
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        for name in ('q_a', 'v_a'):
            expected = torch.empty(RANK, WIDTH)
            nn.init.kaiming_uniform_(expected, a=math.sqrt(5), generator=generator)
            self.assertTrue(torch.equal(expected, getattr(adapter, name)))
        for name in ('q_b', 'v_b'):
            self.assertEqual(torch.count_nonzero(getattr(adapter, name)).item(), 0)
        for name, value in adapter.state_dict().items():
            self.assertTrue(torch.equal(value, self.adapter.state_dict()[name]))

    def test_install_keeps_all_70_existing_processors_and_10625_parameters(self):
        unet = SyntheticUNet()
        before = unet.attn_processors
        scorer_before = list(unet.compatibility[0].scorer.parameters())
        installed = install(unet, self.adapter, self.mask)
        self.assertIs(installed, unet.attention.processor)
        for key, processor in before.items():
            if key != HOOK:
                self.assertIs(unet.attn_processors[key], processor)
        self.assertEqual(sum(p.numel() for p in unet.parameters() if p.requires_grad), 10625)
        self.assertEqual(sum(p.numel() for p in scorer_before), 385)
        self.assertTrue(all(a is b for a, b in zip(scorer_before, unet.compatibility[0].scorer.parameters())))
        with self.assertRaises(TypeError):
            install(unet, SpatialQVAdapter(), self.mask)

    def test_enabled_zero_forward_and_upstream_gradient_neutrality(self):
        upstream = nn.Parameter(torch.randn(WIDTH) * .1)
        hidden = self.hidden.clone().requires_grad_(True)
        states = hidden + upstream.sin()[None, None, :]
        target = torch.randn_like(states)
        baseline = self.attn.processor(self.attn, states)
        base_loss = ((baseline - target).square()).mean()
        base_grads = torch.autograd.grad(base_loss, (hidden, upstream), retain_graph=True)
        actual = self.processor(self.attn, states)
        self.assertTrue(torch.equal(baseline, actual))
        actual_loss = ((actual - target).square()).mean()
        actual_grads = torch.autograd.grad(actual_loss, (hidden, upstream))
        for original, wrapped in zip(base_grads, actual_grads):
            torch.testing.assert_close(original, wrapped, atol=1e-6, rtol=1e-5)
        print('synthetic_fp32_upstream_max_abs_difference=',
              max(float((a - b).abs().max()) for a, b in zip(base_grads, actual_grads)))

    def test_zero_b_gradients_then_effective_update_and_nonzero_a_path(self):
        target = torch.randn_like(self.hidden)
        frozen_before = {name: p.detach().clone() for name, p in self.attn.named_parameters()}
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-4)
        output_before = self.processor(self.attn, self.hidden)
        (output_before - target).square().mean().backward()
        for name in ('q_b', 'v_b'):
            gradient = getattr(self.adapter, name).grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()), 0.)
        for name in ('q_a', 'v_a'):
            self.assertEqual(torch.count_nonzero(getattr(self.adapter, name).grad).item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output_after = self.processor(self.attn, self.hidden)
        (output_after - target).square().mean().backward()
        for name, parameter in self.adapter.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.norm()), 0., name)
        indices = self.processor.query_indices(16, self.hidden.device)
        visible = torch.ones(16, dtype=torch.bool)
        visible[indices] = False
        self.assertTrue(torch.equal(output_before[:, visible], output_after[:, visible]))
        self.assertGreater(float((output_before[:, indices] - output_after[:, indices]).abs().max()), 0.)
        for name, parameter in self.attn.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, frozen_before[name]))

    def test_nonzero_serialization_equivalence_and_exact_visible_values(self):
        with torch.no_grad():
            self.adapter.q_b.normal_(0., .04)
            self.adapter.v_b.normal_(0., .04)
        expected = self.processor(self.attn, self.hidden)
        native = self.attn.processor(self.attn, self.hidden)
        saved = io.BytesIO()
        torch.save(self.adapter.state_dict(), saved)
        saved.seek(0)
        restored = SpatialQVAdapter()
        restored.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
        wrapped = SpatialQVProcessor(AttnProcessor2_0(), restored, self.mask)
        self.assertTrue(torch.equal(expected, wrapped(self.attn, self.hidden)))
        indices = wrapped.query_indices(16, 'cpu')
        visible = torch.ones(16, dtype=torch.bool)
        visible[indices] = False
        self.assertTrue(torch.equal(native[:, visible], expected[:, visible]))
        self.assertGreater(float((native[:, indices] - expected[:, indices]).abs().max()), 0.)
        self.assertNotIn('target_mask', wrapped.state_dict())

    def test_parent_precision_moves_keep_fp32_values_and_parameter_identity(self):
        self.attn.set_processor(self.processor)
        originals = {name: p.detach().clone() for name, p in self.adapter.named_parameters()}
        identities = {name: id(p) for name, p in self.adapter.named_parameters()}
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.attn.to(device='cpu', dtype=dtype)
            for name, parameter in self.adapter.named_parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertTrue(torch.equal(parameter, originals[name]))
                self.assertEqual(id(parameter), identities[name])
            x = self.hidden.to(dtype)
            with torch.no_grad():
                self.assertTrue(torch.equal(self.processor.delegate(self.attn, x), self.attn(x)))

    def test_low_precision_and_autocast_enabled_zero_and_fp32_gradients(self):
        for dtype, autocast in ((torch.float16, False), (torch.bfloat16, False),
                                (torch.float16, True), (torch.float32, True)):
            with self.subTest(dtype=dtype, autocast=autocast):
                attn = attention(dtype)
                adapter = SpatialQVAdapter()
                processor = SpatialQVProcessor(attn.processor, adapter, self.mask)
                hidden = self.hidden.to(dtype).requires_grad_(True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
                    base = attn.processor(attn, hidden)
                    actual = processor(attn, hidden)
                    self.assertTrue(torch.equal(base, actual))
                    loss = actual.float().square().mean()
                loss.backward()
                for name, parameter in adapter.named_parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(adapter.q_b.grad.norm()), 0.)
                self.assertGreater(float(adapter.v_b.grad.norm()), 0.)
                self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_nearest_original_mask_and_update(self):
        mask = torch.zeros(7, 7, dtype=torch.bool)
        mask[1:5, 2:6] = True
        self.processor.update_mask(mask)
        expected = F.interpolate(mask.float()[None, None], (4, 4), mode='nearest').flatten().nonzero().flatten()
        self.assertTrue(torch.equal(expected, self.processor.query_indices(16, 'cpu')))
        mask.fill_(False)
        self.assertGreater(self.processor.target_mask.count_nonzero().item(), 0)
        self.processor.update_mask(mask)
        self.assertTrue(torch.equal(self.processor(self.attn, self.hidden), self.attn.processor(self.attn, self.hidden)))

    def test_reject_unsupported_shapes_masks_inputs_and_settings(self):
        for hidden in (torch.zeros(1, 12, WIDTH), torch.zeros(1, 16, 80),
                       torch.zeros(1, WIDTH, 4, 4)):
            with self.subTest(shape=tuple(hidden.shape)), self.assertRaises(ValueError):
                self.processor(self.attn, hidden)
        for mask in (torch.zeros(4, 5), torch.ones(4, 4) * .5, torch.zeros(0, 0)):
            with self.assertRaises(ValueError):
                self.processor.update_mask(mask)
        for kwargs in ({'encoder_hidden_states': self.hidden},
                       {'attention_mask': torch.ones(1, 16)}, {'temb': torch.zeros(1)}, {'scale': 1.}):
            with self.assertRaises(ValueError):
                self.processor(self.attn, self.hidden, **kwargs)
        settings = {'heads': 20, 'rescale_output_factor': 2., 'residual_connection': True,
                    'is_causal': True, 'upcast_attention': True, 'fused_projections': True,
                    'norm_q': nn.Identity(), 'group_norm': nn.Identity()}
        for name, value in settings.items():
            changed = copy.deepcopy(self.attn)
            setattr(changed, name, value)
            with self.subTest(setting=name), self.assertRaises(ValueError):
                self.processor(changed, self.hidden)
        changed = copy.deepcopy(self.attn)
        changed.to_out[1].p = .1
        with self.assertRaises(ValueError):
            self.processor(changed, self.hidden)
        changed = copy.deepcopy(self.attn)
        changed.to_q.requires_grad_(True)
        with self.assertRaises(ValueError):
            self.processor(changed, self.hidden)
        with self.assertRaises(TypeError):
            SpatialQVProcessor(nn.Identity(), self.adapter, self.mask)


if __name__ == '__main__':
    unittest.main()
