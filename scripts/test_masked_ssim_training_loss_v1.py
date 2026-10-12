"""CPU scientific checks against the unchanged evaluator's SSIM definition."""

import unittest

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.metrics import structural_similarity

from masked_ssim_training_loss_v1 import masked_ssim_loss


def tensor(array):
    return torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1)[None].float() / 255.


def expected_loss(a, b, hole):
    _, similarity = structural_similarity(
        a.astype(np.float64) / 255., b.astype(np.float64) / 255., data_range=1,
        channel_axis=2, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, win_size=11, full=True)
    valid = binary_erosion(hole, structure=np.ones((11, 11), bool), border_value=0)
    return 1. - float(similarity.mean(axis=2)[valid].mean())


class MaskedSSIMTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261012)
        self.a = self.rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.hole = np.zeros((48, 64), bool)
        self.hole[9:42, 10:55] = True
        self.mask = torch.from_numpy(self.hole)[None, None]

    def test_skimage_parity_on_uint8_patterns(self):
        gradient = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None] * 4,
                                   (48, 64, 3)).copy()
        constant = np.full_like(self.a, 127)
        examples = [
            (self.a, self.rng.integers(0, 256, self.a.shape, dtype=np.uint8)),
            (gradient, np.clip(gradient.astype(np.int16) + 13, 0, 255).astype(np.uint8)),
            (constant, np.full_like(constant, 130)),
        ]
        for a, b in examples:
            with self.subTest(pattern=int(a[0, 0, 0])):
                actual = float(masked_ssim_loss(tensor(a), tensor(b), self.mask))
                self.assertAlmostEqual(actual, expected_loss(a, b, self.hole), delta=3e-5)

    def test_identity_loss_zero(self):
        for a in (self.a, np.zeros_like(self.a), np.full_like(self.a, 127), np.full_like(self.a, 255)):
            with self.subTest(value=int(a[0, 0, 0])):
                prediction = tensor(a).requires_grad_(True)
                loss = masked_ssim_loss(prediction, tensor(a), self.mask)
                self.assertAlmostEqual(float(loss.detach()), 0., delta=2e-6)
                loss.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_visible_values_do_not_change_loss(self):
        a, b = tensor(self.a), tensor(np.roll(self.a, 3, axis=1))
        expected = masked_ssim_loss(a, b, self.mask)
        visible = ~self.mask.expand_as(a)
        changed_a, changed_b = a.clone(), b.clone()
        changed_a[visible], changed_b[visible] = -9., 6.
        actual = masked_ssim_loss(changed_a, changed_b, self.mask)
        self.assertTrue(torch.equal(expected, actual))

    def test_known_structural_distortion_increases_loss(self):
        sharp = tensor(self.a)
        blurred = torch.nn.functional.avg_pool2d(sharp, kernel_size=7, stride=1, padding=3)
        self.assertGreater(float(masked_ssim_loss(blurred, sharp, self.mask)), .1)

    def test_finite_nonzero_gradient_only_in_complete_window_support(self):
        hole = self.hole.copy()
        hole[1:4, 1:4] = True  # Missing island too small to support one SSIM window.
        mask = torch.from_numpy(hole)[None, None]
        prediction = tensor(np.roll(self.a, 2, axis=1)).requires_grad_(True)
        original = prediction.detach().clone()
        loss = masked_ssim_loss(prediction, tensor(self.a), mask)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.norm()), 0.)
        centres = binary_erosion(hole, structure=np.ones((11, 11), bool), border_value=0)
        used = binary_dilation(centres, structure=np.ones((11, 11), bool), border_value=0)
        outside = torch.from_numpy(~used)[None, None].expand_as(prediction)
        self.assertEqual(int(torch.count_nonzero(prediction.grad[outside])), 0)
        self.assertTrue(torch.equal(prediction.detach(), original))

    def test_gradient_matches_finite_difference(self):
        prediction = tensor(np.roll(self.a, 1, axis=1)).requires_grad_(True)
        target = tensor(self.a)
        loss = masked_ssim_loss(prediction, target, self.mask)
        loss.backward()
        index = (0, 1, 24, 30)
        analytic = float(prediction.grad[index])
        step = .001
        plus, minus = prediction.detach().clone(), prediction.detach().clone()
        plus[index] += step
        minus[index] -= step
        numerical = float((masked_ssim_loss(plus, target, self.mask)
                           - masked_ssim_loss(minus, target, self.mask)) / (2 * step))
        self.assertAlmostEqual(analytic, numerical, delta=8e-5)

    def test_out_of_range_predictions_are_not_clipped(self):
        target = torch.full((1, 3, 48, 64), .5)
        prediction = torch.full_like(target, 1.2, requires_grad=True)
        loss = masked_ssim_loss(prediction, target, self.mask)
        clipped = masked_ssim_loss(prediction.clamp(0, 1), target, self.mask)
        self.assertGreater(float(loss.detach()), float(clipped.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.norm()), 0.)

    def test_invalid_inputs_and_no_complete_windows(self):
        a = tensor(self.a)
        bad_masks = [torch.zeros_like(self.mask), torch.full_like(self.mask.float(), .5),
                     torch.full_like(self.mask.float(), float('nan')), self.mask[:, :, :-1],
                     self.mask.float().requires_grad_(True)]
        small = torch.zeros_like(self.mask)
        small[:, :, 12:22, 12:22] = True
        bad_masks.append(small)
        for mask in bad_masks:
            with self.subTest(shape=tuple(mask.shape)):
                with self.assertRaises(ValueError):
                    masked_ssim_loss(a, a, mask)
        with self.assertRaises(ValueError):
            masked_ssim_loss(a[:, :, :10], a[:, :, :10], self.mask[:, :, :10])
        with self.assertRaises(ValueError):
            masked_ssim_loss(a[:, :1], a[:, :1], self.mask)
        with self.assertRaises(TypeError):
            masked_ssim_loss(a.to(torch.uint8), a, self.mask)
        nonfinite = a.clone()
        nonfinite[0, 0, 20, 20] = float('inf')
        with self.assertRaises(ValueError):
            masked_ssim_loss(nonfinite, a, self.mask)

    def test_batch_requires_valid_windows_for_every_member(self):
        a = tensor(self.a).repeat(2, 1, 1, 1)
        mask = self.mask.repeat(2, 1, 1, 1)
        mask[1] = False
        with self.assertRaises(ValueError):
            masked_ssim_loss(a, a, mask)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

