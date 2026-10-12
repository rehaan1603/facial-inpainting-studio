"""Differentiable hole SSIM for training; the independent evaluator is unchanged.

Inputs are floating RGB tensors [B, 3, H, W] and binary masks [B, 1, H, W].
Only centres with all 121 pixels inside the hole are averaged, over RGB channels
and the batch. Moments use float32, population covariance, Gaussian sigma 1.5,
and fixed data_range=1 constants. Decoded predictions are never clipped.
"""

import torch
from torch.nn import functional as F


def masked_ssim_loss(prediction, target, mask):
    """Return scalar 1 - fully-contained RGB SSIM, retaining prediction gradients."""
    if not all(isinstance(value, torch.Tensor) for value in (prediction, target, mask)):
        raise TypeError('Prediction, target and mask must be PyTorch tensors')
    if prediction.ndim != 4 or prediction.shape[1] != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [B, 3, H, W] RGB tensors')
    batch, _, height, width = prediction.shape
    if batch < 1 or min(height, width) < 11:
        raise ValueError('Each input must support at least one 11x11 window')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError('Prediction and target must be floating point RGB')
    if prediction.device != target.device:
        raise ValueError('Prediction and target must share a device')
    if mask.shape != (batch, 1, height, width):
        raise ValueError('Expected matching [B, 1, H, W] binary mask')
    if mask.requires_grad:
        raise ValueError('The fixed binary mask must not require gradients')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Prediction and target must contain only finite values')
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Mask must contain only zero and one')

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        hole = mask.to(device=prediction.device, dtype=torch.float32)
        support = F.conv2d(hole, torch.ones((1, 1, 11, 11), device=hole.device,
                                          dtype=torch.float32)) == 121
        if not support.flatten(1).any(dim=1).all():
            raise ValueError('Every batch member needs a fully-contained 11x11 hole window')

        # Unused visible values cannot affect this loss, its numerical centring,
        # or its gradient. No supplied image or mask is modified in place.
        p = torch.where(hole.bool(), prediction.float(), 0.)
        t = torch.where(hole.bool(), target.float(), 0.)
        count = hole.sum(dim=(2, 3), keepdim=True)
        # Subtract a per-channel constant before second moments to reduce fp32
        # cancellation for nearly constant images. Population covariance is
        # translation invariant. Detaching constants avoids cross-pixel graph
        # paths; means are restored for the luminance term below.
        pc = (p.sum(dim=(2, 3), keepdim=True) / count).detach()
        tc = (t.sum(dim=(2, 3), keepdim=True) / count).detach()
        p, t = p - pc, t - tc

        x = torch.arange(-5, 6, dtype=torch.float32, device=p.device)
        gaussian = torch.exp(-x.square() / (2 * 1.5 ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel = (gaussian[:, None] * gaussian[None, :])[None, None].repeat(3, 1, 1, 1)
        average = lambda values: F.conv2d(values, kernel, groups=3)
        pm, tm = average(p), average(t)
        pp, tt = average(p.square()), average(t.square())
        pv, tv = pp - pm.square(), tt - tm.square()
        covariance = average(p * t) - pm * tm
        tolerance = 32 * torch.finfo(torch.float32).eps
        for variance, second, mean in ((pv, pp, pm), (tv, tt, tm)):
            if not torch.isfinite(variance).all():
                raise ValueError('Nonfinite SSIM moments')
            bound = tolerance * (second.abs() + mean.square()).clamp_min(1.)
            if (variance < -bound).any():
                raise ValueError('Negative SSIM variance exceeds floating-point tolerance')
        pv, tv = pv.clamp_min(0.), tv.clamp_min(0.)
        pm, tm = pm + pc, tm + tc
        c1, c2 = .01 ** 2, .03 ** 2
        ssim = ((2 * pm * tm + c1) * (2 * covariance + c2)
                / ((pm.square() + tm.square() + c1) * (pv + tv + c2)))
        selected = ssim.masked_select(support.expand_as(ssim))
        if not torch.isfinite(selected).all():
            raise ValueError('Nonfinite fully-contained SSIM')
        return 1. - selected.mean()

