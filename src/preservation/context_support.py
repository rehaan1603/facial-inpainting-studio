"""Visible-context calibration of mask support. No clean-target interface.

This is an exploratory policy, not calibrated confidence. Patches are shared by
all radii and lie outside their union, preventing a radius from getting an easier
scoring region. Callers must erase them BEFORE any model-dependent preprocessing.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def support(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError('A nonempty, nonfull 2D missing mask is required')
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError('Radius must be a nonnegative integer')
    return mask.copy() if radius == 0 else distance_transform_edt(~mask) <= radius


def probes(mask, radii, *, width=8, count=8, minimum_count=4, gap=3,
           outer_distance=32, placement_seed=20260925):
    """Choose dispersed square patches using mask geometry only, never RGB/truth."""
    largest = support(mask, max(radii))
    distance = distance_transform_edt(~np.asarray(mask, bool))
    allowed = (distance > max(radii) + gap) & (distance <= outer_distance)
    # Full patch footprints must be reliable and separate from every candidate.
    windows = np.lib.stride_tricks.sliding_window_view(allowed, (width, width))
    positions = np.argwhere(windows.all(axis=(-1, -2)))
    rng = np.random.default_rng(placement_seed)
    positions = positions[rng.permutation(len(positions))]
    chosen = []
    for y, x in positions:
        center = np.array([y, x]) + width / 2
        if all(np.linalg.norm(center - (np.array([a, b]) + width / 2)) >= width * 2
               for a, b in chosen):
            chosen.append((int(y), int(x)))
        if len(chosen) >= count:
            break
    hidden = np.zeros_like(largest)
    for y, x in chosen:
        hidden[y:y+width, x:x+width] = True
    assert not np.any(hidden & largest)
    return hidden, {'boxes_yx': chosen, 'count': len(chosen),
                    'eligible': len(chosen) >= minimum_count,
                    'pixels': int(hidden.sum())}


def calibration_input(observed, missing, hidden, radius):
    effective = support(missing, radius) | np.asarray(hidden, bool)
    if np.asarray(observed).shape != (*effective.shape, 3):
        raise ValueError('RGB and mask shapes do not match')
    erased = np.asarray(observed).copy()
    erased[effective] = 0
    return erased, effective


def probe_loss(prediction, observed, hidden):
    if not np.any(hidden):
        raise ValueError('No held-out pixels')
    p, o = np.asarray(prediction), np.asarray(observed)
    if p.shape != o.shape or p.shape != (*hidden.shape, 3) or not np.isfinite(p).all():
        raise ValueError('Invalid prediction')
    return float(np.abs(p.astype(np.float64)[hidden] - o.astype(np.float64)[hidden]).mean())


def choose_radius(losses, radii, eligible=True):
    if not eligible or any(r not in losses or not np.isfinite(losses[r]) for r in radii):
        return 0, 'abstained'
    return min(radii, key=lambda r: (losses[r], r)), 'selected'


def compose(observed, prediction, mask):
    if observed.shape != prediction.shape or observed.shape != (*mask.shape, 3):
        raise ValueError('Composition dimensions do not match')
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    out = observed.copy()
    out[mask] = np.clip(np.rint(prediction[mask]), 0, 255).astype('uint8')
    return out
