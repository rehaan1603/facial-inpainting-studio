"""Explicit editable evidence maps, not a calibrated blind damage detector."""
import numpy as np


def evidence_map(mask, missing=False, degraded_confidence=.5):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError('Require a partial, nonempty 2D damage mask')
    if not 0 <= degraded_confidence <= 1:
        raise ValueError('Confidence must lie in [0,1]')
    confidence = np.ones(mask.shape, dtype=np.float32)
    confidence[mask] = 0 if missing else degraded_confidence
    return confidence


def preserve_observation(observed, generated, mask, confidence):
    observed, generated = np.asarray(observed), np.asarray(generated)
    mask, confidence = np.asarray(mask, bool), np.asarray(confidence)
    if observed.dtype != np.uint8 or generated.dtype != np.uint8 or observed.shape != generated.shape:
        raise ValueError('Matching uint8 RGB images required')
    if observed.ndim != 3 or observed.shape[2] != 3 or mask.shape != observed.shape[:2] or confidence.shape != mask.shape:
        raise ValueError('Invalid geometry')
    if not np.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 1:
        raise ValueError('Invalid confidence')
    if not np.all(confidence[~mask] == 1):
        raise ValueError('Reliable visible pixels must have confidence one')
    result = observed.copy()
    alpha = confidence[mask, None]
    result[mask] = np.rint(alpha * observed[mask].astype(float) + (1-alpha) * generated[mask]).clip(0,255).astype('uint8')
    return result
