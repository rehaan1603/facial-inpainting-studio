"""Compose native model predictions without downsampling the visible photo."""
import numpy as np
from PIL import Image


def compose_prediction(source, mask, prediction):
    observed = np.asarray(source.convert('RGB'))
    binary = np.asarray(mask.convert('L')) >= 128
    if binary.shape != observed.shape[:2]:
        raise ValueError('Output mask and photo dimensions must match.')
    if not np.isfinite(prediction).all():
        raise ValueError('The model returned invalid pixels.')
    generated = Image.fromarray(np.rint(np.clip(prediction, 0, 1) * 255).astype('uint8'))
    generated = np.asarray(generated.resize(source.size, Image.Resampling.LANCZOS))
    result = observed.copy()
    result[binary] = generated[binary]
    return Image.fromarray(result)
