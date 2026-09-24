"""Shared geometry for user photographs and their pixel-aligned evidence maps."""
import numpy as np
from PIL import Image


def _resize_extrema(image, size, reduction):
    """Pool every overlapping input pixel when shrinking; use nearest when growing."""
    if len(size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in size):
        raise ValueError('Resize dimensions must be two positive integers.')
    values = np.asarray(image.convert('L'))
    for axis, target in enumerate((size[1], size[0])):
        length = values.shape[axis]
        if target >= length:
            continue
        # Include the whole source footprint of each output pixel, including
        # fractional boundary pixels. A nearest sample can miss a narrow scratch.
        pooled = []
        for index in range(target):
            start = index * length // target
            stop = ((index + 1) * length + target - 1) // target
            region = values[start:stop, :] if axis == 0 else values[:, start:stop]
            pooled.append(reduction(region, axis=axis))
        values = np.stack(pooled, axis=axis)
    return Image.fromarray(values).resize(size, Image.Resampling.NEAREST)


def resize_binary_mask(mask, size):
    """Return an L 0/255 mask, preserving thin damage when reducing resolution.

    White (at least 128) means replace. Any masked input pixel overlapping a
    downsampled pixel marks that output pixel; upsampling uses nearest neighbour.
    Use the original requested mask for final composition, since a coarse model
    mask can cover nearby reliable pixels as well as the requested damage.
    """
    binary = Image.fromarray((np.asarray(mask.convert('L')) >= 128).astype('uint8') * 255)
    return _resize_extrema(binary, size, np.max)


def require_reference_mask_support(mask, model_resolution):
    """Reject disconnected marks that vanish in SDXL's 8x latent mask sampling.

    This contains a known model limitation without changing the frozen sampler.
    It does not guarantee that every fine boundary is represented in the latent.
    """
    from scipy.ndimage import label
    native = np.asarray(resize_binary_mask(mask, (model_resolution, model_resolution))) >= 128
    components, count = label(native, structure=np.ones((3, 3), dtype=int))
    present = set(np.unique(components[::8, ::8]))
    missing = set(range(1, count + 1)) - present
    if not count or missing:
        raise ValueError('Some marked areas are too thin for reference reconstruction at this size. '
                         'Use Expand mask by 8 px on the main page, paint a wider area, '
                         'or use LaMa for tiny scratches. Review the effective mask before using the result.')


def fit_evidence(source, mask, confidence, size=512):
    if mask.size != source.size or confidence.size != source.size:
        raise ValueError('Image, mask and evidence map dimensions must match.')
    width, height = source.size
    scale = min(size / width, size / height)
    fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((size - fitted[0]) // 2, (size - fitted[1]) // 2)
    shrinking = fitted[0] < width or fitted[1] < height
    # Keep the map and mask paired: if any source pixel is missing, its reduced
    # footprint must not receive a white (fully reliable) evidence value. Taking
    # the minimum is conservative resampling, not probability calibration.
    resized_mask = resize_binary_mask(mask, fitted) if shrinking else mask.convert('L').resize(fitted, Image.Resampling.NEAREST)
    resized_confidence = _resize_extrema(confidence, fitted, np.min) if shrinking else confidence.convert('L').resize(fitted, Image.Resampling.NEAREST)
    outputs = []
    for im, mode, fill in [
        (source.convert('RGB').resize(fitted, Image.Resampling.LANCZOS), 'RGB', (127, 127, 127)),
        (resized_mask, 'L', 0),
        (resized_confidence, 'L', 255),
    ]:
        canvas = Image.new(mode, (size, size), fill)
        canvas.paste(im, offset)
        outputs.append(canvas)
    geometry = {'original_size': [width, height], 'fitted_size': list(fitted),
                'offset': list(offset), 'canvas_size': size,
                'method': 'aspect_preserving_fit',
                'map_downsampling': 'overlap_max_mask_min_evidence' if shrinking else 'not_downsampled'}
    return (*outputs, geometry)
