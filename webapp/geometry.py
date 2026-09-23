"""Shared geometry for user photographs and their pixel-aligned evidence maps."""
from PIL import Image


def fit_evidence(source, mask, confidence, size=512):
    if mask.size != source.size or confidence.size != source.size:
        raise ValueError('Image, mask and evidence map dimensions must match.')
    width, height = source.size
    scale = min(size / width, size / height)
    fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((size - fitted[0]) // 2, (size - fitted[1]) // 2)
    outputs = []
    for im, mode, fill, sampling in [
        (source, 'RGB', (127, 127, 127), Image.Resampling.LANCZOS),
        (mask, 'L', 0, Image.Resampling.NEAREST),
        (confidence, 'L', 255, Image.Resampling.NEAREST),
    ]:
        canvas = Image.new(mode, (size, size), fill)
        canvas.paste(im.convert(mode).resize(fitted, sampling), offset)
        outputs.append(canvas)
    geometry = {'original_size': [width, height], 'fitted_size': list(fitted),
                'offset': list(offset), 'canvas_size': size,
                'method': 'aspect_preserving_fit'}
    return (*outputs, geometry)
