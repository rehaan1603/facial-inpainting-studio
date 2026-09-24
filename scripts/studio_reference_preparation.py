"""CPU-only reference checks and aspect-preserving studio framing.

The frozen restoration reader resizes every photo to a square. Only rectangular
studio uploads are reframed here; square uploads retain their original file and
geometry. This is a deterministic crop, not learned facial alignment.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_face(faces, size, index):
    if len(faces) != 1:
        raise ValueError(
            f'Reference {index} must contain one clear, visible face. Found {len(faces)}. '
            'Use a separate close-up photo of the same person.')
    face = faces[0]
    box = np.asarray(face.bbox, dtype=float)
    score = float(face.det_score)
    width, height = size
    if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(score):
        raise ValueError(f'Reference {index} has an unusable face detection. Choose a clearer photo.')
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    if score < .5 or min(box[2] - box[0], box[3] - box[1]) < 32:
        raise ValueError(f'Reference {index} needs a larger, clear face. Choose a closer photo.')
    return box, score


def frame_rectangular_reference(image, box):
    """Square crop around the face with 40% context per side, without stretching."""
    width, height = image.size
    if width == height:
        raise ValueError('Already-square references must use the unchanged path.')
    left, top, right, bottom = [float(value) for value in box]
    side = max(1, int(math.ceil(max(right - left, bottom - top) * 1.8)))
    # Keep both forehead and chin context. Move the square inside the photograph
    # where possible; extend edge pixels only when the crop exceeds an axis.
    x = int(math.floor((left + right - side) / 2))
    y = int(math.floor((top + bottom - side) / 2))
    x = max(0, min(x, width - side)) if side <= width else max(width - side, min(x, 0))
    y = max(0, min(y, height - side)) if side <= height else max(height - side, min(y, 0))
    padding = [max(0, -x), max(0, -y), max(0, x + side - width), max(0, y + side - height)]
    pixels = np.asarray(image.convert('RGB'))
    padded = np.pad(pixels, ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)), mode='edge')
    crop = Image.fromarray(padded).crop((x + padding[0], y + padding[1], x + padding[0] + side, y + padding[1] + side))
    prepared = crop.resize((512, 512), Image.Resampling.LANCZOS)
    geometry = {
        'operation': 'face_centered_square_crop',
        'crop_box_in_oriented_source': [x, y, x + side, y + side],
        'padding_left_top_right_bottom': padding,
        'padding_mode': 'edge',
        'uniform_scale': 512 / side,
        'prepared_size': [512, 512],
        'scope': 'Deterministic framing without aspect distortion; not learned alignment.',
    }
    return prepared, geometry


def prepare_references(paths, destination, analyser):
    """Validate all photos before inference. The analyser must run on the CPU."""
    if not 3 <= len(paths) <= 4:
        raise ValueError('Provide 3–4 same-person reference photos.')
    paths = [Path(path) for path in paths]
    hashes = [sha(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Use different reference photographs, not duplicate files.')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a new reference-preparation output directory.')
    destination.mkdir(parents=True)
    prepared_paths, entries = [], []
    for index, path in enumerate(paths, 1):
        with Image.open(path) as opened:
            orientation = opened.getexif().get(274, 1)
            image = ImageOps.exif_transpose(opened).convert('RGB').copy()
        # An explicit contiguous BGR copy avoids stride-sensitive native calls.
        faces = analyser.get(np.asarray(image)[:, :, ::-1].copy())
        box, score = validate_face(faces, image.size, index)
        if image.width == image.height and orientation == 1:
            prepared_path = path
            geometry = {'operation': 'unchanged_square', 'prepared_size': list(image.size)}
        else:
            prepared_path = destination / f'reference_{index}.png'
            if image.width == image.height:
                prepared, geometry = image, {'operation': 'exif_orientation_only', 'prepared_size': list(image.size)}
            else:
                prepared, geometry = frame_rectangular_reference(image, box)
            prepared.save(prepared_path)
        entries.append({
            'reference_index': index,
            'original_sha256': hashes[index - 1],
            'prepared_sha256': sha(prepared_path),
            'oriented_source_size': list(image.size),
            'source_exif_orientation': orientation,
            'face_bbox': box.tolist(),
            'detection_score': score,
            'geometry': geometry,
        })
        prepared_paths.append(prepared_path)
    return prepared_paths, entries
