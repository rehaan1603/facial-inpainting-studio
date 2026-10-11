"""Visible-only geometry for a prospective spatial reference attention branch.

No target truth, target eye landmarks, generated scaffold, or dataset annotation
is accepted by this API. Failure explicitly means use the existing baseline.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from src.preservation.structure_transport import similarity_fit, transform_points

ANCHORS = tuple(range(17)) + tuple(range(27, 36)) + tuple(range(48, 68))


def sanitize(observed, mask):
    observed, mask = np.asarray(observed), np.asarray(mask)
    if observed.dtype != np.uint8 or mask.dtype != bool or observed.shape != (*mask.shape, 3):
        raise ValueError('Expected uint8 RGB and matching boolean mask')
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError('Mask needs missing and visible pixels')
    return np.where(mask[..., None], np.uint8(128), observed)


def safe_anchors(shape, mask, margin=8):
    shape = np.asarray(shape, dtype=float)
    if shape.shape != (68, 2) or not np.isfinite(shape).all():
        raise ValueError('Expected finite 68-point coordinates')
    height, width = mask.shape
    distance = distance_transform_edt(~mask)
    indices = []
    for index in ANCHORS:
        x, y = np.floor(shape[index] + .5).astype(int)
        if 0 <= x < width and 0 <= y < height and distance[y, x] >= margin and not mask[y, x]:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def fit_geometry(target_shape, target_bbox, reference_shapes, mask):
    """Fit/validate visible anchors, returning only deployable geometric metadata."""
    target_shape = np.asarray(target_shape, dtype=float)
    bbox = np.asarray(target_bbox, dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= bbox[0]:
        raise ValueError('Invalid target box')
    width = float(bbox[2] - bbox[0])
    anchors = safe_anchors(target_shape, mask)
    result = dict(status='fallback', anchors=anchors.tolist(), references=[], reason=None)
    if len(anchors) < 8:
        result['reason'] = 'Fewer than eight safe visible anchors'
        return result
    validation = anchors[::3]
    training = np.array([i for i in anchors if i not in set(validation)], dtype=int)
    result.update(fit_anchors=training.tolist(), validation_anchors=validation.tolist(), face_width=width)
    accepted = []
    for index, shape in enumerate(reference_shapes):
        row = dict(index=index, eligible=False)
        try:
            shape = np.asarray(shape, dtype=float)
            if shape.shape != (68, 2) or not np.isfinite(shape).all():
                raise ValueError('Missing or invalid reference landmarks')
            fit = similarity_fit(shape[training], target_shape[training])
            held_error = float(np.sqrt(np.square(transform_points(shape[validation], fit) - target_shape[validation]).sum(1).mean()) / width)
            full = similarity_fit(shape[anchors], target_shape[anchors])
            error = float(np.sqrt(np.square(transform_points(shape[anchors], full) - target_shape[anchors]).sum(1).mean()) / width)
            row.update(held_out_normalized_rms=held_error, full_normalized_rms=error)
            if held_error > .04 or error > .04:
                raise ValueError('Visible-anchor error exceeds fixed 0.04 face-width bound')
            registered = transform_points(shape, full)
            eyes = np.stack([registered[36:42].mean(0), registered[42:48].mean(0)])
            row.update(eligible=True, transform=full.tolist(), mapped_eye_centres=eyes.tolist(),
                       reference_eye_brow_envelope=[shape[17:27].tolist(), shape[36:48].tolist()],
                       reliability=float(np.exp(-held_error**2 / (2 * .03**2))))
            accepted.append((index, eyes))
        except (ValueError, TypeError) as exc:
            row['reason'] = str(exc)
        result['references'].append(row)
    if len(accepted) < 2:
        result['reason'] = 'Fewer than two reliable spatial references'
        return result
    disagreement = max(float(np.linalg.norm(a[1] - b[1], axis=1).max() / width)
                       for i, a in enumerate(accepted) for b in accepted[i + 1:])
    result['maximum_eye_centre_disagreement'] = disagreement
    if disagreement > .12:
        result['reason'] = 'Reference eye positions disagree beyond fixed 0.12 face-width bound'
        return result
    result.update(status='usable', reason=None, accepted_reference_indices=[x[0] for x in accepted])
    return result


def estimate_geometry(observed, mask, references, extractor):
    """Only observed/mask/reference arrays cross this inference boundary."""
    safe = sanitize(observed, mask)
    if len(references) != 4:
        raise ValueError('This fixed experiment requires the same four references')
    try:
        target_shape, bbox = extractor.shape(safe)
    except (ValueError, RuntimeError) as exc:
        return dict(status='fallback', reason='Damaged-target detection: ' + str(exc), references=[])
    shapes, failures = [], {}
    for i, rgb in enumerate(references):
        try:
            shapes.append(extractor.shape(rgb)[0])
        except (ValueError, RuntimeError) as exc:
            shapes.append(None)
            failures[str(i)] = str(exc)
    result = fit_geometry(target_shape, bbox, shapes, mask)
    result['reference_detection_failures'] = failures
    return result
