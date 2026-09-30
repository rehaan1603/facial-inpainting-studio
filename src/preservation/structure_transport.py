"""Coordinate-only transport diagnostic; no image detection or model inference.

The prospective method is research/STRUCTURE_TRANSPORT_METHOD_V1.md. Landmarks
use (x, y) pixel-centre coordinates. Image inputs are never modified in place.
"""
from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree


class StructureTransportError(ValueError):
    """Explicit rejected attempt, with metadata suitable for a failure receipt."""

    def __init__(self, reason, metadata=None):
        self.metadata = dict(metadata or {}, status="rejected", reason=reason)
        super().__init__(reason)


def _points(value, name, minimum=1):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < minimum:
        raise StructureTransportError(f"{name} must be an N-by-2 array with N >= {minimum}")
    if not np.isfinite(result).all():
        raise StructureTransportError(f"{name} contains nonfinite coordinates")
    return result


def _mask(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype != np.bool_ or min(value.shape) < 2:
        raise StructureTransportError("missing mask must be a 2D boolean array of size at least 2x2")
    if not value.any():
        raise StructureTransportError("missing mask is empty")
    return value


def _rank_ratio(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(singular[-1] / singular[0]) if singular[0] > 1e-12 else 0.0


def similarity_fit(source, dest):
    """Return a proper (no reflection), positive-scale source-to-dest 2x3 fit."""
    source, dest = _points(source, "source", 3), _points(dest, "dest", 3)
    if source.shape != dest.shape:
        raise StructureTransportError("source/destination landmark shapes differ")
    if min(_rank_ratio(source), _rank_ratio(dest)) < 0.02:
        raise StructureTransportError("degenerate landmark fit: rank ratio below 0.02")
    x, y = source - source.mean(axis=0), dest - dest.mean(axis=0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(2)
    if np.linalg.det(vt.T @ u.T) < 0:
        sign[-1] = -1
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float((singular * sign).sum() / np.square(x).sum())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise StructureTransportError("similarity fit has invalid scale")
    linear = scale * rotation
    return np.column_stack([linear, dest.mean(axis=0) - linear @ source.mean(axis=0)])


def transform_points(points, matrix):
    points = _points(points, "points")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise StructureTransportError("transform must be a finite 2x3 matrix")
    return points @ matrix[:, :2].T + matrix[:, 2]


def build_target_shape(obs68, reference68, missing, anchor_margin=8, minanchors=6):
    """Register references using only safely visible observed anchors; median shape.

    Distance to damage means Euclidean distance to the nearest masked pixel
    centre. This explicit grid convention also applies at fractional coordinates.
    Missing observed landmarks never determine registration.
    """
    observed = _points(obs68, "obs68")
    mask = _mask(missing)
    if observed.shape != (68, 2):
        raise StructureTransportError("obs68 must contain exactly 68 landmarks")
    if not np.isfinite(anchor_margin) or anchor_margin < 0:
        raise StructureTransportError("anchor margin must be finite and nonnegative")
    if not isinstance(minanchors, (int, np.integer)) or minanchors < 3:
        raise StructureTransportError("minanchors must be an integer at least three")
    references = [_points(value, "reference68") for value in reference68]
    if not references or any(value.shape != (68, 2) for value in references):
        raise StructureTransportError("one or more complete 68-point reference shapes are required")
    height, width = mask.shape
    x, y = observed.T
    edge_distance = np.minimum.reduce([x, y, width - 1 - x, height - 1 - y])
    masked_y, masked_x = np.nonzero(mask)
    distance, _ = cKDTree(np.column_stack([masked_x, masked_y])).query(observed)
    in_bounds = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    nearest = np.floor(observed[in_bounds] + 0.5).astype(int)
    visible = np.zeros(68, bool)
    visible[in_bounds] = ~mask[nearest[:, 1], nearest[:, 0]]
    anchors = visible & (distance >= anchor_margin) & (edge_distance >= anchor_margin)
    indices = np.flatnonzero(anchors)
    metadata = {"anchor_indices": indices.tolist(), "anchor_count": len(indices),
                "anchor_margin": float(anchor_margin), "reference_count": len(references),
                "distance_convention": "Euclidean distance to masked pixel centres",
                "consensus": "coordinatewise median of registered reference landmarks"}
    if len(indices) < minanchors:
        raise StructureTransportError("insufficient safely visible registration anchors", metadata)
    metadata["observed_anchor_rank_ratio"] = _rank_ratio(observed[anchors])
    if metadata["observed_anchor_rank_ratio"] < 0.02:
        raise StructureTransportError("observed anchors are degenerate", metadata)
    registered, residuals = [], []
    for number, reference in enumerate(references):
        try:
            matrix = similarity_fit(reference[anchors], observed[anchors])
        except StructureTransportError as exc:
            raise StructureTransportError(f"reference {number} registration failed: {exc}", metadata) from exc
        registered.append(transform_points(reference, matrix))
        residuals.append(float(np.sqrt(np.mean(np.sum(
            (registered[-1][anchors] - observed[anchors]) ** 2, axis=1)))))
    metadata.update(status="complete", registration_rms_pixels=residuals)
    return np.median(np.stack(registered), axis=0), metadata


def transport(scaffold, observed, mask, source68, desired68,
              regularization=0.05, max_displacement_fraction=0.1):
    """Solve an inverse warp with fixed-zero displacement outside the missing mask.

    At destination q, displacement d(q) approximates source p minus q. Thus
    remapping samples scaffold(q+d(q)), moving a feature from p towards q.
    Every admitted constraint has all four bilinear support pixels in the same
    mask component. Components without constraints consequently stay unchanged.
    """
    import cv2

    mask = _mask(mask)
    scaffold, observed = np.asarray(scaffold), np.asarray(observed)
    height, width = mask.shape
    for name, rgb in [("scaffold", scaffold), ("observed", observed)]:
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise StructureTransportError(f"{name} must be matching HxWx3 RGB uint8")
    source, desired = _points(source68, "source68"), _points(desired68, "desired68")
    if source.shape != desired.shape:
        raise StructureTransportError("source/destination landmark shapes differ")
    if not np.isfinite(regularization) or regularization <= 0:
        raise StructureTransportError("regularization must be finite and positive")
    if not np.isfinite(max_displacement_fraction) or max_displacement_fraction <= 0:
        raise StructureTransportError("maximum displacement fraction must be finite and positive")
    labels, components = ndimage.label(mask)  # default four-connected structure
    ys, xs = np.nonzero(mask)
    count = len(xs)
    indices = np.full(mask.shape, -1, dtype=np.int64)
    indices[ys, xs] = np.arange(count)
    ar, ac, av, rhs, accepted, rejected = [], [], [], [], [], Counter()
    for number, (p, q) in enumerate(zip(source, desired)):
        if not ((p >= 0).all() and (q >= 0).all()
                and p[0] <= width - 1 and q[0] <= width - 1
                and p[1] <= height - 1 and q[1] <= height - 1):
            rejected["outside_image"] += 1
            continue
        pn, qn = np.floor(p + 0.5).astype(int), np.floor(q + 0.5).astype(int)
        component = labels[qn[1], qn[0]]
        if component == 0 or labels[pn[1], pn[0]] != component:
            rejected["outside_or_different_mask_component"] += 1
            continue
        x0, y0 = np.floor(q).astype(int)
        if x0 + 1 >= width or y0 + 1 >= height:
            rejected["bilinear_support_outside_image"] += 1
            continue
        support_x, support_y = [x0, x0 + 1, x0, x0 + 1], [y0, y0, y0 + 1, y0 + 1]
        if not np.all(labels[support_y, support_x] == component):
            rejected["bilinear_support_not_fully_masked"] += 1
            continue
        dx, dy = q - [x0, y0]
        weights = [(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
        ar.extend([len(rhs)] * 4)
        ac.extend(indices[support_y, support_x].tolist())
        av.extend(weights)
        rhs.append(p - q)
        accepted.append(number)
    metadata = {"mask_pixels": count, "mask_components": int(components),
                "constraint_count": len(accepted), "constraint_indices": accepted,
                "rejected_constraints": dict(rejected), "regularization": float(regularization),
                "max_displacement_fraction": float(max_displacement_fraction),
                "interpolation": "OpenCV INTER_LINEAR, BORDER_REPLICATE; original visible RGB restored",
                "outside_displacement": "zero (Dirichlet)"}
    if not accepted:
        raise StructureTransportError("no admissible within-component landmark constraints", metadata)
    constraint = sparse.coo_matrix((av, (ar, ac)), shape=(len(rhs), count)).tocsr()
    rhs = np.asarray(rhs)
    # The diagonal remains four even at image/mask boundaries: missing neighbours
    # are fixed-zero Dirichlet values, not a degree-normalized graph Laplacian.
    lr, lc, lv = list(range(count)), list(range(count)), [4.0] * count
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        origin = np.flatnonzero(inside)
        neighbor = indices[ny[inside], nx[inside]]
        keep = neighbor >= 0
        lr.extend(origin[keep].tolist())
        lc.extend(neighbor[keep].tolist())
        lv.extend([-1.0] * int(keep.sum()))
    laplacian = sparse.coo_matrix((lv, (lr, lc)), shape=(count, count)).tocsr()
    if np.array_equal(rhs, np.zeros_like(rhs)):
        displacement = np.zeros((count, 2))
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                displacement = spsolve(
                    (constraint.T @ constraint + regularization * laplacian).tocsc(),
                    constraint.T @ rhs)
        except (RuntimeError, ValueError, MatrixRankWarning) as exc:
            raise StructureTransportError(f"displacement solve failed: {exc}", metadata) from exc
    if displacement.shape != (count, 2) or not np.isfinite(displacement).all():
        raise StructureTransportError("displacement solution is nonfinite or malformed", metadata)
    maximum = float(np.linalg.norm(displacement, axis=1).max())
    metadata["maximum_displacement_pixels"] = maximum
    if maximum > max_displacement_fraction * width:
        raise StructureTransportError("maximum displacement exceeds frozen bound", metadata)
    field = np.zeros((height, width, 2), dtype=np.float64)
    field[mask] = displacement
    du_dy, du_dx = np.gradient(field[:, :, 0])
    dv_dy, dv_dx = np.gradient(field[:, :, 1])
    determinant = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    changed = mask & np.any(field != 0, axis=2)
    minimum = float(determinant[changed].min()) if changed.any() else 1.0
    metadata.update(minimum_changed_inverse_jacobian=minimum,
                    changed_mask_pixels=int(changed.sum()),
                    constraint_rms_residual_pixels=float(np.sqrt(np.mean(np.sum(
                        (constraint @ displacement - rhs) ** 2, axis=1)))))
    if not np.isfinite(determinant).all() or minimum <= 0:
        raise StructureTransportError("inverse warp has a nonpositive or nonfinite Jacobian", metadata)
    if not changed.any():
        result = scaffold.copy()
    else:
        grid_y, grid_x = np.indices(mask.shape, dtype=np.float64)
        result = cv2.remap(scaffold, (grid_x + field[:, :, 0]).astype(np.float32),
                           (grid_y + field[:, :, 1]).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    result[~mask] = observed[~mask]
    metadata.update(status="complete", exact_outside_preservation=bool(
        np.array_equal(result[~mask], observed[~mask])), zero_displacement=not bool(changed.any()))
    return result, metadata
