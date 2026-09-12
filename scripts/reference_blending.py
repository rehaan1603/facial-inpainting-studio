"""Optional, established Poisson boundary harmonization for reference inpainting.

Contract: generated and observed are equally shaped H x W x 3 uint8 RGB numpy
arrays; mask is an H x W boolean array where True means reconstructed pixels.
No unobscured target image is accepted or used. The returned RGB array preserves
observed pixels outside the mask exactly, including holes within the mask.

This is a postprocessing baseline using OpenCV NORMAL_CLONE, not a novel method.
It may change generated colour or contrast; callers should offer it explicitly.
"""
import numpy as np


def harmonize_reference(generated, observed, mask, mode="poisson"):
    """Return ``(rgb_uint8, metadata)`` with explicit handling of unusable masks.

    ``mode='hard'`` selects exact paste-back without requiring OpenCV. With
    ``mode='poisson'``, empty masks return the observed image unchanged. Masks
    without a four-pixel known boundary collar, or without any full 3 x 3 interior,
    use hard paste-back and report their fallback reason. Other OpenCV errors propagate
    with context, rather than silently substituting a different algorithm.
    """
    if mode not in {"poisson", "hard"}:
        raise ValueError("mode must be 'poisson' or 'hard'.")
    for name, array in (("generated", generated), ("observed", observed)):
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
            raise TypeError(f"{name} must be a uint8 numpy RGB array.")
        if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
            raise ValueError(f"{name} must have nonempty shape H x W x 3.")
    if generated.shape != observed.shape:
        raise ValueError("Generated and observed image shapes must match.")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a boolean numpy array (True means reconstruct).")
    if mask.shape != observed.shape[:2]:
        raise ValueError("Mask shape must match the image's H x W dimensions.")

    result = observed.copy()
    result[mask] = generated[mask]
    metadata = {
        "requested_mode": mode,
        "mode": "hard",
        "fallback": False,
        "fallback_reason": None,
        "mask_pixels": int(mask.sum()),
        "outside_mask": "Observed RGB pixels preserved exactly by final composition.",
        "target_rgb_used": False,
    }
    if not mask.any():
        metadata.update(mode="unchanged", fallback=mode == "poisson", fallback_reason="empty_mask")
        return result, metadata
    if mode == "hard":
        return result, metadata

    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        metadata.update(fallback=True, fallback_reason="mask_touches_image_border")
        return result, metadata
    height, width = mask.shape
    has_interior = False
    if height >= 3 and width >= 3:
        interior = np.ones((height - 2, width - 2), dtype=bool)
        for y in range(3):
            for x in range(3):
                interior &= mask[y : y + height - 2, x : x + width - 2]
        has_interior = bool(interior.any())
    if not has_interior:
        metadata.update(fallback=True, fallback_reason="mask_has_no_3x3_interior")
        return result, metadata
    if mask[:5, :].any() or mask[-5:, :].any() or mask[:, :5].any() or mask[:, -5:].any():
        metadata.update(fallback=True, fallback_reason="insufficient_known_boundary_collar")
        return result, metadata

    try:
        import cv2
    except ImportError as error:
        raise ImportError("Poisson harmonization requires OpenCV in reference_env. Use mode='hard' to paste without OpenCV.") from error

    binary = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    # OpenCV uses destination pixels at the solve's inner boundary. Solving on
    # exactly the hole would retain its submitted occlusion colour at that edge.
    # OpenCV also erodes its gradient mask three times. A four-pixel collar
    # keeps those destination gradients outside the submitted hole. Restrict
    # the final result back to the requested mask after the solve.
    # See modules/photo/src/seamless_cloning_impl.cpp, computeDerivatives.
    solve_mask = cv2.dilate(binary, np.ones((9, 9), np.uint8))
    collar = (solve_mask > 0) & ~mask
    # Extend the generated edge into that collar. Copying observed colours into
    # the source would encode the hard colour seam in the source gradients.
    _, nearest_labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    source_y, source_x = np.nonzero(mask)
    nearest_indices = nearest_labels[collar] - 1
    if np.any(nearest_indices < 0) or np.any(nearest_indices >= len(source_y)):
        raise RuntimeError("OpenCV returned invalid nearest-pixel labels for the boundary collar.")
    extended_source = generated.copy()
    extended_source[collar] = generated[source_y[nearest_indices], source_x[nearest_indices]]
    x, y, box_width, box_height = cv2.boundingRect(solve_mask)
    centre = (x + box_width // 2, y + box_height // 2)
    try:
        blended_bgr = cv2.seamlessClone(
            np.ascontiguousarray(extended_source[:, :, ::-1]),
            np.ascontiguousarray(observed[:, :, ::-1]),
            solve_mask,
            centre,
            cv2.NORMAL_CLONE,
        )
    except cv2.error as error:
        raise RuntimeError(f"OpenCV Poisson harmonization failed for {width} x {height} image, mask bounding box {(x, y, box_width, box_height)}: {error}") from error
    blended_rgb = blended_bgr[:, :, ::-1]
    if blended_rgb.shape != observed.shape or blended_rgb.dtype != np.uint8:
        raise RuntimeError("OpenCV returned an unexpected image shape or dtype.")
    # The destination includes the submitted occlusion within the mask. The
    # Poisson solve uses its known boundary, not a clean target photograph.
    result[mask] = blended_rgb[mask]
    metadata.update(mode="poisson", technique="OpenCV seamlessClone NORMAL_CLONE (Poisson boundary harmonization)", computational_collar_pixels=4, source_collar="Nearest generated pixel inside the requested mask", opencv_version=cv2.__version__)
    return result, metadata
