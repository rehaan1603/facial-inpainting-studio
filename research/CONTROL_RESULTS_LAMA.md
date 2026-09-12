# Expanded lama controls

All 28 radius/blending combinations were evaluated on 48 tuning identities. The minimum tuning LPIPS determined one global combination, frozen before assessment inference. The existing 48 development assessment identities were reused; final test remains unused.

Selected radius: **8 pixels**. Selected inward feather width: **0 pixels**.

| Assessment metric | Previous radius 8, hard blend | New selected control |
|---|---:|---:|
| full_face_lpips | 0.063119 | 0.063119 |
| hole_mae | 0.126295 | 0.126295 |
| visible_mae | 0.003298 | 0.003298 |

## Paired changes

New minus previous radius-8 control; lower is better. Confidence intervals resample 48 identities, 2,000 times.

The selected control is unchanged. Zero paired differences and zero-width intervals here mean the same deterministic outputs were reproduced; they do not establish absence of uncertainty for generalization or other controls.
- full_face_lpips: 0.000000, 95% interval [0.000000, 0.000000].
- hole_mae: 0.000000, 95% interval [0.000000, 0.000000].
- visible_mae: 0.000000, 95% interval [0.000000, 0.000000].

## Condition breakdown

| Condition | Old LPIPS | New LPIPS | Old visible MAE | New visible MAE |
|---|---:|---:|---:|---:|
| accurate | 0.05347 | 0.05347 | 0.00276 | 0.00276 |
| under | 0.06211 | 0.06211 | 0.00054 | 0.00054 |
| over | 0.06944 | 0.06944 | 0.00655 | 0.00655 |
| shift | 0.07226 | 0.07226 | 0.00330 | 0.00330 |
| mixed_boundary | 0.05831 | 0.05831 | 0.00334 | 0.00334 |

## Interpretation limits

This is a simple control, not a learned or novel method. Selection optimizes full-face LPIPS and does not enforce a visible-pixel preservation tolerance. Assess the regional errors and condition breakdown before calling the result better overall. Feathering uses only the expanded supplied mask; clean target parsing never defines inference blending.

This remains repeated development assessment, with synthetic textures, unmatched mask areas, and unresolved checkpoint pretraining exposure. The unchanged final test will be required after design decisions are locked. Diffusion sampling variation is not estimated.

## Next step

Both backbone control searches have results. Add area-matched corruptions and real occluder assets before defining a learned contribution.
