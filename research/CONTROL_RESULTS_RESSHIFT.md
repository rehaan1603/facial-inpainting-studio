# Expanded resshift controls

All 28 radius/blending combinations were evaluated on 48 tuning identities. The minimum tuning LPIPS determined one global combination, frozen before assessment inference. The existing 48 development assessment identities were reused; final test remains unused.

Selected radius: **12 pixels**. Selected inward feather width: **4 pixels**.

| Assessment metric | Previous radius 8, hard blend | New selected control |
|---|---:|---:|
| full_face_lpips | 0.036376 | 0.033296 |
| hole_mae | 0.087174 | 0.080133 |
| visible_mae | 0.002757 | 0.004219 |

## Paired changes

New minus previous radius-8 control; lower is better. Confidence intervals resample 48 identities, 2,000 times.

- full_face_lpips: -0.003080, 95% interval [-0.004683, -0.001601].
- hole_mae: -0.007041, 95% interval [-0.009481, -0.004653].
- visible_mae: 0.001462, 95% interval [0.001328, 0.001598].

## Condition breakdown

| Condition | Old LPIPS | New LPIPS | Old visible MAE | New visible MAE |
|---|---:|---:|---:|---:|
| accurate | 0.02629 | 0.03143 | 0.00231 | 0.00382 |
| under | 0.04062 | 0.02610 | 0.00048 | 0.00131 |
| over | 0.03643 | 0.04267 | 0.00538 | 0.00734 |
| shift | 0.04587 | 0.03217 | 0.00278 | 0.00409 |
| mixed_boundary | 0.03266 | 0.03411 | 0.00284 | 0.00454 |

## Interpretation limits

This is a simple control, not a learned or novel method. Selection optimizes full-face LPIPS and does not enforce a visible-pixel preservation tolerance. Assess the regional errors and condition breakdown before calling the result better overall. Feathering uses only the expanded supplied mask; clean target parsing never defines inference blending.

This remains repeated development assessment, with synthetic textures, unmatched mask areas, and unresolved checkpoint pretraining exposure. The unchanged final test will be required after design decisions are locked. Diffusion sampling variation is not estimated.

## Next step

Both backbone control searches have results. Add area-matched corruptions and real occluder assets before defining a learned contribution.
