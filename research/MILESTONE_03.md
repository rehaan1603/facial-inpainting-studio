# Matched backbone development comparison

Both models use identical images, synthetic corruptions, supplied masks, and identity partitions. Each radius is chosen on tuning LPIPS independently; assessment identities are not used for selection.

| Backbone | Selected radius | Assessment LPIPS | Hole MAE | Visible MAE |
|---|---:|---:|---:|---:|
| lama | 8 | 0.06312 | 0.12630 | 0.00330 |
| resshift | 8 | 0.03638 | 0.08717 | 0.00276 |

## Paired assessment differences

ResShift minus LaMa, using each model’s tuning-selected radius. Lower is better for all metrics. Intervals resample 48 identities, with 2,000 replicates.
- full_face_lpips: -0.026743; 95% interval [-0.029235, -0.024384].
- hole_mae: -0.039121; 95% interval [-0.042827, -0.035360].
- visible_mae: -0.000541; 95% interval [-0.000598, -0.000481].

## Limits and next experiment

This compares these checkpoints under this development protocol, not intrinsic architectures or published paper performance. Training data and pretraining exposure differ and remain unresolved. One diffusion seed is fixed per image/family and shared across its controls; sampling variation is not estimated. Exact masks are privileged diagnostics. The synthetic mask families are not area-matched, and final test images remain unused for inference.

Radius 8 is the boundary of the current search. Extend the tuning sweep and add feathered compositing before claiming an optimized correction control. A learned method must improve the reconstruction–preservation curve beyond these controls. No novel method has been evaluated.

## Qualitative inspection

The preselected first assessment identity in outputs/benchmark_v2_resshift/preview.png shows residual synthetic corruption with uncorrected masks. In the eye-region mixed-boundary case, radius 4 produces glasses absent from the target. The exact-mask diagnostic looks more plausible but does not recover every hidden detail. These examples illustrate why aggregate perceptual gains cannot establish identity fidelity. The preview is a diagnostic example, not a measured failure rate.
