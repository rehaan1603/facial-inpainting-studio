# Area-matched development benchmark

The benchmark matches true missing area, false-positive and false-negative mask counts, mask geometry, and synthetic occluder RGB across eye-center, mouth-center, and upper-image locations. All supplied masks have at least 12 pixels of image margin, preventing clipping under the evaluated correction radii.

## Data and protocol

- 48 assessment identities; three area levels, three locations, four mask conditions.
- 432 location/area cases and 5184 model/control metric rows.
- Exact areas: 1,966, 3,932 and 6,554 pixels (approximately 3%, 6% and 10% of a 256-square image).
- Error count is round(0.2 × missing pixels). Under masks remove that count; over masks add it; mixed masks do both. Counts match across locations within each error condition, not across different conditions.
- Controls were frozen from v2: LaMa radius 8/hard blend, ResShift radius 8/hard blend and radius 12/feather 4. No v3 tuning.
- Dataset construction also retains 47 tuning identities, but they are not evaluated here. One tuning identity lacks an eye center; all its area groups are excluded and recorded.
- The same 48 development assessment identities were used previously. This is a controlled diagnostic, not a fresh test set.

## Aggregate assessment

| Backbone | Radius | Feather | LPIPS | Hole MAE | Visible MAE |
|---|---:|---:|---:|---:|---:|
| lama | 8 | 0 | 0.04105 | 0.11448 | 0.00260 |
| resshift | 8 | 0 | 0.02203 | 0.07721 | 0.00216 |
| resshift | 12 | 4 | 0.02686 | 0.08300 | 0.00348 |

## Paired ResShift control changes

Radius 12/feather 4 minus radius 8/hard blend. The 95% intervals resample identities, retaining correlated locations, areas and masks together; 2,000 replicates.

- full_face_lpips: 0.004838, interval [0.004475, 0.005223].
- hole_mae: 0.005795, interval [0.004987, 0.006696].
- visible_mae: 0.001325, interval [0.001239, 0.001427].

The radius-12/feathered setting is worse on all three aggregate metrics here. Its earlier v2 tuning gain does not transfer to this changed protocol. This supports checking robustness across corruption protocols; it does not establish a novel method or prove that area matching alone caused the reversal, because shape and error construction also changed.

## Location breakdown

| Location | Control | LPIPS | Hole MAE | Visible MAE |
|---|---|---:|---:|---:|
| eye_center | lama/8/0 | 0.05118 | 0.13084 | 0.00266 |
| eye_center | resshift/8/0 | 0.02059 | 0.07472 | 0.00207 |
| eye_center | resshift/12/4 | 0.02455 | 0.07880 | 0.00325 |
| mouth_center | lama/8/0 | 0.03958 | 0.12008 | 0.00253 |
| mouth_center | resshift/8/0 | 0.01993 | 0.08209 | 0.00207 |
| mouth_center | resshift/12/4 | 0.02372 | 0.08738 | 0.00341 |
| upper_image | lama/8/0 | 0.03239 | 0.09253 | 0.00261 |
| upper_image | resshift/8/0 | 0.02556 | 0.07481 | 0.00233 |
| upper_image | resshift/12/4 | 0.03232 | 0.08283 | 0.00379 |

## Limits

Locations describe placement centers, not guaranteed full coverage of an anatomical feature. The upper-image location may cover hair, forehead or background. Shapes are translated ellipse-like masks with spatially varying boundary perturbations, not real occluder silhouettes. Equal geometry and texture remove selected confounds; they do not make the surrounding image content identical or establish causal semantic difficulty.

Checkpoint pretraining exposure remains unresolved; only one diffusion seed per case is used. No identity fidelity claim, trained new method, or publication claim is established. Final test data remains unused for inference.

## Provenance and next step

The canonical files are outputs/area_matched_v3_margin12 and outputs/area_matched_v3_margin12_evaluation. The earlier directories without margin12 are superseded engineering artifacts from a boundary-clipping check and must not be used as results.

Next, add held-out realistic occluder assets and finish the pretraining/near-duplicate audit. Any learned correction must be compared against generic learned refinement and blending controls, with a preset preservation tolerance.
