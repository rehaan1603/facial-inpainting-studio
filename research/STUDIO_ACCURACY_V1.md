# Studio generation accuracy audit — 23 September 2026

Four already-observed development identities. Descriptive quality audit; no population accuracy or novelty claim.

Completed scores: **42/44**, including eight unchanged-image controls. Planned generations: **36**. No quality-based retries or discarded failures.

FaceNet/ArcFace values are cosine similarities, not accuracy percentages. ArcFace is a conditioning-related diagnostic; FaceNet is the independent identity evaluator. All metrics retain missing-detection counts. Whole-image metrics can conceal errors in a small mask; masked MAE and local visual review remain essential.

| Condition / mode | Scored | FaceNet ↑ | Gallery ↑ | Mask MAE ↓ | LPIPS ↓ | SSIM ↑ | NIQE ↓ | BRISQUE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| removal/lama | 4/4 | 0.5341 (4/4) | 0.2579 (4/4) | 0.0745 | 0.0381 | 0.9719 | 4.9056 | 9.0318 |
| removal/observed | 4/4 | 0.4802 (3/4) | 0.1833 (3/4) | 0.1970 | 0.0988 | 0.9551 | 4.9444 | 11.0495 |
| removal/reference_1024 | 4/4 | 0.7362 (4/4) | 0.5632 (4/4) | 0.0905 | 0.0290 | 0.9650 | 4.4740 | 9.8668 |
| removal/reference_512 | 3/4 | 0.5168 (3/4) | 0.4149 (3/4) | 0.0821 | 0.0341 | 0.9689 | 4.7448 | 8.3407 |
| removal/resshift | 4/4 | 0.6561 (4/4) | 0.3898 (4/4) | 0.0794 | 0.0328 | 0.9678 | 4.6510 | 9.9749 |
| removal/selection_1024 | 3/4 | 0.6925 (3/4) | 0.5216 (3/4) | 0.0957 | 0.0297 | 0.9653 | 4.4103 | 8.3522 |
| removal/selection_512 | 4/4 | 0.5799 (4/4) | 0.4582 (4/4) | 0.0822 | 0.0361 | 0.9676 | 4.7160 | 9.5765 |
| mixed/evidence_high | 4/4 | 0.8208 (4/4) | 0.5221 (4/4) | 0.0574 | 0.0315 | 0.9727 | 4.1846 | 8.5768 |
| mixed/evidence_low | 4/4 | 0.8139 (4/4) | 0.5136 (4/4) | 0.0511 | 0.0310 | 0.9729 | 4.1735 | 8.6672 |
| mixed/observed | 4/4 | 0.8960 (4/4) | 0.4802 (4/4) | 0.0446 | 0.0519 | 0.9689 | 4.0709 | 7.8703 |
| mixed/refldm_partial | 4/4 | 0.9233 (4/4) | 0.5145 (4/4) | 0.0414 | 0.0364 | 0.9753 | 4.1079 | 6.3834 |

## Paired checks against unchanged input

| Condition / mode | Identity improved | Mask error improved | Perceptual error improved |
|---|---:|---:|---:|
| removal/lama | 3/3 | 4/4 | 4/4 |
| removal/reference_1024 | 3/3 | 4/4 | 4/4 |
| removal/reference_512 | 1/2 | 3/3 | 3/3 |
| removal/resshift | 3/3 | 4/4 | 4/4 |
| removal/selection_1024 | 2/2 | 3/3 | 3/3 |
| removal/selection_512 | 2/3 | 4/4 | 4/4 |
| mixed/evidence_high | 0/4 | 0/4 | 4/4 |
| mixed/evidence_low | 0/4 | 0/4 | 4/4 |
| mixed/refldm_partial | 4/4 | 4/4 | 4/4 |

## Previous 512-pixel reference path: regression check

Exploratory regression check against saved pre-repair generations, added after first audit case scored. Same inputs/masks, seed, adapter scale, steps, resolution and Poisson blending; preprocessing and strength differ.

| Case | Identity delta (new minus old) | Mask MAE delta | LPIPS delta |
|---|---:|---:|---:|
| 1306_removal | -0.2047 | 0.0018 | 0.0057 |
| 2790_removal | N/A | N/A | N/A |
| 1043_removal | -0.1716 | -0.0149 | 0.0042 |
| 787_removal | -0.1596 | 0.0002 | 0.0001 |

## Limits and remaining work

This audit covers each offered generation variant with the exact painted mask, not every mask treatment, seed, damage severity or uploaded image. Partial-damage modes use a fixed user-style evidence weight of 128/255; these are not calibrated confidence predictions. ReF-LDM is deliberately excluded from complete removal because it does not fill missing regions.

All four identities were already used in development. Four independent identities do not establish generalization; no significance or publication-ready novelty claim is made. Better average scores do not guarantee correct expression, gaze, or hidden anatomy. Reserved final-test identities remain untouched.

Per-image metrics, missing detections, output hashes, and paired counts are in `studio_accuracy_v1.json`. Local comparison sheets include clean targets only for evaluation and remain excluded from GitHub. Visual findings are recorded separately after inspection.
