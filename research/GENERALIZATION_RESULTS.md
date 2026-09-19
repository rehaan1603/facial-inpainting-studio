# Generalization results — selection v1

Status: completed small local validation experiment; final-test identities remain reserved.

Reviewed 20 September 2026. **Outcome: mixed/inconclusive.** Mask-aware target FaceNet similarity exceeds the three single-reference selection baselines on average, but falls below all-reference conditioning. All four primary Holm-adjusted p-values are at least 0.50. Withheld-gallery similarity also favors all references. This is evidence of working runtime selection on locally held-out identity labels, not established restoration superiority.

Four locally fresh identity groups × three damage conditions × two seeds × seven policies = 168 logical rows. Identical generation requests are cached, not represented as independent samples. No model was trained in this cycle.

| Policy | FaceNet cosine ↑ (valid/24) | Gallery FaceNet ↑ | NIQE ↓ | BRISQUE ↓ | Hole MAE ↓ |
|---|---:|---:|---:|---:|---:|
| first | 0.7945 (24/24) | 0.5173 | 4.6861 | 10.1750 | 0.0955 |
| random | 0.7959 (24/24) | 0.5094 | 4.6945 | 10.1659 | 0.0929 |
| identity_only | 0.7896 (24/24) | 0.5093 | 4.6813 | 10.1320 | 0.0932 |
| quality_only | 0.7938 (24/24) | 0.4882 | 4.6702 | 10.1499 | 0.0961 |
| all | 0.8180 (24/24) | 0.5304 | 4.7119 | 10.1203 | 0.0900 |
| mask_aware | 0.8113 (24/24) | 0.5015 | 4.6880 | 10.1616 | 0.0944 |
| zero_scale | 0.7810 (24/24) | 0.4212 | 4.7111 | 10.0803 | 0.0792 |

## Primary paired comparisons

Identity means of jointly valid condition/seed pairs. Positive delta favors mask-aware selection. Holm correction covers these four comparisons.

| Compared with | Identities | Paired rows | Mean delta | Bootstrap 95% interval | Exact p / Holm p |
|---|---:|---:|---:|---|---|
| random | 4 | 24 | 0.0154 | [0.003840327262878418, 0.031980566680431366] | 0.1250 / 0.5000 |
| identity_only | 4 | 24 | 0.0216 | [0.009029986957708994, 0.036249185601870224] | 0.1250 / 0.5000 |
| quality_only | 4 | 24 | 0.0175 | [0.0, 0.03608153015375137] | 0.5000 / 0.7500 |
| all | 4 | 24 | -0.0067 | [-0.0165977676709493, 0.005557489891846974] | 0.3750 / 0.7500 |

## Condition breakdown

| Damage | Policy | FaceNet mean | Valid / 8 |
|---|---|---:|---:|
| central_face_mixed_medium | first | 0.6668 | 8 / 8 |
| central_face_mixed_medium | random | 0.6491 | 8 / 8 |
| central_face_mixed_medium | identity_only | 0.6615 | 8 / 8 |
| central_face_mixed_medium | quality_only | 0.6637 | 8 / 8 |
| central_face_mixed_medium | all | 0.6894 | 8 / 8 |
| central_face_mixed_medium | mask_aware | 0.6730 | 8 / 8 |
| central_face_mixed_medium | zero_scale | 0.6251 | 8 / 8 |
| eyes_removal_medium | first | 0.8769 | 8 / 8 |
| eyes_removal_medium | random | 0.8822 | 8 / 8 |
| eyes_removal_medium | identity_only | 0.8542 | 8 / 8 |
| eyes_removal_medium | quality_only | 0.8650 | 8 / 8 |
| eyes_removal_medium | all | 0.8854 | 8 / 8 |
| eyes_removal_medium | mask_aware | 0.8826 | 8 / 8 |
| eyes_removal_medium | zero_scale | 0.8554 | 8 / 8 |
| mouth_removal_medium | first | 0.8398 | 8 / 8 |
| mouth_removal_medium | random | 0.8563 | 8 / 8 |
| mouth_removal_medium | identity_only | 0.8531 | 8 / 8 |
| mouth_removal_medium | quality_only | 0.8527 | 8 / 8 |
| mouth_removal_medium | all | 0.8791 | 8 / 8 |
| mouth_removal_medium | mask_aware | 0.8782 | 8 / 8 |
| mouth_removal_medium | zero_scale | 0.8624 | 8 / 8 |

Rows with failed generation/evaluation or missing FaceNet/ArcFace target similarity: **0**. Full statuses, metric-specific coverage and all complementary metric means are in `generalization_selection_results_v1.json`; the numerical evidence ledger retains every row.

## Interpretation and limitations

Visual review of identity 1306's eye-removal sheet and the central-face mixed-damage sheets for identities 2790, 1043 and 787 (seed 17) found altered eye shape/gaze, exaggerated or changed mouths, softened skin and expression changes. Some reconstructed features remain implausible even when face detection succeeds. This was an engineering inspection, not a blinded human study. All 24 condition/seed contact sheets, including both seeds and every policy, are retained locally; none was removed based on appearance.

Four validation identities, not a final test. Bootstrap intervals unstable at n=4. ArcFace reuses the conditioning encoder. Missing detections remain explicit. No publication-ready superiority claim.

Selection uses damaged input, mask and runtime reference photos only. Dataset identity labels are used for experimental grouping, never by the inference scorer. Unknown pretrained exposure prevents a pretraining-independent generalization claim.

The scorer selects one global FaceID embedding; it does not inject spatial facial details. The all-reference arm has a larger information budget. Zero-scale disables identity contribution within the same adapter graph; it is not a separate LaMa or no-adapter baseline.

Noise may receive an artificially high sharpness score. Pose and visibility are five-landmark proxies, not measured occlusion. Synthetic regional degradation with a supplied mask does not establish blind restoration, natural occlusion robustness, or global deblurring.

Inspect local outputs/generalization_v1/contact_sheets and every failed row before drawing quality conclusions. Final evaluation, external methods, broader identities/severity, human assessment and a justified research contribution remain outstanding.
