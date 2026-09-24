# Studio generation accuracy audit — 24 September 2026

Two newly selected local development identities (3182, 1037). Pretrained exposure unknown. No population accuracy or novelty claim.

Completed scores: **29/30**, including four unchanged-image controls. Planned generations: **26**. No quality-based retries or discarded failures.

FaceNet/ArcFace values are cosine similarities, not accuracy percentages. ArcFace is a conditioning-related diagnostic; FaceNet is the independent identity evaluator. All metrics retain missing-detection counts. Whole-image metrics can conceal errors in a small mask; masked MAE and local visual review remain essential.

| Condition / mode | Scored | FaceNet ↑ | Gallery ↑ | Mask MAE ↓ | LPIPS ↓ | SSIM ↑ | NIQE ↓ | BRISQUE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| removal/lama | 2/2 | 0.6797 (2/2) | 0.5225 (2/2) | 0.0909 | 0.0285 | 0.9699 | 4.5910 | 23.2714 |
| removal/missing_evidence_high | 2/2 | 0.8101 (2/2) | 0.6711 (2/2) | 0.0795 | 0.0281 | 0.9734 | 4.3841 | 24.5910 |
| removal/missing_evidence_low | 2/2 | 0.6372 (2/2) | 0.5002 (2/2) | 0.1436 | 0.0582 | 0.9579 | 4.5016 | 24.6960 |
| removal/observed | 2/2 | 0.5335 (2/2) | 0.4070 (2/2) | 0.2198 | 0.0870 | 0.9516 | 4.7180 | 25.8591 |
| removal/reference_1024 | 1/2 | 0.8021 (1/2) | 0.5509 (1/2) | 0.0755 | 0.0224 | 0.9737 | 4.5752 | 36.7136 |
| removal/reference_256 | 2/2 | 0.7109 (2/2) | 0.4747 (2/2) | 0.2190 | 0.0653 | 0.9533 | 4.4146 | 24.7409 |
| removal/reference_512 | 2/2 | 0.8101 (2/2) | 0.6711 (2/2) | 0.0795 | 0.0281 | 0.9734 | 4.3841 | 24.5910 |
| removal/resshift | 2/2 | 0.7481 (2/2) | 0.5126 (2/2) | 0.0880 | 0.0298 | 0.9695 | 4.2177 | 24.8727 |
| removal/selection_1024 | 2/2 | 0.7964 (2/2) | 0.6327 (2/2) | 0.0734 | 0.0209 | 0.9753 | 4.2975 | 24.9831 |
| removal/selection_256 | 2/2 | 0.7149 (2/2) | 0.5164 (2/2) | 0.2130 | 0.0637 | 0.9533 | 4.4063 | 24.6789 |
| removal/selection_512 | 2/2 | 0.7751 (2/2) | 0.6336 (2/2) | 0.0846 | 0.0279 | 0.9718 | 4.4069 | 24.5598 |
| mixed/evidence_high | 2/2 | 0.9307 (2/2) | 0.7222 (2/2) | 0.0546 | 0.0261 | 0.9765 | 4.1890 | 22.9670 |
| mixed/evidence_low | 2/2 | 0.9470 (2/2) | 0.7181 (2/2) | 0.0514 | 0.0238 | 0.9765 | 4.1973 | 23.0305 |
| mixed/observed | 2/2 | 0.9301 (2/2) | 0.6705 (2/2) | 0.0478 | 0.0423 | 0.9711 | 4.1554 | 22.0742 |
| mixed/refldm_partial | 2/2 | 0.9437 (2/2) | 0.6857 (2/2) | 0.0451 | 0.0274 | 0.9773 | 4.1461 | 21.8405 |

## Paired checks against unchanged input

| Condition / mode | Identity improved | Mask error improved | Perceptual error improved |
|---|---:|---:|---:|
| removal/lama | 2/2 | 2/2 | 2/2 |
| removal/missing_evidence_high | 2/2 | 2/2 | 2/2 |
| removal/missing_evidence_low | 2/2 | 2/2 | 2/2 |
| removal/reference_1024 | 1/1 | 1/1 | 1/1 |
| removal/reference_256 | 2/2 | 1/2 | 2/2 |
| removal/reference_512 | 2/2 | 2/2 | 2/2 |
| removal/resshift | 2/2 | 2/2 | 2/2 |
| removal/selection_1024 | 2/2 | 2/2 | 2/2 |
| removal/selection_256 | 2/2 | 2/2 | 2/2 |
| removal/selection_512 | 2/2 | 2/2 | 2/2 |
| mixed/evidence_high | 1/2 | 0/2 | 2/2 |
| mixed/evidence_low | 1/2 | 0/2 | 2/2 |
| mixed/refldm_partial | 2/2 | 2/2 | 2/2 |

## Limits and remaining work

This audit covers each offered generation variant with the exact painted mask, not every mask treatment, seed, damage severity or uploaded image. Partial-damage modes use a fixed user-style evidence weight of 128/255; missing regions use zero evidence. These are not calibrated confidence predictions. ReF-LDM is deliberately excluded from complete removal because it does not fill missing regions.

Both identities were excluded from earlier local development, including attempted groups, before image access. Two independent identities do not establish generalization; no significance or publication-ready novelty claim is made. Better average scores do not guarantee correct expression, gaze, or hidden anatomy. Reserved final-test identities remain untouched.

Per-image metrics, missing detections, output hashes, and paired counts are in `unfamiliar_studio_accuracy_v2.json`. Local comparison sheets include clean targets only for evaluation and remain excluded from GitHub. Visual findings are recorded separately after inspection.
