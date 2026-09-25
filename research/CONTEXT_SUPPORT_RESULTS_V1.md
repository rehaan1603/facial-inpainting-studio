# Context-guided mask support — results, 25 September 2026

**Decision: Gate failed; do not promote or expand this mechanism or train an adapter on its premise.**

Implemented and tested after freezing the protocol/source signature. Four previously observed removal cases, two seeds, 104 generator calls and 80 comparison rows. Generator and metric definitions remain frozen; inference never reads clean targets or withheld galleries. This is a single-image mechanism screen, not fresh-identity or pretraining-disjoint validation.

Generation coverage: 104/104. Scoring coverage: 80/80. Exact original-mask context preservation: True. Face detection coverage complete: False. No calibration abstentions: True.

Coverage clarification after receipt audit: the false detection flag includes damaged-input controls. All 72 reconstructed rows have successful FaceNet and ArcFace detections. The observed FaceNet mean below uses three identities (ArcFace uses two); every reconstructed arm uses all four. Primary candidate-versus-control contrasts have complete four-identity coverage. See `CONTEXT_SUPPORT_REVIEW_V1.md` for the six damaged-input detection failures and all eight visual reviews.

| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| observed | 0.480200 | 0.183254 | 0.197037 | 0.098775 |
| fixed_r0 | 0.741489 | 0.423009 | 0.062665 | 0.025402 |
| fixed_r2 | 0.726454 | 0.421691 | 0.069612 | 0.026903 |
| fixed_r4 | 0.667186 | 0.386991 | 0.073097 | 0.028558 |
| fixed_r8 | 0.580302 | 0.351568 | 0.080181 | 0.031072 |
| context_selected | 0.611119 | 0.355024 | 0.076645 | 0.029618 |
| random_support | 0.696791 | 0.426119 | 0.072504 | 0.028122 |
| support_mean4 | 0.677294 | 0.394088 | 0.066404 | 0.029478 |
| support_mean5 | 0.694328 | 0.393158 | 0.065967 | 0.029931 |
| seed_mean5 | 0.692329 | 0.375228 | 0.058916 | 0.029259 |

Each mean first averages both seeds within each identity. Detection/metric coverage, ArcFace, NIQE, BRISQUE, SSIM and PSNR remain in the numerical report. The support/seed five-call controls match the proposed policy cost; the four-call support average is a cheaper control.

| Prespecified contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |
|---|---:|---:|---|---:|
| context_selected_minus_fixed_r0/facenet_cosine | 4 | -0.130371 | [-0.23931202664971352, -0.029869444668293] | 1.000000 |
| context_selected_minus_fixed_r0/hole_mae | 4 | 0.013981 | [0.005021702731669464, 0.02327962620003374] | 1.000000 |
| context_selected_minus_fixed_r4/facenet_cosine | 4 | -0.056068 | [-0.11840447410941124, 0.031657829880714417] | 1.000000 |
| context_selected_minus_fixed_r4/hole_mae | 4 | 0.003548 | [1.2745448283793537e-05, 0.007627656132690914] | 1.000000 |
| context_selected_minus_support_mean5/facenet_cosine | 4 | -0.083209 | [-0.13162966817617416, -0.01622946560382843] | 1.000000 |
| context_selected_minus_support_mean5/hole_mae | 4 | 0.010678 | [0.006755606832583343, 0.015028411761138172] | 1.000000 |
| context_selected_minus_seed_mean5/facenet_cosine | 4 | -0.081211 | [-0.1741298995912075, -0.00604681670665741] | 1.000000 |
| context_selected_minus_seed_mean5/hole_mae | 4 | 0.017729 | [0.008618456310805378, 0.028534850871702447] | 1.000000 |

The statistical unit is identity, not seed or output image. Four identities make the intervals unstable and the minimum two-sided exact p-value 0.125. Eight prespecified contrasts use Holm correction. These tests cannot establish publication-level superiority.

| Case | Seed | Chosen radius | Probe error vs hidden-hole MAE Spearman |
|---|---:|---:|---:|
| 1306_removal | 17 | 8 | -0.800000 |
| 1306_removal | 29 | 8 | -0.400000 |
| 2790_removal | 17 | 2 | -0.400000 |
| 2790_removal | 29 | 2 | 0.400000 |
| 1043_removal | 17 | 4 | 0.400000 |
| 1043_removal | 29 | 8 | -0.800000 |
| 787_removal | 17 | 4 | 0.200000 |
| 787_removal | 29 | 8 | -0.800000 |

The correlation is a post-selection diagnostic across four radii within each case/seed, not a calibrated probability, independent trial count or tuning signal. Selection receipts were locked before the final bank was generated. Small visible patches may not predict fidelity of missing facial structures.

Measured summed generator-call time: 27.7 seconds; median call: 0.27 seconds. Model initialization, scoring and file/report work are excluded. A deployment of the policy requires four calibration calls and one final call, not the entire experimental bank.

Local visual sheets contain every fixed-radius arm, selected result and both equal-budget controls for every case/seed. Images remain local. Novelty is unproven: Noise2Self, internal-image inpainting adaptation and mask perturbation have substantial overlap, described in `CONTEXT_SUPPORT_METHOD.md`. No website default, previous experiment, learned adapter or reserved-final identity was changed.
