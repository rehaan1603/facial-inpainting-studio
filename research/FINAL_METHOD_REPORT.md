# Facial Inpainting Studio: method investigation report

Updated 2 October 2026. Current research checkpoint, not a finalized successful method. Existing dataset only. All prior evidence and the local application remain preserved.

## 1. Research question

Can a compact target-reference adapter improve perceptual restoration and corrupted-reference robustness without reducing identity or eye-region fidelity?

## 2. Baseline

Frozen local ReF-LDM, 512 RGB, four references, 50 DDIM steps, guidance 1.5, bfloat16, uniform_trailing. Individually encoded references differ from author benchmark RGB concatenation. Compare identical local inputs/settings, not incompatible published scores.

## 3. Proposed method

Retain the 193-parameter v1 compatibility architecture. Compare a new balanced diffusion-only control with an otherwise identical identity-supervised arm. Both train for 64 updates on the same 16 identities. This experiment isolates the added objective within the new pair; the old 128-update adapter has a different training budget.

## 4. Architecture

Degraded target and reference VQ features produce regional cosine, magnitude, consensus and quality/pose descriptors. A 10-to-16-to-1 MLP generates relative reference attention log-priors in 16 frozen attention layers. Five fixed regions include both eyes, nose, mouth and context. Neutral final-layer initialization reproduces the baseline.

## 5. Mathematical formulation

`b(r,i) = log_softmax_i(MLP(features(target,r,reference_i))) + log(N)`

`attention = softmax(Q K^T / sqrt(d) + b) V`

Reference priors affect the conditional branch, with zero self-attention prior. Relative normalization cannot independently suppress every reference; independent reliability attenuation remains a separate proposed ablation.

## 6. Training objective

`Ldiff = 0.25 * mean(epsilon_error^2) + 0.75 * mean(face_rectangle_epsilon_error^2)`

`L = Ldiff + lambda_id * (1 - cosine(FaceNet(decoded_x0), FaceNet(target)))`

Identity loss runs every fourth update. The frozen FaceNet encoder receives a full aligned image resized to 160, a surrogate that differs from MTCNN evaluation cropping. The identity weight is the first supervised-step diffusion/identity gradient-norm ratio, capped at 10. No development scores select the weight or checkpoint. Backbone and encoder stay frozen. This is not full implementation of consistency, regularization, perceptual and all regional objectives.

| Arm | Updates | Identity weight | Training seconds | Peak allocated MiB |
|---|---:|---:|---:|---:|
| diffusion | 64 | None | 23.61 | 3835.0 |
| identity | 64 | 0.5408350358795648 | 71.42 | 8491.2 |

The identity run exceeds 8 GiB in allocated tensors. It completed on this laptop, but this does not establish comfortable dedicated-VRAM headroom or distinguish shared-memory spillover.

## 7. Experimental protocol

Four observed development identities: 263, 6681, 7400, 10149. Three inference seeds: 17, 29, 43. Two reference conditions: clean and central wrong-person patch. Four methods give 96 generations. Training seed is 20261002; corruption/timestep assignments are independently shuffled. No reserved final identities or new Human Faces dataset used. This task is whole-image blur/noise restoration, not erased-region inpainting.

## 8. Quantitative results

| Method / reference condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (valid n) | Eye crop LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline / clean | 23.860809 | 0.634808 | 0.277250 | 0.914966 | 0.839602 (8/12) | 0.129011 |
| baseline / wrong_patch | 23.846802 | 0.633774 | 0.277689 | 0.905445 | 0.836253 (8/12) | 0.127997 |
| regional_v1 / clean | 23.852219 | 0.634738 | 0.277297 | 0.913288 | 0.842980 (8/12) | 0.128384 |
| regional_v1 / wrong_patch | 23.854192 | 0.633898 | 0.277648 | 0.906632 | 0.839784 (8/12) | 0.128267 |
| diffusion / clean | 23.853310 | 0.634693 | 0.277480 | 0.912959 | 0.838166 (8/12) | 0.129529 |
| diffusion / wrong_patch | 23.845118 | 0.633884 | 0.277700 | 0.897618 | 0.839578 (8/12) | 0.129158 |
| identity / clean | 23.846130 | 0.634577 | 0.277498 | 0.910484 | 0.837578 (8/12) | 0.127565 |
| identity / wrong_patch | 23.861123 | 0.634007 | 0.277913 | 0.904325 | 0.838626 (8/12) | 0.126087 |

These are three-seed means. They must not be compared to the earlier seed-17-only baseline as a before/after effect. All whole-image and FaceNet entries have 12 valid measurements per row; ArcFace has 8.

## 9. Regional results

Identity supervision improves average eye bounding-box LPIPS in both conditions relative to the baseline, but not at every seed. Bounding-box LPIPS includes ten-pixel context and is not strictly pixel-masked LPIPS. Official parsing masks are evaluation-only. Whole-image damage means masked metrics equal full-image metrics; the visible region is empty. Nose, mouth, left/right eye and other-region values remain in JSON.

## 10. Identity results

FaceNet averages decrease relative to baseline in both conditions. With a misleading patch, adding identity loss recovers some of the diffusion-only control drop, but does not reach the baseline. FaceNet is supervised in this arm and therefore is not an independent evaluator. ArcFace does not train the adapter; its incomplete support remains explicit.

## 11. Corruption robustness

Training includes clean, blur, noise and wrong-person patch. This new screen evaluates clean and wrong-person patch only. Previous v1 eight-condition results remain separate. Eye/mouth swaps, irrelevant and mixed references, and one bad among four good references still require a frozen protocol. The latter has five total references and needs a five-reference baseline.

## 12. Ablations

Completed: original local backbone; previous 193-parameter regional adapter; new balanced diffusion-only control; matched identity-supervised arm. The earlier 545-parameter adapter is not rerun in this three-seed panel. Independent reliability gating, global/regional variants, consistency, and the full requested factorial ablations remain to be tested.

## 13. Failures and rejection

96/96 generations and metric rows completed. No candidate meets all per-seed safeguards. None replaces the website baseline. ArcFace target detection fails for identity 7400; seed-17 outputs for 10149 also lack ArcFace detections. These failures already occur in archived baseline scores, not a new evaluator inconsistency. Every new-panel FaceNet output and target detects one face.

## 14. Statistical analysis

JSON reports mean, SD, median and counts by condition/seed/method, plus paired identity deltas, improved/worsened counts and 2,000 identity-cluster bootstrap intervals. Four identity units give exploratory intervals only; no statistical-significance claim. Three inference seeds do not substitute for three training seeds. The 1e-6 tolerance is numerical reproducibility tolerance, not a justified clinical/perceptual non-inferiority margin.

## 15. Limitations

Small reused development cohort; full-image training crop differs from recognition evaluation; one-step decoded x0 supervision differs from final multi-step output; fixed anatomical rectangles; limited corruption coverage; constrained VRAM; unresolved pretraining overlap; no unknown-person generalization evidence.

## 16. Prior-art comparison

Target queries and reference-region selection have prior art. Adding a frozen identity loss also has prior art. Current results do not support a new advantageous mechanism. Published context remains in [paper comparison](PAPER_VS_PROJECT_COMPARISON_20261002.md); local matched evidence is the primary comparison.

## 17. Remaining work and next experiment

Before larger training, align the training recognition crop to fixed training-face detections and compare the same two arms with an image/identity objective closer to final output. Audit gradients across multiple training examples instead of trusting one calibration example. Keep inference evaluators frozen. Independently rejecting unreliable references is the next architectural ablation after the objective test. Do not expand the final evaluation or repeat hyperparameter tuning on it. Larger 20–50-person development, multiple targets, other corruptions, true external portraits and final testing remain pending.

## 18. What the evidence supports

Successful engineering: exact metric reproduction, differentiable frozen identity supervision, matched GPU training, 96 multi-seed generations, retained failures and explicit paired evaluation. The tested identity objective improves average eye crop LPIPS but fails the combined identity/perceptual requirement. The strongest deployable candidate remains the original local ReF-LDM baseline. Work remains in progress; a successful novelty hypothesis is not established by this experiment.

Files: [audit](IDENTITY_COMPATIBILITY_AUDIT_V2.md), [all metrics and statistics](identity_compatibility_v2_results.json), [CSV](identity_compatibility_v2_metrics.csv). Training configs/checkpoints/traces remain under `outputs/identity_compatibility_v2`. No dataset images or weights are published.
