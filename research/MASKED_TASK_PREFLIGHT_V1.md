# Erased-region diagnostic v1

8 October 2026. No training or production change.

## Fixed question

Does the existing restoration baseline, or its unchanged timestep-scaled identity adapter, reconstruct genuinely missing eyes/mouth when all hidden RGB is removed before encoding?

The candidate is the existing v5 `adapter.pt`, SHA256 `5b6ae23cdd3ba60d42b7530867cd020ae3a97001bb5253c0467fa4e0dfa413ee`. It is not retrained or selected by results from this diagnostic. Backbone/source verification uses the existing loader. This comparison is distinct from all earlier blur/noise restoration tables.

## Cases and controls

- Four existing adapter-holdout identities: 263, 6681, 7400, 10149. No new dataset or final-test identities.
- Existing `benchmark_v2.semantic_mask` rule applied to their current targets at 512 pixels: eyes plus eyebrows with 8-pixel padding; mouth/lips with 12-pixel padding. These are **newly instantiated masks using an existing rule**, not previously evaluated case files. This is necessary because the restoration cohort did not contain actual holes; changing cohorts would confound transfer assessment.
- Eight inputs; replace every hidden RGB value with 127. Annotation-derived masks simulate accurate user-supplied masks. Ground-truth anatomy is not passed as a reference feature or used to choose outputs.
- Verify hidden-pixel invariance by changing underlying target pixels, then confirming identical erased inputs. Recompute compatibility diagnostics from erased input and supplied references; do not reuse richer blurred-target diagnostics.
- Preserve existing legacy coverage feature value 1. Changing its definition would be another intervention. Actual mask-aware feature design remains future work.
- Four separately encoded reference latents, unchanged corruption variants and source hash checks. Clean and wrong-patch reference conditions.
- Two arms × four identities × two mask locations × three seeds (17,29,43) × two conditions = 96 planned outputs. A seed-17 clean screen is generated first. Each arm retains DDIM50/guidance1.5/bfloat16 and original reference encoding.
- Fresh unchanged restoration replay must be byte-exact before masked generation. Save raw outputs as well as exact-visible-pixel hard composites.

## Separate native control

Eight ResShift outputs: same cases, seed17, native256, four steps, no references. Masks max-pooled from512 to preserve hole coverage; input resized then holes erased again. Outputs upsampled and hard-composited at512. This is a **functional mask-conditioned control**, not a matched-resolution ReF-LDM performance claim. Original wrapper provenance is redirected into the new experiment directory; historical provenance files remain untouched.

## Evaluation and decision

Frozen existing metric implementations: whole/region PSNR and SSIM, full/crop LPIPS, FaceNet, ArcFace, eye crop LPIPS and exact visible-pixel preservation. Detector failures retain explicit missing values. Eye-region metrics are unavailable for mouth-only masks and must not be filled. Candidate/control means use jointly valid pairs. All numbers are descriptive development results, with only four identities.

A functioning generator must reconstruct facial structure inside the hole; negligible score changes with a visibly blank region do not pass this requirement. If both ReF-LDM arms retain the blank region, prioritize a genuinely mask-conditioned generation path before adapter-loss expansion. No training or default-backend promotion is authorized by a positive whole-image score alone.

Local immutable inputs/settings: `outputs/masked_task_v1/config.json`, `cases.json`, `preparation_receipt.json`. Results, source hashes and replay receipt remain local. Dataset pixels are not added to GitHub.

Code: [experiment](../scripts/audit_masked_task_v1.py), [scoring](../scripts/score_masked_task_v1.py), [native control](../scripts/masked_native_control_v1.py).
