# Focused reference reconstruction: three-person development results

9 October 2026. User requested focus on three usable-reference faces and local test packs. Existing identities263,6681,10149 are fixed across every comparison. No training, new dataset, synthetic references or checkpoint replacement.

## What changed

Two sequential isolated comparisons: (1) existing SDXL generation resolution512→1024, holding references/seed17/steps30/scale0.8/strength1/guidance5 fixed; (2) existing Poisson boundary harmonization applied to the identical1024 raw outputs. Resolution also changes latent-noise dimensions and invokes existing VAE tiling, so equal seeds are seed labels, not identical noise tensors. Original512 and1024 raw/hard outputs are retained.

## Matched results

Every row below uses the SAME three identities, seed17 and clean references. PSNR/SSIM concern the actual hole. LPIPS is a contextual hole crop. Both identity metrics have3/3 valid pairs per row; ArcFace is conditioning-related, not independent. These cosine scores are not calibrated accuracy percentages.

| Hole | Method | Hole PSNR ↑ | Hole SSIM ↑ | Hole crop LPIPS ↓ | FaceNet ↑ | ArcFace ↑ |
|---|---|---:|---:|---:|---:|---:|
| eyes | 512 hard | 11.615221 | 0.408938 | 0.459634 | 0.709557 | 0.738128 |
| mouth | 512 hard | 12.831615 | 0.536433 | 0.556723 | 0.871033 | 0.849463 |
| eyes | 1024 hard | 12.037955 | 0.397760 | 0.373688 | 0.739026 | 0.787957 |
| mouth | 1024 hard | 13.349169 | 0.512804 | 0.450581 | 0.900900 | 0.885638 |
| eyes | 1024 blended | 17.019171 | 0.428353 | 0.193782 | 0.764280 | 0.787376 |
| mouth | 1024 blended | 20.662422 | 0.549327 | 0.238383 | 0.912414 | 0.895708 |

## Interpretation

The final tested settings improve all five average metrics relative to 512 hard composition on this three-person clean-reference screen. Resolution alone reduces SSIM; blending removes much of the colour discontinuity. Relative to1024 hard composition, blended eye ArcFace decreases slightly, so this is not a claim that every stage improves every metric. Eye colour, gaze and expression remain imperfect in visual comparisons. Do not promote these observations to population-level accuracy or a novelty claim.

The broader512 run attempted48 cases:36 outputs and12 reference-detection failures, all for identity7400. The1024 screen attempted8 cases:6 outputs and2 analogous failures. None were replaced or silently dropped. Three-face focus was explicitly requested by the user; results on those three must remain labelled development results.

## Reproduce in the local website

Select **Use 1–4 reference photos**, add all four photos, choose **1024 px**, enable **Match colour at edges**, select **Use mask exactly**, and enable **Ignore opaque cover colour**. Use the damaged input and its matching mask from the same test folder. This routes through the existing neutralizing wrapper at strength1. The clean original is for evaluation only and must not be included as a reference.

Test-pack location: `Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09/`.

## Remaining focused work

Keep these cases fixed while testing identity/gaze preservation and robustness to a misleading reference. A visual improvement and higher whole-face cosine do not establish correct eye geometry. Repeat a selected candidate across the previously fixed seeds and corrupted references before changing general website defaults. No global default or model checkpoint has changed in this experiment.

[Audit](MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md) · [SDXL protocol](MASKED_SDXL_PREFLIGHT_V1.md) · [Resolution protocol](MASKED_SDXL_RESOLUTION_PREFLIGHT_V1.md) · [Numerical receipt](focused_reference_v1_results.json)

## Actual website verification

3 recorded website runs; 3 complete; 3 byte-exact matches to the corresponding offline candidate. See JSON for individual jobs.

## Lower reference-strength probe: retained as experimental

Only FaceID scale changed from 0.8 to 0.4 on the same three eye cases. Mean hole PSNR/SSIM/LPIPS and identity improve, but person10149 loses substantial FaceNet similarity. The lower gain is not selected as a universal three-person setting. No per-person scale selection is presented as an automatic method.

| Person | FaceNet at 0.8 | FaceNet at 0.4 | Delta |
|---|---:|---:|---:|
| 263 | 0.728110 | 0.779740 | +0.051630 |
| 6681 | 0.766272 | 0.845577 | +0.079305 |
| 10149 | 0.798458 | 0.735698 | -0.062761 |

## Reference integration check

A read-only inspection found correct normalized face embeddings, conditional/unconditional ordering and all four references reaching attention as 64 projected tokens. No concrete integration defect was found. The reference path supplies identity vectors, not explicit gaze, iris colour or spatial correspondence. The author describes Portrait as not requiring a LoRA; adding one blindly would not address the observed failure. [Author model card](https://huggingface.co/h94/IP-Adapter-FaceID).

The source image is currently reduced to 512 before optional 1024 inference. That limits retained source detail for high-resolution uploads; these frozen tests already use 512 inputs, so changing source preprocessing would require a separate test.
