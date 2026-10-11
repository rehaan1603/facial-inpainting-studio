# Dense CLIP geometry v3 â€” feasibility results

11 October 2026. This is a frozen-feature geometry diagnostic, not an inpainting training run or a new reconstruction result.

## Method and scope

The existing local ViT-H/14 image encoder supplies penultimate 16x16 patch features (1280 dimensions). Unlike the compressed adapter tokens, these features retain patch coordinates. No face detector or target eye landmarks are used. Missing target pixels are sanitized before encoding; only visible grid centres away from the hole participate in matching. Global CLIP self-attention still mixes observed context, so these descriptors are not claimed to have purely local receptive fields.

Same eight training people, four original references each, and original masks. No development cases or hidden truth enter matching. Mutual cosine matches use a fixed confidence margin; held-out pairs validate a proper similarity transform. Wrong-person controls use the next training identity's four references. Full [prospective protocol](DENSE_FEATURE_GEOMETRY_PREFLIGHT_V3.md) was saved before execution.

## Validation

Five synthetic unit tests passed: known feature permutation, ambiguity/opposition rejection, transform recovery and order invariance, incoherent geometry rejection, and hole exclusion. The initial real-image GPU probe passed, including exact feature invariance when hidden RGB changed. The encoder has **632,076,800 frozen parameters and zero trainable parameters**. There is no optimizer, backward pass or adapter checkpoint.

## Measured outcomes

All 40 images were encoded (eight sanitized targets and 32 references), plus one target re-encoding for invariance. Execution took 184.20 seconds. Peak allocated VRAM was **1,302,118,912 bytes (1.30 GB)**; reserved peak was 1,442,840,576 bytes. The 3.5 GB cap passed.

**0/32 correct-reference pairs** passed every geometric rule, and **0/8 targets** reached two usable references. Fourteen correct pairs had fewer than 12 mutual matches; 18 had enough matches but failed the held-out/fit/coverage checks. **0/32 wrong-person pairs** passed; all had too few mutual matches. That negative-control result alone is not success because every positive pair also failed.

Post-hoc diagnosis, without changing transforms or gates: identity1281/reference0 had 67 matches and 84% fitting inliers; 16/23 held-out matches were within12px, but the maximum held-out error was454.81px, producing RMS142.5px. Other pairs had widespread errors. Thus some coherent matches coexist with severe mismatches. The32px grid also limits precision; grid quantization alone does not explain errors of hundreds of pixels. This is evidence for correspondence outlier handling, not for raising the acceptance threshold.

## Decision

Do not train on this geometry or replace the application path. Baseline reconstruction metrics remain unchanged; no new SSIM, LPIPS or identity improvement is claimed. A next controlled experiment should address correspondence ambiguity/outliers, while keeping held-out validation and wrong-person controls fixed. A passing visible-region registration still needs validation of missing-eye extrapolation.

[Photo-free evidence and detailed failures](dense_feature_geometry_v3_results.json). Full local feature tensors and receipts remain under `outputs/spatial_reference_correspondence_v3/geometry`; they are not published.
