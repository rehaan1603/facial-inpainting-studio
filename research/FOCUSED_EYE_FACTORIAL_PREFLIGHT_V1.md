# Fixed guidance / appearance interaction

10 October 2026. Freeze before the final three calls in a two-by-two inference comparison. The prior isolated arms are already complete: CFG 5 / no appearance, CFG 3 / no appearance, CFG 5 / appearance 0.2. No additional weight search or per-person settings.

Complete the remaining arm: **CFG 3 / appearance 0.2**. This adds ONE fixed appearance branch relative to the existing CFG 3 candidate. Retain persons 263, 6681 and 10149, four real clean references in unchanged order, seed 17, DDIM 30, FaceID scale 0.8, strength 1, model 1024 / output 512, the same masks and Poisson finish. Reuse the hash-verified cached reference features from the appearance-only experiment. No new encoder weights, training, target-derived descriptions, target pixels in generation or reference selection.

Rationale: lower guidance improves all six means but loses person-10149 identity; appearance 0.2 at CFG 5 improves that person's six measures but reduces aggregate SSIM. Testing their interaction is justified; an improvement is not assumed. The values 3 and 0.2 were fixed in earlier preflights, not retuned for different people.

Use nested experimental wrappers around the unchanged generator. First replay person 263 at CFG 3 with appearance switched OFF. Require byte-exact raw and finished agreement with the existing CFG 3 output before candidates. Save all source/feature/input/reference hashes, scales, shapes, failures, raw/hard/final images, timing and memory in a fresh directory. Preserve prior experiments, checkpoints and website defaults.

Use the unchanged scorer. Report all four arms against the original CFG 5 control AND the three-person CFG 3 control, with coverage and per-person regressions. Require improved hole SSIM and no deterioration in PSNR, hole/eye LPIPS, FaceNet or ArcFace; per-person failures and visual colour/gaze/brow errors remain explicit. A small numerical pass cannot establish exact missing-eye recovery or promote a universal default. Broader fixed-seed and misleading-reference checks follow only a promising result.
