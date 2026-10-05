# Expanded reference-encoding diagnostic

5 October 2026. Protocol recorded before generation.

## Question and scope

Does concatenating reference RGB images before VQ encoding, instead of concatenating independently encoded latents, improve the available official-input development cases? This isolates encoding, not a new adapter or a reproduced paper benchmark.

Select every official metadata row whose only flag is `pilot_development_overlap`, with matching reference identity labels. The current audit yields five targets: 11491, 19122, 24589, 27308 and 27474. Four belong to identity 6681 and one to identity 263. Both identities have already been observed in project development. The 24589 target has one reference and is an implementation-equivalence control, not evidence of a multi-reference gain.

## Frozen comparison

- Both encodings use the same frozen EMA backbone, official 512-pixel degraded input, first up-to-five references in official CSV order, 100 DDIM uniform-trailing steps, guidance 1.5, float32 and author torchvision output conversion.
- Inference seeds: 17, 29 and 43. Generate both arms for every target/seed: 30 images, 15 pairs, only two identity units.
- Preserve the same starting noise in each pair. Clear reference caches between runs. No reference substitution, synthetic references, training, checkpoint selection or altered detection thresholds.
- Replay target 11491/seed17 and verify both existing smoke image hashes. Require byte-identical results for the one-reference control. Retain failures rather than substituting easier cases.
- Preserve config, input/reference/output hashes and per-run receipts. Keep dataset images local.

## Evaluation and decision

Use the existing frozen local PSNR, RGB SSIM, AlexNet LPIPS, FaceNet and ArcFace implementations. Retain detector failures and use only valid matched pairs for identity contrasts. Score degraded inputs as an additional control. Eye-region LPIPS requires verified geometry between the official HQ target and local HQ component annotations; mark it unavailable if alignment cannot be verified.

Report all per-target/seed results, separate the single-reference control from multi-reference comparisons, and aggregate targets/seeds within identities before discussing identity replication. Do not infer statistical population confidence from two identities or count seeds as independent people. Record memory and runtime. No production change follows automatically: broader matched local corruption checks remain necessary even if this diagnostic improves.

No reserved-final identity or additional Human Faces dataset is used. No large training is authorized by passing this diagnostic.
