# Fixed face-crop identity supervision: v3

3 October 2026. Existing dataset, separate from production and previous experiment outputs.

## Change and controlled comparison

Changed only the training recognition crop: frozen target MTCNN bounds replace full-image resizing. Preserve the v2 193-parameter architecture, initialization seed, 64-update sample/timestep schedule, optimizer, periodic supervision and loss weight 0.5408350358795648. Both target and predicted training image use the same fixed box. Evaluation still independently detects target/output faces. Ground-truth training crops never enter inference.

All 16 training targets have one detected face. Antialiased bilinear tensor resizing matches crop bounds, output size and standardization, but avoids PIL uint8 rounding. Minimum crop embedding agreement with the evaluator on training targets: 0.99996465.

Two crop checks pass: invalid boxes are rejected and gradients remain inside the crop. Backbone and recognition encoder stay frozen.

## Execution and loss diagnostics

64 GPU updates completed in 93.56 seconds; peak allocated tensor memory 8489.4 MiB. This exceeds 8 GiB and does not establish dedicated-VRAM headroom.

Identity supervision ran at 16 steps across 12 of the 16 training identities. Diffusion training still covers all 16. The inherited every-fourth-update schedule does not give equal identity-supervision coverage.

Weighted identity/diffusion gradient norm ratio: 0.256 to 56.541. Opposing gradient directions: 4/16 checks. This diagnoses the risk of one-example weight calibration; it does not by itself prove why a final image improves or worsens.

## Matched results

Four observed development identities, inference seeds 17/29/43, clean and wrong-person-patch references. 24 fresh cropped-candidate images; 72 controls reused only after byte-hash verification. All 96 rows are evaluated under the same frozen metrics. Reused controls are not new generation successes.

| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (n) | Eye crop LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline / clean | 23.860809 | 0.634808 | 0.277250 | 0.914966 | 0.839602 (8/12) | 0.129011 |
| baseline / wrong_patch | 23.846802 | 0.633774 | 0.277689 | 0.905445 | 0.836253 (8/12) | 0.127997 |
| regional_v1 / clean | 23.852219 | 0.634738 | 0.277297 | 0.913288 | 0.842980 (8/12) | 0.128384 |
| regional_v1 / wrong_patch | 23.854192 | 0.633898 | 0.277648 | 0.906632 | 0.839784 (8/12) | 0.128267 |
| identity_full / clean | 23.846130 | 0.634577 | 0.277498 | 0.910484 | 0.837578 (8/12) | 0.127565 |
| identity_full / wrong_patch | 23.861123 | 0.634007 | 0.277913 | 0.904325 | 0.838626 (8/12) | 0.126087 |
| identity_crop / clean | 23.858696 | 0.634656 | 0.277598 | 0.911069 | 0.840483 (8/12) | 0.128583 |
| identity_crop / wrong_patch | 23.854215 | 0.633859 | 0.278305 | 0.903844 | 0.840528 (8/12) | 0.128553 |

Higher PSNR/SSIM/cosine and lower LPIPS are better. FaceNet is training supervision, not an independent metric. Eye crop LPIPS includes context and is not strictly pixel-masked. This remains full-image restoration, not erased-eye completion.

## Decision and next work

Cropped-candidate promotion checks: `{"promote": false, "violations": ["clean/17/ssim", "clean/17/lpips", "clean/17/facenet_cosine", "clean/29/psnr", "clean/29/lpips", "clean/29/facenet_cosine", "clean/29/eye_bbox_lpips", "clean/43/psnr", "clean/43/ssim", "clean/43/facenet_cosine", "wrong_patch/17/lpips", "wrong_patch/17/facenet_cosine", "wrong_patch/17/eye_bbox_lpips", "wrong_patch/29/lpips", "wrong_patch/29/eye_bbox_lpips", "wrong_patch/43/psnr", "wrong_patch/43/ssim", "wrong_patch/43/lpips"], "incomplete_support": ["clean/17/arcface_conditioning_cosine", "clean/29/arcface_conditioning_cosine", "clean/43/arcface_conditioning_cosine", "wrong_patch/17/arcface_conditioning_cosine", "wrong_patch/29/arcface_conditioning_cosine", "wrong_patch/43/arcface_conditioning_cosine"], "lpips_improves_both_conditions": false}`.

Any metric regressions, incomplete supports and seed-dependent results remain visible. No production replacement follows a failing screen. The next objective experiment should balance supervision across all training identities and examine bounded gradient contributions across multiple training examples, separately from any architecture change. Independently suppressing all misleading references remains a subsequent gating ablation.

Only four observed identities and one training seed are available here. Three inference seeds do not establish training-seed robustness. Final held-out data and the separate Human Faces work remain unused. Current numeric tolerance is 1e-6 reproducibility tolerance, not a perceptual non-inferiority margin.

[All measurements and paired statistics](identity_crop_v3_results.json), [CSV](identity_crop_v3_metrics.csv), [provenance](identity_crop_v3_provenance.json). The v2 report is preserved in [previous objective experiment](FINAL_METHOD_REPORT.md).
