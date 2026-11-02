# Completed spatial-capacity experiment — reviewed 2 November 2026

Both matched 32-update GPU arms and all21 evaluation records completed. Neutral raw/final replay passed. The candidate fails13 fixed numerical gates. It is not promoted.

| Metric | CFG3 baseline | Fixed spatial control | Learned spatial |
|---|---:|---:|---:|
| hole_psnr | 17.894556 | 17.904439 | 17.906345 |
| hole_ssim | 0.454540 | 0.444849 | 0.444869 |
| hole_lpips | 0.178454 | 0.176619 | 0.176377 |
| eye_lpips | 0.152351 | 0.152006 | 0.151868 |
| facenet_cosine | 0.784486 | 0.782534 | 0.782839 |
| arcface_conditioning_cosine | 0.816156 | 0.804875 | 0.805402 |

## Visual review

All six raw/finished comparison sheets reviewed on2 November2026. Person263 still has a changed gaze, blue/cyan iris appearance and different brow shape;6681 retains overly vivid cyan irises;10149 has different eye opening, gaze and brow structure. Raw outputs already contain these mismatches. No clear anatomical improvement from the added spatial parameters is visible at the displayed full-face and eye-crop scales. Finishing corrects global tone and preserves visible pixels, but does not recover the true missing structure.

## Decision and next work

Stop this recipe under its existing failed gates. Keep CFG3 and all historical checkpoints. Added capacity alone did not solve reference-to-target anatomical correspondence. Repeated three-person results are developmental, not a generalization benchmark. The next separately specified test reduces guidance3 to2 with no training or architecture change; it follows the earlier measured CFG5-to3 gains and requires stronger identity safeguards. If it fails, do not expand or promote it. Larger independent validation, corrupted-reference tests, uncertainty intervals and final manuscript remain outstanding.

Sources: outputs/spatial_capacity_v6/supervision/receipt.json; outputs/spatial_capacity_v6_eval/{summary.json,paired_results.json,report.md}. Full photo-free numerical report: SPATIAL_CAPACITY_V6_NUMERICAL_REPORT.md.
