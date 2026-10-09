# Fixed guidance and appearance interaction: all four arms

10 October 2026. Same three development people, four clean real references, seed 17, masks, DDIM 30, FaceID scale 0.8, strength 1, 1024 inference / 512 output and Poisson finish. Guidance is fixed to 5 or 3; appearance branch is absent or fixed at 0.2. The last arm adds only appearance relative to the already tested CFG 3 control. No training or per-person/seed/scale selection.

## Three-person means

| Arm | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---:|---:|---:|---:|---:|---:|
| CFG 5 | 17.019171 | 0.428353 | 0.193782 | 0.171193 | 0.764280 | 0.787376 |
| CFG 3 | 17.894556 | 0.454540 | 0.178454 | 0.152351 | 0.784486 | 0.816156 |
| CFG 5 + appearance | 17.243635 | 0.423499 | 0.180554 | 0.150591 | 0.782868 | 0.791934 |
| CFG 3 + appearance | 17.903762 | 0.441567 | 0.175686 | 0.143247 | 0.784508 | 0.810635 |

## Every person and every arm

| Person | Arm | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---|---:|---:|---:|---:|---:|---:|
| 263 | CFG 5 | 16.413578 | 0.494264 | 0.168389 | 0.206952 | 0.728110 | 0.802763 |
| 263 | CFG 3 | 17.429831 | 0.538117 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | CFG 5 + appearance | 16.234086 | 0.470865 | 0.163522 | 0.198791 | 0.716685 | 0.781189 |
| 263 | CFG 3 + appearance | 17.204263 | 0.510011 | 0.148285 | 0.188544 | 0.729421 | 0.794773 |
| 6681 | CFG 5 | 18.910703 | 0.493393 | 0.147842 | 0.122778 | 0.766272 | 0.786206 |
| 6681 | CFG 3 | 19.859689 | 0.512666 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | CFG 5 + appearance | 19.086401 | 0.486212 | 0.133289 | 0.114821 | 0.791263 | 0.795364 |
| 6681 | CFG 3 + appearance | 19.747170 | 0.495842 | 0.136687 | 0.092935 | 0.856496 | 0.835235 |
| 10149 | CFG 5 | 15.733232 | 0.297403 | 0.265115 | 0.183849 | 0.798458 | 0.773159 |
| 10149 | CFG 3 | 16.394149 | 0.312838 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | CFG 5 + appearance | 16.410418 | 0.313421 | 0.244853 | 0.138160 | 0.840654 | 0.799248 |
| 10149 | CFG 3 + appearance | 16.759855 | 0.318848 | 0.242087 | 0.148261 | 0.767606 | 0.801898 |

## Paired final-arm differences

| Control | Metric | Control mean | Final mean | Delta | Valid pairs |
|---|---|---:|---:|---:|---:|
| CFG 5 | hole_psnr | 17.019171 | 17.903762 | +0.884591 | 3/3 |
| CFG 5 | hole_ssim | 0.428353 | 0.441567 | +0.013214 | 3/3 |
| CFG 5 | hole_lpips | 0.193782 | 0.175686 | -0.018096 | 3/3 |
| CFG 5 | eye_lpips | 0.171193 | 0.143247 | -0.027946 | 3/3 |
| CFG 5 | facenet_cosine | 0.764280 | 0.784508 | +0.020228 | 3/3 |
| CFG 5 | arcface_conditioning_cosine | 0.787376 | 0.810635 | +0.023259 | 3/3 |
| CFG 3 | hole_psnr | 17.894556 | 17.903762 | +0.009206 | 3/3 |
| CFG 3 | hole_ssim | 0.454540 | 0.441567 | -0.012974 | 3/3 |
| CFG 3 | hole_lpips | 0.178454 | 0.175686 | -0.002768 | 3/3 |
| CFG 3 | eye_lpips | 0.152351 | 0.143247 | -0.009104 | 3/3 |
| CFG 3 | facenet_cosine | 0.784486 | 0.784508 | +0.000022 | 3/3 |
| CFG 3 | arcface_conditioning_cosine | 0.816156 | 0.810635 | -0.005521 | 3/3 |

## Numerical decision

- Final arm against CFG 5: PASS on matched mean requirements.
- Final arm against CFG 3: FAIL on matched mean requirements.

Keep all per-person tradeoffs visible. A mean pass is not sufficient for universal deployment. The neutral CFG 3 / appearance-off replay matches the saved CFG 3 raw and final images byte-for-byte. Failed attempts, missing identity measurements and original controls remain available. All completed outputs preserve visible pixels exactly.

All three original/four-arm contact sheets were inspected. The combined arm is not selected: SSIM and ArcFace worsen against CFG 3 alone, and local eye fidelity is unresolved. Three development people and one inference seed cannot establish generalization; SSIM and identity cosines are not accuracy percentages. Whole-image restoration ReF-LDM scores belong to a different task/protocol. See FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md and the separate three-seed guidance-only results.

- Person 263: Appearance at CFG 3 changes iris colour/gaze and does not recover original eyebrow shape; SSIM falls relative to CFG 3 alone.
- Person 6681: Brow/eye structure and colour remain mismatched; added appearance lowers SSIM relative to CFG 3 alone.
- Person 10149: Eyelid, gaze and eyebrow discrepancies persist; FaceNet remains below the CFG 5 control.
