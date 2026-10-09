# Focused eye reconstruction: lower guidance diagnostic

9 October 2026. Three repeated-development people, eyes only, clean four-reference sets, seed17. Only diffusion guidance changes5 to3. FaceID scale stays0.8;1024 inference,30DDIM steps and Poisson finishing remain fixed. No training, new references or checkpoint changes.

The neutral guidance5 replay precedes the three candidates. Original generator and website defaults remain unchanged.

## Matched means

| Metric | Guidance5 control | Guidance3 candidate | Delta | Valid pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.019171 | 17.894556 | +0.875386 | 3 |
| hole_ssim | 0.428353 | 0.454540 | +0.026187 | 3 |
| hole_lpips | 0.193782 | 0.178454 | -0.015328 | 3 |
| eye_lpips | 0.171193 | 0.152351 | -0.018842 | 3 |
| facenet_cosine | 0.764280 | 0.784486 | +0.020206 | 3 |
| arcface_conditioning_cosine | 0.787376 | 0.816156 | +0.028780 | 3 |

## Every person

| Person | Metric | Control | Candidate | Delta |
|---|---|---:|---:|---:|
| 263 | hole_psnr | 16.413578 | 17.429831 | +1.016253 |
| 263 | hole_ssim | 0.494264 | 0.538117 | +0.043853 |
| 263 | hole_lpips | 0.168389 | 0.144067 | -0.024322 |
| 263 | eye_lpips | 0.206952 | 0.175436 | -0.031516 |
| 263 | facenet_cosine | 0.728110 | 0.750827 | +0.022717 |
| 263 | arcface_conditioning_cosine | 0.802763 | 0.821858 | +0.019095 |
| 6681 | hole_psnr | 18.910703 | 19.859689 | +0.948986 |
| 6681 | hole_ssim | 0.493393 | 0.512666 | +0.019273 |
| 6681 | hole_lpips | 0.147842 | 0.136960 | -0.010883 |
| 6681 | eye_lpips | 0.122778 | 0.118169 | -0.004610 |
| 6681 | facenet_cosine | 0.766272 | 0.840004 | +0.073732 |
| 6681 | arcface_conditioning_cosine | 0.786206 | 0.840881 | +0.054675 |
| 10149 | hole_psnr | 15.733232 | 16.394149 | +0.660917 |
| 10149 | hole_ssim | 0.297403 | 0.312838 | +0.015435 |
| 10149 | hole_lpips | 0.265115 | 0.254336 | -0.010779 |
| 10149 | eye_lpips | 0.183849 | 0.163447 | -0.020401 |
| 10149 | facenet_cosine | 0.798458 | 0.762628 | -0.035830 |
| 10149 | arcface_conditioning_cosine | 0.773159 | 0.785730 | +0.012571 |

## Decision

Combined mean numerical gate: **PASS**. Visual inspection and per-person identity errors remain separate. This report does not automatically select the candidate or change website defaults.

Whole-face similarity is not iris-colour accuracy. ArcFace conditions the generator; FaceNet is separate. LPIPS uses the fixed contextual crops from the existing evaluator. Detection failures remain explicit.

Raw guidance5 controls already show iris-colour/gaze changes before Poisson finishing; boundary blending alone cannot solve the observed generation errors.

Visual comparisons are local in `outputs/focused_eye_guidance_v1/comparisons`. Candidate test files are labelled Experimental in each existing three-person folder.
