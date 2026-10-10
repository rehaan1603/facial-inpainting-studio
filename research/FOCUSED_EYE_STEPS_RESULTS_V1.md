# Fixed50-step erased-eye reconstruction check

10 October 2026. Three fixed development people, seed17. Only DDIM30 to50 changes against CFG3; references, masks, FaceID0.8, strength1, 1024/512, precision and Poisson finishing stay fixed. No FreeU, appearance branch, training or website change.

Completed 9/9 image records, including six reused controls; one additional byte-exact replay.

| Metric | CFG5 /30 | CFG3 /30 | CFG3 /50 |
|---|---:|---:|---:|
| hole_psnr | 17.019171 (3/3) | 17.894556 (3/3) | 17.743326 (3/3) |
| hole_ssim | 0.428353 (3/3) | 0.454540 (3/3) | 0.448495 (3/3) |
| hole_lpips | 0.193782 (3/3) | 0.178454 (3/3) | 0.184688 (3/3) |
| eye_lpips | 0.171193 (3/3) | 0.152351 (3/3) | 0.162697 (3/3) |
| facenet_cosine | 0.764280 (3/3) | 0.784486 (3/3) | 0.784870 (3/3) |
| arcface_conditioning_cosine | 0.787376 (3/3) | 0.816156 (3/3) | 0.812475 (3/3) |

## Every person

| Person | Arm | Hole SSIM | PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---|---:|---:|---:|---:|---:|---:|
| 263 | control5 | 0.494264 | 16.413578 | 0.168389 | 0.206952 | 0.728110 | 0.802763 |
| 263 | control3 | 0.538117 | 17.429831 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | steps50 | 0.528320 | 17.249747 | 0.152101 | 0.186600 | 0.737147 | 0.818645 |
| 6681 | control5 | 0.493393 | 18.910703 | 0.147842 | 0.122778 | 0.766272 | 0.786206 |
| 6681 | control3 | 0.512666 | 19.859689 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | steps50 | 0.506488 | 19.628436 | 0.144069 | 0.132962 | 0.840603 | 0.830989 |
| 10149 | control5 | 0.297403 | 15.733232 | 0.265115 | 0.183849 | 0.798458 | 0.773159 |
| 10149 | control3 | 0.312838 | 16.394149 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | steps50 | 0.310679 | 16.351795 | 0.257895 | 0.168530 | 0.776861 | 0.787791 |

## Decision

Numerical replication gate: FAIL.
Visual inspection is recorded separately. A failed gate stops this fixed recipe. Keep every negative result; do not choose50 steps for a favored person or mix the new seed17 average with the previous three-seed average. The previous identity regression for10149 remains explicit.

The frozen hole SSIM protocol and contextual crop LPIPS remain unchanged. FaceNet is separate from generation conditioning; ArcFace is conditioning-family similarity. Whole-image SSIM and cosine values are not reconstruction accuracy percentages. More steps change the DDIM timestep grid and add computation; they do not add missing reference spatial information. These three people have repeatedly been used for development.

[Frozen protocol](FOCUSED_EYE_STEPS_PREFLIGHT_V1.md). Raw and finished images, timings, hashes and every attempted result are retained locally.

## Completed visual review

- Person 263: More denoising retains frontal gaze and unnatural iris contrast; eyebrow size/shape differs and SSIM falls.
- Person 6681: No clear correction of iris/gaze or brow shape; hole SSIM, PSNR and contextual LPIPS worsen.
- Person 10149: FaceNet improves versus CFG3 but iris/gaze/expression remain wrong; SSIM and perceptual metrics worsen.

Rejected step-budget recipe; no replication or website change.
