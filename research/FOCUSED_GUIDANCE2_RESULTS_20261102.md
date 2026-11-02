# CFG2 versus CFG3 — 2 November 2026

Fixed three-person development comparison; seed17, four real references,30DDIM steps,1024 inference/512 export, FaceID0.8 and Poisson finishing. Only guidance3-to2 changes. No training or website promotion. Raw and final neutral CFG3 replay is pixel-exact.

| Metric | CFG3 | CFG2 | Change |
|---|---:|---:|---:|
| psnr | 29.387906 | 29.619483 | +0.231577 |
| ssim | 0.961286 | 0.961307 | +0.000021 |
| lpips | 0.022612 | 0.022335 | -0.000277 |
| hole_psnr | 17.894556 | 18.126134 | +0.231577 |
| hole_ssim | 0.454540 | 0.454104 | -0.000437 |
| hole_lpips | 0.178454 | 0.179539 | +0.001085 |
| eye_lpips | 0.152351 | 0.144884 | -0.007466 |
| facenet_cosine | 0.784486 | 0.805466 | +0.020980 |
| arcface_conditioning_cosine | 0.816156 | 0.807838 | -0.008318 |

## Every person

| Person | Metric | CFG3 | CFG2 | Change |
|---|---|---:|---:|---:|
| 263 | psnr | 29.062505 | 29.164696 | +0.102191 |
| 263 | ssim | 0.968875 | 0.968325 | -0.000551 |
| 263 | lpips | 0.016392 | 0.016522 | +0.000130 |
| 263 | hole_psnr | 17.429831 | 17.532022 | +0.102191 |
| 263 | hole_ssim | 0.538117 | 0.527635 | -0.010482 |
| 263 | hole_lpips | 0.144067 | 0.144333 | +0.000267 |
| 263 | eye_lpips | 0.175436 | 0.181598 | +0.006162 |
| 263 | facenet_cosine | 0.750827 | 0.802069 | +0.051242 |
| 263 | arcface_conditioning_cosine | 0.821858 | 0.816860 | -0.004997 |
| 6681 | psnr | 31.642592 | 32.175521 | +0.532929 |
| 6681 | ssim | 0.967531 | 0.967627 | +0.000096 |
| 6681 | lpips | 0.015800 | 0.014669 | -0.001131 |
| 6681 | hole_psnr | 19.859689 | 20.392618 | +0.532929 |
| 6681 | hole_ssim | 0.512666 | 0.515515 | +0.002849 |
| 6681 | hole_lpips | 0.136960 | 0.137775 | +0.000815 |
| 6681 | eye_lpips | 0.118169 | 0.102251 | -0.015918 |
| 6681 | facenet_cosine | 0.840004 | 0.866102 | +0.026098 |
| 6681 | arcface_conditioning_cosine | 0.840881 | 0.855765 | +0.014884 |
| 10149 | psnr | 27.458621 | 27.518233 | +0.059611 |
| 10149 | ssim | 0.947451 | 0.947969 | +0.000517 |
| 10149 | lpips | 0.035643 | 0.035812 | +0.000169 |
| 10149 | hole_psnr | 16.394149 | 16.453760 | +0.059611 |
| 10149 | hole_ssim | 0.312838 | 0.319162 | +0.006323 |
| 10149 | hole_lpips | 0.254336 | 0.256508 | +0.002172 |
| 10149 | eye_lpips | 0.163447 | 0.150804 | -0.012643 |
| 10149 | facenet_cosine | 0.762628 | 0.748228 | -0.014400 |
| 10149 | arcface_conditioning_cosine | 0.785730 | 0.750890 | -0.034840 |

Numerical gate: FAIL. All six raw/finished comparison sheets reviewed; no promotion.

- 10149/facenet_cosine/identity_guard
- 10149/arcface_conditioning_cosine/identity_guard
- hole_ssim/mean
- hole_lpips/mean
- arcface_conditioning_cosine/mean

All three are repeated development cases. Whole SSIM includes preserved visible pixels; cosine is not accuracy. These results do not estimate unfamiliar-person reconstruction performance. If any gate fails, stop this recipe; no per-person parameter choice or extra seed search.

## Visual review

- 263: CFG2 softens some eye detail but gaze and brow shape still differ from ground truth.
- 6681: CFG2 reduces vivid cyan appearance somewhat; iris shape and gaze remain inaccurate.
- 10149: Eye opening and gaze still differ, with reduced FaceNet and ArcFace versus CFG3.

Stop recipe: five failed numerical gates; do not replicate or promote.
