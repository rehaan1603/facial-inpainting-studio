# Hole-local reference appearance: fixed three-person screen

10 October 2026. Same inputs, four references, cached appearance features, CFG3, DDIM30, identity0.8, appearance0.2, 1024 inference / 512 export and Poisson finishing. Only the appearance query routing changes versus the global-appearance control. The full joint-reference attention is retained; no native per-reference mask path, RGB transfer, training or website change.

Completed 12/12 records, including 9 hash-verified reused controls. The two raw/finished full-mask and appearance-off replays are additional verification calls.

| Metric | Original CFG5 | Current CFG3 | Global appearance | Local appearance |
|---|---:|---:|---:|---:|
| hole_psnr | 17.019171 (3/3) | 17.894556 (3/3) | 17.903762 (3/3) | 17.904116 (3/3) |
| hole_ssim | 0.428353 (3/3) | 0.454540 (3/3) | 0.441567 (3/3) | 0.444884 (3/3) |
| hole_lpips | 0.193782 (3/3) | 0.178454 (3/3) | 0.175686 (3/3) | 0.176645 (3/3) |
| eye_lpips | 0.171193 (3/3) | 0.152351 (3/3) | 0.143247 (3/3) | 0.152241 (3/3) |
| facenet_cosine | 0.764280 (3/3) | 0.784486 (3/3) | 0.784508 (3/3) | 0.782640 (3/3) |
| arcface_conditioning_cosine | 0.787376 (3/3) | 0.816156 (3/3) | 0.810635 (3/3) | 0.805353 (3/3) |

## Matched differences against Current CFG3

| Metric | Control | Local candidate | Delta | Pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.894556 | 17.904116 | +0.009560 | 3/3 |
| hole_ssim | 0.454540 | 0.444884 | -0.009656 | 3/3 |
| hole_lpips | 0.178454 | 0.176645 | -0.001809 | 3/3 |
| eye_lpips | 0.152351 | 0.152241 | -0.000110 | 3/3 |
| facenet_cosine | 0.784486 | 0.782640 | -0.001846 | 3/3 |
| arcface_conditioning_cosine | 0.816156 | 0.805353 | -0.010803 | 3/3 |

## Matched differences against Global appearance

| Metric | Control | Local candidate | Delta | Pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.903762 | 17.904116 | +0.000354 | 3/3 |
| hole_ssim | 0.441567 | 0.444884 | +0.003317 | 3/3 |
| hole_lpips | 0.175686 | 0.176645 | +0.000959 | 3/3 |
| eye_lpips | 0.143247 | 0.152241 | +0.008994 | 3/3 |
| facenet_cosine | 0.784508 | 0.782640 | -0.001868 | 3/3 |
| arcface_conditioning_cosine | 0.810635 | 0.805353 | -0.005282 | 3/3 |

## Matched differences against Original CFG5

| Metric | Control | Local candidate | Delta | Pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.019171 | 17.904116 | +0.884945 | 3/3 |
| hole_ssim | 0.428353 | 0.444884 | +0.016531 | 3/3 |
| hole_lpips | 0.193782 | 0.176645 | -0.017137 | 3/3 |
| eye_lpips | 0.171193 | 0.152241 | -0.018952 | 3/3 |
| facenet_cosine | 0.764280 | 0.782640 | +0.018360 | 3/3 |
| arcface_conditioning_cosine | 0.787376 | 0.805353 | +0.017978 | 3/3 |

## Every person

| Person | Arm | Hole SSIM | PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---|---:|---:|---:|---:|---:|---:|
| 263 | Original CFG5 | 0.494264 | 16.413578 | 0.168389 | 0.206952 | 0.728110 | 0.802763 |
| 263 | Current CFG3 | 0.538117 | 17.429831 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | Global appearance | 0.510011 | 17.204263 | 0.148285 | 0.188544 | 0.729421 | 0.794773 |
| 263 | Local appearance | 0.503064 | 17.073415 | 0.153591 | 0.192277 | 0.732108 | 0.783381 |
| 6681 | Original CFG5 | 0.493393 | 18.910703 | 0.147842 | 0.122778 | 0.766272 | 0.786206 |
| 6681 | Current CFG3 | 0.512666 | 19.859689 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | Global appearance | 0.495842 | 19.747170 | 0.136687 | 0.092935 | 0.856496 | 0.835235 |
| 6681 | Local appearance | 0.516028 | 20.072223 | 0.131452 | 0.108263 | 0.853041 | 0.843804 |
| 10149 | Original CFG5 | 0.297403 | 15.733232 | 0.265115 | 0.183849 | 0.798458 | 0.773159 |
| 10149 | Current CFG3 | 0.312838 | 16.394149 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | Global appearance | 0.318848 | 16.759855 | 0.242087 | 0.148261 | 0.767606 | 0.801898 |
| 10149 | Local appearance | 0.315560 | 16.566711 | 0.244893 | 0.156183 | 0.762770 | 0.788875 |

## Decision and limits

Numerical replication gate: FAIL.
The report reuses the existing strict non-worsening gate (zero tolerance), whereas the frozen preflight allows 1e-6 numerical tolerance. All failed mean differences exceed 1e-6; either rule rejects this recipe. The protocol and results have not been adjusted.

The decision also requires visual review. A numerical failure stops replication of this fixed recipe and prevents website promotion. Every failed attempt and per-person regression remains in the source record; no best-person or parameter selection follows.

The unchanged scorer measures hole SSIM over fully contained11x11 windows and LPIPS on contextual crops. Whole-image SSIM is dominated by preserved pixels. FaceNet is separate from generation conditioning; ArcFace belongs to the conditioning family. These three development people and seed17 are not an unknown-client accuracy estimate.

Routing is local at each attention operation, but subsequent UNet mixing can propagate its influence. It does not provide spatial correspondence between reference eyes and target eyes. Original visible RGB is preserved by final compositing, with raw images retained separately. No checkpoints, dataset or evaluation settings changed.

See [fixed preflight](FOCUSED_EYE_LOCAL_APPEARANCE_PREFLIGHT_V1.md). Joint-reference processor calculations use the installed Diffusers source recorded in config.json. This differs from [native per-reference IP-Adapter masks](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter#masking).

## Completed visual review

- Person 263: Hole-local appearance still changes eyebrow shape and iris/gaze; numerical SSIM and both identity measures drop versus CFG3.
- Person 6681: Hole SSIM and identity improve versus CFG3, but irises/gaze and brow texture remain mismatched.
- Person 10149: Slight hole SSIM/identity gains versus CFG3 do not recover original eyelid expression, gaze or iris texture.

Combined gate failed; no replication, per-person oracle routing or website integration.
