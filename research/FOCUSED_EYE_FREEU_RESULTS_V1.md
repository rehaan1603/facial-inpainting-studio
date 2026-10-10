# Fixed FreeU decoder experiment on three development people

10 October 2026. Installed Diffusers FreeU with author SDXL tuple b1=1.3, b2=1.4, s1=0.9, s2=0.2. Only decoder feature reweighting changes against CFG3. Existing masks, four references, DDIM30, FaceID0.8, full strength, 1024 model / 512 export and Poisson finishing are unchanged. No training or production changes.

Seeds: [17]. Completed 9/9 records, including 6 preserved controls. Every failed attempt and missing detection remains explicit.

| Metric | Original CFG5 | Current CFG3 | FreeU candidate |
|---|---:|---:|---:|
| hole_psnr | 17.019171 (3/3) | 17.894556 (3/3) | 16.219278 (3/3) |
| hole_ssim | 0.428353 (3/3) | 0.454540 (3/3) | 0.399476 (3/3) |
| hole_lpips | 0.193782 (3/3) | 0.178454 (3/3) | 0.202284 (3/3) |
| eye_lpips | 0.171193 (3/3) | 0.152351 (3/3) | 0.176091 (3/3) |
| facenet_cosine | 0.764280 (3/3) | 0.784486 (3/3) | 0.743002 (3/3) |
| arcface_conditioning_cosine | 0.787376 (3/3) | 0.816156 (3/3) | 0.770318 (3/3) |

## Matched differences against control3

| Metric | Control | Candidate | Delta | Valid pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.894556 | 16.219278 | -1.675279 | 3/3 |
| hole_ssim | 0.454540 | 0.399476 | -0.055064 | 3/3 |
| hole_lpips | 0.178454 | 0.202284 | +0.023830 | 3/3 |
| eye_lpips | 0.152351 | 0.176091 | +0.023741 | 3/3 |
| facenet_cosine | 0.784486 | 0.743002 | -0.041484 | 3/3 |
| arcface_conditioning_cosine | 0.816156 | 0.770318 | -0.045838 | 3/3 |

## Matched differences against control5

| Metric | Control | Candidate | Delta | Valid pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.019171 | 16.219278 | -0.799893 | 3/3 |
| hole_ssim | 0.428353 | 0.399476 | -0.028877 | 3/3 |
| hole_lpips | 0.193782 | 0.202284 | +0.008502 | 3/3 |
| eye_lpips | 0.171193 | 0.176091 | +0.004898 | 3/3 |
| facenet_cosine | 0.764280 | 0.743002 | -0.021278 | 3/3 |
| arcface_conditioning_cosine | 0.787376 | 0.770318 | -0.017058 | 3/3 |

## Every attempted image

| Person | Seed | Arm | Hole SSIM | Hole PSNR | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 263 | 17 | control5 | 0.494264 | 16.413578 | 0.168389 | 0.206952 | 0.728110 | 0.802763 |
| 263 | 17 | control3 | 0.538117 | 17.429831 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | 17 | freeu | 0.448166 | 14.879700 | 0.203402 | 0.221988 | 0.708972 | 0.761478 |
| 6681 | 17 | control5 | 0.493393 | 18.910703 | 0.147842 | 0.122778 | 0.766272 | 0.786206 |
| 6681 | 17 | control3 | 0.512666 | 19.859689 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | 17 | freeu | 0.466362 | 18.719752 | 0.145963 | 0.163552 | 0.792767 | 0.807736 |
| 10149 | 17 | control5 | 0.297403 | 15.733232 | 0.265115 | 0.183849 | 0.798458 | 0.773159 |
| 10149 | 17 | control3 | 0.312838 | 16.394149 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | 17 | freeu | 0.283901 | 15.058380 | 0.257488 | 0.142734 | 0.727268 | 0.741740 |

## Decision and scope

Combined numerical gate: FAIL. Per-person identity guard: FAIL.
The fixed screen does not authorize website promotion. Visual review is recorded separately in visual_review.json. If the screen fails, seeds29/43 are not spent on this recipe. No best-person, best-seed or parameter selection is used.

Hole SSIM uses the unchanged Gaussian RGB map and fully contained11x11 windows. Whole-image SSIM is dominated by unchanged visible pixels; it is not missing-eye accuracy. Contextual LPIPS is a bounding-box crop, not a pixel-masked perceptual measure. Small anatomical-eye masks may have very few SSIM windows. FaceNet is independent of generation conditioning; ArcFace uses the conditioning recognition model family. This development test does not measure unseen-client accuracy.

FreeU adds no local reference RGB correspondence. Eyelid, iris, gaze and eyebrow fidelity still require visual inspection. CPU offload allows the frozen decoder to run on the laptop GPU; actual runtime and peak memory are retained, with no overclocking.

Sources: [FreeU paper](https://arxiv.org/abs/2309.11497), [author SDXL recipe](https://github.com/ChenyangSi/FreeU#sdxl). See [audit and frozen preflight](FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md).

## Completed visual review

- Person 263: FreeU exaggerates dark eyebrows and eyeliner; saturated frontal irises and gaze remain wrong.
- Person 6681: Eye colour and contrast become exaggerated, with small eyebrows and incorrect gaze.
- Person 10149: Heavier dark eye/brow contrast and iris/gaze mismatch persist relative to both original and CFG3.

Rejected fixed recipe; no replication or website integration.
