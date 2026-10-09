# Registered reference eye/brow texture: rejected prototype

10 October 2026. Three development people, seed 17 and the unchanged CFG 3 generated scaffold. Four real reference photos; the evaluation original is never a generation input. Fixed visible-anchor alignment, open-eye threshold, residual threshold, bounded colour offset and feathered transfer are specified in FOCUSED_EYE_TEXTURE_PREFLIGHT_V1.md. No training or new diffusion generation occurred. The files labelled raw contain the post-transfer image.

## Matched three-person means

| Metric | CFG 3 control | Texture transfer | Delta | Valid pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.894556 | 16.045742 | -1.848815 | 3/3 |
| hole_ssim | 0.454540 | 0.407623 | -0.046918 | 3/3 |
| hole_lpips | 0.178454 | 0.230474 | +0.052020 | 3/3 |
| eye_lpips | 0.152351 | 0.215844 | +0.063494 | 3/3 |
| facenet_cosine | 0.784486 | 0.761451 | -0.023035 | 3/3 |
| arcface_conditioning_cosine | 0.816156 | 0.721999 | -0.094157 | 3/3 |

## Every person

| Person | Metric | CFG 3 control | Transfer | Delta |
|---|---|---:|---:|---:|
| 263 | hole_psnr | 17.429831 | 16.767164 | -0.662668 |
| 263 | hole_ssim | 0.538117 | 0.505528 | -0.032590 |
| 263 | hole_lpips | 0.144067 | 0.201712 | +0.057645 |
| 263 | eye_lpips | 0.175436 | 0.271572 | +0.096136 |
| 263 | facenet_cosine | 0.750827 | 0.795458 | +0.044632 |
| 263 | arcface_conditioning_cosine | 0.821858 | 0.773287 | -0.048570 |
| 6681 | hole_psnr | 19.859689 | 16.152757 | -3.706933 |
| 6681 | hole_ssim | 0.512666 | 0.415394 | -0.097272 |
| 6681 | hole_lpips | 0.136960 | 0.195345 | +0.058385 |
| 6681 | eye_lpips | 0.118169 | 0.153883 | +0.035715 |
| 6681 | facenet_cosine | 0.840004 | 0.767850 | -0.072153 |
| 6681 | arcface_conditioning_cosine | 0.840881 | 0.717438 | -0.123443 |
| 10149 | hole_psnr | 16.394149 | 15.217305 | -1.176844 |
| 10149 | hole_ssim | 0.312838 | 0.301947 | -0.010891 |
| 10149 | hole_lpips | 0.254336 | 0.294364 | +0.040028 |
| 10149 | eye_lpips | 0.163447 | 0.222078 | +0.058630 |
| 10149 | facenet_cosine | 0.762628 | 0.721044 | -0.041584 |
| 10149 | arcface_conditioning_cosine | 0.785730 | 0.675273 | -0.110457 |

## Decision and explanation

Rejected; not added to the website. All six mean measures worsen against CFG 3. All three comparisons were inspected and contain local alignment/appearance errors. A good fit to visible global anchors does not establish correspondence for the hidden eye or eyebrow. An open-eye test also does not verify pose, expression, makeup or local occlusion compatibility.

- Person 263: Registered detail produces misaligned/double eyebrow structure and mismatched gaze.
- Person 6681: Transferred eye detail is clipped or displaced; local eye geometry does not match the target.
- Person 10149: Transferred eyelid expression and eye/brow positioning remain inconsistent with the original.

Zero-transfer bitmap equivalence, exact regeneration from saved geometry and exact visible-pixel preservation pass for all three. FaceNet and ArcFace are available for all three; no failed measurement was omitted. The original checkpoints, inputs, references and raw score files remain unchanged.

Next spatial work must establish local pose/expression correspondence and confidence before transferring reference detail, with a safe fallback. Keep the optional CFG 3 comparison available while this is investigated; do not claim hidden iris colour or gaze has been recovered exactly. Corrupted-reference and independent-client evaluation remain pending. SSIM and identity cosine are not accuracy percentages.
