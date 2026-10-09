# Fixed reference appearance branch: three-person results

Same frozen three-person erased-eye inputs, four real references in unchanged order, seed 17, CFG 5, DDIM 30, scale 0.8, strength 1, 1024 inference / 512 output and Poisson finishing. Only a fixed-scale (0.2) pretrained Plus Face visual-reference branch is added. No training; baseline models/checkpoints and website defaults are preserved.

Attempted 3/3; completed 3/3. Failures remain in the source records. Original targets are evaluation-only.

## Matched means

| Metric | Control | Appearance addition | Delta | Valid pairs |
|---|---:|---:|---:|---:|
| hole_psnr | 17.019171 | 17.243635 | +0.224464 | 3/3 |
| hole_ssim | 0.428353 | 0.423499 | -0.004854 | 3/3 |
| hole_lpips | 0.193782 | 0.180554 | -0.013228 | 3/3 |
| eye_lpips | 0.171193 | 0.150591 | -0.020602 | 3/3 |
| facenet_cosine | 0.764280 | 0.782868 | +0.018588 | 3/3 |
| arcface_conditioning_cosine | 0.787376 | 0.791934 | +0.004558 | 3/3 |

## Every person

| Person | Metric | Control | Appearance addition | Delta |
|---|---|---:|---:|---:|---:|
| 263 | hole_psnr | 16.413578 | 16.234086 | -0.179492 |
| 263 | hole_ssim | 0.494264 | 0.470865 | -0.023399 |
| 263 | hole_lpips | 0.168389 | 0.163522 | -0.004867 |
| 263 | eye_lpips | 0.206952 | 0.198791 | -0.008161 |
| 263 | facenet_cosine | 0.728110 | 0.716685 | -0.011424 |
| 263 | arcface_conditioning_cosine | 0.802763 | 0.781189 | -0.021573 |
| 6681 | hole_psnr | 18.910703 | 19.086401 | +0.175698 |
| 6681 | hole_ssim | 0.493393 | 0.486212 | -0.007180 |
| 6681 | hole_lpips | 0.147842 | 0.133289 | -0.014553 |
| 6681 | eye_lpips | 0.122778 | 0.114821 | -0.007958 |
| 6681 | facenet_cosine | 0.766272 | 0.791263 | +0.024991 |
| 6681 | arcface_conditioning_cosine | 0.786206 | 0.795364 | +0.009158 |
| 10149 | hole_psnr | 15.733232 | 16.410418 | +0.677187 |
| 10149 | hole_ssim | 0.297403 | 0.313421 | +0.016018 |
| 10149 | hole_lpips | 0.265115 | 0.244853 | -0.020262 |
| 10149 | eye_lpips | 0.183849 | 0.138160 | -0.045688 |
| 10149 | facenet_cosine | 0.798458 | 0.840654 | +0.042196 |
| 10149 | arcface_conditioning_cosine | 0.773159 | 0.799248 | +0.026089 |

## Decision and scope

Combined mean numerical gate: FAIL.
- Person 263: FAIL on all six non-worsening measures.
- Person 6681: FAIL on all six non-worsening measures.
- Person 10149: PASS on all six non-worsening measures.

Visual review remains a separate requirement. No automatic production selection. The zero-appearance replay must match both original raw and finished control images byte-for-byte and has passed. All completed outputs preserve every visible pixel. SSIM/identity cosine are not accuracy percentages; three development people and one seed are insufficient for generalization. ArcFace is also used for conditioning. CLIP face-crop features are appearance cues, not exact spatial iris-pixel correspondence.

Official model assets and feature provenance are stored in outputs/focused_eye_appearance_v1/config.json. See [frozen protocol](FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md), [official model repository](https://huggingface.co/h94/IP-Adapter) and [adapter documentation](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter).

## Visual review and recorded encoding failure

All three original/control/candidate comparisons were inspected. No universal selection: average SSIM decreases and matching iris/gaze/brow reconstruction remains unresolved.
- Person 263: Eye colour and gaze mismatch persist; eyebrow texture changes but is not a faithful match.
- Person 6681: Eyes remain oversaturated; the appearance addition does not restore matching original brow/eye structure.
- Person 10149: All six numerical measures improve, but gaze and eyebrow shape still visibly differ from the original.

The first attempt failed during reference encoding with a Half/Float mismatch before producing a candidate. Its config, runner snapshot, cropped references and log are preserved in outputs/focused_eye_appearance_v1_encoding_failure. The retry explicitly converts the CPU encoder to float32 and uses the required vision hidden states; neutral raw/final replay then passes byte-exactly. No prior checkpoint or experiment was overwritten.

Next, complete the fixed CFG 3 / appearance 0.2 interaction arm specified in FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md. This is not a new scale search or per-person selection.
