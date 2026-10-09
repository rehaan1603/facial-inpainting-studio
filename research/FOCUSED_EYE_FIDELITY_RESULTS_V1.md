# Three-person eye fidelity: guidance and sampler experiments

9 October 2026. Same persons 263, 6681 and 10149, erased-eye masks, four clean references and seed 17. All arms use 1024 inference / 512 output, 30 steps, FaceID scale 0.8, strength 1 and Poisson finishing. No training.

Each candidate changes ONE setting relative to the saved DDIM guidance 5 control. The guidance 3 candidate retains DDIM. The Euler candidate retains guidance 5. No per-person or best-seed selection.

## Mean values, three matched cases

| Method | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---:|---:|---:|---:|---:|---:|
| DDIM CFG5 | 17.019171 | 0.428353 | 0.193782 | 0.171193 | 0.764280 | 0.787376 |
| DDIM CFG3 | 17.894556 | 0.454540 | 0.178454 | 0.152351 | 0.784486 | 0.816156 |
| Euler CFG5 | 17.075214 | 0.434059 | 0.186383 | 0.168720 | 0.754315 | 0.783776 |

## Per-person comparisons

| Person | Method | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace |
|---|---|---:|---:|---:|---:|---:|---:|
| 263 | DDIM CFG5 | 16.413578 | 0.494264 | 0.168389 | 0.206952 | 0.728110 | 0.802763 |
| 263 | DDIM CFG3 | 17.429831 | 0.538117 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | Euler CFG5 | 16.452729 | 0.499072 | 0.166523 | 0.201617 | 0.704234 | 0.802095 |
| 6681 | DDIM CFG5 | 18.910703 | 0.493393 | 0.147842 | 0.122778 | 0.766272 | 0.786206 |
| 6681 | DDIM CFG3 | 19.859689 | 0.512666 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | Euler CFG5 | 19.002982 | 0.499596 | 0.140671 | 0.121181 | 0.769614 | 0.780287 |
| 10149 | DDIM CFG5 | 15.733232 | 0.297403 | 0.265115 | 0.183849 | 0.798458 | 0.773159 |
| 10149 | DDIM CFG3 | 16.394149 | 0.312838 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | Euler CFG5 | 15.769931 | 0.303509 | 0.251956 | 0.183364 | 0.789096 | 0.768944 |

## Combined numerical gate
- DDIM CFG3: PASS on matched means.
- Euler CFG5: FAIL on matched means.

## Visual review and selection

All three comparison sheets were inspected. **Neither candidate is promoted to a universal website default.** Lower guidance is promising for masked-region structure, but it does not solve exact iris colour, gaze or brow recovery. Euler fails the combined identity requirement.
- Person 263: CFG 3 reduces exaggerated contrast, but generated iris colour and gaze still differ from the original. Euler also retains these discrepancies.
- Person 6681: CFG 3 is closer in tone than CFG 5, but the eyes remain more saturated and eyebrow/eye structure is imperfect.
- Person 10149: All arms retain a visibly incorrect gaze/eye appearance. CFG 3 lowers FaceNet by 0.035830 despite higher hole SSIM and ArcFace.

## Anatomical eye SSIM diagnostic

These values already exist in the frozen scorer. They refer to annotated eye pixels inside the damage mask, rather than the larger erased rectangle that also contains brows and skin. Only fully contained 11-by-11 SSIM windows contribute. Very small valid-window counts make these local values unstable; they are supplementary and do not replace the primary frozen hole SSIM.

| Person | Method | Anatomical eye SSIM | Valid window centres |
|---|---|---:|---:|
| 263 | DDIM CFG5 | 0.189793 | 219 |
| 263 | DDIM CFG3 | 0.208927 | 219 |
| 263 | Euler CFG5 | 0.186948 | 219 |
| 6681 | DDIM CFG5 | 0.202704 | 66 |
| 6681 | DDIM CFG3 | 0.251216 | 66 |
| 6681 | Euler CFG5 | 0.202655 | 66 |
| 10149 | DDIM CFG5 | 0.032124 | 472 |
| 10149 | DDIM CFG3 | 0.082073 | 472 |
| 10149 | Euler CFG5 | 0.028440 | 472 |

These gates do not replace visual review or per-person identity checks. Neither research runner modifies the website. SSIM is not a colour-specific or identity-accuracy percentage. ArcFace also conditions FaceID.

The guidance5 neutral replay is raw- and finished-byte-exact. All original controls remain available. Evaluation-only targets never enter generation. Failures and missing detector measurements remain in source results.

Preflights: FOCUSED_EYE_GUIDANCE_PREFLIGHT_V1.md and FOCUSED_EYE_SCHEDULER_PREFLIGHT_V1.md. Local image comparisons: outputs/focused_eye_fidelity_v1.
