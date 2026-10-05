# Expanded official-input reference encoding results

5 October 2026. Completed observed-development diagnostic. No training or website changes.

## Scope and integrity

Generated 30 images across five official targets and seeds 17/29/43, and scored all 30 outputs plus five degraded-input controls. Both original smoke replays are byte-identical. The three single-reference pairs are byte-identical. Two paired-summary tests pass.

Four multi-reference targets belong to identity 6681. The single-reference control belongs to identity 263. Thus the multi-reference contrast has 12 paired outputs but only **one identity**, not four or twelve independent people. Reference selection is the first up-to-five official CSV entries; these local metrics and the frozen EMA loader are not an exact published benchmark reproduction.

## Matched multi-reference results

| Metric | Separate latent encoding | Joint RGB encoding | Joint minus separate | Valid pairs | Joint wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 22.814315 | 22.809751 | -0.004564 | 12/12 | 6 / 6 |
| ssim | 0.546588 | 0.546070 | -0.000517 | 12/12 | 4 / 8 |
| lpips | 0.281699 | 0.282950 | +0.001251 | 12/12 | 2 / 10 |
| facenet_cosine | 0.899630 | 0.903725 | +0.004094 | 12/12 | 6 / 6 |
| arcface_conditioning_cosine | 0.775136 | 0.775924 | +0.000788 | 9/12 | 4 / 5 |
| eye_bbox_lpips | 0.098759 | 0.099599 | +0.000840 | 12/12 | 5 / 7 |

Higher PSNR/SSIM/identity similarity and lower LPIPS/eye LPIPS are preferable. Joint encoding improves mean FaceNet and ArcFace but worsens mean PSNR, SSIM, LPIPS and eye-region LPIPS. The initial favorable single-case result does not repeat as an overall quality gain. Keep production preprocessing unchanged.

ArcFace lacks target detection for 11491, leaving 9/12 valid multi-reference pairs. No failures were substituted or thresholds changed. FaceNet has all 12 pairs. Eye annotations use the same HQ indices; resized local HQ images differ by approximately one intensity level on average, with phase-correlation offsets below 0.012 pixels across all five targets. This supports geometry alignment while documenting source compression differences. Eye LPIPS includes ten-pixel bounding-box context.

## Resource and visual review

| Encoding | Peak allocated GiB, multi-reference | Mean seconds per generation |
|---|---:|---:|
| joint_rgb | 7.093 | 9.400 |
| separate_latents | 2.387 | 7.271 |

Timing excludes initial model loading and includes per-run conditioning, sampling and PNG save. Peak tensor allocation excludes other processes and some driver memory. It is not a quality ceiling imposed by VRAM.

Inspected all five three-seed comparison sheets. Both methods restore sharper texture than the inputs. The outputs still alter eyelids, eye color/gaze, skin texture and expression. Encoding differences are small; no consistently superior visual result emerged. The one-reference control remains visibly identical between arms.

## Decision and next step

Do not switch production to joint RGB encoding on this evidence. Test oracle removal of the deliberately corrupted reference under the original four-identity matched local protocol before training an independent rejection gate. Oracle removal would only establish feasibility; a deployable selector must use degraded inputs and supplied references without corruption labels or evaluation targets.

Reserved final identities and the additional Human Faces dataset were not used. Local visual sheets: `outputs/refldm_encoding_v2/comparison_*_all_seeds.png`. Raw numeric evidence: [results](refldm_encoding_v2_results.json). Frozen protocol: [preflight](REFLDM_ENCODING_SCREEN_V2_PREFLIGHT.md).
