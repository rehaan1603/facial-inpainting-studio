# Spatial reference correspondence v1: prospective fixed protocol

11 October2026. Written after code/architecture audit and before implementation,
GPU feature extraction, optimization or new development generation. Candidate
parameters and numeric targets below are plans, not measured improvements.

## Single intervention

Add spatial reference features only at
`mid_block.attentions.0.transformer_blocks.0.attn1.processor` in existing SDXL.
Keep the CFG3 FaceID-only baseline. No global appearance adapter, FreeU, sampler,
step-count, prompt, mask or finishing changes.

Reference hidden features come from the same frozen UNet at fixed timestep500,
using real reference images and fixed noise seed20261011+reference index. Encode
references independently with zero hole mask, same text/FaceID context, and their
own image latent. Capture this layer's normalized input. Each retained source
grid cell has its actual pixel-centre coordinate. Apply the existing layer's
frozen K/V projections. Feature extraction is a hypothesis inspired by DIFT,
not a claimed reproduction of its published extraction recipe.

Register source coordinates using visible target jaw/nose/mouth landmarks only.
Sanitize every missing pixel to128 before detection, even if the caller supplied
hidden RGB. Never use target eye/brow predictions, clean target pixels,
anatomical annotations or the generated evaluation output to fit alignment.
References may use all their visible landmarks. Same fixed supplied photos
remain in FaceID; spatially unreliable references contribute no spatial tokens.

Fit a proper similarity transform (no reflection), at least8 visible safe
anchors,8-pixel boundary margin, nondegenerate rank ratio>=0.02. Deterministically
hold out every third safe anchor for validation. Require held-out RMS divided by
detected target-face width<=0.04 and full-anchor RMS<=0.04. Require at least two
of the four references to pass and pairwise mapped eye-centre disagreement<=0.12
target-face widths. Failed detection or geometry gives exact baseline fallback,
recorded explicitly. These are engineering reliability gates, not proofs that
hidden eye shape/gaze matches.

Keep at most128 source grid cells per accepted reference within its detected
eye/brow envelope with one feature-cell margin. Deterministic row-major uniform
subsampling if needed. Joint reference attention receives Gaussian coordinate
bias `-squared_distance/(2*0.08^2)` in normalized target-image coordinates,
plus log reliability from held-out visible-anchor error. Attention also uses
the frozen target query/source key dot product. Do not materialize a full
query-by-source-by-channel tensor. Only supplied-mask query centres receive the
extra branch. A masked-query residual uses20 learned head gates
`0.2*tanh(g)`, initialized to zero. Exactly20 additional trainable scalars.
Empty/invalid banks bypass to the unmodified delegate.

## Mandatory stages and stopping rules

1. Complete the saved audit before implementation.
2. Unit tests: hidden-pixel invariance; safe-anchor exclusions; proper transforms;
   failed/ambiguous geometry fallback; spatial coordinates change attention;
   zero state equals unmodified processor; outside-mask output unchanged;
   joint reference/token permutation invariance; finite nonzero gate gradients;
   exactly20 trainable parameters and no pretrained gradients.
3. CPU geometry audit on the eight existing training cases, with all failures
   retained. All eight must have usable geometry to retain the declared fixed
   dataset. Otherwise stop before GPU training and diagnose. Do not drop failed
   cases or change thresholds to pass. A separate justified revision is needed.
4. Only after those pass, cache reference features and run one bounded512 direct-
   UNet backward feasibility probe, batch1, timeout300 seconds. Check neutral
   output tolerance1e-3, finite nonzero gate gradient, frozen parameters, actual
   parameter count, peak allocated/reserved bytes. Target budget7.6 decimal GB
   allocated; stop on OOM, timeout, stale/busy studio or resource failure.
5. If feasible, fixed64-update training and only the final checkpoint. No search,
   person-specific settings, repeated best checkpoints or revised evaluation.
6. Full1024 initial zero-state raw/final byte and pixel replay on person263
   against the saved CFG3 baseline before development candidate generation.
7. Exactly three fixed seed17 candidates, then unchanged scoring and visual review.
   Wider seeds29/43 follow only a passing combined first screen and frozen plan.

## Training, conditional on prerequisites

Reuse the exact existing v2 training manifest SHA256
`b90a1ccddad5cd7c6efafe39209567765e918aabbf0a68022955e3baea2a875f` and its exclusion
list. IDs6792,2250,2083,4164,674,476,1281,8748, four real references each.
Never train on263,6681,10149,7400 or any reserved/final identity. Original target
and masked-latent caches are supervision/training inputs, never development
alignment inputs. Preserve reference order and all source hashes.

Freeze UNet/FaceID2,942,381,892 parameters, VAE, text encoders and InsightFace.
Train only20 gate parameters. AdamW lr0.001, weight_decay0, clip norm1, batch1,
512 training resolution, fp16 frozen weights/bf16 autocast/fp32 gates, existing
gradient checkpointing and temporary Flash/Math backend policy restored after.
Seed20261011. Cycle the fixed eight identities eight times. Timesteps100 for
updates1–16,300 for17–32,500 for33–48,700 for49–64. Noise seed20261011+update index+1.

Loss unchanged from the prior representation study:0.25 full epsilon MSE plus
0.75 normalized masked epsilon MSE. This isolates spatial representation rather
than add the already-negative RGB recipe or several arbitrary losses. Both
weights are explicit and independently disableable in a separately declared
ablation, not changed during this first comparison. No whole-image SSIM loss.
Structural/perceptual/identity objectives remain later isolated options if this
representation has a viable gradient and useful matched outcome.

Save new initial/final gate checkpoints, optimizer/seed/source receipts, all
updates and failures under `outputs/spatial_reference_correspondence_v1` only.
No overwrite/resume of prior checkpoints. Expected6–7.5GB peak is unmeasured;
the actual probe/training receipts must report runtime and peak memory.

## Matched evaluation and acceptance

A: current CFG3. B: the one new20-gate candidate. Same persons263/6681/10149,
references, masks, seed17, DDIM30, CFG3, FaceID0.8, strength1,1024 inference,
512 export, model offload and existing Poisson finishing. Fixed generator/scorer
hashes, supports and input hashes. Reuse only verified control images/scores.
Evaluate with the frozen `score_masked_sdxl_v1.py` and region/identity modules.
Report per-person and means, coverage/failures, anatomical-eye SSIM/LPIPS from
existing regional records, and all zero-fallback cases. Ground truth enters
only the separate scoring process. No hidden-target inference access.

First milestone: mean hole SSIM>0.50 (prefer>0.55); hole LPIPS<=baseline;
hole PSNR>=baseline; mean FaceNet and ArcFace delta>=-0.002, every-person
identity delta>=-0.005; eye LPIPS no higher; zero changed visible pixels;
all three valid detections; same region supports; no severe iris/gaze/lid/brow
artifact on inspected raw/final images. Record strict identity deltas even when
within tolerance. Mean improvement without these requirements is not promotion.
Three-seed replication is required before a more reliable claim.0.80 hole SSIM
is aspirational and is not a promised or threshold-tuned outcome.

If B fails, preserve complete results and diagnose before any larger experiment.
Do not change the app defaults or publish restricted photographs.
