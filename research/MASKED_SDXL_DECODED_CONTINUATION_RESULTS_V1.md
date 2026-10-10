# Masked SDXL decoded-loss continuation: completed results v1

11 October 2026 (India time); protocol and training frozen on 10 October. Actual matched GPU experiment, with every outcome retained. Existing datasets, checkpoints, experiments and production behavior are preserved. Both fresh continuation checkpoints and all images stay local. Code and photo-free numeric evidence can be uploaded.

## Decision

The decoded missing-region RGB loss **does not pass the fixed combined quality/identity gate**. Compared with the equally trained epsilon control, mean hole PSNR/SSIM, hole-crop LPIPS and FaceNet worsen; eye-crop LPIPS and ArcFace improve slightly. Whole-image LPIPS also improves, but this does not reverse the missing-region or identity regressions. Raw and finished visual inspection of both new arms on all three people still finds incorrect gaze, saturated/pale iris colours and inaccurate eye/brow geometry. No new model enters the website and no extra-seed replication is started from this failed recipe. Reconstruction improvement remains work in progress.

## What was actually trained

Both arms start from the same preserved 64-update, 385-parameter checkpoint, SHA256 `0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba`. AdamW resets identically. Each arm completes **16 finite-gradient updates** on the unchanged eight-person cache, two updates per person, timesteps 100/300, matched seeds 20261011..20261026. Training people are 6792,2250,2083,4164,674,476,1281,8748. The three evaluated people 263/6681/10149 and protected 7400 remain outside optimization. Training 512, evaluation 1024, export 512. Four real same-person references remain in the original order.

The shared 10→32→1 scorer has 385 trainable parameters and is used at 70 attention processors. Frozen loaded backbone/adapter parameters remain 3,366,130,756. A bounded 0.5 appearance-token bias uses target/reference statistical compatibility features inside the hole; FaceID/text and the fixed 0.8/0.2 branch scales are unchanged. This is a lightweight local adapter experiment, not full backbone training. The token statistics still provide no verified anatomical correspondence and do not implement a zero-reference abstention route.

- **Epsilon control:** 0.25 full epsilon MSE + 0.75 normalized missing-region epsilon MSE.
- **RGB candidate:** the same objective plus 0.1 decoded missing-region RGB L1, normalized by 3×hole pixels, with no clipping of decoded values.

DDPM alpha and the epsilon x0 formula reconstruct a clean latent estimate. A separate frozen 83,653,863-parameter CPU fp32 VAE decodes it while preserving the gradient through device transfer. The addition of this one loss is the only arm change; no new capacity, identity loss, perceptual loss, reference selection, data or inference tuning is added. Both initial checkpoint files are byte-identical, and the first-forward full/hole/combined epsilon losses match **exactly**. All source/model/cache hashes, loss equations, schedules, frozen-gradient guards and restored attention-backend flags pass verification.

## Gradient/resource prerequisite

A preregistered one-case training-only probe passes before the paired continuations. Unweighted decoded RGB gradient norm 0.0000908112, epsilon gradient norm 0.0000703402, gradient cosine 0.9988125 and nonzero disposable update establish that the new term reaches the 385 parameters. They do not establish improved reconstruction. The probe creates no checkpoint or quality image, and changes no prior checkpoint. See [probe preflight](MASKED_SDXL_DECODED_LOSS_PREFLIGHT_V1.md), [results](MASKED_SDXL_DECODED_LOSS_PROBE_RESULTS_V1.md) and [numeric receipt](masked_sdxl_decoded_loss_probe_v1_evidence.json).

| Saved continuation | Updates | Worker seconds | Supervisor wall seconds | Peak allocated GPU bytes | Peak process working-set bytes |
|---|---:|---:|---:|---:|---:|
| Epsilon |16|81.625|97.344|7,290,791,936|7,437,721,600|
| RGB |16|322.094|338.344|7,290,989,056|7,857,897,472|

Both sequential owned workers finish within the predeclared 900-second per-arm bound. RGB costs about 3.95× the control worker time because of differentiable CPU decoding. This resource result explains compute cost, not the cause of poor eye fidelity. The laptop's 8 GB GPU motivated the bounded 512-resolution pilot; no evidence establishes RAM/VRAM as the sole reason for the numerical/visual limitations. The short optimization schedule, train/evaluation resolution gap and adapter representation are also explicit limitations.

## Evaluation and evidence

The fixed inference recipe is DDIM 30, CFG 3, FaceID 0.8/appearance 0.2, strength 1, seed 17, model CPU offload, 1024 inference and 512 Poisson-finished export. **Seven GPU calls** include one exact enabled starting-state replay and six new candidates; nine hash-verified scored controls are reused. Generation takes 415.516 seconds. **15/15 metric rows**, five arms×three people, have valid whole/hole/eye/identity values and one-face detections. All matched region supports and final visible-pixel preservation pass. This is three development people and one inference seed, not 15 independent identities or unknown-client accuracy.

The initial enabled scorer reproduces both the previous raw and finished PNGs byte/pixel-exactly. Evaluation originals enter only CPU scoring and contact sheets, never training or generation. Hole SSIM uses fully contained 11×11 windows; LPIPS uses fixed contextual hole/eye crops. FaceNet is separate from conditioning; ArcFace is in the conditioning family. Whole-image scores include unchanged visible pixels, so they must not be called missing-eye accuracy or percentages.

[Fixed paired protocol](MASKED_SDXL_DECODED_CONTINUATION_PREFLIGHT_V1.md) · [verified photo-free evidence with all rows, means, traces and hashes](masked_sdxl_decoded_continuation_v1_evidence.json) · [previous 64-update result](MASKED_SDXL_COMPATIBILITY_PILOT_RESULTS_V2.md).

All three existing local test folders contain **Experimental - Decoded RGB 16 updates** with five arms' raw/final images, both comparison sheets, all values and a read-first note. Existing inputs, masks, real references and comparison-only originals remain intact. The illustrated overall and small-batch PPTs are updated locally; public GitHub evidence excludes their dataset photos.

## Next investigation

Stop this exact RGB recipe rather than select a favorable person, metric or seed. Before another training intervention, audit attention selectivity and spatial feature correspondence on training-only cases: global statistical compatibility may be insufficient to select iris/brow structure. The existing pixel-loss gradient works, but these 16 matched updates do not improve the required combination. A capacity/representation change requires its own audit and one fixed comparison; do not add several losses, select new evaluation images or relax the gate. Corrupted/misleading references, unfamiliar people, independent training/inference seeds and reserved final testing remain pending after a promising controlled candidate. This result cannot establish a general failure of all decoded objectives or justify a publication-performance claim.

## Complete measured tables and every failed gate

# Masked SDXL matched decoded-loss continuations

Two matched 16-update continuations start from the same completed 64-update scorer, with reset optimizers and identical eight-person noise/timestep schedule. The only arm difference is added 0.1 decoded-hole RGB L1. Evaluation uses people 263, 6681 and 10149, four existing clean references, the preserved damaged eye inputs and masks, inference seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, strength 1, model offload, 1024 inference / 512 output and Poisson finishing. Current CFG3 and local appearance control images and scores are hash-verified reuses. The shared initial scorer stays enabled for a byte-exact replay against the prior 64-update result. Six new candidates and nine reused controls provide fifteen scored records.

Numerical gate: FAIL. Completed 15/15 records. Candidate generation attempts: 6/6. Neutral replay: PASS. Visual review: FAIL.

## Missing-region means and metric coverage

| Metric | Current CFG3 | Neutral local appearance | Prior 64-update scorer | Epsilon continuation | RGB continuation |
|---|---:|---:|---:|---:|---:|
| hole_psnr | 17.894556 (3/3) | 17.904116 (3/3) | 17.907794 (3/3) | 17.906163 (3/3) | 17.906019 (3/3) |
| hole_ssim | 0.454540 (3/3) | 0.444884 (3/3) | 0.445016 (3/3) | 0.444992 (3/3) | 0.444860 (3/3) |
| hole_lpips | 0.178454 (3/3) | 0.176645 (3/3) | 0.176358 (3/3) | 0.176786 (3/3) | 0.176811 (3/3) |
| eye_lpips | 0.152351 (3/3) | 0.152241 (3/3) | 0.152193 (3/3) | 0.152312 (3/3) | 0.151757 (3/3) |
| facenet_cosine | 0.784486 (3/3) | 0.782640 (3/3) | 0.784304 (3/3) | 0.782793 (3/3) | 0.781355 (3/3) |
| arcface_conditioning_cosine | 0.816156 (3/3) | 0.805353 (3/3) | 0.805454 (3/3) | 0.804817 (3/3) | 0.805360 (3/3) |

## Every person and outcome

| Person | Arm | Status | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace conditioning family |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | complete | 17.429831 | 0.538117 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | Neutral local appearance | complete | 17.073415 | 0.503064 | 0.153591 | 0.192277 | 0.732108 | 0.783381 |
| 263 | Prior 64-update scorer | complete | 17.076728 | 0.503317 | 0.153536 | 0.192332 | 0.736823 | 0.783789 |
| 263 | Epsilon continuation | complete | 17.074920 | 0.503161 | 0.153580 | 0.192866 | 0.736709 | 0.783357 |
| 263 | RGB continuation | complete | 17.073116 | 0.503147 | 0.153678 | 0.192239 | 0.732177 | 0.782899 |
| 6681 | Current CFG3 | complete | 19.859689 | 0.512666 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | Neutral local appearance | complete | 20.072223 | 0.516028 | 0.131452 | 0.108263 | 0.853041 | 0.843804 |
| 6681 | Prior 64-update scorer | complete | 20.078855 | 0.516160 | 0.130310 | 0.107882 | 0.853206 | 0.845153 |
| 6681 | Epsilon continuation | complete | 20.076735 | 0.516292 | 0.131407 | 0.107629 | 0.848552 | 0.843784 |
| 6681 | RGB continuation | complete | 20.076820 | 0.515945 | 0.131105 | 0.107713 | 0.848478 | 0.845412 |
| 10149 | Current CFG3 | complete | 16.394149 | 0.312838 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | Neutral local appearance | complete | 16.566711 | 0.315560 | 0.244893 | 0.156183 | 0.762770 | 0.788875 |
| 10149 | Prior 64-update scorer | complete | 16.567798 | 0.315570 | 0.245228 | 0.156366 | 0.762884 | 0.787419 |
| 10149 | Epsilon continuation | complete | 16.566834 | 0.315523 | 0.245372 | 0.156441 | 0.763117 | 0.787310 |
| 10149 | RGB continuation | complete | 16.568120 | 0.315489 | 0.245650 | 0.155319 | 0.763411 | 0.787769 |

## RGB gates against all four controls

RGB hole SSIM must strictly improve on each control mean, including fresh epsilon as the causal loss control. All other five metric means must be no worse, using zero tolerance. Every person must retain FaceNet and ArcFace cosine at least control minus 0.005 against each control. All six metrics require 3/3 valid pairs.

| Control | Metric | Control mean | RGB mean | Delta | Valid pairs | Gate |
|---|---|---:|---:|---:|---:|---|
| Current CFG3 | hole_psnr | 17.894556 | 17.906019 | +0.011462 | 3/3 | PASS |
| Current CFG3 | hole_ssim | 0.454540 | 0.444860 | -0.009680 | 3/3 | FAIL |
| Current CFG3 | hole_lpips | 0.178454 | 0.176811 | -0.001643 | 3/3 | PASS |
| Current CFG3 | eye_lpips | 0.152351 | 0.151757 | -0.000594 | 3/3 | PASS |
| Current CFG3 | facenet_cosine | 0.784486 | 0.781355 | -0.003131 | 3/3 | FAIL |
| Current CFG3 | arcface_conditioning_cosine | 0.816156 | 0.805360 | -0.010796 | 3/3 | FAIL |
| Neutral local appearance | hole_psnr | 17.904116 | 17.906019 | +0.001903 | 3/3 | PASS |
| Neutral local appearance | hole_ssim | 0.444884 | 0.444860 | -0.000024 | 3/3 | FAIL |
| Neutral local appearance | hole_lpips | 0.176645 | 0.176811 | +0.000166 | 3/3 | FAIL |
| Neutral local appearance | eye_lpips | 0.152241 | 0.151757 | -0.000484 | 3/3 | PASS |
| Neutral local appearance | facenet_cosine | 0.782640 | 0.781355 | -0.001285 | 3/3 | FAIL |
| Neutral local appearance | arcface_conditioning_cosine | 0.805353 | 0.805360 | +0.000007 | 3/3 | PASS |
| Prior 64-update scorer | hole_psnr | 17.907794 | 17.906019 | -0.001775 | 3/3 | FAIL |
| Prior 64-update scorer | hole_ssim | 0.445016 | 0.444860 | -0.000156 | 3/3 | FAIL |
| Prior 64-update scorer | hole_lpips | 0.176358 | 0.176811 | +0.000453 | 3/3 | FAIL |
| Prior 64-update scorer | eye_lpips | 0.152193 | 0.151757 | -0.000436 | 3/3 | PASS |
| Prior 64-update scorer | facenet_cosine | 0.784304 | 0.781355 | -0.002949 | 3/3 | FAIL |
| Prior 64-update scorer | arcface_conditioning_cosine | 0.805454 | 0.805360 | -0.000094 | 3/3 | FAIL |
| Epsilon continuation | hole_psnr | 17.906163 | 17.906019 | -0.000144 | 3/3 | FAIL |
| Epsilon continuation | hole_ssim | 0.444992 | 0.444860 | -0.000132 | 3/3 | FAIL |
| Epsilon continuation | hole_lpips | 0.176786 | 0.176811 | +0.000025 | 3/3 | FAIL |
| Epsilon continuation | eye_lpips | 0.152312 | 0.151757 | -0.000555 | 3/3 | PASS |
| Epsilon continuation | facenet_cosine | 0.782793 | 0.781355 | -0.001437 | 3/3 | FAIL |
| Epsilon continuation | arcface_conditioning_cosine | 0.804817 | 0.805360 | +0.000543 | 3/3 | PASS |

| Control | Person | Identity metric | Delta | Minimum allowed delta | Gate |
|---|---|---|---:|---:|---|
| Current CFG3 | 263 | facenet_cosine | -0.018650 | -0.005000 | FAIL |
| Current CFG3 | 263 | arcface_conditioning_cosine | -0.038958 | -0.005000 | FAIL |
| Current CFG3 | 6681 | facenet_cosine | +0.008474 | -0.005000 | PASS |
| Current CFG3 | 6681 | arcface_conditioning_cosine | +0.004531 | -0.005000 | PASS |
| Current CFG3 | 10149 | facenet_cosine | +0.000783 | -0.005000 | PASS |
| Current CFG3 | 10149 | arcface_conditioning_cosine | +0.002039 | -0.005000 | PASS |
| Neutral local appearance | 263 | facenet_cosine | +0.000069 | -0.005000 | PASS |
| Neutral local appearance | 263 | arcface_conditioning_cosine | -0.000482 | -0.005000 | PASS |
| Neutral local appearance | 6681 | facenet_cosine | -0.004563 | -0.005000 | PASS |
| Neutral local appearance | 6681 | arcface_conditioning_cosine | +0.001608 | -0.005000 | PASS |
| Neutral local appearance | 10149 | facenet_cosine | +0.000640 | -0.005000 | PASS |
| Neutral local appearance | 10149 | arcface_conditioning_cosine | -0.001105 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | facenet_cosine | -0.004646 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | arcface_conditioning_cosine | -0.000890 | -0.005000 | PASS |
| Prior 64-update scorer | 6681 | facenet_cosine | -0.004728 | -0.005000 | PASS |
| Prior 64-update scorer | 6681 | arcface_conditioning_cosine | +0.000259 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | facenet_cosine | +0.000526 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | arcface_conditioning_cosine | +0.000350 | -0.005000 | PASS |
| Epsilon continuation | 263 | facenet_cosine | -0.004532 | -0.005000 | PASS |
| Epsilon continuation | 263 | arcface_conditioning_cosine | -0.000458 | -0.005000 | PASS |
| Epsilon continuation | 6681 | facenet_cosine | -0.000074 | -0.005000 | PASS |
| Epsilon continuation | 6681 | arcface_conditioning_cosine | +0.001628 | -0.005000 | PASS |
| Epsilon continuation | 10149 | facenet_cosine | +0.000294 | -0.005000 | PASS |
| Epsilon continuation | 10149 | arcface_conditioning_cosine | +0.000459 | -0.005000 | PASS |

## Failures, detections and visible pixels

| Arm | Complete | Failed | Blocked | Reused | Visible exact |
|---|---:|---:|---:|---:|---:|
| Current CFG3 | 3/3 | 0 | 0 | 3 | 3/3 |
| Neutral local appearance | 3/3 | 0 | 0 | 3 | 3/3 |
| Prior 64-update scorer | 3/3 | 0 | 0 | 3 | 3/3 |
| Epsilon continuation | 3/3 | 0 | 0 | 0 | 3/3 |
| RGB continuation | 3/3 | 0 | 0 | 0 | 3/3 |

## Metric supports for every person

| Person | Arm | Hole pixels | Valid 11x11 SSIM centres | Eye pixels | Visible pixels | Visible pixels changed |
|---|---|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Current CFG3 | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Current CFG3 | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Neutral local appearance | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Neutral local appearance | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Neutral local appearance | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Prior 64-update scorer | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Prior 64-update scorer | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Prior 64-update scorer | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Epsilon continuation | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Epsilon continuation | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Epsilon continuation | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | RGB continuation | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | RGB continuation | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | RGB continuation | 20516 | 17466 | 1672 | 241628 | 0 |

LPIPS support is the fixed hole or eye bounding box plus 10 pixels of context, with no resizing. The exact crop boxes, metric status strings and full target/output face detections are recorded for every person in paired_results.json.

| Person | Arm | FaceNet target/output detection | ArcFace target/output detection |
|---|---|---|---|
| 263 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Neutral local appearance | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Neutral local appearance | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Neutral local appearance | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Prior 64-update scorer | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Prior 64-update scorer | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Prior 64-update scorer | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Epsilon continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Epsilon continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Epsilon continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | RGB continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | RGB continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | RGB continuation | ok (1) / ok (1) | ok (1) / ok (1) |

## Decision and limits

Every completed, failed and blocked attempt is retained. Negative mean differences and per-person regressions remain in the tables. A failed numerical gate stops this recipe. Even a numerical pass requires completed visual review of every person before proposing replication. This script grants no replication or website promotion.

FaceNet is separate from generation conditioning; ArcFace is in the conditioning family. Cosines and SSIM are measurements, not accuracy percentages. Three development people and one inference seed do not estimate unknown-client accuracy. These learned reference tokens do not establish anatomical eye correspondence or verify recovery of the true hidden face. Training at 512 and evaluation at 1024 is an explicit pilot limitation. Final visible RGB is preserved by compositing; raw outputs are retained and shown separately.

Generation did not open evaluation originals. The CPU evaluator and contact-sheet report use originals only for evaluation. Photographs and image artifacts remain local. Existing models, results, metrics and source files are preserved.

Failed gate count: 17.
- control3/hole_ssim/mean
- control3/facenet_cosine/mean
- control3/arcface_conditioning_cosine/mean
- control3/263/facenet_cosine/identity_guard
- control3/263/arcface_conditioning_cosine/identity_guard
- local/hole_ssim/mean
- local/hole_lpips/mean
- local/facenet_cosine/mean
- prior/hole_psnr/mean
- prior/hole_ssim/mean
- prior/hole_lpips/mean
- prior/facenet_cosine/mean
- prior/arcface_conditioning_cosine/mean
- epsilon/hole_psnr/mean
- epsilon/hole_ssim/mean
- epsilon/hole_lpips/mean
- epsilon/facenet_cosine/mean

## Saved visual review

- Person 263: Inspected epsilon and RGB raw/finished sheets. Both continue to reconstruct a different gaze from the sideways-looking target. Iris hue, lid shape and brow angles remain inaccurate. RGB does not provide a clear visual correction over epsilon or prior64; finishing restores visible colour but not missing-eye geometry.
- Person 6681: Inspected epsilon and RGB raw/finished sheets. Both retain unnaturally saturated cyan irises, especially the eye on the right of the image, and mismatched lid/iris structure versus the target. The lower eye-crop LPIPS mean does not establish a visible fix. Raw global colour shift remains outside the preserved finished pixels.
- Person 10149: Inspected epsilon and RGB raw/finished sheets. Both retain paler green/blue irises, narrower eye opening and flatter/thinner brow geometry relative to the target. Neither continuation visibly repairs these details. Final visible pixels remain preserved; raw images show global generation colour changes.

## Whole-image values

These values cover all 512 x 512 RGB pixels. They do not replace missing-region scores because final compositing preserves most visible pixels.

| Metric | Current CFG3 | Neutral local appearance | Prior 64-update scorer | Epsilon continuation | RGB continuation |
|---|---:|---:|---:|---:|---:|
| psnr | 29.387906 (3/3) | 29.397466 (3/3) | 29.401143 (3/3) | 29.399513 (3/3) | 29.399369 (3/3) |
| ssim | 0.961286 (3/3) | 0.960728 (3/3) | 0.960739 (3/3) | 0.960732 (3/3) | 0.960727 (3/3) |
| lpips | 0.022612 (3/3) | 0.022462 (3/3) | 0.022468 (3/3) | 0.022481 (3/3) | 0.022452 (3/3) |

| Person | Arm | Whole PSNR | Whole SSIM | Whole LPIPS |
|---|---|---:|---:|---:|
| 263 | Current CFG3 | 29.062505 | 0.968875 | 0.016392 |
| 263 | Neutral local appearance | 28.706089 | 0.966771 | 0.017950 |
| 263 | Prior 64-update scorer | 28.709402 | 0.966786 | 0.017956 |
| 263 | Epsilon continuation | 28.707594 | 0.966772 | 0.017993 |
| 263 | RGB continuation | 28.705791 | 0.966774 | 0.017948 |
| 6681 | Current CFG3 | 31.642592 | 0.967531 | 0.015800 |
| 6681 | Neutral local appearance | 31.855126 | 0.967712 | 0.015194 |
| 6681 | Prior 64-update scorer | 31.861758 | 0.967728 | 0.015160 |
| 6681 | Epsilon continuation | 31.859638 | 0.967722 | 0.015174 |
| 6681 | RGB continuation | 31.859723 | 0.967708 | 0.015128 |
| 10149 | Current CFG3 | 27.458621 | 0.947451 | 0.035643 |
| 10149 | Neutral local appearance | 27.631183 | 0.947701 | 0.034242 |
| 10149 | Prior 64-update scorer | 27.632270 | 0.947704 | 0.034288 |
| 10149 | Epsilon continuation | 27.631306 | 0.947702 | 0.034278 |
| 10149 | RGB continuation | 27.632593 | 0.947700 | 0.034280 |
