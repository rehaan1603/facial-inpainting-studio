# Masked SDXL compatibility pilot v2

10 October 2026. **Real GPU training completed: 64 fixed updates on eight training identities, 385 shared trainable parameters.** The three development-test people were excluded. Matched reconstruction evaluation and visual review are complete. Training slightly improves all six batch means over its neutral local-appearance control, but does not meet the stronger current CFG3 acceptance gate.

The original input-cache attempt failed because a profile reference for person 8570 had no face detection at the unchanged buffalo_l640 settings. A full 64-reference audit found only person2250 had four encodable original references. The retained recovery manifest uses eight original identities and unchanged targets, selecting four real same-person references by detector eligibility/numeric HQ order and inherited duplicate/protected exclusions before optimization. No test photo/reference, source image or old manifest was replaced. [Recovery plan](MASKED_SDXL_REFERENCE_ELIGIBILITY_RECOVERY_V2.md), [cache plan](MASKED_SDXL_COMPATIBILITY_DATA_PREFLIGHT_V2.md), [frozen pilot](MASKED_SDXL_COMPATIBILITY_PILOT_PREFLIGHT_V2.md).

Training identities: **6792, 2250, 2083, 4164, 674, 476, 1281, 8748**. Each contributed eight updates, two per timestep 100/300/500/700, with independently seeded noise. The cache contains eight targets,32 references,16 CPU VAE encodes and nine tensor files (85,867,168 bytes), prepared in 160.547 seconds. No evaluation photo/annotation was decoded during cache preparation or training.

| Training evidence | Actual result |
|---|---:|
| Fixed completed / finite-gradient updates | 64 / 64 |
| Unique shared trainable parameters | 385 |
| Loaded frozen UNet plus adapter parameters | 3,366,130,756 |
| Wrapped attention processors | 70 |
| Real-input initial neutral maximum absolute difference | 0.0 |
| Frozen parameter gradients | Absent |
| Peak allocated / reserved GPU memory (bytes) | 7,290,601,472 / 7,564,427,264 |
| Worker elapsed / supervisor wall seconds | 420.828 / 438.125 |
| Mean forward-backward-update seconds | 4.318609 |
| Aggregate gradient norm minimum / maximum | 0.000002526 / 0.000520897 |
| Prior attention-backend flags restored | Yes |

Only bounded target-query/reference-token appearance compatibility is learned. FaceID/text/original attention/projection weights stay frozen. The direct nine-channel UNet is trained at 512 using actual forward-noised clean latents, the supplied hole mask and unnoised masked-image latents. Objective is 0.25 full epsilon MSE plus 0.75 normalized hole epsilon MSE, with AdamW 1e-4 and fixed final-update checkpoint selection. It does not optimize image-space SSIM, RGB, perceptual or identity loss. Losses at different timesteps/noise realizations are not a paired before/after reconstruction-quality comparison.

All 64 traces, source/model/config/protocol/checkpoint hashes and memory/timing counters are retained in [photo-free evidence](masked_sdxl_compatibility_pilot_v2_evidence.json). Initial/last checkpoints and embeddings remain in the new local v2 output directory. Existing checkpoints, production defaults and the original training data remain available.

Both enabled and disabled initial-state raw/final inference replays passed byte and pixel equality against prior local 263. Exactly **five GPU generation calls** were completed: two controls and three trained candidates. Nine scored records contain three new candidates and six reused controls; these are not nine independent generated images. FaceNet/ArcFace and all six metrics are valid for every record; supplied metric supports match and final visible RGB is exact. All three raw and finished comparisons were inspected. No replication or website promotion follows this failed combined gate.

# Fixed masked SDXL compatibility training pilot

One 385-parameter scorer trained for 64 fixed updates at 512 on eight existing training identities. Evaluation uses people 263, 6681 and 10149, four existing clean references, the preserved damaged eye inputs and masks, inference seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, strength 1, model offload, 1024 inference / 512 output and Poisson finishing. Current CFG3 and local appearance control images and scores are hash-verified reuses. The initial scorer is loaded and disabled for one separate local-appearance 263 neutral replay before three trained candidates.

Numerical gate: FAIL. Completed 9/9 records. Candidate generation attempts: 3/3. Neutral replay: PASS. Visual review: FAIL.

## Mean values and metric coverage

| Metric | Current CFG3 | Local appearance control | Trained pilot |
|---|---:|---:|---:|
| hole_psnr | 17.894556 (3/3) | 17.904116 (3/3) | 17.907794 (3/3) |
| hole_ssim | 0.454540 (3/3) | 0.444884 (3/3) | 0.445016 (3/3) |
| hole_lpips | 0.178454 (3/3) | 0.176645 (3/3) | 0.176358 (3/3) |
| eye_lpips | 0.152351 (3/3) | 0.152241 (3/3) | 0.152193 (3/3) |
| facenet_cosine | 0.784486 (3/3) | 0.782640 (3/3) | 0.784304 (3/3) |
| arcface_conditioning_cosine | 0.816156 (3/3) | 0.805353 (3/3) | 0.805454 (3/3) |

## Every person and outcome

| Person | Arm | Status | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace conditioning family |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | complete | 17.429831 | 0.538117 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | Local appearance control | complete | 17.073415 | 0.503064 | 0.153591 | 0.192277 | 0.732108 | 0.783381 |
| 263 | Trained compatibility pilot | complete | 17.076728 | 0.503317 | 0.153536 | 0.192332 | 0.736823 | 0.783789 |
| 6681 | Current CFG3 | complete | 19.859689 | 0.512666 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | Local appearance control | complete | 20.072223 | 0.516028 | 0.131452 | 0.108263 | 0.853041 | 0.843804 |
| 6681 | Trained compatibility pilot | complete | 20.078855 | 0.516160 | 0.130310 | 0.107882 | 0.853206 | 0.845153 |
| 10149 | Current CFG3 | complete | 16.394149 | 0.312838 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | Local appearance control | complete | 16.566711 | 0.315560 | 0.244893 | 0.156183 | 0.762770 | 0.788875 |
| 10149 | Trained compatibility pilot | complete | 16.567798 | 0.315570 | 0.245228 | 0.156366 | 0.762884 | 0.787419 |

## Matched gates against both controls

Hole SSIM must strictly improve on each control mean. All other five metric means must be no worse, using zero tolerance. Every person must retain FaceNet and ArcFace cosine at least control minus 0.005 against each control. All six metrics require 3/3 valid pairs.

| Control | Metric | Control mean | Pilot mean | Delta | Valid pairs | Gate |
|---|---|---:|---:|---:|---:|---|
| Current CFG3 | hole_psnr | 17.894556 | 17.907794 | +0.013237 | 3/3 | PASS |
| Current CFG3 | hole_ssim | 0.454540 | 0.445016 | -0.009525 | 3/3 | FAIL |
| Current CFG3 | hole_lpips | 0.178454 | 0.176358 | -0.002096 | 3/3 | PASS |
| Current CFG3 | eye_lpips | 0.152351 | 0.152193 | -0.000157 | 3/3 | PASS |
| Current CFG3 | facenet_cosine | 0.784486 | 0.784304 | -0.000182 | 3/3 | FAIL |
| Current CFG3 | arcface_conditioning_cosine | 0.816156 | 0.805454 | -0.010702 | 3/3 | FAIL |
| Local appearance control | hole_psnr | 17.904116 | 17.907794 | +0.003677 | 3/3 | PASS |
| Local appearance control | hole_ssim | 0.444884 | 0.445016 | +0.000132 | 3/3 | PASS |
| Local appearance control | hole_lpips | 0.176645 | 0.176358 | -0.000287 | 3/3 | PASS |
| Local appearance control | eye_lpips | 0.152241 | 0.152193 | -0.000048 | 3/3 | PASS |
| Local appearance control | facenet_cosine | 0.782640 | 0.784304 | +0.001665 | 3/3 | PASS |
| Local appearance control | arcface_conditioning_cosine | 0.805353 | 0.805454 | +0.000101 | 3/3 | PASS |

| Control | Person | Identity metric | Delta | Minimum allowed delta | Gate |
|---|---|---|---:|---:|---|
| Current CFG3 | 263 | facenet_cosine | -0.014004 | -0.005000 | FAIL |
| Current CFG3 | 263 | arcface_conditioning_cosine | -0.038068 | -0.005000 | FAIL |
| Current CFG3 | 6681 | facenet_cosine | +0.013202 | -0.005000 | PASS |
| Current CFG3 | 6681 | arcface_conditioning_cosine | +0.004273 | -0.005000 | PASS |
| Current CFG3 | 10149 | facenet_cosine | +0.000257 | -0.005000 | PASS |
| Current CFG3 | 10149 | arcface_conditioning_cosine | +0.001690 | -0.005000 | PASS |
| Local appearance control | 263 | facenet_cosine | +0.004715 | -0.005000 | PASS |
| Local appearance control | 263 | arcface_conditioning_cosine | +0.000408 | -0.005000 | PASS |
| Local appearance control | 6681 | facenet_cosine | +0.000165 | -0.005000 | PASS |
| Local appearance control | 6681 | arcface_conditioning_cosine | +0.001350 | -0.005000 | PASS |
| Local appearance control | 10149 | facenet_cosine | +0.000114 | -0.005000 | PASS |
| Local appearance control | 10149 | arcface_conditioning_cosine | -0.001455 | -0.005000 | PASS |

## Failures, detections and visible pixels

| Arm | Complete | Failed | Blocked | Reused | Visible exact |
|---|---:|---:|---:|---:|---:|
| Current CFG3 | 3/3 | 0 | 0 | 3 | 3/3 |
| Local appearance control | 3/3 | 0 | 0 | 3 | 3/3 |
| Trained compatibility pilot | 3/3 | 0 | 0 | 0 | 3/3 |

## Metric supports for every person

| Person | Arm | Hole pixels | Valid 11x11 SSIM centres | Eye pixels | Visible pixels | Visible pixels changed |
|---|---|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Current CFG3 | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Current CFG3 | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Local appearance control | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Local appearance control | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Local appearance control | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Trained compatibility pilot | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Trained compatibility pilot | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Trained compatibility pilot | 20516 | 17466 | 1672 | 241628 | 0 |

LPIPS support is the fixed hole or eye bounding box plus 10 pixels of context, with no resizing. The exact crop boxes, metric status strings and full target/output face detections are recorded for every person in paired_results.json.

| Person | Arm | FaceNet target/output detection | ArcFace target/output detection |
|---|---|---|---|
| 263 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Current CFG3 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Local appearance control | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Local appearance control | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Local appearance control | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Trained compatibility pilot | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Trained compatibility pilot | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Trained compatibility pilot | ok (1) / ok (1) | ok (1) / ok (1) |

## Decision and limits

Every completed, failed and blocked attempt is retained. Negative mean differences and per-person regressions remain in the tables. A failed numerical gate stops this recipe. Even a numerical pass requires completed visual review of every person before proposing replication. This script grants no replication or website promotion.

FaceNet is separate from generation conditioning; ArcFace is in the conditioning family. Cosines and SSIM are measurements, not accuracy percentages. Three development people and one inference seed do not estimate unknown-client accuracy. These learned reference tokens do not establish anatomical eye correspondence or verify recovery of the true hidden face. Training at 512 and evaluation at 1024 is an explicit pilot limitation. Final visible RGB is preserved by compositing; raw outputs are retained and shown separately.

Generation did not open evaluation originals. The CPU evaluator and contact-sheet report use originals only for evaluation. Photographs and image artifacts remain local. Existing models, results, metrics and source files are preserved.

Failed gate count: 5.
- control3/hole_ssim/mean
- control3/facenet_cosine/mean
- control3/arcface_conditioning_cosine/mean
- control3/263/facenet_cosine/identity_guard
- control3/263/arcface_conditioning_cosine/identity_guard

## Saved visual review

- Person 263: Finished and raw comparisons reviewed. The trained result is visually almost unchanged from local appearance. Gaze remains much more frontal than the original sideways gaze; iris saturation and brow/lid shape remain mismatched. The numerical identity guard against current CFG3 fails. Not acceptable as a repaired reconstruction.
- Person 6681: Finished and raw comparisons reviewed. Training produces small numerical gains, but bright cyan irises and pupil/highlight artifacts remain different from the original more muted eyes. Brow and eyelid detail are not recovered exactly. Not visually established as a clear repair.
- Person 10149: Finished and raw comparisons reviewed. Predicted eyes remain narrower and paler than the original wide-eyed expression, with flatter brows/lid texture. Appearance-control differences are visually tiny; eyeLPIPS and ArcFace regress slightly versus that control. Not acceptable as true hidden-eye recovery.

## What this establishes and the next intervention

The module has a real gradient/update path and small directional improvements over the same neutral architecture; all three hole SSIM values rise over that control. The improvements are small and not replicated. Person10149's hole/eye LPIPS and ArcFace worsen against the local control, and person 263 loses 0.014004 FaceNet/0.038068 ArcFace against current CFG3. Do not describe all persons/metrics as improved. The current CFG3 remains the stronger accepted quality control.

Code inspection identifies plausible limitations rather than a proven single cause: epsilon MSE does not supervise decoded iris colour/gaze/brow geometry directly; ten pooled feature statistics and learned reference tokens do not establish anatomical correspondence; joint attention must retain a fixed appearance contribution even when every reference token is unreliable. Increasing GPU usage alone does not remove these limitations.

The recommended next isolated experiment is a differentiable decoded masked-region reconstruction objective, initially one bounded gradient/resource probe, preserving the 385-parameter module, data, optimizer, references and evaluation. Audit decoder and loss gradients before training. Identity remains a mandatory independent guard; add no simultaneous capacity/gating/perceptual changes. A separately versioned residual reliability gate is a later alternative because token reweighting cannot abstain from a wholly misleading branch. These are future experiments, not implemented or measured benefits.

Local comparison copies are in each existing three-person set under **Experimental - Trained compatibility64 updates** (folder name includes spaces). Photos/embeddings/checkpoints stay local; numbers, hash pins, source/data metadata and code are on GitHub.
