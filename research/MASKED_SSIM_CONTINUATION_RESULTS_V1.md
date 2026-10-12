# Masked SSIM continuation results v1

Two fixed 32-update continuations of the same 64-update scorer compare epsilon loss with epsilon plus 0.1 fully-contained masked structural loss. This export copies saved evaluator values and recomputes only arithmetic means and paired differences. It does not read photographs or recalculate image metrics.

Numerical gate: **FAIL**. Visual review: **PASS**. Generation: complete; attempts 6/6. No replication or website promotion is authorized by this export.

Three fixed development people (263, 6681, 10149), seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, strength 1, 1024 inference, 512 export, and Poisson finishing. Controls are the preserved CFG3, local appearance, prior 64-update scorer, and the matched fresh epsilon continuation.

## Metric definitions and availability

PSNR is in dB; SSIM and identity cosines are measurements, not accuracy percentages. Hole metrics use the original supplied mask. Eye PSNR and SSIM are copied from `regions.eyes`, the annotated eye region intersected with the hole. Regional SSIM averages fully contained 11 × 11 windows. Hole and eye LPIPS are fixed bounding-box crops with 10 pixels of context and no resizing. Whole-image values include visible pixels preserved by compositing and cannot replace the hole results. `NA` denotes unavailable finite values; exact original status strings and metric source fields are retained in the JSON.

FaceNet is separate from generation conditioning. ArcFace belongs to the conditioning family.

## All metric means

| Metric | Current CFG3 | Neutral local appearance | Prior 64-update scorer | Epsilon continuation | SSIM continuation |
|---|---:|---:|---:|---:|---:|
| whole_psnr | 29.387906 (3/3) | 29.397466 (3/3) | 29.401143 (3/3) | 29.399063 (3/3) | 29.397788 (3/3) |
| whole_ssim | 0.961286 (3/3) | 0.960728 (3/3) | 0.960739 (3/3) | 0.960730 (3/3) | 0.960730 (3/3) |
| whole_lpips | 0.022612 (3/3) | 0.022462 (3/3) | 0.022468 (3/3) | 0.022491 (3/3) | 0.022463 (3/3) |
| hole_psnr | 17.894556 (3/3) | 17.904116 (3/3) | 17.907794 (3/3) | 17.905714 (3/3) | 17.904439 (3/3) |
| hole_ssim | 0.454540 (3/3) | 0.444884 (3/3) | 0.445016 (3/3) | 0.444932 (3/3) | 0.444849 (3/3) |
| hole_lpips | 0.178454 (3/3) | 0.176645 (3/3) | 0.176358 (3/3) | 0.176961 (3/3) | 0.176619 (3/3) |
| eye_psnr | 12.836770 (3/3) | 13.000542 (3/3) | 13.010521 (3/3) | 13.005253 (3/3) | 13.006092 (3/3) |
| eye_ssim | 0.180739 (3/3) | 0.191435 (3/3) | 0.191265 (3/3) | 0.190813 (3/3) | 0.190973 (3/3) |
| eye_lpips | 0.152351 (3/3) | 0.152241 (3/3) | 0.152193 (3/3) | 0.152136 (3/3) | 0.152006 (3/3) |
| facenet_cosine | 0.784486 (3/3) | 0.782640 (3/3) | 0.784304 (3/3) | 0.785901 (3/3) | 0.782534 (3/3) |
| arcface_conditioning_cosine | 0.816156 (3/3) | 0.805353 (3/3) | 0.805454 (3/3) | 0.805024 (3/3) | 0.804875 (3/3) |

## Whole-image outcomes

| Person | Arm | Status | whole_psnr | whole_ssim | whole_lpips |
|---|---|---|---:|---:|---:|
| 263 | Current CFG3 | complete | 29.062505 | 0.968875 | 0.016392 |
| 263 | Neutral local appearance | complete | 28.706089 | 0.966771 | 0.017950 |
| 263 | Prior 64-update scorer | complete | 28.709402 | 0.966786 | 0.017956 |
| 263 | Epsilon continuation | complete | 28.707853 | 0.966769 | 0.017973 |
| 263 | SSIM continuation | complete | 28.706897 | 0.966781 | 0.017971 |
| 6681 | Current CFG3 | complete | 31.642592 | 0.967531 | 0.015800 |
| 6681 | Neutral local appearance | complete | 31.855126 | 0.967712 | 0.015194 |
| 6681 | Prior 64-update scorer | complete | 31.861758 | 0.967728 | 0.015160 |
| 6681 | Epsilon continuation | complete | 31.857771 | 0.967713 | 0.015279 |
| 6681 | SSIM continuation | complete | 31.854169 | 0.967709 | 0.015188 |
| 10149 | Current CFG3 | complete | 27.458621 | 0.947451 | 0.035643 |
| 10149 | Neutral local appearance | complete | 27.631183 | 0.947701 | 0.034242 |
| 10149 | Prior 64-update scorer | complete | 27.632270 | 0.947704 | 0.034288 |
| 10149 | Epsilon continuation | complete | 27.631566 | 0.947709 | 0.034223 |
| 10149 | SSIM continuation | complete | 27.632299 | 0.947702 | 0.034230 |

## Hole and eye outcomes

| Person | Arm | Status | hole_psnr | hole_ssim | hole_lpips | eye_psnr | eye_ssim | eye_lpips |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | complete | 17.429831 | 0.538117 | 0.144067 | 11.635754 | 0.208927 | 0.175436 |
| 263 | Neutral local appearance | complete | 17.073415 | 0.503064 | 0.153591 | 12.141085 | 0.256504 | 0.192277 |
| 263 | Prior 64-update scorer | complete | 17.076728 | 0.503317 | 0.153536 | 12.149122 | 0.255811 | 0.192332 |
| 263 | Epsilon continuation | complete | 17.075179 | 0.503081 | 0.153606 | 12.149601 | 0.256379 | 0.192507 |
| 263 | SSIM continuation | complete | 17.074223 | 0.503183 | 0.153477 | 12.146275 | 0.256152 | 0.192466 |
| 6681 | Current CFG3 | complete | 19.859689 | 0.512666 | 0.136960 | 13.651156 | 0.251216 | 0.118169 |
| 6681 | Neutral local appearance | complete | 20.072223 | 0.516028 | 0.131452 | 14.025806 | 0.247020 | 0.108263 |
| 6681 | Prior 64-update scorer | complete | 20.078855 | 0.516160 | 0.130310 | 14.039787 | 0.247752 | 0.107882 |
| 6681 | Epsilon continuation | complete | 20.074868 | 0.516120 | 0.131151 | 14.021698 | 0.246053 | 0.108546 |
| 6681 | SSIM continuation | complete | 20.071266 | 0.515978 | 0.130475 | 14.028629 | 0.247598 | 0.108047 |
| 10149 | Current CFG3 | complete | 16.394149 | 0.312838 | 0.254336 | 13.223401 | 0.082073 | 0.163447 |
| 10149 | Neutral local appearance | complete | 16.566711 | 0.315560 | 0.244893 | 12.834735 | 0.070779 | 0.156183 |
| 10149 | Prior 64-update scorer | complete | 16.567798 | 0.315570 | 0.245228 | 12.842653 | 0.070232 | 0.156366 |
| 10149 | Epsilon continuation | complete | 16.567094 | 0.315594 | 0.246126 | 12.844460 | 0.070007 | 0.155357 |
| 10149 | SSIM continuation | complete | 16.567827 | 0.315386 | 0.245903 | 12.843373 | 0.069168 | 0.155505 |

## Identity and visible-pixel outcomes

| Person | Arm | Status | facenet_cosine | arcface_conditioning_cosine | Visible pixels changed |
|---|---|---|---:|---:|---:|
| 263 | Current CFG3 | complete | 0.750827 | 0.821858 | 0 |
| 263 | Neutral local appearance | complete | 0.732108 | 0.783381 | 0 |
| 263 | Prior 64-update scorer | complete | 0.736823 | 0.783789 | 0 |
| 263 | Epsilon continuation | complete | 0.737025 | 0.783848 | 0 |
| 263 | SSIM continuation | complete | 0.736576 | 0.783125 | 0 |
| 6681 | Current CFG3 | complete | 0.840004 | 0.840881 | 0 |
| 6681 | Neutral local appearance | complete | 0.853041 | 0.843804 | 0 |
| 6681 | Prior 64-update scorer | complete | 0.853206 | 0.845153 | 0 |
| 6681 | Epsilon continuation | complete | 0.857640 | 0.842922 | 0 |
| 6681 | SSIM continuation | complete | 0.847911 | 0.843916 | 0 |
| 10149 | Current CFG3 | complete | 0.762628 | 0.785730 | 0 |
| 10149 | Neutral local appearance | complete | 0.762770 | 0.788875 | 0 |
| 10149 | Prior 64-update scorer | complete | 0.762884 | 0.787419 | 0 |
| 10149 | Epsilon continuation | complete | 0.763039 | 0.788303 | 0 |
| 10149 | SSIM continuation | complete | 0.763114 | 0.787585 | 0 |

## Official fixed gates

Hole SSIM must strictly improve against every control. Hole PSNR, hole/eye LPIPS and mean FaceNet/ArcFace must be no worse (zero tolerance), with 3/3 valid pairs. Each person must retain both identity cosines within 0.005 of each control. Complete records, exact neutral replay, matched supports and zero final visible changed pixels are required. Whole-image and eye PSNR/SSIM are supplementary diagnostics; the exporter adds no new selection gate.

| Control | Metric | Control mean | SSIM mean | Delta | Pairs | Gate |
|---|---|---:|---:|---:|---:|---|
| Current CFG3 | hole_psnr | 17.894556 | 17.904439 | +0.009882 | 3/3 | PASS |
| Current CFG3 | hole_ssim | 0.454540 | 0.444849 | -0.009691 | 3/3 | FAIL |
| Current CFG3 | hole_lpips | 0.178454 | 0.176619 | -0.001835 | 3/3 | PASS |
| Current CFG3 | eye_lpips | 0.152351 | 0.152006 | -0.000345 | 3/3 | PASS |
| Current CFG3 | facenet_cosine | 0.784486 | 0.782534 | -0.001952 | 3/3 | FAIL |
| Current CFG3 | arcface_conditioning_cosine | 0.816156 | 0.804875 | -0.011281 | 3/3 | FAIL |
| Neutral local appearance | hole_psnr | 17.904116 | 17.904439 | +0.000322 | 3/3 | PASS |
| Neutral local appearance | hole_ssim | 0.444884 | 0.444849 | -0.000035 | 3/3 | FAIL |
| Neutral local appearance | hole_lpips | 0.176645 | 0.176619 | -0.000027 | 3/3 | PASS |
| Neutral local appearance | eye_lpips | 0.152241 | 0.152006 | -0.000235 | 3/3 | PASS |
| Neutral local appearance | facenet_cosine | 0.782640 | 0.782534 | -0.000106 | 3/3 | FAIL |
| Neutral local appearance | arcface_conditioning_cosine | 0.805353 | 0.804875 | -0.000478 | 3/3 | FAIL |
| Prior 64-update scorer | hole_psnr | 17.907794 | 17.904439 | -0.003355 | 3/3 | FAIL |
| Prior 64-update scorer | hole_ssim | 0.445016 | 0.444849 | -0.000167 | 3/3 | FAIL |
| Prior 64-update scorer | hole_lpips | 0.176358 | 0.176619 | +0.000260 | 3/3 | FAIL |
| Prior 64-update scorer | eye_lpips | 0.152193 | 0.152006 | -0.000187 | 3/3 | PASS |
| Prior 64-update scorer | facenet_cosine | 0.784304 | 0.782534 | -0.001771 | 3/3 | FAIL |
| Prior 64-update scorer | arcface_conditioning_cosine | 0.805454 | 0.804875 | -0.000579 | 3/3 | FAIL |
| Epsilon continuation | hole_psnr | 17.905714 | 17.904439 | -0.001275 | 3/3 | FAIL |
| Epsilon continuation | hole_ssim | 0.444932 | 0.444849 | -0.000083 | 3/3 | FAIL |
| Epsilon continuation | hole_lpips | 0.176961 | 0.176619 | -0.000342 | 3/3 | PASS |
| Epsilon continuation | eye_lpips | 0.152136 | 0.152006 | -0.000131 | 3/3 | PASS |
| Epsilon continuation | facenet_cosine | 0.785901 | 0.782534 | -0.003367 | 3/3 | FAIL |
| Epsilon continuation | arcface_conditioning_cosine | 0.805024 | 0.804875 | -0.000149 | 3/3 | FAIL |

| Control | Person | Identity metric | Delta | Minimum | Gate |
|---|---|---|---:|---:|---|
| Current CFG3 | 263 | facenet_cosine | -0.014251 | -0.005000 | FAIL |
| Current CFG3 | 263 | arcface_conditioning_cosine | -0.038733 | -0.005000 | FAIL |
| Current CFG3 | 6681 | facenet_cosine | +0.007908 | -0.005000 | PASS |
| Current CFG3 | 6681 | arcface_conditioning_cosine | +0.003036 | -0.005000 | PASS |
| Current CFG3 | 10149 | facenet_cosine | +0.000486 | -0.005000 | PASS |
| Current CFG3 | 10149 | arcface_conditioning_cosine | +0.001855 | -0.005000 | PASS |
| Neutral local appearance | 263 | facenet_cosine | +0.004468 | -0.005000 | PASS |
| Neutral local appearance | 263 | arcface_conditioning_cosine | -0.000257 | -0.005000 | PASS |
| Neutral local appearance | 6681 | facenet_cosine | -0.005130 | -0.005000 | FAIL |
| Neutral local appearance | 6681 | arcface_conditioning_cosine | +0.000113 | -0.005000 | PASS |
| Neutral local appearance | 10149 | facenet_cosine | +0.000343 | -0.005000 | PASS |
| Neutral local appearance | 10149 | arcface_conditioning_cosine | -0.001290 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | facenet_cosine | -0.000247 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | arcface_conditioning_cosine | -0.000665 | -0.005000 | PASS |
| Prior 64-update scorer | 6681 | facenet_cosine | -0.005294 | -0.005000 | FAIL |
| Prior 64-update scorer | 6681 | arcface_conditioning_cosine | -0.001237 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | facenet_cosine | +0.000229 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | arcface_conditioning_cosine | +0.000165 | -0.005000 | PASS |
| Epsilon continuation | 263 | facenet_cosine | -0.000449 | -0.005000 | PASS |
| Epsilon continuation | 263 | arcface_conditioning_cosine | -0.000723 | -0.005000 | PASS |
| Epsilon continuation | 6681 | facenet_cosine | -0.009729 | -0.005000 | FAIL |
| Epsilon continuation | 6681 | arcface_conditioning_cosine | +0.000994 | -0.005000 | PASS |
| Epsilon continuation | 10149 | facenet_cosine | +0.000075 | -0.005000 | PASS |
| Epsilon continuation | 10149 | arcface_conditioning_cosine | -0.000719 | -0.005000 | PASS |

Failed official gates: 20.

- control3/hole_ssim/mean
- control3/facenet_cosine/mean
- control3/arcface_conditioning_cosine/mean
- control3/263/facenet_cosine/identity_guard
- control3/263/arcface_conditioning_cosine/identity_guard
- local/hole_ssim/mean
- local/facenet_cosine/mean
- local/arcface_conditioning_cosine/mean
- local/6681/facenet_cosine/identity_guard
- prior/hole_psnr/mean
- prior/hole_ssim/mean
- prior/hole_lpips/mean
- prior/facenet_cosine/mean
- prior/arcface_conditioning_cosine/mean
- prior/6681/facenet_cosine/identity_guard
- epsilon/hole_psnr/mean
- epsilon/hole_ssim/mean
- epsilon/facenet_cosine/mean
- epsilon/arcface_conditioning_cosine/mean
- epsilon/6681/facenet_cosine/identity_guard

## Coverage, support and failures

| Arm | Complete | Failed | Blocked | Missing | Reused |
|---|---:|---:|---:|---:|---:|
| Current CFG3 | 3/3 | 0 | 0 | 0 | 3 |
| Neutral local appearance | 3/3 | 0 | 0 | 0 | 3 |
| Prior 64-update scorer | 3/3 | 0 | 0 | 0 | 3 |
| Epsilon continuation | 3/3 | 0 | 0 | 0 | 0 |
| SSIM continuation | 3/3 | 0 | 0 | 0 | 0 |

| Person | Arm | Hole pixels | Hole SSIM centres | Eye pixels | Eye SSIM centres | FaceNet target/output | ArcFace target/output |
|---|---|---:|---:|---:|---:|---|---|
| 263 | Current CFG3 | 18000 | 14950 | 1349 | 219 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Neutral local appearance | 18000 | 14950 | 1349 | 219 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Prior 64-update scorer | 18000 | 14950 | 1349 | 219 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Epsilon continuation | 18000 | 14950 | 1349 | 219 | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | SSIM continuation | 18000 | 14950 | 1349 | 219 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Current CFG3 | 17388 | 14278 | 859 | 66 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Neutral local appearance | 17388 | 14278 | 859 | 66 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Prior 64-update scorer | 17388 | 14278 | 859 | 66 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Epsilon continuation | 17388 | 14278 | 859 | 66 | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | SSIM continuation | 17388 | 14278 | 859 | 66 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Current CFG3 | 20516 | 17466 | 1672 | 472 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Neutral local appearance | 20516 | 17466 | 1672 | 472 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Prior 64-update scorer | 20516 | 17466 | 1672 | 472 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Epsilon continuation | 20516 | 17466 | 1672 | 472 | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | SSIM continuation | 20516 | 17466 | 1672 | 472 | ok (1) / ok (1) | ok (1) / ok (1) |

## Training, probe and resources

| Stage | Status | Completed updates | Seconds | Peak GPU allocated bytes | Peak GPU reserved bytes | Peak working-set bytes |
|---|---|---:|---:|---:|---:|---:|
| Disposable structural-loss probe | passed | NA | 33.937000000005355 | 7337905664 | 7767851008 | 7437795328 |
| epsilon | complete | 32 | 228.59399999998277 | 7290791936 | 7616856064 | 7438245888 |
| ssim | complete | 32 | 1458.6879999999946 | 7290989056 | 7616856064 | 7773536256 |

Probe gradient cosine: 0.996861; structural/epsilon gradient norm ratio: 10.333099. These assess gradients, not generated-image quality.

All 32 recorded identity/timestep/noise schedules match: PASS.

Both arms use 385 trainable scorer parameters and a reset AdamW optimizer. Training remains at 512; evaluation runs at 1024. The JSON retains every saved update, loss, gradient/resource record, config, checkpoint hash, and source pin.

## Supervision and recovery

- `.\outputs\masked_sdxl_ssim_continuation_v1\recovery_supervision_v1\receipt.json`: complete. Original supervisor WinError 5 on atomic receipt replace after epsilon completed, before SSIM launch
- `.\outputs\masked_sdxl_ssim_continuation_v1\supervision\receipt.json`: running. Sequential matched epsilon/SSIM continuations; no quality-based arm selection
- `.\outputs\masked_sdxl_ssim_continuation_v1\supervision\receipt.json.partial`: running. Sequential matched epsilon/SSIM continuations; no quality-based arm selection
- `.\outputs\masked_sdxl_ssim_loss_supervisor_v1\receipt.json`: finished. Saved supervision record.

Original supervisor records and supplemental recovery records remain separate and retain their recorded statuses. A stale original status is not rewritten as successful supervision.

## Visual review and limits

- Person 263: Raw and finished outputs retain forward-facing gaze instead of the original sideways gaze, lighter blue/green irises, and altered angular brows. SSIM and epsilon continuations look nearly identical at full-face and crop scales; no clear new seam or structural artifact is visible. This is not a recovery of the original eye geometry.
- Person 6681: Both raw and finished SSIM retain excessively cyan iris colour with an irregular bright patch at the image-right eye, altered lids and sparse/light eyebrows. These are also present in the controls; no clear extra artifact appears in SSIM. Finished compositing restores surrounding appearance, but does not fix the eye details.
- Person 10149: The SSIM result retains upward/pale gaze, narrower eye opening and flatter/thinner brow shapes than the original. Its visible geometry is nearly indistinguishable from the local/prior/epsilon controls; no clear additional artifact is seen. The raw full-image colour shift remains in all arms and is removed outside the hole by finishing. The original expression is not restored.

All outcomes are retained, including regressions, failures, blocked records and unavailable metrics. Three development identities and one inference seed do not estimate unknown-client accuracy. Learned reference tokens do not establish anatomical correspondence or recovery of the true hidden face. A failed fixed gate stops this recipe; even a pass requires visual review before a later replication proposal. Photographs, generated images and contact sheets remain local and are not included in these exports.

## Provenance

Exported at 2026-10-12T17:37:21.931884+00:00. Source evaluation: `.\outputs\masked_sdxl_ssim_continuation_eval_v1`. The companion JSON records hashes of every consumed evidence file, the pinned report/scorer/source files and the continuation checkpoints. Saved source files were checked unchanged at export. The exporter does not reopen private images or rehash large model assets; their original recorded pins are retained as provenance, not represented as newly verified.
