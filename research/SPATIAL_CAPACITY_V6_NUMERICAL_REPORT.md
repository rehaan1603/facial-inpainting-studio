# Masked SDXL matched spatial-capacity continuations

Two matched 32-update continuations start from the same completed 64-update scorer, with reset optimizers and identical eight-person noise/timestep schedule. Both use the same structural loss; only spatial adapter optimization differs. Evaluation uses people 263, 6681 and 10149, four existing clean references, the preserved damaged eye inputs and masks, inference seed 17, DDIM 30, CFG 3, FaceID 0.8, appearance 0.2, strength 1, model offload, 1024 inference / 512 output and Poisson finishing. Current CFG3 and local appearance control images and scores are hash-verified reuses. The shared initial scorer stays enabled for a byte-exact replay against the prior 64-update result. Six new candidates and fifteen reused controls provide twenty-one scored records.

Numerical gate: FAIL. Completed 21/21 records. Candidate generation attempts: 6/6. Neutral replay: PASS. Visual review: PENDING.

## Missing-region means and metric coverage

| Metric | Current CFG3 | Neutral local appearance | Prior 64-update scorer | Epsilon continuation | SSIM continuation | Fixed spatial control | Learned spatial adapter |
|---|---:|---:|---:|---:|---:|---:|---:|
| hole_psnr | 17.894556 (3/3) | 17.904116 (3/3) | 17.907794 (3/3) | 17.905714 (3/3) | 17.904439 (3/3) | 17.904439 (3/3) | 17.906345 (3/3) |
| hole_ssim | 0.454540 (3/3) | 0.444884 (3/3) | 0.445016 (3/3) | 0.444932 (3/3) | 0.444849 (3/3) | 0.444849 (3/3) | 0.444869 (3/3) |
| hole_lpips | 0.178454 (3/3) | 0.176645 (3/3) | 0.176358 (3/3) | 0.176961 (3/3) | 0.176619 (3/3) | 0.176619 (3/3) | 0.176377 (3/3) |
| eye_lpips | 0.152351 (3/3) | 0.152241 (3/3) | 0.152193 (3/3) | 0.152136 (3/3) | 0.152006 (3/3) | 0.152006 (3/3) | 0.151868 (3/3) |
| facenet_cosine | 0.784486 (3/3) | 0.782640 (3/3) | 0.784304 (3/3) | 0.785901 (3/3) | 0.782534 (3/3) | 0.782534 (3/3) | 0.782839 (3/3) |
| arcface_conditioning_cosine | 0.816156 (3/3) | 0.805353 (3/3) | 0.805454 (3/3) | 0.805024 (3/3) | 0.804875 (3/3) | 0.804875 (3/3) | 0.805402 (3/3) |

## Every person and outcome

| Person | Arm | Status | Hole PSNR | Hole SSIM | Hole crop LPIPS | Eye crop LPIPS | FaceNet | ArcFace conditioning family |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 263 | Current CFG3 | complete | 17.429831 | 0.538117 | 0.144067 | 0.175436 | 0.750827 | 0.821858 |
| 263 | Neutral local appearance | complete | 17.073415 | 0.503064 | 0.153591 | 0.192277 | 0.732108 | 0.783381 |
| 263 | Prior 64-update scorer | complete | 17.076728 | 0.503317 | 0.153536 | 0.192332 | 0.736823 | 0.783789 |
| 263 | Epsilon continuation | complete | 17.075179 | 0.503081 | 0.153606 | 0.192507 | 0.737025 | 0.783848 |
| 263 | SSIM continuation | complete | 17.074223 | 0.503183 | 0.153477 | 0.192466 | 0.736576 | 0.783125 |
| 263 | Fixed spatial control | complete | 17.074223 | 0.503183 | 0.153477 | 0.192466 | 0.736576 | 0.783125 |
| 263 | Learned spatial adapter | complete | 17.072002 | 0.503121 | 0.153210 | 0.191844 | 0.732312 | 0.783716 |
| 6681 | Current CFG3 | complete | 19.859689 | 0.512666 | 0.136960 | 0.118169 | 0.840004 | 0.840881 |
| 6681 | Neutral local appearance | complete | 20.072223 | 0.516028 | 0.131452 | 0.108263 | 0.853041 | 0.843804 |
| 6681 | Prior 64-update scorer | complete | 20.078855 | 0.516160 | 0.130310 | 0.107882 | 0.853206 | 0.845153 |
| 6681 | Epsilon continuation | complete | 20.074868 | 0.516120 | 0.131151 | 0.108546 | 0.857640 | 0.842922 |
| 6681 | SSIM continuation | complete | 20.071266 | 0.515978 | 0.130475 | 0.108047 | 0.847911 | 0.843916 |
| 6681 | Fixed spatial control | complete | 20.071266 | 0.515978 | 0.130475 | 0.108047 | 0.847911 | 0.843916 |
| 6681 | Learned spatial adapter | complete | 20.080616 | 0.516084 | 0.130554 | 0.108017 | 0.853298 | 0.844468 |
| 10149 | Current CFG3 | complete | 16.394149 | 0.312838 | 0.254336 | 0.163447 | 0.762628 | 0.785730 |
| 10149 | Neutral local appearance | complete | 16.566711 | 0.315560 | 0.244893 | 0.156183 | 0.762770 | 0.788875 |
| 10149 | Prior 64-update scorer | complete | 16.567798 | 0.315570 | 0.245228 | 0.156366 | 0.762884 | 0.787419 |
| 10149 | Epsilon continuation | complete | 16.567094 | 0.315594 | 0.246126 | 0.155357 | 0.763039 | 0.788303 |
| 10149 | SSIM continuation | complete | 16.567827 | 0.315386 | 0.245903 | 0.155505 | 0.763114 | 0.787585 |
| 10149 | Fixed spatial control | complete | 16.567827 | 0.315386 | 0.245903 | 0.155505 | 0.763114 | 0.787585 |
| 10149 | Learned spatial adapter | complete | 16.566417 | 0.315401 | 0.245369 | 0.155744 | 0.762907 | 0.788022 |

## Capacity gates against all six controls

Candidate hole SSIM must strictly improve on each control mean, including fresh optimizer-frozen spatial weights as the causal capacity control. All other five metric means must be no worse, using zero tolerance. Every person must retain FaceNet and ArcFace cosine at least control minus 0.005 against each control. All six metrics require 3/3 valid pairs.

| Control | Metric | Control mean | Capacity mean | Delta | Valid pairs | Gate |
|---|---|---:|---:|---:|---:|---|
| Current CFG3 | hole_psnr | 17.894556 | 17.906345 | +0.011789 | 3/3 | PASS |
| Current CFG3 | hole_ssim | 0.454540 | 0.444869 | -0.009672 | 3/3 | FAIL |
| Current CFG3 | hole_lpips | 0.178454 | 0.176377 | -0.002077 | 3/3 | PASS |
| Current CFG3 | eye_lpips | 0.152351 | 0.151868 | -0.000482 | 3/3 | PASS |
| Current CFG3 | facenet_cosine | 0.784486 | 0.782839 | -0.001647 | 3/3 | FAIL |
| Current CFG3 | arcface_conditioning_cosine | 0.816156 | 0.805402 | -0.010754 | 3/3 | FAIL |
| Neutral local appearance | hole_psnr | 17.904116 | 17.906345 | +0.002229 | 3/3 | PASS |
| Neutral local appearance | hole_ssim | 0.444884 | 0.444869 | -0.000016 | 3/3 | FAIL |
| Neutral local appearance | hole_lpips | 0.176645 | 0.176377 | -0.000268 | 3/3 | PASS |
| Neutral local appearance | eye_lpips | 0.152241 | 0.151868 | -0.000373 | 3/3 | PASS |
| Neutral local appearance | facenet_cosine | 0.782640 | 0.782839 | +0.000199 | 3/3 | PASS |
| Neutral local appearance | arcface_conditioning_cosine | 0.805353 | 0.805402 | +0.000048 | 3/3 | PASS |
| Prior 64-update scorer | hole_psnr | 17.907794 | 17.906345 | -0.001449 | 3/3 | FAIL |
| Prior 64-update scorer | hole_ssim | 0.445016 | 0.444869 | -0.000147 | 3/3 | FAIL |
| Prior 64-update scorer | hole_lpips | 0.176358 | 0.176377 | +0.000019 | 3/3 | FAIL |
| Prior 64-update scorer | eye_lpips | 0.152193 | 0.151868 | -0.000325 | 3/3 | PASS |
| Prior 64-update scorer | facenet_cosine | 0.784304 | 0.782839 | -0.001466 | 3/3 | FAIL |
| Prior 64-update scorer | arcface_conditioning_cosine | 0.805454 | 0.805402 | -0.000052 | 3/3 | FAIL |
| Epsilon continuation | hole_psnr | 17.905714 | 17.906345 | +0.000631 | 3/3 | PASS |
| Epsilon continuation | hole_ssim | 0.444932 | 0.444869 | -0.000063 | 3/3 | FAIL |
| Epsilon continuation | hole_lpips | 0.176961 | 0.176377 | -0.000583 | 3/3 | PASS |
| Epsilon continuation | eye_lpips | 0.152136 | 0.151868 | -0.000268 | 3/3 | PASS |
| Epsilon continuation | facenet_cosine | 0.785901 | 0.782839 | -0.003062 | 3/3 | FAIL |
| Epsilon continuation | arcface_conditioning_cosine | 0.805024 | 0.805402 | +0.000377 | 3/3 | PASS |
| SSIM continuation | hole_psnr | 17.904439 | 17.906345 | +0.001906 | 3/3 | PASS |
| SSIM continuation | hole_ssim | 0.444849 | 0.444869 | +0.000019 | 3/3 | PASS |
| SSIM continuation | hole_lpips | 0.176619 | 0.176377 | -0.000241 | 3/3 | PASS |
| SSIM continuation | eye_lpips | 0.152006 | 0.151868 | -0.000138 | 3/3 | PASS |
| SSIM continuation | facenet_cosine | 0.782534 | 0.782839 | +0.000305 | 3/3 | PASS |
| SSIM continuation | arcface_conditioning_cosine | 0.804875 | 0.805402 | +0.000527 | 3/3 | PASS |
| Fixed spatial control | hole_psnr | 17.904439 | 17.906345 | +0.001906 | 3/3 | PASS |
| Fixed spatial control | hole_ssim | 0.444849 | 0.444869 | +0.000019 | 3/3 | PASS |
| Fixed spatial control | hole_lpips | 0.176619 | 0.176377 | -0.000241 | 3/3 | PASS |
| Fixed spatial control | eye_lpips | 0.152006 | 0.151868 | -0.000138 | 3/3 | PASS |
| Fixed spatial control | facenet_cosine | 0.782534 | 0.782839 | +0.000305 | 3/3 | PASS |
| Fixed spatial control | arcface_conditioning_cosine | 0.804875 | 0.805402 | +0.000527 | 3/3 | PASS |

| Control | Person | Identity metric | Delta | Minimum allowed delta | Gate |
|---|---|---|---:|---:|---|
| Current CFG3 | 263 | facenet_cosine | -0.018515 | -0.005000 | FAIL |
| Current CFG3 | 263 | arcface_conditioning_cosine | -0.038142 | -0.005000 | FAIL |
| Current CFG3 | 6681 | facenet_cosine | +0.013294 | -0.005000 | PASS |
| Current CFG3 | 6681 | arcface_conditioning_cosine | +0.003587 | -0.005000 | PASS |
| Current CFG3 | 10149 | facenet_cosine | +0.000279 | -0.005000 | PASS |
| Current CFG3 | 10149 | arcface_conditioning_cosine | +0.002292 | -0.005000 | PASS |
| Neutral local appearance | 263 | facenet_cosine | +0.000204 | -0.005000 | PASS |
| Neutral local appearance | 263 | arcface_conditioning_cosine | +0.000334 | -0.005000 | PASS |
| Neutral local appearance | 6681 | facenet_cosine | +0.000256 | -0.005000 | PASS |
| Neutral local appearance | 6681 | arcface_conditioning_cosine | +0.000664 | -0.005000 | PASS |
| Neutral local appearance | 10149 | facenet_cosine | +0.000137 | -0.005000 | PASS |
| Neutral local appearance | 10149 | arcface_conditioning_cosine | -0.000853 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | facenet_cosine | -0.004512 | -0.005000 | PASS |
| Prior 64-update scorer | 263 | arcface_conditioning_cosine | -0.000073 | -0.005000 | PASS |
| Prior 64-update scorer | 6681 | facenet_cosine | +0.000092 | -0.005000 | PASS |
| Prior 64-update scorer | 6681 | arcface_conditioning_cosine | -0.000686 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | facenet_cosine | +0.000022 | -0.005000 | PASS |
| Prior 64-update scorer | 10149 | arcface_conditioning_cosine | +0.000602 | -0.005000 | PASS |
| Epsilon continuation | 263 | facenet_cosine | -0.004713 | -0.005000 | PASS |
| Epsilon continuation | 263 | arcface_conditioning_cosine | -0.000132 | -0.005000 | PASS |
| Epsilon continuation | 6681 | facenet_cosine | -0.004342 | -0.005000 | PASS |
| Epsilon continuation | 6681 | arcface_conditioning_cosine | +0.001546 | -0.005000 | PASS |
| Epsilon continuation | 10149 | facenet_cosine | -0.000132 | -0.005000 | PASS |
| Epsilon continuation | 10149 | arcface_conditioning_cosine | -0.000282 | -0.005000 | PASS |
| SSIM continuation | 263 | facenet_cosine | -0.004264 | -0.005000 | PASS |
| SSIM continuation | 263 | arcface_conditioning_cosine | +0.000591 | -0.005000 | PASS |
| SSIM continuation | 6681 | facenet_cosine | +0.005386 | -0.005000 | PASS |
| SSIM continuation | 6681 | arcface_conditioning_cosine | +0.000552 | -0.005000 | PASS |
| SSIM continuation | 10149 | facenet_cosine | -0.000207 | -0.005000 | PASS |
| SSIM continuation | 10149 | arcface_conditioning_cosine | +0.000437 | -0.005000 | PASS |
| Fixed spatial control | 263 | facenet_cosine | -0.004264 | -0.005000 | PASS |
| Fixed spatial control | 263 | arcface_conditioning_cosine | +0.000591 | -0.005000 | PASS |
| Fixed spatial control | 6681 | facenet_cosine | +0.005386 | -0.005000 | PASS |
| Fixed spatial control | 6681 | arcface_conditioning_cosine | +0.000552 | -0.005000 | PASS |
| Fixed spatial control | 10149 | facenet_cosine | -0.000207 | -0.005000 | PASS |
| Fixed spatial control | 10149 | arcface_conditioning_cosine | +0.000437 | -0.005000 | PASS |

## Failures, detections and visible pixels

| Arm | Complete | Failed | Blocked | Reused | Visible exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current CFG3 | 3/3 | 0 | 0 | 3 | 3/3 |
| Neutral local appearance | 3/3 | 0 | 0 | 3 | 3/3 |
| Prior 64-update scorer | 3/3 | 0 | 0 | 3 | 3/3 |
| Epsilon continuation | 3/3 | 0 | 0 | 3 | 3/3 |
| SSIM continuation | 3/3 | 0 | 0 | 3 | 3/3 |
| Fixed spatial control | 3/3 | 0 | 0 | 0 | 3/3 |
| Learned spatial adapter | 3/3 | 0 | 0 | 0 | 3/3 |

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
| 263 | SSIM continuation | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | SSIM continuation | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | SSIM continuation | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Fixed spatial control | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Fixed spatial control | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Fixed spatial control | 20516 | 17466 | 1672 | 241628 | 0 |
| 263 | Learned spatial adapter | 18000 | 14950 | 1349 | 244144 | 0 |
| 6681 | Learned spatial adapter | 17388 | 14278 | 859 | 244756 | 0 |
| 10149 | Learned spatial adapter | 20516 | 17466 | 1672 | 241628 | 0 |

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
| 263 | SSIM continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | SSIM continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | SSIM continuation | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Fixed spatial control | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Fixed spatial control | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Fixed spatial control | ok (1) / ok (1) | ok (1) / ok (1) |
| 263 | Learned spatial adapter | ok (1) / ok (1) | ok (1) / ok (1) |
| 6681 | Learned spatial adapter | ok (1) / ok (1) | ok (1) / ok (1) |
| 10149 | Learned spatial adapter | ok (1) / ok (1) | ok (1) / ok (1) |

## Decision and limits

Every completed, failed and blocked attempt is retained. Negative mean differences and per-person regressions remain in the tables. A failed numerical gate stops this recipe. Even a numerical pass requires completed visual review of every person before proposing replication. This script grants no replication or website promotion.

FaceNet is separate from generation conditioning; ArcFace is in the conditioning family. Cosines and SSIM are measurements, not accuracy percentages. Three development people and one inference seed do not estimate unknown-client accuracy. These learned reference tokens do not establish anatomical eye correspondence or verify recovery of the true hidden face. Training at 512 and evaluation at 1024 is an explicit pilot limitation. Final visible SSIM is preserved by compositing; raw outputs are retained and shown separately.

Generation did not open evaluation originals. The CPU evaluator and contact-sheet report use originals only for evaluation. Photographs and image artifacts remain local. Existing models, results, metrics and source files are preserved.

Failed gate count: 13.
- control3/hole_ssim/mean
- control3/facenet_cosine/mean
- control3/arcface_conditioning_cosine/mean
- control3/263/facenet_cosine/identity_guard
- control3/263/arcface_conditioning_cosine/identity_guard
- local/hole_ssim/mean
- prior/hole_psnr/mean
- prior/hole_ssim/mean
- prior/hole_lpips/mean
- prior/facenet_cosine/mean
- prior/arcface_conditioning_cosine/mean
- epsilon/hole_ssim/mean
- epsilon/facenet_cosine/mean

## Whole-image values

These values cover all 512 x 512 SSIM pixels. They do not replace missing-region scores because final compositing preserves most visible pixels.

| Metric | Current CFG3 | Neutral local appearance | Prior 64-update scorer | Epsilon continuation | SSIM continuation | Fixed spatial control | Learned spatial adapter |
|---|---:|---:|---:|---:|---:|---:|---:|
| psnr | 29.387906 (3/3) | 29.397466 (3/3) | 29.401143 (3/3) | 29.399063 (3/3) | 29.397788 (3/3) | 29.397788 (3/3) | 29.399695 (3/3) |
| ssim | 0.961286 (3/3) | 0.960728 (3/3) | 0.960739 (3/3) | 0.960730 (3/3) | 0.960730 (3/3) | 0.960730 (3/3) | 0.960731 (3/3) |
| lpips | 0.022612 (3/3) | 0.022462 (3/3) | 0.022468 (3/3) | 0.022491 (3/3) | 0.022463 (3/3) | 0.022463 (3/3) | 0.022436 (3/3) |

| Person | Arm | Whole PSNR | Whole SSIM | Whole LPIPS |
|---|---|---:|---:|---:|
| 263 | Current CFG3 | 29.062505 | 0.968875 | 0.016392 |
| 263 | Neutral local appearance | 28.706089 | 0.966771 | 0.017950 |
| 263 | Prior 64-update scorer | 28.709402 | 0.966786 | 0.017956 |
| 263 | Epsilon continuation | 28.707853 | 0.966769 | 0.017973 |
| 263 | SSIM continuation | 28.706897 | 0.966781 | 0.017971 |
| 263 | Fixed spatial control | 28.706897 | 0.966781 | 0.017971 |
| 263 | Learned spatial adapter | 28.704676 | 0.966773 | 0.017975 |
| 6681 | Current CFG3 | 31.642592 | 0.967531 | 0.015800 |
| 6681 | Neutral local appearance | 31.855126 | 0.967712 | 0.015194 |
| 6681 | Prior 64-update scorer | 31.861758 | 0.967728 | 0.015160 |
| 6681 | Epsilon continuation | 31.857771 | 0.967713 | 0.015279 |
| 6681 | SSIM continuation | 31.854169 | 0.967709 | 0.015188 |
| 6681 | Fixed spatial control | 31.854169 | 0.967709 | 0.015188 |
| 6681 | Learned spatial adapter | 31.863519 | 0.967724 | 0.015100 |
| 10149 | Current CFG3 | 27.458621 | 0.947451 | 0.035643 |
| 10149 | Neutral local appearance | 27.631183 | 0.947701 | 0.034242 |
| 10149 | Prior 64-update scorer | 27.632270 | 0.947704 | 0.034288 |
| 10149 | Epsilon continuation | 27.631566 | 0.947709 | 0.034223 |
| 10149 | SSIM continuation | 27.632299 | 0.947702 | 0.034230 |
| 10149 | Fixed spatial control | 27.632299 | 0.947702 | 0.034230 |
| 10149 | Learned spatial adapter | 27.630890 | 0.947696 | 0.034232 |
