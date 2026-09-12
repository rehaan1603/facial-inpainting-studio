# Completed local experiments

> Historical first-stage single-image results at the 1,500-update refiner budget. These tables remain unchanged. See TRAINING_EXTENSION_RESULTS.md for the completed 6,000-update study and REFERENCE_IMPLEMENTATION.md for the separate reference-assisted implementation; neither changes this original test protocol.

These tables are generated from raw metrics; lower is better for all three measures. Visible MAE measures change to genuinely unoccluded pixels. It is not an identity-recognition score.

## Controlled development assessment

48 previously used development identity labels, 432 location/area cases, four supplied-mask conditions. Learned LaMa results average three matched seeds; ResShift learned results use predeclared seed 17 only. Fixed controls use one deterministic LaMa run or one diffusion seed per case.

| backbone | method | seeds | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---:|---:|---:|
| lama | generic | 3 | 0.058448 | 0.170009 | 0.000143 |
| lama | preservation_weighted | 3 | 0.061238 | 0.180471 | 0.000106 |
| resshift | generic | 1 | 0.034955 | 0.089615 | 0.000167 |
| resshift | preservation_weighted | 1 | 0.037621 | 0.096317 | 0.000117 |
| lama | dilation 8, feather 0 | 1 | 0.041050 | 0.114483 | 0.002601 |
| resshift | dilation 8, feather 0 | 1 | 0.022025 | 0.077208 | 0.002155 |
| resshift | dilation 12, feather 4 | 1 | 0.026863 | 0.083003 | 0.003481 |

### Seed-level results

| backbone | method | seed | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---:|---:|---:|
| lama | generic | 17 | 0.053541 | 0.154347 | 0.000167 |
| lama | generic | 29 | 0.061267 | 0.177678 | 0.000112 |
| lama | generic | 43 | 0.060535 | 0.178001 | 0.000149 |
| lama | preservation_weighted | 17 | 0.057812 | 0.169722 | 0.000116 |
| lama | preservation_weighted | 29 | 0.058716 | 0.171019 | 0.000097 |
| lama | preservation_weighted | 43 | 0.067186 | 0.200672 | 0.000104 |
| resshift | generic | 17 | 0.034955 | 0.089615 | 0.000167 |
| resshift | preservation_weighted | 17 | 0.037621 | 0.096317 | 0.000117 |

### Weighted minus generic, paired development changes

Intervals are percentile bootstrap intervals over 48 identity labels (2,000 replicates), keeping masks, locations and training seeds together. They condition on these three fitted seeds rather than estimating population-wide training uncertainty.

- lama, full_face_lpips: +0.002790, 95% interval [+0.002026, +0.003611].
- lama, hole_mae: +0.010463, 95% interval [+0.007509, +0.013698].
- lama, visible_mae: -0.000037, 95% interval [-0.000054, -0.000025].
- resshift, full_face_lpips: +0.002666, 95% interval [+0.001290, +0.004147].
- resshift, hole_mae: +0.006702, 95% interval [+0.004328, +0.009190].
- resshift, visible_mae: -0.000050, 95% interval [-0.000076, -0.000031].

## Frozen object-composite test

64 HQ test identity labels and 64 LaPa test images; four error conditions; 12 fixed, photographed COCO object cutouts from six categories. These assets were absent from refiner training. This is a synthetic paired composite test, not real occluded-face capture. Its 5,632 rows include unchanged-input controls. LaPa uses a ground-truth-landmark-defined crop for consistent face framing, unavailable in an unconstrained deployment; cross-source numbers also reflect this preprocessing difference.

| dataset | backbone | method | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---:|---:|---:|
| celebahq | lama | dilate8 | 0.056023 | 0.114938 | 0.004027 |
| celebahq | lama | generic | 0.101323 | 0.190607 | 0.000673 |
| celebahq | lama | preservation_weighted | 0.104247 | 0.197939 | 0.000414 |
| celebahq | lama | supplied | 0.077732 | 0.181530 | 0.000562 |
| celebahq | lama | unchanged | 0.134737 | 0.312717 | 0.000000 |
| celebahq | resshift | dilate8 | 0.030016 | 0.072053 | 0.003176 |
| celebahq | resshift | generic | 0.074026 | 0.126087 | 0.000589 |
| celebahq | resshift | preservation_weighted | 0.079410 | 0.134838 | 0.000384 |
| celebahq | resshift | supplied | 0.064128 | 0.148660 | 0.000500 |
| celebahq | resshift | tuned | 0.035920 | 0.078663 | 0.005084 |
| celebahq | resshift | unchanged | 0.134737 | 0.312717 | 0.000000 |
| lapa | lama | dilate8 | 0.056342 | 0.088750 | 0.002807 |
| lapa | lama | generic | 0.141428 | 0.200826 | 0.000302 |
| lapa | lama | preservation_weighted | 0.143486 | 0.207690 | 0.000176 |
| lapa | lama | supplied | 0.095689 | 0.173030 | 0.000383 |
| lapa | lama | unchanged | 0.173268 | 0.321709 | 0.000000 |
| lapa | resshift | dilate8 | 0.065207 | 0.106281 | 0.003350 |
| lapa | resshift | generic | 0.129096 | 0.173218 | 0.000356 |
| lapa | resshift | preservation_weighted | 0.134553 | 0.185485 | 0.000213 |
| lapa | resshift | supplied | 0.097288 | 0.172216 | 0.000433 |
| lapa | resshift | tuned | 0.078724 | 0.119097 | 0.005816 |
| lapa | resshift | unchanged | 0.173268 | 0.321709 | 0.000000 |

### Accurate-mask stratum

| dataset | backbone | method | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---:|---:|---:|
| celebahq | lama | dilate8 | 0.054466 | 0.118380 | 0.003744 |
| celebahq | lama | generic | 0.107452 | 0.205671 | 0.000608 |
| celebahq | lama | preservation_weighted | 0.111577 | 0.224430 | 0.000348 |
| celebahq | lama | supplied | 0.028875 | 0.079157 | 0.000000 |
| celebahq | lama | unchanged | 0.134737 | 0.312717 | 0.000000 |
| celebahq | resshift | dilate8 | 0.028726 | 0.072391 | 0.002990 |
| celebahq | resshift | generic | 0.078027 | 0.133436 | 0.000545 |
| celebahq | resshift | preservation_weighted | 0.085858 | 0.148538 | 0.000336 |
| celebahq | resshift | supplied | 0.015970 | 0.055401 | 0.000000 |
| celebahq | resshift | tuned | 0.035453 | 0.079045 | 0.005055 |
| celebahq | resshift | unchanged | 0.134737 | 0.312717 | 0.000000 |
| lapa | lama | dilate8 | 0.054193 | 0.090748 | 0.002688 |
| lapa | lama | generic | 0.150272 | 0.231558 | 0.000278 |
| lapa | lama | preservation_weighted | 0.152774 | 0.236777 | 0.000147 |
| lapa | lama | supplied | 0.030724 | 0.060977 | 0.000000 |
| lapa | lama | unchanged | 0.173268 | 0.321709 | 0.000000 |
| lapa | resshift | dilate8 | 0.062769 | 0.107496 | 0.002981 |
| lapa | resshift | generic | 0.136870 | 0.189425 | 0.000338 |
| lapa | resshift | preservation_weighted | 0.141657 | 0.204674 | 0.000183 |
| lapa | resshift | supplied | 0.035165 | 0.069375 | 0.000000 |
| lapa | resshift | tuned | 0.078945 | 0.122257 | 0.005814 |
| lapa | resshift | unchanged | 0.173268 | 0.321709 | 0.000000 |

### Weighted minus generic on the new test

Bootstrap over 64 HQ identity labels or 64 LaPa images, conditional on the fixed assets. LaPa identity independence is unresolved; intervals may understate uncertainty if identities repeat.

- celebahq/lama/weighted-minus-generic, full_face_lpips: +0.002925, 95% interval [-0.000127, +0.005952].
- celebahq/lama/weighted-minus-generic, hole_mae: +0.007332, 95% interval [-0.002503, +0.016484].
- celebahq/lama/weighted-minus-generic, visible_mae: -0.000258, 95% interval [-0.000352, -0.000177].
- celebahq/resshift/weighted-minus-generic, full_face_lpips: +0.005384, 95% interval [+0.002039, +0.008715].
- celebahq/resshift/weighted-minus-generic, hole_mae: +0.008751, 95% interval [+0.002872, +0.014981].
- celebahq/resshift/weighted-minus-generic, visible_mae: -0.000204, 95% interval [-0.000275, -0.000142].
- lapa/lama/weighted-minus-generic, full_face_lpips: +0.002058, 95% interval [-0.001274, +0.005396].
- lapa/lama/weighted-minus-generic, hole_mae: +0.006864, 95% interval [-0.002617, +0.016077].
- lapa/lama/weighted-minus-generic, visible_mae: -0.000125, 95% interval [-0.000182, -0.000077].
- lapa/resshift/weighted-minus-generic, full_face_lpips: +0.005456, 95% interval [+0.001924, +0.009148].
- lapa/resshift/weighted-minus-generic, hole_mae: +0.012267, 95% interval [+0.004706, +0.020180].
- lapa/resshift/weighted-minus-generic, visible_mae: -0.000143, 95% interval [-0.000203, -0.000088].

## Compute and reproducibility

Six 119,057-parameter refiner runs, 1,500 steps and batch size 16 each: 431.6 seconds recorded training-loop time in these sessions; maximum allocated memory 400.0 MiB. Cache preparation, dataset audits, downloads, checkpoint startup, and evaluation are excluded. This is not total project energy or wall time.

No generative backbone was retrained. Checkpoints, reviewed manifests, threshold selections, frozen test protocol and raw hashes are recorded locally. Training losses were still improving at the last budgeted step, so convergence is not established.

## Scientific limits

The weighted loss is a standard control, not an established novel contribution. No nearest-method reproduction (VCNet/OSOR), soft-alpha learned control, identity fidelity evaluation, blinded human study, real paired capture, or comprehensive pretraining-membership audit is provided. Final test inference is now complete; these cases must not be treated as unused for subsequent method development. The engineering pipeline and scoped experiments are complete. An IEEE submission is not ready merely because this package runs.

Object-protocol QA verified all 128 cases: true masks are nonempty and do not touch the image boundary; mask polarity and unchanged visible pixels pass. Missing area ranges from 2.28% to 22.89%; 5 LaPa crops require black padding outside the source image. Location, area and texture differ from development, so differences cannot be attributed to object appearance alone. Category/condition breakdowns, including sample counts, are in outputs/object_test/strata.csv.

## Warm inference on this laptop

Warm single-image inference including preprocessing tensors and CPU/GPU transfers, excluding image disk IO and model loading. One fixed development example, no background evaluation running. Refiner remains resident for all measurements. Not total project compute.

| Backbone | Refiner | Median ms | P95 ms | Allocated MiB |
|---|---|---:|---:|---:|
| lama | False | 21.56 | 22.05 | 246.4 |
| lama | True | 23.23 | 24.29 | 246.4 |
| resshift | False | 234.54 | 262.09 | 1006.8 |
| resshift | True | 272.98 | 300.56 | 1006.8 |
