# Training-budget and compositing follow-up

The six original refiners have now each received 6,000 training updates, four times the first-stage budget. Their initial checkpoints and metrics are preserved. All results below reuse the 48 development assessment identity labels; no test inference was performed in this extension.

## Reconstruction and preservation

| backbone | variant | budget | compositor | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---|---:|---:|---:|
| lama | generic | 1500 | hard | 0.058448 | 0.170009 | 0.000143 |
| lama | generic | 6000 | hard | 0.075708 | 0.246777 | 0.000058 |
| lama | generic | 6000 | probability_alpha | 0.075543 | 0.247879 | 0.000044 |
| lama | preservation_weighted | 1500 | hard | 0.061238 | 0.180471 | 0.000106 |
| lama | preservation_weighted | 6000 | hard | 0.065194 | 0.194623 | 0.000074 |
| lama | preservation_weighted | 6000 | probability_alpha | 0.065054 | 0.196436 | 0.000050 |
| resshift | generic | 1500 | hard | 0.034955 | 0.089615 | 0.000167 |
| resshift | generic | 6000 | hard | 0.065051 | 0.158915 | 0.000063 |
| resshift | generic | 6000 | probability_alpha | 0.065292 | 0.159978 | 0.000052 |
| resshift | preservation_weighted | 1500 | hard | 0.037621 | 0.096317 | 0.000117 |
| resshift | preservation_weighted | 6000 | hard | 0.035559 | 0.085840 | 0.000077 |
| resshift | preservation_weighted | 6000 | probability_alpha | 0.037318 | 0.088473 | 0.000050 |

## Paired changes

Each contrast is first named setting minus second; lower is better. The 95% intervals use 2,000 paired resamples of identity labels, retaining locations, mask errors and training seeds together. Three seeds are evaluated on LaMa and predefined seed 17 on ResShift; intervals do not estimate uncertainty over all possible training seeds.

### lama/generic/6000-minus-1500-hard

- full_face_lpips: +0.017260, interval [+0.015926, +0.018556].
- hole_mae: +0.076768, interval [+0.070154, +0.083108].
- visible_mae: -0.000084, interval [-0.000103, -0.000067].

### lama/generic/soft-minus-hard-6000

- full_face_lpips: -0.000165, interval [-0.000223, -0.000105].
- hole_mae: +0.001102, interval [+0.001045, +0.001155].
- visible_mae: -0.000014, interval [-0.000017, -0.000012].

### lama/preservation_weighted/6000-minus-1500-hard

- full_face_lpips: +0.003956, interval [+0.002640, +0.005222].
- hole_mae: +0.014152, interval [+0.009723, +0.018195].
- visible_mae: -0.000032, interval [-0.000047, -0.000017].

### lama/preservation_weighted/soft-minus-hard-6000

- full_face_lpips: -0.000141, interval [-0.000218, -0.000059].
- hole_mae: +0.001813, interval [+0.001714, +0.001905].
- visible_mae: -0.000024, interval [-0.000029, -0.000019].

### lama/weighted-minus-generic-6000

- full_face_lpips: -0.010514, interval [-0.011240, -0.009834].
- hole_mae: -0.052154, interval [-0.056964, -0.047403].
- visible_mae: +0.000016, interval [+0.000006, +0.000029].

### resshift/generic/6000-minus-1500-hard

- full_face_lpips: +0.030096, interval [+0.027743, +0.032512].
- hole_mae: +0.069299, interval [+0.062038, +0.076706].
- visible_mae: -0.000105, interval [-0.000128, -0.000084].

### resshift/generic/soft-minus-hard-6000

- full_face_lpips: +0.000242, interval [+0.000186, +0.000299].
- hole_mae: +0.001063, interval [+0.001008, +0.001121].
- visible_mae: -0.000010, interval [-0.000013, -0.000008].

### resshift/preservation_weighted/6000-minus-1500-hard

- full_face_lpips: -0.002062, interval [-0.004035, -0.000095].
- hole_mae: -0.010477, interval [-0.014735, -0.006506].
- visible_mae: -0.000040, interval [-0.000070, -0.000013].

### resshift/preservation_weighted/soft-minus-hard-6000

- full_face_lpips: +0.001760, interval [+0.001535, +0.001983].
- hole_mae: +0.002632, interval [+0.002482, +0.002779].
- visible_mae: -0.000027, interval [-0.000034, -0.000022].

### resshift/weighted-minus-generic-6000

- full_face_lpips: -0.029492, interval [-0.031297, -0.027598].
- hole_mae: -0.073074, interval [-0.080043, -0.066171].
- visible_mae: +0.000015, interval [+0.000001, +0.000032].

## Interpretation

The budget comparison retains the same training and threshold-selection rules, with a new tuning-selected checkpoint and threshold for each extended run. It measures the resulting longer-training pipeline, not a fixed-threshold weight-only intervention. The soft-compositing comparison shares each checkpoint, threshold, candidate and effective mask exactly. It blends using the learned segmentation score; this is a useful simple control, but does not substitute for a reconstruction-trained alpha head or an OSOR reproduction.

A smaller tuning loss is not proof of convergence or better reconstructed faces. The complete learning histories and reconstruction contrasts above determine which first-stage conclusions survive. Data/identity and pretraining-exposure limitations remain. The original test sample is already observed evidence; future confirmatory testing needs reserved cases and frozen choices.

## Training records

| Variant | Seed | Best update | Initial tuning loss | Best tuning loss |
|---|---:|---:|---:|---:|
| generic | 17 | 6000 | 0.029542 | 0.002152 |
| generic | 29 | 5500 | 0.027991 | 0.002250 |
| generic | 43 | 6000 | 0.022326 | 0.002766 |
| preservation_weighted | 17 | 6000 | 0.032983 | 0.002098 |
| preservation_weighted | 29 | 6000 | 0.021453 | 0.001833 |
| preservation_weighted | 43 | 5500 | 0.022799 | 0.002344 |
