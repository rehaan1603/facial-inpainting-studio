# Extended development evaluation v1

Frozen before extended scoring and new reference-count generation, 14 September 2026.

## Population and design

Retain all twelve cases from reference_diagnostics_v1 and all 96 original v3 outputs. Add one-reference and two-reference runs using the first one/two references in the previously frozen manifest order, with scale 0.8, strength 0.99, seed 17, nominal 30 steps, 512 pixels, identical prompts, DDIM and existing frozen inference code/runtime. This creates 24 additional candidates, each scored with hard and Poisson composition. Compare against the existing four-reference condition, and retain the original zero-scale control. The manifest order was chosen before output quality was known. This ablates count for nested fixed subsets, not optimal reference selection; reference count also changes the adapter token count. Do not claim learned mask-aware fusion or ranking has been implemented.

The original strength contrast changes effective step count (29 vs 30); describe it as a combined setting contrast. Compositor comparisons share a generated candidate. No training, clean-target conditioning or candidate reranking is allowed. Generation errors must be recorded; no replacement cases or silent omissions.

## Measurements

- ArcFace-style InsightFace buffalo_l w600k_r50 cosine, original 112-pixel five-landmark alignment, CPU ONNX, detection size 640x640 and threshold 0.5, independently detect each output/target, require exactly one face. This is the SAME recognition checkpoint as conditioning: report it as a conditioning-encoder diagnostic, never an independent verifier. Preserve detector failures. FaceNet remains the independently chosen evaluator; its pretraining overlap remains unknown.
- FaceNet uses the frozen identity diagnostic v1 preprocessing and weights. Score new count outputs with the same implementation settings. Do not tune detection thresholds after seeing results.
- PyIQA 0.1.16 NIQE and BRISQUE, named `niqe` and `brisque` original defaults, RGB [0,1] full 512x512 image, CPU, no user crop/resizing. Record effective library configuration. NIQE internally uses 96-pixel blocks and crops to a block multiple; scores do not cover every border pixel. Lower is better; these natural-image statistics are secondary proxies, not identity or human preference measures. No NIQE on tiny mask crops.
- RGB SSIM: scikit-image structural_similarity, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, window size 11. Whole-image RGB PSNR and hole-only RGB PSNR (hole MSE over all masked channels), hole/visible MAE, and whole-image AlexNet LPIPS. Outside-mask preservation checked against the submitted input, not inferred from aggregate scores.
- Score clean targets and observed masked inputs as contextual quality controls. These do not represent competing reconstruction methods. Nonfinite scores and detector failures remain explicit. No FID on this tiny sample and no low-FAR verification claim.

## Analysis

All configuration means include metric-specific valid/total counts. Pair reference on/off, .99/1 settings, Poisson/hard, and 2/1, 4/1, 4/2 reference counts by identity. Use 10,000 identity bootstrap resamples, seed 20260914, percentile 95% intervals. For each metric/contrast also report two-sided exact paired sign-flip p-values (all 2^n flips, n<=12), with Holm correction across ALL reported metric/contrast tests in this extended report. These remain exploratory development analyses: post hoc metric choice and prior use of these cases prohibit confirmatory claims. Never count 144 scored rows as 144 independent identities.

## Limits and next stage

This closes metric and fixed-subset ablation gaps on a small development set. Fresh identity/gallery separation, larger/several mask locations, multiple generation seeds, external baselines, proposed selection/fusion training, and a held-out final evaluation remain separate work. Do not change website defaults automatically from these results.

Sources: https://github.com/chaofengc/IQA-PyTorch ; https://github.com/deepinsight/insightface ; https://github.com/timesler/facenet-pytorch
