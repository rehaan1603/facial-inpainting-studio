# Reconstruction and Visible-Region Preservation under Inaccurate Facial Inpainting Masks

> Historical first-stage single-image manuscript. The original measurements and text below are retained. TRAINING_EXTENSION_RESULTS.md contains subsequent budget/compositing evidence, and REFERENCE_IMPLEMENTATION.md describes the separate reference-assisted baseline. A submission manuscript must be revised around all relevant evidence and the final supported contribution.

**Technical draft; exploratory study. Human author review and a specific venue are still required. No new-method or acceptance claim is made.**

## Abstract

Inpainting under inaccurate masks requires balancing removal of missed occlusion against changes to valid image content. We implement a local evaluation pipeline using frozen LaMa and ResShift backbones, identity-label-separated development partitions, controlled area-matched corruptions, and two capacity-matched mask-refinement controls. Six 119,057-parameter refiners are trained on reviewed CelebAMask-HQ training images with three random seeds. On 48 development identity labels, increasing the negative-region loss weight from one to four changes mean LaMa LPIPS from 0.05845 to 0.06124, hole MAE from 0.17001 to 0.18047, and visible-region MAE from 0.000143 to 0.000106. Thus the weighted control preserves visible pixels more closely while worsening completion. A separately frozen exploratory test covers 64 CelebAMask-HQ test identity labels and 64 LaPa test images with 12 photographed object cutouts. The package records source hashes, exclusions, fitted checkpoints, selection rules and raw metrics. These results characterize a trade-off under the tested budgets; they do not establish superiority, novel mask refinement, real-capture performance, or reliable recovery of identity.

## 1. Introduction

An inaccurate inpainting mask can fail in two directions. False negatives leave obstructing pixels available to the model and may preserve them in the output. False positives authorize edits to genuinely visible pixels. Optimizing a single full-image reconstruction score can hide the second error because the damaged area and the valid area are mixed. We therefore report perceptual reconstruction, true-hole reconstruction and visible-region change separately.

The study asks whether a simple learned segmentation control can improve this balance relative to mask dilation and blending. Its contributions are an auditable local experimental implementation, a controlled diagnostic matching selected corruption factors across placement locations, and a documented comparison of generic and cost-weighted refinement. They are implementation and experimental contributions; originality relative to the complete robust-mask literature has not been established.

## 2. Relation to prior work

LaMa uses large-mask inpainting with Fourier convolutions [1]. ResShift provides a diffusion restoration framework and a released face-inpainting configuration [2]. We use frozen released models through documented local wrappers; this is not a full reproduction of their published training or evaluation. VCNet already integrates mask prediction and blind completion [3], so predicting a correction mask is not a new idea. Recent effect-aware soft-alpha removal also overlaps the broader refinement setting [4]. The present binary weighted segmentation control does not reproduce or supersede those methods. The full VCNet method and training sections were subsequently inspected (CLOSEST_METHOD_UPDATE.md); no numerical reproduction was performed. The broader reviewed literature, access limitations and unresolved nearest comparisons are documented in PUBLICATION_AUDIT.md.

## 3. Problem and controls

Let x be a clean RGB target, T the true hidden-pixel mask, and o an independently sampled occluder texture. The observed image is y = (1 − T) x + T o. The model receives only y and an erroneous supplied mask M. It never receives x, T, clean parsing or a target embedding. T is available for synthetic supervision and scoring. Deployment assumes the supplied image has already been appropriately framed as a face crop.

A refiner predicts p = sigmoid(g(y, M)). It receives RGB and one mask channel at 128-square resolution. A three-level encoder-decoder uses widths 16, 32 and 64, paired 3×3 convolution/GroupNorm/SiLU blocks, average pooling and bilinear skip-connected upsampling. The network has 119,057 trainable parameters. Probabilities are bilinearly upsampled to 256 square and thresholded to yield an effective mask E. Frozen inpainting produces a candidate F(y,E), and hard compositing returns E F(y,E) + (1 − E)y. Hence pixels outside E are exactly retained before output quantization.

The training loss is region-balanced binary cross entropy: L = (mean over T of softplus(−z) + λ mean over its complement of softplus(z)) / (1 + λ). Empty-region denominators are clamped to one. Generic refinement uses λ = 1; preservation-weighted refinement uses λ = 4. Both variants see the same seed-specific sampled data and have identical architecture, optimizer and update budgets. This conventional cost weighting is an ablation, not a novel loss. Frozen threshold candidates {0.2, 0.35, 0.5, 0.65, 0.8} minimize FNR + 4 FPR on the same tuning cases for both variants, with ties favoring the larger threshold. This common threshold objective means the experiment compares training-loss weights conditional on a preservation-weighted selection rule.

Simple controls include unchanged input, supplied-mask inpainting, dilation by eight pixels, and the v2-selected ResShift dilation-12/feather-4 setting. Feathering scales a distance-transform alpha inside the expanded mask. LaMa’s tuned radius-8/hard-composite setting equals its dilation-eight control. No clean-mask oracle is treated as a deployable method.

## 4. Data and protocol

All 52,168 supplied face images were decoded and hashed. Exact-file duplicate quarantine removed 224 files. A global 64-bit pHash screen over the remaining images proposed 496 cross-split or cross-source pairs within Hamming distance six. Visual triage identified eleven near-identical-photograph pairs and one scene-sequence pair for conservative additional quarantine. The reviewed manifests retain 29,926 HQ and 21,994 LaPa images. The screen does not exhaustively detect cropped or flipped duplicates and does not recognize identities. The review was performed with assistant visual inspection and awaits independent human replication.

Training uses 24,133 reviewed HQ training images resized to 128 square. Random unions of one to three rectangles or ellipses are filled with solid colors, low-frequency random colors or transformed HQ training-image textures. Supplied masks are independently eroded/dilated and translated. AdamW uses learning rate 0.0003, default weight decay 0.01, batch size 16, mixed precision, gradient norm clipping at one, and 1,500 updates per run. Checkpoints are selected every 250 steps by their respective training objective on 48 fixed tuning examples. All six runs finish at the allocated budget; decreasing final tuning losses mean convergence is not established. Each run samples with replacement rather than guaranteeing an epoch over all source images.

The v2 development protocol contains 96 distinct HQ validation identity labels: 48 tuning and 48 assessment, excluding a previous 32-identity pilot. The area-matched diagnostic reuses the 48 assessment identity labels with three locations, three exact missing areas and four mask conditions. Areas are 1,966, 3,932 and 6,554 pixels. Under and over errors remove or add round(0.2 × missing area) pixels; the mixed condition does both. Error counts are matched across locations within a condition, not across conditions. Translated masks, error patterns and textures share canonical geometry, with margins preventing correction-mask clipping. Placement centers are eye, mouth and upper-image locations; they do not guarantee pure anatomical coverage. The changed protocol is a diagnostic on reused development identities, not an untouched test.

The new object-composite protocol was frozen in a hashed JSON file before test inference. It selects 64 HQ test identity labels and 64 LaPa test images by fixed hash ordering. HQ images are resized; LaPa faces are cropped using the 106-landmark bounding square enlarged by 1.5 before resizing. This annotation-derived framing is privileged preprocessing, not an input to mask correction, and complicates direct cross-dataset interpretation. Twelve COCO object cutouts are reused across faces, with one asset per face and four supplied-mask conditions: accurate, erosion four, dilation four and translation (6, −6). Objects are resized and placed near the crop center, using alpha ≥ 128 as an opaque silhouette. These photographed assets were held out from local refiner training; composites remain synthetic. True missing area ranges from 2.28% to 22.89%, and five LaPa crops require black padding outside the source. Object appearance, area and placement differ from development, preventing attribution to texture alone.

LPIPS uses the original AlexNet implementation [7], scored on RGB in [−1,1]. Hole and visible MAE average absolute RGB errors within T and its complement respectively. Empty auxiliary regions are reported unavailable. Percentile bootstrap intervals use 2,000 paired resamples of HQ identity labels or LaPa images, keeping associated conditions together. Three learned seeds are evaluated on LaMa development; predefined seed 17 alone is used for ResShift transfer and the object test. Intervals condition on these fitted seeds and the 12 object assets. There is no multiple-comparison correction or preset practical-effect margin; analyses are exploratory.

## 5. Results

### Controlled development comparison

| backbone | method | seeds | LPIPS | Hole MAE | Visible MAE |
|---|---|---|---:|---:|---:|
| lama | generic | 3 | 0.058448 | 0.170009 | 0.000143 |
| lama | preservation_weighted | 3 | 0.061238 | 0.180471 | 0.000106 |
| resshift | generic | 1 | 0.034955 | 0.089615 | 0.000167 |
| resshift | preservation_weighted | 1 | 0.037621 | 0.096317 | 0.000117 |
| lama | dilation 8, feather 0 | 1 | 0.041050 | 0.114483 | 0.002601 |
| resshift | dilation 8, feather 0 | 1 | 0.022025 | 0.077208 | 0.002155 |
| resshift | dilation 12, feather 4 | 1 | 0.026863 | 0.083003 | 0.003481 |

The weighted control has lower visible-region MAE and worse LPIPS and hole MAE than the generic control in the aggregate on both backbones. It does not dominate the reconstruction–preservation trade-off. Compared with fixed dilation, both learned controls preserve much more visible content while leaving substantially more hole error. This is evidence against presenting the current weighted model as an overall improvement. It does not prove that every weighting, architecture, training budget or corruption distribution will fail.

The earlier v2-selected ResShift radius-12/feather-4 setting also regresses against radius-eight/hard blending on the area-matched diagnostic. Since shape and perturbation construction changed along with matching, this reversal cannot be attributed causally to missing-area matching alone.

### Object-composite test

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

On all four dataset/backbone combinations, weighted refinement has lower aggregate visible MAE and worse aggregate LPIPS and hole MAE than generic refinement. Dilation-eight produces much lower hole error in these cases, with more visible-region change. These are descriptive aggregate directions, not a declaration of statistical or practical superiority. All frozen controls are reported, including unchanged input, so reduced visible error cannot by itself be mistaken for successful completion. Accurate-mask strata and paired confidence intervals are generated in FINAL_RESULTS.md and final_results.json. No settings are retuned after this test. Subsequent work must treat these images as observed evidence and reserve new evaluation cases.

## 6. Limitations and reproducibility

The study uses modest sample sizes, only twelve object assets, coarse polygon cutouts and synthetic occlusion. Real paired capture, human preference, independent identity-fidelity evaluation and nearest robust-mask method reproduction are absent. LaPa identity independence and all relevant backbone/evaluator pretraining memberships are unresolved. The LaMa export is an engineering checkpoint with recorded checksum, not independently established numerical equivalence to an original author model. The ResShift VAE filename is not proof of its training split. Results at 256 square and 1,500 refiner updates cannot establish general conclusions about high-resolution or converged training.

Source/configuration hashes, checkpoint hashes, reviewed manifests, masks and raw metrics are retained locally. Training and warm inference efficiency are recorded separately, with exclusions stated. Scientific tables and plots regenerate from raw files using report_final.py. Third-party images and checkpoints are not included in a public release; individual attribution and dataset permissions require author review. Sixteen analytic tests cover corruption independence, metric regions, threshold selection, mask geometry, duplicate search and refiner gradients. They validate implementation invariants, not scientific effectiveness.

## 7. Conclusion

The completed local experiment demonstrates how preservation gains can coexist with worse completion under inaccurate masks. Standard negative-region weighting does not yield an overall improvement in this development setting, and a validation-selected morphological control changes rank under a revised corruption protocol. The package provides a reproducible basis for further research. It does not yet justify a new-method IEEE submission; the next scientific advance must resolve a demonstrated limitation and survive stronger controls and independent evaluation.

## References

[1] LaMa authors. Resolution-robust Large Mask Inpainting with Fourier Convolutions, WACV 2022. Official code: https://github.com/advimman/lama

[2] Z. Yue, J. Wang and C. C. Loy. ResShift official journal-branch code and face-inpainting configuration. https://github.com/zsyOAOA/ResShift . Source commit bb03b7d21614cace01787e097c8a6ab6b945227d. Cite the final relevant article metadata before submission.

[3] Y. Wang, Y.-C. Chen, X. Tao and J. Jia. VCNet: A Robust Approach to Blind Image Inpainting, 2020. https://arxiv.org/abs/2003.06816

[4] OSOR authors. OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal, 2026 preprint. https://arxiv.org/html/2606.28094v1 . No experimental reproduction in this project.

[5] CelebAMask-HQ / MaskGAN dataset and paper, CVPR 2020. https://github.com/switchablenorms/CelebAMask-HQ ; CelebA dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

[6] LaPa dataset, associated AAAI 2020 paper. https://github.com/jd-opensource/lapa-dataset

[7] R. Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, CVPR 2018. https://github.com/richzhang/PerceptualSimilarity

[8] COCO dataset, official download and individual-image licensing metadata. https://cocodataset.org/#download ; https://cocodataset.org/#termsofuse

## Author preparation note

This draft was generated with an AI assistant. Human authors must verify the methods, citations and data permissions, add authorship and contribution information, apply the chosen venue’s template and disclosure policy, and decide whether the scientific contribution is sufficient. Nothing has been submitted.
