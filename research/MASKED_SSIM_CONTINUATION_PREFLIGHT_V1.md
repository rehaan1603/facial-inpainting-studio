# Direct masked structural-loss continuation v1

12 October2026. Written before implementation/probe/training. Separate experiment from the failed correspondence sequence; it does not reuse failed alignments or change their decisions.

## Audited hypothesis and single change

The current385-parameter compatibility scorer was trained with epsilon MSE; the subsequent0.1 masked RGB L1 experiment failed quality/identity gates. Neither objective directly compares local luminance, contrast and structure. Test whether a differentiable fully-contained masked SSIM term produces a useful distinct gradient and improves actual generated hole SSIM. This is a loss experiment, not a solution or claim of spatial correspondence. Limited scorer capacity and eight training identities are serious limitations.

Compare two new32-update continuations from the same immutable64-update checkpoint SHA2560f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba. Control: .25 full epsilon MSE + .75 hole epsilon MSE. Candidate: same + .1*(1 - masked SSIM). Weight.1 is fixed before execution to keep the structural penalty subordinate to denoising; no sweep or development tuning. No RGB L1, new identity/perceptual term, capacity change, reference selection or alignment is added. Zero weight is the causal ablation.

Training-only SSIM: RGB channels independently, Gaussian11x11 sigma1.5, C1=.01^2/C2=.03^2, population moments, average over channels and centres whose whole11x11 support is inside the original hole. Original VAE decode maps /2+.5 without clipping; float32 moments, variance clamped only against tiny numerical negatives. Constants use fixed data_range1. This differentiable training loss is separate from the unchanged skimage evaluator. Validate agreement on bounded uint8 examples and finite gradients/visible-pixel independence before GPU.

Same eight training identities6792,2250,2083,4164,674,476,1281,8748, original cache, masks and four references. Exclude263,6681,10149,7400 and every existing exclusion. Same385 trainable parameters, frozen3,366,130,756 backbone/adapter parameters, frozen CPU fp32 VAE83,653,863. Same512 training; fp16 UNet/bf16 forward/fp32 scorer and loss; differentiable CPU VAE path already audited. Same AdamW reset lr1e-4, betas.9/.999, eps1e-8,wd.01,clip1. Ordered eight cases repeat four times; updates1â€“16 at t100,17â€“32 at t300; noise20261010+update number, same in both arms. No early stopping on quality, no selected best checkpoint; compare final checkpoints only.

## Required stages

1. New masked-SSIM unit tests and training-only CPU data/preflight hashes.
2. Disposable GPU probe on original training case6792, t300, noise20261011: finite nonzero structural and epsilon gradients, gradient cosine/norm ratio, frozen weights without gradients, nonzero disposable update, exact old checkpoint hashes. No checkpoint/generation. Bound600seconds and7.8GB peak allocated. Restore attention flags.
3. If probe passes, sequential epsilon/candidate training, each bounded1800seconds. Preserve every trace/failure, both exact initial checkpoints and final checkpoints in new folders. Never overwrite prior experiments. GPU is used for UNet gradient computation; CPU VAE is required to fit the8GB laptop, not evidence of quality limitations.
4. Matched evaluation on same three development people263/6681/10149, seed17, fixed DDIM30/CFG3/FaceID.8/appearance.2/strength1/1024inference/512export/Poisson finishing. Reuse hash-verified CFG3, local appearance and prior64-update controls. Include an exact enabled initial-state replay, then all six new outputs. Keep old scorer/metric formulas unchanged; originals enter only scoring. Report per-person and means for hole/eye SSIM,PSNR,LPIPS,FaceNet,ArcFace,detector coverage and visible changed pixels. Inspect raw/final gaze,iris,brow and seams.

## Decision rule

Candidate mean hole SSIM must improve against BOTH current CFG3 and matched epsilon. Hole PSNR must not decline; hole/eye LPIPS must not rise; mean FaceNet/ArcFace must not decline. Every-person identity cosine must remain within.005 of each control, all3 detector pairs valid, zero final visible pixels changed, and visual artifacts must not worsen. The same gates are also reported against preserved local appearance and prior64-update controls; promotion requires all four comparisons to pass. These are scientific assessment gates, not promises. Publish all failed conditions. No promotion or multi-seed expansion after failure. A pass requires later three-seed replication before a robust claim. No whole-image SSIM is substituted for the hole metric.

New loss/probe/training/evaluation scripts use separate names and outputs; old generators, score_masked_sdxl_v1.py, region_metrics_v1.py and existing checkpoints stay byte-identical. Training and evaluation scripts must be saved and source-pinned before optimization.
