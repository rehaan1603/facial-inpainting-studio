# Matched computation-path prerequisite v3

14 October 2026. Written before implementation/execution. Supersedes no earlier results. V1 and v2 both failed native-model scorer-gradient neutrality. Original native repeat is exact, while scorer-only/all-parameter requests agree. Historical control reuse is abandoned, not retroactively approved.

## New paired control

Use the v2 numerical implementation unchanged in both arms. The control retains its seeded rank-4 A matrices and zero B matrices as frozen parameters; only the 385-parameter scorer trains. The candidate trains both scorer and 10,240 new parameters. Both execute the same attention-difference function. At initialization, original64 scorer weights, target/reference data, prediction, epsilon loss and decoded structural loss must match. The historical native model stays a separate comparator and its earlier failed gradient test remains recorded.

## Prerequisite probe

Same training-only person6792, t300, noise20261011, frozen fp16 backbone with bf16 autocast, frozen CPU fp32 VAE, FLASH/MATH backend policy. No evaluation original. First record the native gradient, then the frozen-zero matched path's prediction and epsilon/structural scorer gradients, then the trainable-zero path. Require exact predictions and initial scalar losses. Require original-scorer gradients between the matched arms to meet unchanged atol1e-7/rtol1e-4 for both losses. Report exactness/max differences and retain gradient vectors. Record the mismatch against the historical native path separately without treating it as a pass.

The candidate must deliver finite nonzero epsilon and SSIM gradients to each Q.B/V.B, permit one disposable in-memory AdamW step, and leave native UNet/VAE weights without gradients. Control adapter tensors must remain unchanged. Preserve original source/checkpoint hashes and backend flags. Fixed limit600 wall seconds and7.8e9 allocated GPU bytes. No checkpoint or quality score. Any failure stops this prerequisite and forbids training.

If the probe passes, perform the same pre-training enabled-zero1024 replay required by v1, byte/pixel exact against original64 person263/seed17 raw/final outputs. No hidden evaluation original enters generation. Only after replay passes, freeze new trainer/evaluator/report code and a fresh two-arm training plan. Both arms must train anew from original64 with the same32-update SSIM objective, eight people/schedule/optimizer from v1. The historical SSIM32 trace is contextual, not the causal control. No optimization beyond the disposable probe occurs under this prerequisite-only document. Quality gates, masks, references, metrics and protection of old artifacts remain unchanged.
