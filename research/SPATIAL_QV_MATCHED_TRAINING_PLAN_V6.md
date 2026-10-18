# Spatial capacity matched experiment plan — 19 October 2026

Status: planned; training has not started. Requires successful V6 full GPU prerequisite and exact 1024 raw/final replay, followed by implementation review and source freezing before either optimizer runs.

## One change and hypothesis
The 385-parameter shared reference scorer may lack spatial capacity to correct eye structure. Add the tested 10,240-parameter rank-4 Q/V residual at `up_blocks.1.attentions.2.transformer_blocks.1.attn1`. Its outputs affect original-mask queries only; downstream spatial changes still require visible-pixel compositing. Hypothesis: learning spatial Q/V corrections improves missing-region SSIM without losing identity. This is a hypothesis, not a result.

## Two fresh arms
Both start from immutable original 64-update scorer SHA `0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba`, identical seeded A/zero B spatial tensors and identical autograd graphs with 10,625 parameters. Control optimizer updates only the original 385; all spatial tensors must stay byte-identical, without weight decay. Candidate optimizer updates all 10,625. Compute/clear all gradients in both arms; clip only optimizer parameters to norm 1. Log both total and optimized gradient norms. Preserve the native baseline separately.

32 updates per arm; training people in existing order: 6792, 2250, 2083, 4164, 674, 476, 1281, 8748. Preserve existing masks/four references/cache/manifest. Exclude all existing evaluation identities. Training resolution 512; noise seed 20261010+update (1-based); timestep 100 first 16 updates, 300 last 16. AdamW lr1e-4, betas(.9,.999), eps1e-8, weight_decay .01. Same .25 full epsilon MSE + .75 normalized-hole epsilon MSE + .1 fully-contained masked SSIM, with identical operation order in both arms. Native UNet, VAE and adapters remain frozen. No extra loss, data or model changes.

## Required implementation
Create new training/supervisor/evaluation/scoring/report scripts from the existing masked SSIM continuation scripts; do not edit their pinned historical versions. Use the tested spatial module and robust unique-temp receipt saves. Validate exact initial weights, first predictions/losses and initial scorer gradients; control spatial weights unchanged each step; frozen native gradients absent; all losses/gradients finite; old checkpoint/source hashes unchanged. Save both scorer and spatial final checkpoints in fresh arm-specific folders, never overwrite.

Freeze all training/evaluation/report implementation hashes before optimization. Preflight must fail closed if GPU probe/replay, source pins, datasets, version or control receipts fail. 1,800 seconds and 7.8e9 CUDA allocated bytes per arm; measure actual use. Historical SSIM32 took about24minutes; two fresh arms may take roughly50minutes plus image generation/scoring. No automatic resume, quality-based early selection or extra epochs.

## Evaluation and gates
Same three development people263/6681/10149, seed17, masks/references, DDIM30, CFG3, identity scale.8, appearance.2, strength1, 1024 inference/512 export, model offload and Poisson compositing. Generate three images per new arm, retain all15 historical rows:21 total records. Never read pristine evaluation targets during generation. Score unchanged whole/hole/eye metrics, FaceNet, ArcFace, detector coverage and support matching. Inspect all raw/final outputs, not just best cases. Final visible pixels must be exact.

Primary comparison: capacity vs fresh optimizer-frozen control. Require strictly improved mean hole SSIM; hole PSNR not lower; hole/eye LPIPS not worse; mean FaceNet and ArcFace not lower; per-person identity decline no worse than .005; complete paired coverage and visual non-worsening. Retain comparisons to every historical arm; do not promote a candidate that fails current baseline quality/identity safeguards. Full-image SSIM is secondary because most pixels are copied. No test-set selection, accuracy percentage or generalization claim from three repeatedly inspected development people.

After this gate only: freeze candidate, evaluate new identities/multiple seeds and corrupted/misleading references without retuning on those outcomes. Publication readiness remains future validation work. Website defaults change only after measured benefit and regression checks.
