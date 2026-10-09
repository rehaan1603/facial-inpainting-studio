# Mask-conditioned reference baseline: fixed diagnostic

8 October 2026. Continue the frozen erased-region cases from `masked_task_v1`; no training or production edits.

Run existing `reference_inpaint.reconstruct` with four unchanged references, FaceID scale0.8, strength1, DDIM30, guidance5, model512, model CPU offload and hard composition. Use the same eight erased inputs, clean/wrong-patch reference conditions, and seeds17/29/43: **48 planned attempts**. Start with the seed17 clean screen, then finish the fixed grid. Reuse the runtime only to avoid repeated model loads.

Do not substitute reference photos, lower detection thresholds, remove failed identities, or choose favorable seeds. A reference-face detection failure is an explicit failed attempt. The first launch stopped at such a failure; the continuation records those failures and continues other cases without changing the generator or references. Non-reference runtime failures still stop the run.

Save raw and hard-composited outputs, model metadata, source/input hashes, timing and memory. Verify exact visible-pixel preservation. Frozen regional metrics include hole PSNR/SSIM, contextual hole/eye crop LPIPS, FaceNet and ArcFace with detection coverage. ArcFace conditions FaceID and is not an independent identity metric for this backend. FaceNet is separate from conditioning here.

Compare only matching case/seed subsets to prior methods, with each method's valid counts shown. ResShift has only seed17 and uses native256 with no references; ReF-LDM uses its own DDIM50/guidance1.5 latent-reference conditioning. These are task-capability comparisons, not reproductions of published head-to-head benchmarks.

This is a small repeated-development screen with annotation-derived masks and unresolved pretraining exposure. Generating eyes is necessary but insufficient: incorrect gaze, identity, expression or boundary artifacts remain failures to investigate. No training or website-default promotion follows automatically.

Code: [runner](../scripts/run_masked_sdxl_v1.py), [scorer](../scripts/score_masked_sdxl_v1.py). Local inputs and immutable configuration: `outputs/masked_sdxl_v1/config.json`. Dataset images stay local.
