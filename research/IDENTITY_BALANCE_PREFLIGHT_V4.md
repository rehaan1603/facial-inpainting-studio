# Balanced identity supervision and gradient cap — v4

3 October 2026. Status: completed. See [measured results](IDENTITY_BALANCE_RESULTS_V4.md); the frozen plan below is retained.

## Frozen comparison

Keep the v3 193-parameter adapter, frozen ReF-LDM and FaceNet, fixed training face crops, 64-update sample/timestep schedule, initialization seed 20261002, AdamW learning rate 0.001 and diffusion objective. Train two new arms from neutral initialization. Both apply identity supervision on every update: four updates per each of 16 training identities.

- balanced_fixed: inherited identity weight 0.5408350358795648.
- balanced_capped: at each update use min(inherited weight, diffusion gradient norm / max(identity gradient norm, 1e-12)). Norms are detached; the identity gradient contribution cannot exceed the diffusion gradient norm before total-gradient clipping. This is an experimental threshold, not an optimized or validated choice. Opposing gradients are measured, not projected.

This pair isolates the cap under balanced supervision. Comparison with v3 also changes supervision frequency and is not a pure coverage ablation. One training seed only.

## Evaluation

Use observed development identities 263, 6681, 7400 and 10149; inference seeds 17, 29, 43; clean and wrong-person-patch references. Generate 24 fresh images per candidate. Reuse 72 prior controls per evaluation only with byte-hash verification. Repeated controls are not independent observations. Frozen metrics include PSNR, SSIM, LPIPS, FaceNet, ArcFace, eye-region LPIPS, NIQE and BRISQUE. The unchanged joint gate includes missing ArcFace support explicitly. Reserved final identities and Human Faces data remain excluded.

## Execution and provenance

Training, evaluation and scoring run sequentially with logs and failure status in outputs/identity_balance_v4. No production model replacement. Every training update records both gradient norms, effective weight and gradient cosine. Configurations record code and crop hashes; completed training records checkpoint hash. A failed stage stops the pipeline instead of silently skipping a candidate.

The v3 scoring destination was corrected: it had written its public snapshot to the v2 filename. Each public snapshot was restored from its corresponding local scores.json; original local numeric records were preserved.
