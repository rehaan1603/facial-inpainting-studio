# Fixed FreeU screen: audit and frozen experiment plan

10 October 2026. Complete this protocol before candidate inference. This is one isolated inference experiment on the existing three-person erased-eye development batch. Preserve all checkpoints, model assets, controls, images, masks, reference orders and prior results. No training, new weights, dataset replacement or website default change is authorized by a successful numerical screen alone.

## Audit: what currently reconstructs the missing region

The small-batch reference path is the existing pretrained **SDXL inpainting + IP-Adapter FaceID Portrait** implementation in [reference_inpaint.py](../scripts/reference_inpaint.py), using a nine-channel inpainting UNet. The pipeline receives the damaged RGB image and binary mask, encoded image latents and masked-image latents. Four real supplied reference photographs each supply a normalized 512-dimensional InsightFace `buffalo_l` recognition embedding. Each reference remains separate through the learned Portrait projection and contributes its own reference tokens; the four embeddings are not averaged into one photograph. The four-reference conditional/unconditional embedding bank has shape `[2,4,512]` and the recorded Portrait projection contributes 16 tokens per photograph, 64 per conditional branch.

The pretrained UNet, VAE, text encoders, Portrait projection/attention weights and face encoder are frozen for this experiment: no optimizer, backward pass or parameter update occurs. **Locally trainable/optimized parameters: 0.** This means inference-only use, rather than a claim that every loaded PyTorch tensor's default `requires_grad` flag has already been enumerated. The runner must disable parameter gradients explicitly or preserve the pipeline's inference-only execution. No new checkpoint is created.

The actual eye-hole inputs are already erased and are not blur/noise restoration targets. SDXL generates at 1024, after the existing generator first prepares a 512-pixel source; output is reduced to 512. Existing Poisson boundary harmonization then operates on the generated pixels and the damaged observation. A final exact composition preserves every pixel outside the submitted mask. Both raw and finished outputs are retained. Poisson processing can alter colour and contrast inside the hole, so its benefit must not be attributed to new learned anatomy.

Portrait identity embeddings do not explicitly retain iris RGB texture, gaze, eyebrow geometry or target-reference pixel correspondence. The higher whole-face identity score and hole SSIM therefore do not establish that these missing details are correct. Changing all conditioning, sampling and finishing settings together would prevent causal comparison.

## Current matched baseline and SSIM pitfalls

Frozen identities: **263, 6681, 10149**. Fixed seeds: **17, 29, 43**. Four clean real references per identity, unchanged order. Erased-eye masks, DDIM 30 steps, FaceID scale 0.8, strength 1, 1024 inference / 512 export and Poisson finish are identical within the existing CFG 5 versus CFG 3 comparison. The original unoccluded target is evaluation-only.

Saved nine-pair results from `outputs/focused_eye_guidance_replication_v2/summary.json`:

| Metric | CFG 5 preserved control | CFG 3 current comparison |
|---|---:|---:|
| Whole-image SSIM | 0.958459 | 0.959748 |
| Hole SSIM | 0.414146 | 0.433098 |
| Anatomical-eye SSIM, diagnostic | 0.105131 | 0.134083 |
| Hole PSNR | 17.017847 | 17.821481 |
| Hole contextual LPIPS | 0.201441 | 0.191851 |
| Eye contextual LPIPS | 0.159577 | 0.144985 |
| FaceNet cosine | 0.769196 | 0.798866 |
| ArcFace conditioning-family cosine | 0.760720 | 0.785079 |

These are nine seed/identity pairs from three repeatedly examined people, not nine independent identities. Person 10149's three-seed FaceNet mean drops from 0.785449 at CFG 5 to 0.771830 at CFG 3 despite improved hole SSIM. This existing trade-off must remain visible.

The evaluator remains [score_masked_sdxl_v1.py](../scripts/score_masked_sdxl_v1.py), [region_metrics_v1.py](../scripts/region_metrics_v1.py) and [extended_evaluation_metrics.py](../scripts/extended_evaluation_metrics.py):

- RGB PSNR uses normalized squared error, with hole and whole-image values kept separate. Exact zero error retains the explicit infinite-status marker.
- SSIM uses RGB channels, Gaussian weighting, sigma 1.5, data range 1, sample-covariance disabled and an 11-pixel window. Regional SSIM averages only centres whose full 11-by-11 window lies inside that region. Mask geometry and the erosion rule remain unchanged.
- Whole-image SSIM near 0.96 is dominated by preserved visible pixels; it does not mean 96% of erased eye detail is accurately recovered. Hole SSIM includes skin and brows in the hole, so it may improve while anatomical eyes remain poor.
- Existing anatomical-eye masks come from fixed evaluation annotations and are intersected with the hole. For person 6681 the left-eye mask has only **four** valid SSIM centres and the right eye has 62; report these counts and avoid treating the tiny left-eye estimate as a robust scalar. There is no dedicated iris-colour, gaze or eyebrow-fidelity metric in the frozen primary scorer.
- LPIPS is frozen AlexNet on RGB in [-1,1]. Hole and eye LPIPS are shared bounding-box crops with ten pixels of context, without resizing. These are contextual crop measures, not pixel-masked LPIPS. A joint eyes crop also includes the space between the eyes.
- FaceNet is a separately conditioned VGGFace2 recognition model using standardized MTCNN 160-pixel crops. ArcFace uses the same `buffalo_l` recognition family that supplies generation identity conditioning; it is useful but not an independent identity benchmark for this backend. Exactly one detected face is required. Missing or multiple detections remain null with coverage, not zero or selectively removed.
- Compare identical valid candidate/control pairs, retaining all attempted failures. Reused score caches cannot establish freshness if scorer/model/input provenance changes; use a new output/scoring directory, frozen hashes and the same evaluation environment. Do not alter the evaluator merely to raise SSIM.

The clean targets and annotation components may be opened by the evaluator, never by the candidate generation path. Do not choose prompts, seeds, references, local alignment transforms or FreeU factors using the hidden target pixels. Visual comparison with the target is a diagnostic after generation, not a candidate selector.

## Previous interventions already tried

Do not combine or silently promote these failed candidates:

| Isolated change | Observed result |
|---|---|
| Reference Plus Face appearance branch at fixed 0.2, CFG 5 | Three-person seed-17 hole SSIM fell 0.428353 to 0.423499 despite improved mean LPIPS/identity. |
| Same appearance branch added to CFG 3 | Hole SSIM fell 0.454540 to 0.441567 and ArcFace fell 0.816156 to 0.810635 versus CFG 3 alone. |
| Visible-anchor registered reference eye/brow RGB texture | All six primary mean metrics worsened: hole SSIM 0.454540 to 0.407623. Visual double brows, clipped/displaced eye details and expression mismatch were recorded. |
| Euler scheduler at CFG 5 | Recorded identity regressions; not selected. |

Sources: [appearance](FOCUSED_EYE_APPEARANCE_RESULTS_V1.md), [factorial](FOCUSED_EYE_FACTORIAL_RESULTS_V1.md), [texture](FOCUSED_EYE_TEXTURE_RESULTS_V1.md), [guidance replication](FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md). FreeU is separate from each of these mechanisms.

## One intervention: fixed official FreeU recipe

**Hypothesis:** rebalancing SDXL decoder backbone and skip features may reduce unwanted high-frequency facial artifacts while retaining structural information, improving hole SSIM without losing identity. This is a testable inference hypothesis; it does not create missing reference spatial correspondence or guarantee iris/gaze recovery.

Use the locally installed Diffusers FreeU implementation on the existing pipeline. Fixed factors, taken from the **author repository's SDXL section**:

```python
pipeline.enable_freeu(b1=1.3, b2=1.4, s1=0.9, s2=0.2)
```

Do not sweep these factors or choose them per person. FreeU rescales backbone features and filters skip features in selected decoder stages. The installed library implementation must be inspected and its version/source hash recorded before inference; this is not a claim of byte-identical implementation to the paper's reference code.

Primary sources checked 10 October 2026: [FreeU paper](https://arxiv.org/abs/2309.11497), [author code and SDXL factors](https://github.com/ChenyangSi/FreeU#sdxl), [Diffusers integration](https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu). The current Diffusers guide also shows a different community SDXL factor set; this experiment deliberately fixes the author README's 1.3/1.4/0.9/0.2 recipe. Neither source establishes masked facial inpainting superiority for our inputs.

Everything else stays fixed at the existing **CFG 3** control: same inputs/reference order, DDIM 30, scale 0.8, strength 1, mask, 1024/512 resolutions, generator seed, precision, CPU-offload mode, VAE tiling, prompts and Poisson composition. FreeU OFF must be replayed first and match the saved person-263 seed-17 CFG 3 raw and finished output pixels exactly. Abort before candidate generation on replay mismatch. A library update, scheduler replacement, second adapter or new sampling budget is a separate experiment.

## Files, compute and execution order

Create a fresh runner/scorer/report/verifier scoped to FreeU, wrapping the existing frozen generator and evaluator. Suggested names: `scripts/run_focused_eye_freeu_v1.py`, `scripts/score_focused_eye_freeu_v1.py`, `scripts/report_focused_eye_freeu_v1.py` and `scripts/verify_focused_eye_freeu_v1.py`. Save under new `outputs/focused_eye_freeu_v1/` with fail-if-exists behavior. Preserve original source checkpoints and folders. The run config must include hashes of this protocol, generator, wrappers, installed FreeU source, scheduler config, model/adapter assets, case manifest and each input/mask/reference.

1. Inspect the installed FreeU implementation and API; confirm its use does not replace the reference attention processors or reference projection.
2. Set FreeU OFF, replay person 263 / seed 17; compare both raw and finished pixels to the preserved CFG 3 output, then record the comparison.
3. Freeze the config and run exactly three FreeU ON candidates: all three fixed identities at seed 17. Save every attempted result, timings, peak memory, raw output, finished output and metadata; no retries with altered settings after seeing scores.
4. Free inference memory, then score through the frozen CPU evaluation environment and record six-metric gates and all detector records. Build equally sized original/control/candidate views and inspect eye geometry, iris/gaze, brows, colour and seams.
5. Only if the preregistered seed-17 numerical, per-person identity and visual gates below pass, repeat the same fixed FreeU factors on all three identities at seeds 29 and 43. Preserve reused CFG 3 and CFG 5 controls after finished/raw hash checks. Complete all six replication outputs; do not stop after favorable seeds.

Initial budget: one neutral replay plus three candidate GPU calls. Conditional replication: six more calls. No gradient graph, optimizer state or training checkpoint is needed. Existing 30-step runs took about 30 seconds per image after initial loading, with peak allocated tensor memory **6,535,683,072 bytes (about 6.09 GiB)**. FreeU adds filtering/rescaling, so measure actual runtime and peak allocation rather than promising no overhead from paper claims. Existing model CPU offload uses the GPU for denoising while keeping the laptop within memory limits. Concurrent GPU jobs risk contention; execute this batch serially and preserve the local website session.

Runtime separation remains: generation in `cache/reference_env_v2/Scripts/python.exe`; scoring in `cache/evaluation_env_v1/Scripts/python.exe`, as configured in `configs/local.json`. The original generation and evaluation environments remain unchanged.

## Frozen evaluation and decision gates

Report contrasts against **both** the current CFG 3 control and preserved CFG 5 baseline, without mixing a seed-17 aggregate with a nine-pair aggregate. Primary contrast: FreeU ON versus FreeU OFF at CFG 3. Secondary contrast: the preserved CFG 5 baseline. Include complete per-person and per-seed tables and all individual regressions.

For conditional replication after seed 17:

- All three candidates complete, visible changed pixels are exactly zero, and valid paired FaceNet/ArcFace coverage is 3/3. Missing coverage fails the gate rather than creating a favorable subset.
- Compared with both CFG 3 and CFG 5, mean hole SSIM strictly improves; mean hole PSNR, hole contextual LPIPS, eye contextual LPIPS, FaceNet and ArcFace do not worsen. Use the existing strict gate with no revised post-result success margin.
- Each person's FaceNet and ArcFace cosine is at least its matched CFG 3 control minus **0.005**. This pragmatic development-screen allowance is fixed before results, is not a scientifically validated perceptual threshold, and does not erase the requirement that mean identity must not worsen. Disclose every delta and other per-person regressions.
- Visual gate: no new displaced/clipped/duplicated eye structure or brows, no obvious worsened gaze/iris/colour mismatch, and no new seam artifact versus the control in the equally sized comparisons. A prettier different face is not successful reconstruction.

For a fully replicated promising candidate, apply the same mean gates over all nine fixed pairs, require the per-person FaceNet loss not to exceed 0.005 on three-seed means and report per-seed outcomes. Also check whether identity has been preserved against CFG 5, particularly person 10149 whose current CFG 3 control already regressed. A candidate that only beats a weaker CFG 3 identity score must not be described as restoring the CFG 5 identity level. Exact raw/finished controls and all negative attempts stay available regardless of selection.

Anatomical eye SSIM, per-eye PSNR/SSIM/crop LPIPS, boundary diagnostics and whole-image metrics remain supplementary outputs of the frozen scorer; retain SSIM-window counts, and do not resize/expand tiny evaluation components to obtain a better value. The six-metric gate prevents improving SSIM by smoothing away detail at the cost of LPIPS or identity. Do not average identities and inference seeds as independent subjects for significance claims; three identities support descriptive development findings only.

This screen does not establish unknown-client accuracy, population performance, robustness to corrupted references or a new scientific contribution. If it fails, retain and report it; do not activate it in the website. If it passes, corrupted-reference and independently reserved identity evaluation remain separate required work. No best-seed presentation, accuracy percentage, revised evaluation protocol or automatic default promotion follows from this small batch.
