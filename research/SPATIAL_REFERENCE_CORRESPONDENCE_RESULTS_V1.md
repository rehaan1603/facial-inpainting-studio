# Spatial reference correspondence v1 — prerequisite results

11 October 2026. Status: isolated implementation and CPU checks complete; real-image geometry gate failed. Not installed in the application, trained or promoted.

## What changed

The audit found that the four references become 64 learned FaceID tokens without explicit anatomical coordinates. A scalar token compatibility bias cannot itself supply spatial correspondence. The selected experiment adds reference feature coordinates at one frozen SDXL self-attention layer, with only **20 trainable head gates**. It preserves existing FaceID/text conditioning and limits its residual to masked query positions. All 2,942,381,892 parameters in the audited UNet plus FaceID projection/processors remain frozen. Other encoders remain frozen too.

Implementation: `scripts/spatial_reference_attention_v1.py` and `scripts/spatial_reference_geometry_v1.py`. The feature-cache/training runner is deliberately not implemented yet: its required geometry failed. The existing generator, evaluator, checkpoints and app are unchanged.

## Measured prerequisite outcomes

The CPU audit used the same eight training identities, four supplied real references per person and original erased-eye masks. Hidden target pixels were replaced by 128 before detection. Target eye/brow landmarks were excluded from alignment; held-out visible anchors assessed fit. No development identity was used.

| Training identity | Outcome |
|---|---|
|6792|No face detected in erased target|
|2250|Pass; three eligible references|
|2083|No face detected in erased target|
|4164|Zero eligible references; held-out normalized RMS 0.07986–0.12638|
|674|No face detected in erased target|
|476|No face detected in erased target|
|1281|Pass; three eligible references|
|8748|Zero eligible references; held-out normalized RMS 0.04466–0.12256|

Only **2/8** cases pass. The fixed error bound is 0.04 face widths, with at least two eligible references per target. The four detector failures and two alignment failures remain in the denominator. These are engineering coverage results, not reconstruction accuracy. Residuals do not establish whether pose, expression or landmark error is the sole cause.

The preregistered decision is **stop before GPU training and diagnose geometry**. Do not remove failed identities or relax thresholds after observing these results. Local detailed evidence: `outputs/spatial_reference_correspondence_v1/geometry/receipt.json`; public photo-free summary: [JSON](spatial_reference_correspondence_v1_results.json).

## Tests and resource status

CPU unittest discovery passed **18 tests**: eight new attention tests, six new geometry tests and four existing spatial-reference rejection tests. Checks include zero-gate exact equality, baseline fallback, finite gate gradients, exactly 20 trainable parameters, frozen attention weights without gradients, masked-query-only changes, coordinate sensitivity, joint-token permutation invariance, hidden-pixel invariance and rejection of unsafe geometry. Actual 1280-channel attention neutrality was checked on synthetic tensors. This is not an end-to-end GPU or image-quality validation.

Expected training peak was estimated at 6–7.5 GB before execution. **Candidate GPU peak, full-model backward feasibility and generation neutrality are not measured.** No GPU feature extraction, candidate training, checkpoint or new development reconstruction was produced.

## Matched comparison status

| Method / seed 17, three development people | Hole PSNR ↑ | Hole SSIM ↑ | Hole LPIPS ↓ | FaceNet ↑ | ArcFace ↑ |
|---|---:|---:|---:|---:|---:|
|Preserved CFG3 baseline, previously recorded|17.894556|0.454540|0.178454|0.784486|0.816156|
|Spatial correspondence v1|Not generated|Not generated|Not generated|Not generated|Not generated|

Baseline three-seed mean hole SSIM is 0.433098. The baseline numbers above are existing evidence, not a fresh reproduction in this prerequisite run. Eye metrics, visible-pixel preservation and visual gaze/iris/brow assessment remain required for a future generated candidate. No improvement is claimed.

## Next experiment, before training

Investigate detector-independent matching of **visible** target and reference features, with robust held-out correspondence checks. DIFT-style frozen features are a candidate, not an established solution. This addresses the observed prerequisite failure instead of teaching the adapter incorrect eye coordinates. First write a separate protocol revision specifying feature layer, mask exclusion, mutual-match confidence and geometric validation; test all eight original cases and preserve this failed v1 result. Only then consider a bounded GPU gradient/VRAM probe and the fixed matched training/evaluation sequence. Missing-eye geometry remains uncertain even when visible anchors agree.

See [audit and ranked options](SPATIAL_CORRESPONDENCE_AUDIT_20261011.md) and [unchanged prospective protocol](SPATIAL_REFERENCE_CORRESPONDENCE_PREFLIGHT_V1.md). [DIFT](https://arxiv.org/abs/2306.03881) and [CorrFill](https://arxiv.org/abs/2501.02355) motivate correspondence research; they do not establish novelty or performance of this implementation.
