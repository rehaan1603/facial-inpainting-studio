# Dense correspondence checkpoint — 12 October 2026

The fixed resolution-only experiment is complete. Small-shift localization improved on synthetic controls, but real-reference coverage still fails. No inpainting candidate was trained, promoted or newly scored.

## Controlled comparison

Same eight training targets, known shifts, cosine/margin matcher and geometric acceptance rules. Synthetic shifted observations are diagnostic controls, not independent references or new people.

| Descriptor | Identity controls | 8px shift controls | 32px shift controls | Real reference pairs passing | Targets with >=2 references |
|---|---:|---:|---:|---:|---:|
|CLIP224,16x16 grid|8/8|0/8|2/8|0/32|0/8|
|DINO518,37x37 grid|8/8|0/8|7/8|0/32|0/8|
|DINO1036,74x74 grid|8/8|8/8|3/8|0/32|0/8|

The1036 run reduces known8px transform error to **1.0811px for every case**, versus5.84–8px at518. It worsens32px gate acceptance from7/8 to3/8, despite fitted-transform errors of2.59px for seven cases and4.04px for one. A correct approximate fitted transform does not imply that held-out correspondences are trustworthy. All negative outcomes remain in this comparison;1036 is not selected as globally better.

All32 wrong-person pairs were rejected at each dense-feature setting. With zero correct-reference coverage, this does not establish a useful identity discriminator. Gate acceptance is not a reconstruction accuracy percentage.

## Real references at1036

| Training identity | Eligible references / 4 |
|---|---:|
|6792|0|
|2250|0|
|2083|0|
|4164|0|
|674|0|
|476|0|
|1281|0|
|8748|0|

Twenty-four pairs had fewer than12 mutual confident matches. Eight had enough matches but failed fixed fit/held-out/coverage conditions. For example1281/reference0 has held-out RMS8.6px but only25% fitting inliers, so it still fails. Increasing sampling resolution does not solve reference-pose and feature-ambiguity problems under this protocol. Those causes are hypotheses; this test does not isolate them individually.

## Resources and safeguards

22,056,576 frozen DINO parameters; **zero trainable parameters**, no optimizer/backward/checkpoint. Torch SDPA, fp16, batch1. Peak allocated **120,979,456 bytes (121 MB)**; exact reserved peak and71.203-second execution are in the receipt. This is feature extraction cost, not SDXL training VRAM.

First-target finite-feature and exact hidden-pixel invariance checks passed. All eight targets, original masks, same-person references and exclusions were hash-verified. Four v5 unit tests passed, alongside five v3 and four v4 tests. An independent source audit confirmed correct patch order, coordinates and mask handling. Pretrained positional embeddings were interpolated to74x74; actual patch features were computed at1036, not upsampled from the37grid. Original512px images contain no additional photographic detail after resizing.

## Research decision

Stop this fixed feature-only sequence as specified in the [v5 protocol](DENSE_DINO_RESOLUTION_PREFLIGHT_V5.md). Preserve v3, v3a, v4 and v5 failures. Do not relax gates, choose best cases, promote synthetic controls as real-reference success or start adapter training on these correspondences. Application, generation settings, baseline, checkpoints and frozen evaluation code are unchanged. **There are no new hole SSIM, LPIPS, FaceNet or ArcFace values.**

The next research direction should address pose-dependent anatomical matching and confidence, rather than more input-resolution sweeps or more GPU allocation. A separately audited model that predicts regional correspondence from visible target context and reference structure could be trained with training-only supervision, but would require a fresh protocol, explicit pose/occlusion controls and validation of missing-region extrapolation. This is proposed work, not an implemented or validated solution.

## Evidence

- [v3 CLIP results](DENSE_FEATURE_GEOMETRY_RESULTS_V3.md)
- [known-warp diagnostic](DENSE_EQUIVARIANCE_RESULTS_V3A.md)
- [v4 native DINO results](DENSE_DINO_GEOMETRY_RESULTS_V4.md)
- [v5 complete photo-free metrics and provenance](dense_dino_geometry_v5_results.json)
- Local raw receipt/features: `outputs/spatial_reference_correspondence_v5/geometry`

Pretrained model: [DINOv2-small](https://huggingface.co/facebook/dinov2-small), pinned revision and model hash in the v4/v5 protocols. Its published model capabilities are not a performance claim for this facial-inpainting project.
