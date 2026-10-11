# Masked-eye spatial correspondence audit

11 October 2026. Audited commit `e0273d1`, branch `codex/local-studio-checkpoint`.
Completed before changing research implementation. No new quality result follows
from this audit. The original checkpoint, generator and evaluator stay frozen.

## Exact current path

`scripts/reference_inpaint.py::reconstruct` calls the installed SDXL inpainting
pipeline with an explicit mask. It detects each supplied reference independently
with frozen InsightFace buffalo_l. Four normalized 512-dimensional embeddings
form `[1,4,512]`; classifier-free guidance prepends zeros to give `[2,4,512]`.
References are neither averaged nor selected using the hidden evaluation target.

Frozen `IPAdapterFaceIDImageProjection` expands each embedding to 16 learned
2048-dimensional tokens. The bank is `[2,4,16,2048]`, flattened jointly to
64 reference tokens by attention. These learned tokens are not a 4-by-4 image
grid and must not be assigned invented eye/eyebrow coordinates.

The installed `IPAdapterAttnProcessor2_0` routes this bank through separate
`to_k_ip`/`to_v_ip` projections at 70 `attn2` processors in cross-attention down,
mid and up blocks. Text attention is separate. Actual inspected widths are
640 (10 heads) and 1280 (20 heads), with head dimension 64 and reference input
dimension 2048. Example names:

- `down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor`
- `mid_block.attentions.0.transformer_blocks.0.attn2.processor`
- `mid_block.attentions.0.transformer_blocks.9.attn2.processor`

Target spatial positions remain in the flattened UNet query grid. At 1024
inference, the VAE latent is 128-by-128; attention grids are 64-by-64 and
32-by-32. At 512 training they are 32-by-32 and 16-by-16. The nine-channel
UNet input concatenates noisy latent (4), mask (1), and masked-image latent (4).
The pipeline thus knows the hole location, but FaceID reference tokens have no
explicit correspondence to the target grid.

The preserved generator's default is CFG5. The research CFG3 baseline uses
`run_focused_eye_guidance_v1.py::GuidanceProxy`, which changes only that argument.
DDIM30, FaceID0.8, strength1, 1024 inference, 512 export and Poisson finishing
remain fixed. `reference_blending.py::harmonize_reference` copies visible pixels
exactly after compositing. The 0.961286 whole SSIM is not the eye-hole SSIM.

## Parameter and memory audit

Read-only instantiation on the meta device, followed by loading the existing
FaceID adapter on CPU, measured these parameter counts (no GPU allocation):

| Scope | Parameters |
|---|---:|
| SDXL UNet before FaceID | 2,567,478,084 |
| UNet including FaceID projection and its attention weights | 2,942,381,892 |
| FaceID projection alone, included above | 34,116,608 |
| Prior two-adapter UNet, including appearance branch | 3,366,130,756 |
| Proposed additional head gates at one 1280-channel / 20-head layer | 20 |

The new experiment would freeze all 2,942,381,892 loaded UNet/FaceID parameters.
VAE, both text encoders and InsightFace also remain frozen, with preprocessing
on CPU/offloaded. They are not included in that UNet-only count. Existing
safetensor stored-element counts are VAE83,653,863, text encoder123,060,557 and
second text encoder694,659,917; text counts include stored position-ID buffers
and are therefore not presented as exact trainable-parameter counts.

Prior completed two-branch training peaked at 7,290,989,056 allocated bytes.
The proposed single-layer, FaceID-only branch removes the appearance adapter.
A 6–7.5 decimal-GB training peak is an estimate, not a measured guarantee.
CPU feature caching and at most512 reference tokens cap additional bank storage
to about2.62 MB for float16 keys+values at width1280. Activations, allocator
reservation and desktop use still determine whether the run fits. No 1024
backward feasibility is assumed. GPU checked idle: RTX5070 Laptop,8151 MiB.

## Why previous changes were marginal or harmful

The 385-parameter `CompatibilityScorer` in
`sdxl_token_compatibility_probe_v1.py` consumes ten statistics of query/key
vectors. It has no coordinates or anatomical labels. Its second adapter
compresses CLIP features `[2,4,257,1280]` to 16 learned appearance tokens per
reference. The processor changes their logits at masked queries, but cannot
locate a corresponding iris or eyebrow in the source image. At zero scorer
weights it reproduces the neutral appearance branch, which already differs
from and underperforms CFG3 on SSIM/ArcFace.

`run_focused_eye_texture_v1.py` already tried reference eye/brow registration
using visible landmarks, then direct pixel transfer. It lowered hole SSIM
0.454540 to0.407623 and produced double brows, shifted detail and expression
errors. Its good global anchor fit did not prove hidden-eye correspondence.
The new proposal must use soft feature guidance and reliability checks rather
than repeat hard RGB copying or trust inferred hidden target landmarks.

The decoded0.1 RGB-L1 continuation worsened hole SSIM and FaceNet against the
matched epsilon continuation at about3.95 times the worker cost. More loss terms
or more GPU utilization alone is not an evidenced remedy.

## Frozen scoring definitions

Inspected `score_masked_sdxl_v1.py`, `region_metrics_v1.py` and
`extended_evaluation_metrics.py`. RGB arrays are scored at512 pixels.

- Hole PSNR: RGB MSE inside the unchanged binary mask, normalized by255,
  then minus10 log10. Perfect zero MSE is recorded with an explicit status.
- Hole SSIM: Gaussian11-by-11 windows, sigma1.5, population covariance,
  RGB[0,1]. Average only centres whose entire window lies inside the hole.
- Hole LPIPS: AlexNet LPIPS on the fixed region bounding box plus10 pixels of
  context. This is contextual crop LPIPS, not pixel-masked LPIPS.
- Anatomical eye metrics: existing annotation intersection with the hole,
  scoring only. Small regions may have no valid SSIM centres and stay null.
- FaceNet: MTCNN single-face crop160, VGGFace2 InceptionResnetV1 embedding cosine.
- ArcFace: buffalo_l normalized embedding cosine, one detected face required.
  ArcFace also conditions FaceID, so it is not an independent identity encoder.
- Missing/multiple-face detections remain explicit nulls with coverage counts.
- Visible preservation: count changed pixels against the damaged input outside
  the exact supplied mask, required zero.

## Ranked candidate mechanisms

| Rank | Mechanism | Benefit hypothesis | Risk / complexity | Extra memory |
|---|---|---|---|---|
|1|Coordinate-biased frozen UNet reference patch attention at one masked-query layer,20 zero gates|Reference tokens retain actual spatial locations and native feature channels|Moderate integration; visible-anchor errors and reference pose/gaze mismatch remain|Small capped bank; one extra local attention operation|
|2|Dense diffusion-feature correspondence with mutual matching and confidence|Could improve nonrigid alignment beyond global landmarks|Higher implementation/runtime; damaged eyes have no observed feature match|Larger feature cache and matching matrices|
|3|Regional FaceID/CLIP crop embeddings with region labels|Simple left/right region routing|Still loses within-eye spatial layout; close to already-negative compressed-token tests|Small adapter, but weaker justification|

Select rank1 as a bounded feasibility experiment, conditional on training-only
geometry reliability. This tests explicit positional reference information in
the native feature space; it does not assert correct hidden gaze is identifiable.
No clean development target, annotation or best-output selection may create
the alignment, reference selection or conditioning.

## Files and sequence

Add isolated files rather than alter hash-pinned baselines:
`scripts/spatial_reference_geometry_v1.py` for sanitized visible-anchor alignment;
`scripts/spatial_reference_attention_v1.py` for the local attention residual;
`scripts/test_spatial_reference_attention_v1.py` and `scripts/test_spatial_reference_geometry_v1.py` for invariance/neutrality;
`scripts/audit_spatial_reference_geometry_v1.py` for training-only prerequisites.
Conditional later files: `probe_spatial_reference_attention_v1.py`, feature-cache,
training and matched-evaluation runners. Reuse the existing frozen generator,
scorer and original training manifest rather than overwrite them.

One proposed hook is
`mid_block.attentions.0.transformer_blocks.0.attn1.processor`. The frozen self-
attention channel space provides spatial reference features. Capture reference
hidden states with the same frozen UNet, project with its existing K/V weights,
and attach registered coordinate metadata. Preserve FaceID/text attention.
Only masked target queries receive a bounded residual, multiplied by20 zero-
initialized head gates. Zero state must reproduce the current CFG3 baseline.

Training is conditional on CPU geometry/mechanism tests and a bounded single-
backward VRAM probe. The full protocol is in
`SPATIAL_REFERENCE_CORRESPONDENCE_PREFLIGHT_V1.md`.

## Related work checked 11 October2026

[DIFT, NeurIPS2023](https://arxiv.org/abs/2306.03881) motivates using frozen
diffusion features for correspondence; it does not validate our specific
inpainting layer or missing-eye geometry. Its [official implementation](https://github.com/Tsingularity/dift)
uses diffusion feature extraction.

[CorrFill, WACV2025](https://arxiv.org/abs/2501.02355) already uses correspondence
constraints and self-attention masking for reference inpainting. Thus explicit
correspondence alone is not a new research claim. Our proposed small face-specific
mechanism would need controlled evidence and independent evaluation before any
novelty or superiority claim. No published number is treated as a matched result.
