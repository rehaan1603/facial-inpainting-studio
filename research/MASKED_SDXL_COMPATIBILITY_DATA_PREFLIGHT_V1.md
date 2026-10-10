# Masked SDXL compatibility cache: data preflight v1

Status: source preparation only. The preparation script has not been run; no model
training, GPU work, training-photo decoding, or evaluation-photo/annotation decoding
is part of writing this plan. Training and evaluation require their own frozen
protocol. This cache provides inputs for a later pilot; it provides no evidence of
better reconstruction.

## Fixed data boundary

Use only the 16 `role=train` cases, in their existing order, from
`outputs/reference_intervention_pilot_v1/manifest.json`. Its SHA256 is
`f01f52bae44001bc9c23d36c0b4c5b48cd8e60e8f19cb1ba717be085143f117e`.
The training identities are `8570, 6792, 4410, 2250, 3930, 8173, 2083, 7506,
4625, 38, 4164, 674, 8660, 476, 1281, 8748`. Each has one existing clean HQ
target and exactly four existing real HQ references. Preserve the target/reference
order and IDs. Do not use the old manifest's degradation or reference corruption.

Exclude all manifest `excluded_identities`, all `adapter_holdout` identities, and
explicitly `263, 6681, 10149, 7400`. Do not substitute cases after a failure. The
source CSV and near-duplicate fingerprint file must still match the hashes frozen
in the old manifest. Verify each selected record against the original reviewed
CSV's identity, split, HQ ID, path, and source SHA. Verify all 80 source photo
hashes before any image is decoded. Require every source path to resolve inside
the configured HQ image directory; target and reference IDs and hashes must be
distinct. Existing partition and near-duplicate controls are inherited rather
than rerun using reserved photographs.

The local cache contains sensitive derived image/face features. Keep it local;
do not put its tensors, photographs, aligned crops, or embeddings in public
artifacts. Existing data, model files, checkpoints, output folders, evaluation
code, and production defaults remain preserved.

## Fixed target and mask procedure

Read `configs/local.json` for local paths. Decode only the 16 training targets and
64 training references. Targets use EXIF-transposed RGB and one Lanczos resize to
512 by 512, then the actual local Diffusers `VaeImageProcessor` with resizing
disabled converts to `[1,3,512,512]` float32 in `[-1,1]`. Its conversion divides
RGB values by 255, then multiplies by two and subtracts one.

Reuse the exact `semantic_mask` function body from the pinned
`scripts/benchmark_v2.py`, executed in an isolated namespace rather than importing
the benchmark's evaluation/backbone modules. Call it only on each selected
training target HQ ID with `family='eyes', size=512`. It unions `l_eye, r_eye,
l_brow, r_brow` annotations and fills their bounding rectangle with the existing
8-pixel padding. Record hashes/presence of those four target annotation files;
missing individual components follow the existing function's behavior, while an
empty or extreme mask stops preparation. No holdout/evaluation annotations are
opened. Record the mask's bounding rectangle, coverage, and binary-byte hash.

Set `masked_image = target_normalized * (mask < 0.5)` so missing pixels are zero
in the normalized tensor. Encode clean target and masked image separately through
the official local SDXL VAE. Use CPU float32 weights/input/posterior and
`latent_dist.sample(generator=CPU_seeded_generator)`, then multiply by the pinned
VAE `scaling_factor` (expected `0.13025`). The latent mask is nearest-neighbor
interpolation of the binary mask to `[1,1,64,64]`; no blur or latent-mask dilation.
Use independent deterministic seeds derived from the manifest seed and target HQ
ID for target and masked-image samples. Posterior and sampled values must be
finite. VAE sampling, rather than posterior mode, is frozen by this plan.

## Fixed reference and text features

For every reference, reproduce `reference_inpaint.py`: native EXIF-transposed RGB
converted to BGR; CPU InsightFace `buffalo_l`, detection/recognition only,
`ctx_id=-1`, `det_size=(640,640)`; require exactly one detected face and a finite
512-dimensional `normed_embedding`. Keep all four embeddings separate and retain
their manifest order. Cache the zero unconditional and four positive embeddings
as CPU float32 `[2,4,512]`; the later SDXL path must cast this bank to its dtype.
Never encode the clean target as a reference. A failure stops the entire cache
and remains in the receipt; do not select a face, lower a threshold, or drop a
reference.

For appearance, reproduce `run_focused_eye_appearance_v1.py`: InsightFace's
five-point `face_align.norm_crop` on BGR with `image_size=224`, convert back to RGB,
default `CLIPImageProcessor()`, and local
`CLIPVisionModelWithProjection.vision_model(..., output_hidden_states=True)` on
CPU float32, selecting `hidden_states[-2]`. Encode unconditional features by
passing `zeros_like(processed_pixels)` through that same encoder. Keep four
photographs separate; cache CPU float32 `[2,4,257,1280]`, ordered unconditional
then positive. This retains the pre-cast features; the old inference runner cast
the bank to float16 after CPU float32 computation. Do not save crop images.

Use the positive prompt from `reference_inpaint.py` verbatim:

> A realistic portrait photograph of a person, natural facial features, consistent lighting, detailed skin.

Load the two official local CLIP tokenizers/text encoders in CPU float16, as the
current baseline loads its text components in float16. Construct the existing
Diffusers inpainting pipeline with `unet=None, vae=None` solely to call its actual
`encode_prompt` method with this prompt, no classifier-free guidance, no LoRA,
no textual-inversion additions, `clip_skip=None`, and one image per prompt.
Cache positive `[1,77,2048]` and pooled `[1,1280]` float16 tensors once. The two
penultimate hidden states are concatenated (768 plus 1280 channels); pooled
features come from the second projected text encoder. Cache `[1,6]` float16 SDXL
time IDs `[512,512,0,0,512,512]` for the later direct UNet. No UNet weights are
loaded and no inference/scheduler step is performed.

This is the current local computation, not a claim of exact historical replay.
The focused appearance run recorded Diffusers 0.32.2 and Transformers 4.49.0;
the current `reference_env_v2` lock records Diffusers 0.40.0 and Transformers
5.17.0. The current named `CLIPImageProcessor` uses `TorchvisionBackend`, while
the older implementation differed. CPU and GPU float16 kernels may also differ
in rounding. Record actual
versions and implementation-source hashes and never change environments in this
preparation script.

## Asset pins and cache contract

Verify every consumed model asset against existing provenance before loading:
official SDXL revision `115134f363124c53c7d878647567d04daf26e41e` from
`research/osor_downloads.json`; FaceID Portrait revision
`43907e6f44d079bf1a9102d9a6e56aef7a219bae` from
`research/reference_adapter_provenance.json`; all existing `buffalo_l` ONNX
hashes from `research/reference_faces_provenance.json`; appearance encoder and
Plus Face adapter revision `018e402774aeeddd60609b4ecdb7e298259dc729` from the
external `eye_appearance_models_v1/receipt.json`. The adapters are hash-verified
for future consumer compatibility, not loaded. Reject extra ONNX files or a new
model-package manifest in the `buffalo_l` directory, because FaceAnalysis scans
that directory before filtering requested model tasks. The text encoders' default
`model.safetensors` aliases must hash-match their pinned `model.fp16.safetensors`
downloads. Load the pinned VAE's explicit `fp16` file into float32 weights.
Record provenance/source hashes, source-file hashes, component implementation
hashes, local version/CPU/device facts, dtype/shape rules, and elapsed stages.

Create only `outputs/masked_sdxl_compatibility_pilot_v1/cache`, failing if it
already exists. Save `text_conditioning.pt`, exactly 16 per-identity tensor files,
and an atomic `receipt.json`. Receipts start as running, record each completed
case, and finish as complete only when counts, shapes, finite values, bank
ordering, and hashes pass. On any exception, record the failure and preserve
partial files; never automatically resume or overwrite a partial cache. Later
training must reject a receipt that is not complete and hash-verify all its files.
No image previews, evaluation metrics, checkpoints, generated references, or
training updates are created.

Consumer schema: `receipt.common_condition.path` identifies the shared tensor
file with `prompt_embeds`, `pooled_prompt_embeds`, and `add_time_ids`.
Each `receipt.records` entry has `identity`, `tensor_path`, `sha256`, source
target/reference records, and `visible_fraction` (one minus the 512-pixel mask
coverage) plus `latent_visible_fraction` (one minus nearest-mask coverage at 64).
Each per-identity tensor file has exactly these keys:

| Key | CPU shape | Dtype |
| --- | --- | --- |
| `target_latent` | `[1,4,64,64]` | float32 |
| `masked_latent` | `[1,4,64,64]` | float32 |
| `mask` | `[1,1,64,64]` | float32, binary; one means hole |
| `identity` | `[2,4,512]` | float32; unconditional then positive |
| `appearance` | `[2,4,257,1280]` | float32; unconditional then positive |

The receipt includes the pinned original scheduler configuration and requires
`prediction_type='epsilon'`. Later training is responsible for its own fixed
update count, noise/timestep schedule, SDXL casts, checkpoint rules, and evaluation
freeze; this preparation script sets no training hyperparameters.

## Execution and expected cost

After source review, the explicit preparation command is:

```powershell
& 'C:/Users/rehaa/.cache/facial-inpainting/reference_env_v2/Scripts/python.exe' scripts/prepare_masked_sdxl_compatibility_pilot_v1.py --prepare
```

The script disables CUDA before importing ML packages and uses four Torch CPU
threads plus one OpenCV thread. Models load in separate text/VAE/vision stages
and are released between stages. The largest encoder has approximately 2.53 GB
of float32 weights; additional activation and loading memory are required.
Budget several GB of system RAM and approximately 165-175 MiB of new cache disk
space, dominated by the four-reference positive/unconditional float32 appearance
banks. Preparation includes 32 CPU VAE encodes and 128 reference-image passes
through CLIP vision (64 positive, 64 unconditional) plus CPU face analysis.
Wall time and peak RAM have not been measured; they must be recorded on the
actual run. The synthetic UNet gradient probe is not an estimate of this CPU
preparation cost.

This is a 16-identity engineering cache with one mask/target per identity, using
already-exposed project training data and pretrained components. It cannot
establish generalization, true hidden-eye recovery, or publication-scale quality.
