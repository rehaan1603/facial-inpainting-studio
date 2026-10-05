# Author protocol audit and timestep identity experiment

4 October 2026. Status: v5 training/evaluation completed. Official benchmark download and initial metadata audit completed. No official benchmark performance claim.

## Confirmed source behavior

Checked cached author revision af6690c19fdc6421802fd7996510bcfef259bfd1. In ldm/models/diffusion/ddpm.py the decoded identity loss is multiplied by sqrt_alphas_cumprod[t]. configs/refldm.yaml enables this option and sets the author loss weight to 0.1. The code uses epsilon prediction and L2 loss. This concrete implementation is the reference for the experiment.

Our recent candidates use FaceNet rather than the author's ArcFace training loss, a fixed training face crop, a 193-parameter adapter, and a regional/global epsilon objective. They do not reproduce the author's full training procedure. Their inherited identity coefficient is 0.5408350358795648.

## Frozen v5 comparison

Single change relative to v4 balanced_fixed: effective identity weight = inherited coefficient times sqrt_alphas_cumprod[t]. Keep all 64 updates, sample/timestep ordering, initial seed, training data, optimizer, model and evaluation settings. No gradient-norm cap or reference-rejection mechanism is added in this arm. Log every timestep factor and effective weight. This tests the scaling principle, not the whole paper's loss implementation.

Evaluate 24 fresh candidate images across the existing four observed identities, seeds 17/29/43 and clean/wrong-patch references. Keep all metrics and detector failures. Compare against the archived v4 fixed-weight arm as well as the frozen local baseline. Do not use the final reserved identities.

## Official benchmark preparation

Author release: https://github.com/ChiWeiHsiao/ref-ldm/releases/tag/0.1.0

Download CelebA-Test-Ref.zip (2,036,927,061 bytes) into local cache. Keep source metadata and validate the archive before extraction. High-quality reference photos are already available locally in CelebAMask-HQ. Do not assume their filenames match until the official mapping is inspected.

The paper's settings include 100 DDIM steps and five references, compared with our 50 steps and four individually encoded references. Benchmark preparation must verify preprocessing, reference encoding, model weights and metric code. Downloading the official data alone does not establish a reproduced result.

Before evaluating any adapter, audit overlap between official benchmark identities and all local training/development/final identities. Do not call previously observed identities an untouched test set. Keep the current corruption experiment and official benchmark results separate.

## Sources

- Author paper: https://arxiv.org/html/2412.05043v1
- Author code: https://github.com/ChiWeiHsiao/ref-ldm
- Local scripts: train_identity_timestep_v5.py, evaluate_identity_timestep_v5.py, score_identity_timestep_v5.py

## Source-level reproduction checklist

| Component | Author release / paper | Current local screen | Required action |
|---|---|---|---|
| DDIM steps | Demo defaults to 50; paper reports 100 | 50 | Distinguish demo reproduction from paper protocol |
| Reference encoding | Concatenate RGB references in width, then encode | Encode each reference separately, concatenate latents | Run author encoding as its own baseline configuration |
| Numeric precision | Demo does not enable autocast | bfloat16 autocast | Measure precision sensitivity and report any resource-driven deviation |
| Output conversion | torchvision to_pil_image | Explicit multiply, round, uint8 | Keep author output conversion in author-mode baseline |
| Identity training | WebFace R50 ONNX, central 87.5% crop, area resize 112 | VGGFace2 FaceNet, fixed MTCNN target box, antialiased bilinear 160 | Treat as different losses, not equivalent identity implementations |
| Identity-loss scaling | 0.1 times sqrt_alphas_cumprod[t] | v5 inherited 0.540835 times same timestep function | This isolates scaling, not full loss reproduction |
| Backbone | EMA inference | Strict model loading and EMA copied | Check matching weights and outputs |
| Test/evaluation scripts | Cached README still lists testing inference script as unreleased; no standalone metric script found | Frozen local metrics | Avoid claiming exact published metric reproduction until preprocessing and metric definitions agree |

The benchmark archive is being prepared without decoding test images. First audit identity overlap using metadata. The official release is not automatically an untouched adapter test set because local experiments already used CelebA identities.

## Completed benchmark intake

The 2,036,927,061-byte official archive passed ZIP CRC validation. Locally computed SHA-256: e168c38e91d24dbd5f4cce12e0e115dd8a9ea641e90dd7218a00fee6cf0fad5e. The release supplied no digest, so this is a local integrity receipt, not independently authenticated content hashing.

The official mapping contains 2,533 targets across 1,723 CelebA identity labels. Against the current pilot manifest, 13 rows overlap training identities, 5 overlap development identities, and 468 touch the broader 1,206-identity exclusion list. Counts can overlap. Excluded identities are not synonymous with the eight reserved final identities.

111 rows have at least one official reference whose CelebA identity label differs from the target. This is a metadata disagreement, not proof that the photo shows the wrong person. The scan considers every listed reference, not only a first-five subset. All listed reference files exist locally. 1,968 rows have none of these initial flags; they are not yet certified untouched across the entire project history.

No benchmark image pixels have been decoded. Next: audit historical experiment exposure, resolve identifier/identity-label discrepancies, and preserve the official mapping for reproduction. Filtering flagged rows would create a separate local subset and must never be presented as the full published benchmark.

V5 results are in IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md. The candidate does not pass the joint gate. Do not expand this training simply because one identity metric improves.
