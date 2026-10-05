# Balanced identity supervision: timestep_scaled

Measured development screen; one training seed, four identities and three inference seeds.

| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (n) | Eye LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline / clean | 23.860809 | 0.634808 | 0.277250 | 0.914966 | 0.839602 (8) | 0.129011 |
| baseline / wrong_patch | 23.846802 | 0.633774 | 0.277689 | 0.905445 | 0.836253 (8) | 0.127997 |
| timestep_scaled / clean | 23.859487 | 0.635006 | 0.277431 | 0.916227 | 0.837028 (8) | 0.130181 |
| timestep_scaled / wrong_patch | 23.848519 | 0.633869 | 0.278230 | 0.904139 | 0.840036 (8) | 0.126473 |

Joint promotion gate: False

Failures and incomplete support:

```json
{
  "promote": false,
  "violations": [
    "clean/17/lpips",
    "clean/17/facenet_cosine",
    "clean/29/lpips",
    "clean/29/eye_bbox_lpips",
    "clean/43/psnr",
    "clean/43/lpips",
    "clean/43/facenet_cosine",
    "clean/43/eye_bbox_lpips",
    "wrong_patch/17/lpips",
    "wrong_patch/29/ssim",
    "wrong_patch/29/lpips",
    "wrong_patch/29/facenet_cosine",
    "wrong_patch/29/eye_bbox_lpips",
    "wrong_patch/43/psnr",
    "wrong_patch/43/lpips",
    "wrong_patch/43/facenet_cosine"
  ],
  "incomplete_support": [
    "clean/17/arcface_conditioning_cosine",
    "clean/29/arcface_conditioning_cosine",
    "clean/43/arcface_conditioning_cosine",
    "wrong_patch/17/facenet_cosine",
    "wrong_patch/17/arcface_conditioning_cosine",
    "wrong_patch/29/arcface_conditioning_cosine",
    "wrong_patch/43/arcface_conditioning_cosine"
  ],
  "lpips_improves_both_conditions": false
}
```

FaceNet is supervised during training. ArcFace is independent of that loss but has incomplete detection coverage. These repeated development images do not establish generalization to arbitrary client photographs. No production promotion is automatic.
## Single-change comparison against v4 fixed weight

| Condition | Metric | Candidate minus fixed | Paired identities |
|---|---|---:|---:|
| clean | psnr | -0.002009 | 4 |
| clean | ssim | 0.000215 | 4 |
| clean | lpips | -0.000135 | 4 |
| clean | facenet_cosine | 0.003522 | 4 |
| clean | arcface_conditioning_cosine | -0.001814 | 3 |
| clean | eye_bbox_lpips | 0.000899 | 4 |
| wrong_patch | psnr | -0.014845 | 4 |
| wrong_patch | ssim | 0.000134 | 4 |
| wrong_patch | lpips | 0.000431 | 4 |
| wrong_patch | facenet_cosine | -0.001680 | 4 |
| wrong_patch | arcface_conditioning_cosine | 0.001310 | 3 |
| wrong_patch | eye_bbox_lpips | -0.003591 | 4 |