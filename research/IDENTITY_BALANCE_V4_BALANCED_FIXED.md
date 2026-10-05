# Balanced identity supervision: balanced_fixed

Measured development screen; one training seed, four identities and three inference seeds.

| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (n) | Eye LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline / clean | 23.860809 | 0.634808 | 0.277250 | 0.914966 | 0.839602 (8) | 0.129011 |
| baseline / wrong_patch | 23.846802 | 0.633774 | 0.277689 | 0.905445 | 0.836253 (8) | 0.127997 |
| balanced_fixed / clean | 23.861496 | 0.634790 | 0.277566 | 0.912705 | 0.838758 (8) | 0.129282 |
| balanced_fixed / wrong_patch | 23.863364 | 0.633735 | 0.277799 | 0.901731 | 0.838312 (8) | 0.130064 |

Joint promotion gate: False

Failures and incomplete support:

```json
{
  "promote": false,
  "violations": [
    "clean/17/psnr",
    "clean/17/ssim",
    "clean/17/lpips",
    "clean/17/facenet_cosine",
    "clean/29/psnr",
    "clean/29/eye_bbox_lpips",
    "clean/43/facenet_cosine",
    "wrong_patch/17/ssim",
    "wrong_patch/17/lpips",
    "wrong_patch/17/facenet_cosine",
    "wrong_patch/17/eye_bbox_lpips",
    "wrong_patch/29/facenet_cosine",
    "wrong_patch/29/eye_bbox_lpips",
    "wrong_patch/43/lpips",
    "wrong_patch/43/facenet_cosine",
    "wrong_patch/43/eye_bbox_lpips"
  ],
  "incomplete_support": [
    "clean/17/arcface_conditioning_cosine",
    "clean/29/arcface_conditioning_cosine",
    "clean/43/arcface_conditioning_cosine",
    "wrong_patch/17/arcface_conditioning_cosine",
    "wrong_patch/29/arcface_conditioning_cosine",
    "wrong_patch/43/arcface_conditioning_cosine"
  ],
  "lpips_improves_both_conditions": false
}
```

FaceNet is supervised during training. ArcFace is independent of that loss but has incomplete detection coverage. These repeated development images do not establish generalization to arbitrary client photographs. No production promotion is automatic.