# Balanced identity supervision and bounded gradients: v4

3 October 2026. Completed development experiment; production model unchanged.

| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (valid n) | Eye LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline / clean | 23.860809 | 0.634808 | 0.277250 | 0.914966 | 0.839602 (8) | 0.129011 |
| baseline / wrong_patch | 23.846802 | 0.633774 | 0.277689 | 0.905445 | 0.836253 (8) | 0.127997 |
| balanced_fixed / clean | 23.861496 | 0.634790 | 0.277566 | 0.912705 | 0.838758 (8) | 0.129282 |
| balanced_fixed / wrong_patch | 23.863364 | 0.633735 | 0.277799 | 0.901731 | 0.838312 (8) | 0.130064 |
| balanced_capped / clean | 23.865872 | 0.634648 | 0.277072 | 0.911838 | 0.837150 (8) | 0.128431 |
| balanced_capped / wrong_patch | 23.849157 | 0.633897 | 0.277997 | 0.907001 | 0.838076 (8) | 0.128413 |

## Verification

Both arms completed 64 GPU updates, with four identity-supervised updates for every training identity. Each generated 24 fresh images. All 144 reused control records match their original image hashes and metric values exactly; these represent the same 72 controls counted twice, not 144 independent samples.

- balanced_fixed: identity/reconstruction gradient ratio 0.052–55.213; joint promotion gate **False**.
- balanced_capped: identity/reconstruction gradient ratio 0.082–1.000; joint promotion gate **False**.

## Interpretation and remaining work

The fixed-versus-capped pair isolates loss weighting with identical supervision coverage. Comparison with v3 also changes identity-supervision frequency. FaceNet is supervised and is not an independent identity metric. ArcFace coverage is incomplete and remains explicit in the gate. One training seed and four repeatedly observed development identities cannot establish generalization to unknown client photographs. Full-image blur/noise restoration is tested here, not erased-eye inpainting.

Review each failed criterion before choosing the next experiment. A successful gradient cap is an optimization check, not evidence of improved visual quality. Do not replace the production model automatically. More training seeds, broader corruptions and a separately held-out assessment remain necessary after a candidate passes the development criteria.