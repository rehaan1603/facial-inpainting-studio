# DINOv2-small geometry v4 — results

Recorded 12 October 2026. Frozen DINOv2 improved one synthetic translation control, but **accepted no genuine reference pairs under the fixed protocol**. The prerequisite for spatial-adapter training did not pass. No reconstructed images or new reconstruction metrics were produced.

## Method and input assumptions

The [prospective protocol](DENSE_DINO_GEOMETRY_PREFLIGHT_V4.md) replaced CLIP-H/14 at 224 pixels with frozen DINOv2-small at 518 pixels. The downloaded model is pinned to `facebook/dinov2-small` revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, with weights SHA256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`. No executable model code was downloaded; loading used `trust_remote_code=False`.

The eight existing training people, original four same-person references per target, masks and exclusions were retained. Every full 512x512 observation is sanitized inside the hole before resizing to 518x518 with PIL bicubic interpolation. There is no centre crop or face detector. Normalization uses the pinned ImageNet mean [.485,.456,.406] and standard deviation [.229,.224,.225]. The downloaded processor's default 224-pixel centre crop is not used.

The final normalized hidden state, excluding CLS, gives a genuine 37x37 grid of 384-dimensional patch descriptors. Coordinates are `(index + 0.5) * 512 / 37`, approximately 13.837838 pixels apart in the original image. Feature maps are not interpolated. This bundled feature-extractor replacement changes both representation and sampling resolution; it is not an isolated resolution-only comparison. Global encoder self-attention still mixes sanitized observed context, so patch descriptors are not claimed to have strictly local receptive fields.

Mutual cosine similarity >=0.6, bidirectional margin >=0.02, coordinate support, mask margin, held-out split and all RANSAC/coverage/scale/residual gates remained unchanged. The next training person's four references supply the same 32 wrong-person control pairs. The first target passed exact hidden-RGB feature invariance after sanitization.

## Genuine reference coverage

| Training identity | Mutual matches for references 0, 1, 2, 3 | Accepted references / 4 | Case usable (at least 2 references) |
|---|---|---:|---|
| 6792 | 2, 7, 6, 4 | 0 | No |
| 2250 | 17, 23, 36, 33 | 0 | No |
| 2083 | 35, 13, 30, 16 | 0 | No |
| 4164 | 5, 13, 5, 13 | 0 | No |
| 674 | 10, 14, 8, 10 | 0 | No |
| 476 | 6, 8, 5, 12 | 0 | No |
| 1281 | 48, 23, 12, 13 | 0 | No |
| 8748 | 14, 12, 9, 6 | 0 | No |

Overall, **0/32** same-person pairs passed and **0/8** cases were usable. Fourteen same-person pairs had fewer than 12 mutual matches; the other 18 failed the fixed fit/held-out/coverage gates. No case or reference was removed to improve this result.

Wrong-person acceptance was **0/32 (0%)**. This satisfies that numerical control in isolation, but does not demonstrate identity specificity when genuine reference acceptance is also zero. Facial geometry can correspond between different people; this geometry test is not an identity-verification metric. The combined prerequisite failed because real-reference coverage requires all eight cases to be usable.

## Known-shift diagnostic

These controls use translations of sanitized training observations, not additional photographs or independent subjects. Padding and propagated holes are excluded. Identity controls reuse the same feature banks and are not independent encoder replays. The known-transform error and held-out error measure different things, as described in the [CLIP diagnostic](DENSE_EQUIVARIANCE_RESULTS_V3A.md).

| Control | CLIP v3a gate passes / 8 | DINO v4 gate passes / 8 | DINO known-transform RMS |
|---|---:|---:|---|
| Identity | 8 | 8 | 0 pixels |
| +8px translation | 0 | 0 | Approximately 5.8378–8 pixels |
| +32px translation | 2 | 7 | Approximately 4.3243 pixels |

| Training identity | +8px held-out RMS | +8px gate | +32px held-out RMS | +32px gate |
|---|---:|---|---:|---|
| 6792 | 14.263717 | Fail | 10.314115 | Pass |
| 2250 | 17.675523 | Fail | 10.782357 | Pass |
| 2083 | 17.864571 | Fail | 8.344530 | Pass |
| 4164 | 20.030183 | Fail | 14.731503 | Fail |
| 674 | 18.424223 | Fail | 9.601919 | Pass |
| 476 | 19.764387 | Fail | 11.491699 | Pass |
| 1281 | 25.206835 | Fail | 2.824637 | Pass |
| 8748 | 15.887512 | Fail | 11.175062 | Pass |

RMS values are in original-image pixels, rounded to six decimal places. All 32-pixel fits approximated displacement by two patch cells (about 27.6757 pixels), leaving about 4.3243 pixels of known-transform error. Improved gate coverage therefore does not mean more exact translation estimates than CLIP's full-cell controls. The 8-pixel controls still failed for every person.

## Measured resources and decision

| Field | Measured value |
|---|---:|
| Frozen model parameters | 22,056,576 |
| Trainable parameters | 0 |
| Peak GPU allocated memory | 71,557,120 bytes |
| Peak GPU reserved memory | 79,691,776 bytes |
| Elapsed time | 16.015 seconds |
| Same-person pair acceptance | 0/32 |
| Wrong-person pair acceptance | 0/32 |
| Usable training cases | 0/8 |
| Combined prerequisite | Fail |
| Training updates / reconstruction generations | 0 / 0 |

The resource ceiling was not the blocking factor in this diagnostic. Results support investigating finer descriptor sampling, but do not show that additional GPU use alone will resolve correspondence between different poses and appearances. No gates were relaxed, no settings were selected using development reconstruction scores, and no spatial-adapter training or application promotion followed v4.

A separate v5 protocol investigates the same frozen model at 1036-pixel input with a finer actual patch grid. Its results must be evaluated separately; no v5 performance is included here. Current PSNR, SSIM, LPIPS, FaceNet and ArcFace reconstruction results remain unchanged.

## Evidence and implementation

- Complete local receipt: `outputs/spatial_reference_correspondence_v4/geometry/receipt.json`.
- Receipt SHA256: `d32343216057bceac000f3a9678589c7e1a40f1513fddc78bbc15b8e61399d04`.
- The receipt retains all 64 same/wrong-person pair outcomes, all 24 controls, model/config/preprocessor hashes, library versions, source hashes and feature-file hashes. Feature tensors remain local.
- [Extraction and evaluation runner](../scripts/audit_dino_geometry_v4.py), [native grid mapping](../scripts/dino_geometry_v4.py), [unchanged matcher and geometric fit](../scripts/dense_feature_geometry_v3.py), [grid tests](../scripts/test_dino_geometry_v4.py).
- Feature-method reference: [DINOv2 paper](https://arxiv.org/abs/2304.07193). This experiment is not a reproduction of a facial-inpainting benchmark from that paper.

