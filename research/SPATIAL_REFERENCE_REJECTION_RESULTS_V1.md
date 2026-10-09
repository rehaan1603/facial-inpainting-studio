# Spatial reference rejection results

6 October 2026. Fixed development diagnostic, no training or deployment.

48 intervention outputs, 24 reused baseline outputs and eight neutral baseline replays. Fixed central-quarter rejection versus a seeded equal-area corner control. Oracle location knowledge is unavailable to an automatic client-image method.

## Interpretation and visual review

All 48 interventions and all 72 metric rows completed. All eight neutral replays are byte-exact. Four mechanism checks and two paired-statistics checks pass; 24 baseline metric dictionaries remain exactly unchanged. Each intervention made 800 modified attention calls (16 layers x 50 sampling steps), masking exactly one quarter of reference 1's tokens at each spatial resolution.

Corrupted-reference FaceNet improves on average (0.90932 to 0.91303 on 11 matched pairs), but only four pairs improve and seven worsen. LPIPS increases by 0.000911 and eye crop LPIPS by 0.005332. With clean references, FaceNet decreases and LPIPS increases. The spatial oracle does not consistently outperform the equal-area corner control: its corrupted-reference eye LPIPS is 0.133329 versus 0.126227 for the corner mask. Hard exclusion is not ready for deployment or learned-gate expansion.

FaceNet detection is unavailable for identity 263, oracle_region/seed17/wrong_patch and random_corner/seed43/wrong_patch. Consequently oracle-versus-baseline FaceNet has 11 pairs, while oracle-versus-corner has 10. Each comparison recomputes both means on its own matched subset. ArcFace retains eight valid pairs in these contrasts. Missing values were not filled or thresholds changed.

Inspected all eight local comparison sheets, covering both conditions and all three seeds for every identity. Reconstructions are sharper than degraded inputs, but eyelid/gaze, facial texture and mouth/expression differences from the original remain. Differences between policies are often subtle, and no consistently better oracle reconstruction is apparent. This visual review is descriptive, not a blinded human study.

The next mechanism question is whether rejecting attention globally across every target query removes useful central facial information. A separately fixed query-local or softer attenuation control may test that hypothesis. No hyperparameter sweep, training or new final-data evaluation was performed after observing these results. Corruption can also spread beyond the selected tokens through the frozen encoders' receptive fields, so the current test is not perfect excision of corrupt evidence.

[Verification receipt and local sheet paths](spatial_reference_rejection_v1_verification.json). Existing application and model checkpoints are unchanged.

## oracle_region versus baseline: clean

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.870604 | 23.860809 | 0.009796 | 12/12 | 7 / 5 |
| ssim | 0.635101 | 0.634808 | 0.000293 | 12/12 | 7 / 5 |
| lpips | 0.278549 | 0.277250 | 0.001299 | 12/12 | 3 / 9 |
| facenet_cosine | 0.910219 | 0.914966 | -0.004746 | 12/12 | 5 / 7 |
| arcface_conditioning_cosine | 0.841355 | 0.839602 | 0.001753 | 8/12 | 5 / 3 |
| eye_bbox_lpips | 0.130156 | 0.129011 | 0.001145 | 12/12 | 5 / 7 |

## random_corner versus baseline: clean

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.859714 | 23.860809 | -0.001095 | 12/12 | 4 / 8 |
| ssim | 0.634914 | 0.634808 | 0.000105 | 12/12 | 8 / 4 |
| lpips | 0.279079 | 0.277250 | 0.001829 | 12/12 | 2 / 10 |
| facenet_cosine | 0.918199 | 0.914966 | 0.003233 | 12/12 | 7 / 5 |
| arcface_conditioning_cosine | 0.843477 | 0.839602 | 0.003875 | 8/12 | 4 / 4 |
| eye_bbox_lpips | 0.131632 | 0.129011 | 0.002622 | 12/12 | 7 / 5 |

## oracle_region versus random_corner: clean

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.870604 | 23.859714 | 0.010891 | 12/12 | 9 / 3 |
| ssim | 0.635101 | 0.634914 | 0.000188 | 12/12 | 7 / 5 |
| lpips | 0.278549 | 0.279079 | -0.000530 | 12/12 | 8 / 4 |
| facenet_cosine | 0.910219 | 0.918199 | -0.007979 | 12/12 | 7 / 5 |
| arcface_conditioning_cosine | 0.841355 | 0.843477 | -0.002122 | 8/12 | 3 / 5 |
| eye_bbox_lpips | 0.130156 | 0.131632 | -0.001477 | 12/12 | 6 / 6 |

## oracle_region versus baseline: wrong_patch

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.864187 | 23.846802 | 0.017385 | 12/12 | 9 / 3 |
| ssim | 0.634164 | 0.633774 | 0.000389 | 12/12 | 8 / 4 |
| lpips | 0.278600 | 0.277689 | 0.000911 | 12/12 | 5 / 7 |
| facenet_cosine | 0.913034 | 0.909319 | 0.003715 | 11/12 | 4 / 7 |
| arcface_conditioning_cosine | 0.839151 | 0.836253 | 0.002897 | 8/12 | 4 / 4 |
| eye_bbox_lpips | 0.133329 | 0.127997 | 0.005332 | 12/12 | 6 / 6 |

## random_corner versus baseline: wrong_patch

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.853208 | 23.846802 | 0.006406 | 12/12 | 6 / 6 |
| ssim | 0.634320 | 0.633774 | 0.000545 | 12/12 | 8 / 4 |
| lpips | 0.278343 | 0.277689 | 0.000654 | 12/12 | 5 / 7 |
| facenet_cosine | 0.906629 | 0.910035 | -0.003406 | 11/12 | 4 / 7 |
| arcface_conditioning_cosine | 0.838965 | 0.836253 | 0.002711 | 8/12 | 5 / 3 |
| eye_bbox_lpips | 0.126227 | 0.127997 | -0.001770 | 12/12 | 7 / 5 |

## oracle_region versus random_corner: wrong_patch

| Metric | Intervention | Control | Delta | Pairs | Wins / losses |
|---|---:|---:|---:|---:|---:|
| psnr | 23.864187 | 23.853208 | 0.010979 | 12/12 | 7 / 5 |
| ssim | 0.634164 | 0.634320 | -0.000156 | 12/12 | 5 / 7 |
| lpips | 0.278600 | 0.278343 | 0.000257 | 12/12 | 5 / 7 |
| facenet_cosine | 0.914000 | 0.911985 | 0.002015 | 10/12 | 6 / 4 |
| arcface_conditioning_cosine | 0.839151 | 0.838965 | 0.000186 | 8/12 | 3 / 5 |
| eye_bbox_lpips | 0.133329 | 0.126227 | 0.007103 | 12/12 | 4 / 8 |

## Decision and limits

Complete joint screen passed: **False**. LPIPS improves in both conditions: **False**.

Per-seed regressions and missing support remain in JSON. Four observed identities do not establish population generalization. Corrupt information can spread outside the rejected token window through receptive fields. No production change follows this diagnostic. Visual inspection is recorded separately.

[Fixed protocol](SPATIAL_REFERENCE_REJECTION_PREFLIGHT_V1.md) · [Full values, pairs and bootstrap intervals](spatial_reference_rejection_v1_results.json)
