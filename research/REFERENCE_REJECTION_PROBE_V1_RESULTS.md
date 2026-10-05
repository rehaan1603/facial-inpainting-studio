# Reference rejection feasibility results

5 October 2026. Existing-dataset causal diagnostic. No adapter training or production changes.

## Completed work

Generated 24 intervention images across four observed identities and seeds 17/29/43. Replayed eight baseline images, all byte-exact. Reused 24 hash-verified baseline images and their existing metrics, giving 48 scored rows with 24 newly computed metric rows. All four three-seed comparison sheets were inspected.

The drop-first arm removes the reference whose central patch was deliberately replaced in the wrong-patch protocol. This uses known corruption location and is an oracle feasibility test. The same three-reference output is also compared with the clean four-reference baseline to expose the cost of removing valid information. The zero-reference arm zeros four latent conditioning references, preserving the author CFG null-reference cache convention. It does not remove attention tokens.

## Matched contrasts

Every identity score below uses the same available pairs for both arms. Incomplete identity detection is never filled with zero or dropped without its denominator. Higher PSNR/SSIM/identity similarity and lower LPIPS/eye LPIPS are preferred.

### drop_first/baseline/clean

| Metric | Intervention | Control | Difference | Valid pairs | Improved / worse |
|---|---:|---:|---:|---:|---:|
| psnr | 23.883503 | 23.860809 | +0.022694 | 12/12 | 8 / 4 |
| ssim | 0.635503 | 0.634808 | +0.000694 | 12/12 | 6 / 6 |
| lpips | 0.283732 | 0.277250 | +0.006482 | 12/12 | 0 / 12 |
| facenet_cosine | 0.911535 | 0.914966 | -0.003431 | 12/12 | 6 / 6 |
| arcface_conditioning_cosine | 0.843235 | 0.839602 | +0.003633 | 8/12 | 4 / 4 |
| eye_bbox_lpips | 0.128235 | 0.129011 | -0.000775 | 12/12 | 5 / 7 |

### drop_first/baseline/wrong_patch

| Metric | Intervention | Control | Difference | Valid pairs | Improved / worse |
|---|---:|---:|---:|---:|---:|
| psnr | 23.883503 | 23.846802 | +0.036701 | 12/12 | 8 / 4 |
| ssim | 0.635503 | 0.633774 | +0.001729 | 12/12 | 7 / 5 |
| lpips | 0.283732 | 0.277689 | +0.006043 | 12/12 | 1 / 11 |
| facenet_cosine | 0.911535 | 0.905445 | +0.006090 | 12/12 | 5 / 7 |
| arcface_conditioning_cosine | 0.843235 | 0.836253 | +0.006982 | 8/12 | 5 / 3 |
| eye_bbox_lpips | 0.128235 | 0.127997 | +0.000239 | 12/12 | 7 / 5 |

### zero_references/baseline/clean

| Metric | Intervention | Control | Difference | Valid pairs | Improved / worse |
|---|---:|---:|---:|---:|---:|
| psnr | 23.931416 | 23.860809 | +0.070608 | 12/12 | 7 / 5 |
| ssim | 0.639604 | 0.634808 | +0.004796 | 12/12 | 7 / 5 |
| lpips | 0.296944 | 0.277250 | +0.019695 | 12/12 | 0 / 12 |
| facenet_cosine | 0.873014 | 0.916065 | -0.043052 | 11/12 | 1 / 10 |
| arcface_conditioning_cosine | 0.817608 | 0.839602 | -0.021994 | 8/12 | 2 / 6 |
| eye_bbox_lpips | 0.150586 | 0.129011 | +0.021575 | 12/12 | 2 / 10 |

### zero_references/baseline/wrong_patch

| Metric | Intervention | Control | Difference | Valid pairs | Improved / worse |
|---|---:|---:|---:|---:|---:|
| psnr | 23.931416 | 23.846802 | +0.084615 | 12/12 | 6 / 6 |
| ssim | 0.639604 | 0.633774 | +0.005830 | 12/12 | 7 / 5 |
| lpips | 0.296944 | 0.277689 | +0.019256 | 12/12 | 0 / 12 |
| facenet_cosine | 0.873014 | 0.905757 | -0.032744 | 11/12 | 2 / 9 |
| arcface_conditioning_cosine | 0.817608 | 0.836253 | -0.018646 | 8/12 | 2 / 6 |
| eye_bbox_lpips | 0.150586 | 0.127997 | +0.022589 | 12/12 | 0 / 12 |

### drop_first/zero_references

| Metric | Intervention | Control | Difference | Valid pairs | Improved / worse |
|---|---:|---:|---:|---:|---:|
| psnr | 23.883503 | 23.931416 | -0.047914 | 12/12 | 7 / 5 |
| ssim | 0.635503 | 0.639604 | -0.004101 | 12/12 | 5 / 7 |
| lpips | 0.283732 | 0.296944 | -0.013213 | 12/12 | 11 / 1 |
| facenet_cosine | 0.912763 | 0.873014 | +0.039749 | 11/12 | 10 / 1 |
| arcface_conditioning_cosine | 0.843235 | 0.817608 | +0.025628 | 8/12 | 6 / 2 |
| eye_bbox_lpips | 0.128235 | 0.150586 | -0.022350 | 12/12 | 11 / 1 |

## Interpretation

Removing the known corrupted reference improves mean FaceNet (0.905445 to 0.911535), PSNR and SSIM compared with the misleading-reference baseline. However, LPIPS worsens in 11/12 pairs (0.277689 to 0.283732) and mean eye LPIPS also worsens (0.127997 to 0.128235). Mean FaceNet improvement does not imply consistent per-case improvement: five pairs improve and seven worsen. ArcFace has only 8/12 valid pairs. The combined quality requirement is not met.

With clean references, dropping the first photograph worsens LPIPS in all 12 pairs and reduces average FaceNet. Zeroing all reference latents improves some pixel scores but degrades identity and perceptual quality. FaceNet has 11/12 valid zero-reference comparisons; baseline means in those tables therefore differ from the full 12-output baseline mean.

Visual inspection found seed-dependent eye shape/gaze, lip, texture and color changes under all arms. Dropping a reference gives small, inconsistent visible changes; zero-reference outputs can become softer or more generic. Sharper appearance alone does not establish facial fidelity.

## Decision and remaining mechanism work

Do not train or deploy a whole-reference rejection rule on the premise that it solves the identity/perceptual trade-off. This oracle result shows that a reference containing misleading material can still supply valuable uncorrupted detail. A narrower next diagnostic is spatially restricting rejection to corrupted reference regions while preserving the rest, with a neutral-equivalence test, equal-size random-region controls and an explicitly labelled oracle-location control. Only if such a feasibility check helps should a target/reference compatibility estimator learn that localization without corruption labels or clean target information at inference.

The next mechanism must meet the original identity, perceptual and eye-region requirements across identities/seeds. Final validation, broader corruption severity, independent training seeds, source-aligned paper comparisons and unknown-client evaluation remain. Reserved final identities and additional Human Faces data were not accessed.

## Execution recovery and provenance

All image scoring completed before a NumPy integer in the summary prevented JSON export. The export was repaired by converting counts to ordinary integers. A regression test passes. Recovery reused all 48 completed metric rows after checking image hashes and row identities, rather than rerunning models or changing scores. The initial failure log and recovery log remain separately preserved in outputs. This was a summary serialization failure, not a model-generation failure.

Scripts: `evaluate_reference_rejection_v1.py`, `score_reference_rejection_v1.py`, `test_reference_rejection_summary_v1.py`. [Frozen preflight](REFERENCE_REJECTION_PROBE_V1_PREFLIGHT.md). [All results](reference_rejection_v1_results.json). Local visual sheets: `outputs/reference_rejection_v1/comparison_*_all_seeds.png`.
