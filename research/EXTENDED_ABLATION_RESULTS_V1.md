# Extended evaluation and ablations — development results

All 144 outputs (72 generated candidates) are scored across twelve identity labels. Rows and compositors are correlated, not independent subjects. ArcFace uses the conditioning checkpoint; FaceNet is the separate evaluator.

| Configuration | FaceNet | ArcFace* | NIQE ↓ | BRISQUE ↓ | SSIM ↑ | Hole PSNR ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| refs4_scale0.0_strength1.0_hard | 0.4257 (11/12) | 0.5651 (11/12) | 4.0158 (12/12) | 10.9061 (12/12) | 0.9509 (12/12) | 12.3156 (12/12) | 0.0647 (12/12) |
| refs4_scale0.0_strength1.0_poisson | 0.4827 (11/12) | 0.5795 (11/12) | 4.1977 (12/12) | 10.6332 (12/12) | 0.9635 (12/12) | 17.3043 (12/12) | 0.0341 (12/12) |
| refs4_scale0.0_strength0.99_hard | 0.4454 (11/12) | 0.5557 (12/12) | 4.0296 (12/12) | 11.0542 (12/12) | 0.9573 (12/12) | 14.8020 (12/12) | 0.0555 (12/12) |
| refs4_scale0.0_strength0.99_poisson | 0.4942 (11/12) | 0.5663 (12/12) | 4.1861 (12/12) | 10.8449 (12/12) | 0.9651 (12/12) | 17.8444 (12/12) | 0.0325 (12/12) |
| refs4_scale0.8_strength1.0_hard | 0.6085 (11/12) | 0.6783 (11/12) | 3.9791 (12/12) | 10.8347 (12/12) | 0.9502 (12/12) | 12.4916 (12/12) | 0.0619 (12/12) |
| refs4_scale0.8_strength1.0_poisson | 0.6596 (11/12) | 0.6869 (11/12) | 4.1805 (12/12) | 10.6697 (12/12) | 0.9620 (12/12) | 17.1730 (12/12) | 0.0329 (12/12) |
| refs4_scale0.8_strength0.99_hard | 0.6378 (11/12) | 0.6668 (12/12) | 3.9936 (12/12) | 10.9113 (12/12) | 0.9562 (12/12) | 14.7687 (12/12) | 0.0545 (12/12) |
| refs4_scale0.8_strength0.99_poisson | 0.6588 (11/12) | 0.6818 (12/12) | 4.1622 (12/12) | 10.7342 (12/12) | 0.9629 (12/12) | 17.1960 (12/12) | 0.0327 (12/12) |
| refs1_scale0.8_strength0.99_hard | 0.6002 (11/12) | 0.6526 (12/12) | 3.9803 (12/12) | 11.0641 (12/12) | 0.9554 (12/12) | 14.3373 (12/12) | 0.0559 (12/12) |
| refs1_scale0.8_strength0.99_poisson | 0.6125 (11/12) | 0.6548 (12/12) | 4.1546 (12/12) | 10.9078 (12/12) | 0.9621 (12/12) | 16.7277 (12/12) | 0.0339 (12/12) |
| refs2_scale0.8_strength0.99_hard | 0.6249 (11/12) | 0.6680 (12/12) | 3.9925 (12/12) | 10.9753 (12/12) | 0.9563 (12/12) | 14.8386 (12/12) | 0.0545 (12/12) |
| refs2_scale0.8_strength0.99_poisson | 0.6482 (11/12) | 0.6769 (12/12) | 4.1771 (12/12) | 10.7917 (12/12) | 0.9628 (12/12) | 17.1642 (12/12) | 0.0327 (12/12) |

*ArcFace is a conditioning-encoder diagnostic, not independent identity verification. Cosine similarities are not recognition percentages. NIQE and BRISQUE are full-image statistical quality proxies; neither establishes facial correctness.

## Paired FaceNet ablations

| A minus B | Paired n | Difference | Descriptive 95% interval | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| refs4_scale0.8_strength1.0_hard minus refs4_scale0.0_strength1.0_hard | 11 | +0.1827 | [0.0869, 0.3018] | 0.0010 | 0.1533 |
| refs4_scale0.8_strength1.0_poisson minus refs4_scale0.0_strength1.0_poisson | 11 | +0.1769 | [0.0873, 0.2866] | 0.0020 | 0.2891 |
| refs4_scale0.8_strength0.99_hard minus refs4_scale0.0_strength0.99_hard | 11 | +0.1924 | [0.0780, 0.3087] | 0.0146 | 1.0000 |
| refs4_scale0.8_strength0.99_poisson minus refs4_scale0.0_strength0.99_poisson | 11 | +0.1646 | [0.0492, 0.2903] | 0.0264 | 1.0000 |
| refs4_scale0.0_strength0.99_hard minus refs4_scale0.0_strength1.0_hard | 11 | +0.0196 | [-0.0560, 0.0939] | 0.6309 | 1.0000 |
| refs4_scale0.0_strength0.99_poisson minus refs4_scale0.0_strength1.0_poisson | 11 | +0.0115 | [-0.0970, 0.1115] | 0.8398 | 1.0000 |
| refs4_scale0.0_strength1.0_poisson minus refs4_scale0.0_strength1.0_hard | 11 | +0.0569 | [0.0001, 0.1119] | 0.0918 | 1.0000 |
| refs4_scale0.0_strength0.99_poisson minus refs4_scale0.0_strength0.99_hard | 11 | +0.0488 | [0.0066, 0.0904] | 0.0566 | 1.0000 |
| refs4_scale0.8_strength0.99_hard minus refs4_scale0.8_strength1.0_hard | 11 | +0.0293 | [-0.0296, 0.1013] | 0.4658 | 1.0000 |
| refs4_scale0.8_strength0.99_poisson minus refs4_scale0.8_strength1.0_poisson | 11 | -0.0008 | [-0.0640, 0.0776] | 0.9844 | 1.0000 |
| refs4_scale0.8_strength1.0_poisson minus refs4_scale0.8_strength1.0_hard | 11 | +0.0511 | [0.0206, 0.0807] | 0.0098 | 1.0000 |
| refs4_scale0.8_strength0.99_poisson minus refs4_scale0.8_strength0.99_hard | 11 | +0.0210 | [-0.0011, 0.0430] | 0.1064 | 1.0000 |
| refs2_scale0.8_strength0.99_hard minus refs1_scale0.8_strength0.99_hard | 11 | +0.0247 | [-0.0227, 0.0786] | 0.4268 | 1.0000 |
| refs2_scale0.8_strength0.99_poisson minus refs1_scale0.8_strength0.99_poisson | 11 | +0.0357 | [-0.0074, 0.0919] | 0.2158 | 1.0000 |
| refs4_scale0.8_strength0.99_hard minus refs1_scale0.8_strength0.99_hard | 11 | +0.0376 | [-0.0118, 0.0988] | 0.2617 | 1.0000 |
| refs4_scale0.8_strength0.99_poisson minus refs1_scale0.8_strength0.99_poisson | 11 | +0.0463 | [-0.0150, 0.1184] | 0.2422 | 1.0000 |
| refs4_scale0.8_strength0.99_hard minus refs2_scale0.8_strength0.99_hard | 11 | +0.0129 | [-0.0123, 0.0368] | 0.3428 | 1.0000 |
| refs4_scale0.8_strength0.99_poisson minus refs2_scale0.8_strength0.99_poisson | 11 | +0.0105 | [-0.0191, 0.0411] | 0.5205 | 1.0000 |

Holm correction covers all 180 metric/contrast tests; bootstrap intervals are descriptive and unadjusted. Full per-identity data, coverage, quality-control scores and every metric contrast are retained in JSON. Detection failures are not imputed.

## Limits

Fixed nested reference subsets confound photo count with the evidence in those photos; this is not optimal selection or learned fusion. One seed, small eye masks, twelve development identities, and unknown pretraining overlap remain limitations. The strength comparison also changes effective denoising steps. No FID, low-FAR recognition, final-test or novelty claim is supported.

## Reproduction

Run prepare_extended_evaluation.py with the project Python; run run_reference_count_ablation.py with reference_env_v2 Python; run evaluate_extended_diagnostics.py --suite original and --suite count with evaluation_env_v1 Python, then analyze_extended_ablations.py. Original local images/checkpoints and frozen manifests are required. Public hosting and app defaults are unchanged.
