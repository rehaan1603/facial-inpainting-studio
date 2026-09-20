# Aligned local-feature fusion: development diagnostic

Four previously observed development identities, mixed damage, seeds 17 and 29. Six matched policies; 48 scored rows. Native aligned reference VAE features are injected into masked-image conditioning after the first denoising step. Global FaceID context remains identical. No trainable parameters and no reserved-final use.

| Policy | FaceNet target | Gallery | LPIPS | Hole MAE | NIQE | BRISQUE |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.6894 | 0.4877 | 0.0318 | 0.0905 | 4.6736 | 10.1050 |
| regional_global | 0.6807 | 0.4664 | 0.0316 | 0.0908 | 4.6697 | 10.1545 |
| single | 0.6925 | 0.4953 | 0.0328 | 0.0958 | 4.6580 | 10.1855 |
| equal | 0.6755 | 0.4857 | 0.0323 | 0.0947 | 4.6733 | 10.1462 |
| quality | 0.6803 | 0.4830 | 0.0326 | 0.0949 | 4.6762 | 10.1837 |
| damage | 0.6846 | 0.4892 | 0.0324 | 0.0949 | 4.6687 | 10.1616 |

Primary contrasts: damage-conditioned minus each control. Seeds are averaged within identity before uncertainty estimates; five tests use Holm correction.

| Control | Delta | 95% identity bootstrap CI | Exact p | Holm p |
|---|---:|---|---:|---:|
| global | -0.0048 | [-0.018076851963996887, 0.011861711740493774] | 0.625 | 1.000 |
| regional_global | 0.0039 | [-0.015616059303283691, 0.020875893533229828] | 0.625 | 1.000 |
| single | -0.0079 | [-0.02117672562599182, 0.005391255021095276] | 0.500 | 1.000 |
| equal | 0.0091 | [0.0054392218589782715, 0.012817516922950745] | 0.125 | 0.625 |
| quality | 0.0043 | [-0.004052862524986267, 0.015221372246742249] | 0.750 | 1.000 |

All metrics and detection coverage are retained in local_latent_results_v1.json. Four identity units cannot support precise inference (minimum two-sided exact p = 0.125). This mixed-only mean is not directly comparable to the historical three-condition mean of 0.8180. Five-point similarity alignment does not solve 3D pose, expression or occlusion. Native feature injection is out of the frozen UNet training distribution; successful execution does not prove useful conditioning.

Zero-gain passthrough and spatial injection are unit-tested; an independent full GPU zero-gain equivalence run has not yet been performed. No learned adapter or candidate reranking has been run in this phase.
