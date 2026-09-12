# Reference contribution: development diagnostic

All 48 candidates and 96 scored rows completed on twelve development identity labels. The same generator, seed and references were used in every paired contrast. These results do not constitute a final test, an identity-fidelity evaluation or a novel-method claim.

| Configuration | LPIPS | Hole MAE | Visible MAE | Inner boundary MAE |
|---|---:|---:|---:|---:|
| scale0.0_strength1.0_hard | 0.064723 | 0.214502 | 0.000000 | 0.179364 |
| scale0.0_strength1.0_poisson | 0.034146 | 0.090450 | 0.000000 | 0.031852 |
| scale0.0_strength0.99_hard | 0.055528 | 0.145325 | 0.000000 | 0.112950 |
| scale0.0_strength0.99_poisson | 0.032460 | 0.084491 | 0.000000 | 0.029716 |
| scale0.8_strength1.0_hard | 0.061918 | 0.205922 | 0.000000 | 0.178809 |
| scale0.8_strength1.0_poisson | 0.032886 | 0.090855 | 0.000000 | 0.031422 |
| scale0.8_strength0.99_hard | 0.054500 | 0.144198 | 0.000000 | 0.109736 |
| scale0.8_strength0.99_poisson | 0.032674 | 0.091549 | 0.000000 | 0.029592 |

Lower is better for these error measures. Exact outside-mask preservation is imposed by composition, not learned by the model.

## Paired LPIPS differences

| Contrast (A minus B) | Mean | 95% identity bootstrap interval |
|---|---:|---|
| Reference on minus off; strength 1.0; hard | -0.002806 | [-0.005373, -0.000328] |
| Reference on minus off; strength 1.0; poisson | -0.001260 | [-0.003611, 0.001056] |
| Reference on minus off; strength 0.99; hard | -0.001029 | [-0.005538, 0.002711] |
| Reference on minus off; strength 0.99; poisson | 0.000214 | [-0.004225, 0.003962] |
| Strength .99 minus 1; scale 0.0; hard | -0.009195 | [-0.013477, -0.004529] |
| Strength .99 minus 1; scale 0.0; poisson | -0.001686 | [-0.003812, 0.000446] |
| Poisson minus hard; scale 0.0; strength 1.0 | -0.030578 | [-0.038066, -0.023010] |
| Poisson minus hard; scale 0.0; strength 0.99 | -0.023069 | [-0.028768, -0.017620] |
| Strength .99 minus 1; scale 0.8; hard | -0.007418 | [-0.011563, -0.002863] |
| Strength .99 minus 1; scale 0.8; poisson | -0.000212 | [-0.002807, 0.002212] |
| Poisson minus hard; scale 0.8; strength 1.0 | -0.029032 | [-0.036561, -0.021485] |
| Poisson minus hard; scale 0.8; strength 0.99 | -0.021826 | [-0.027093, -0.016672] |

Negative differences favour A. All other paired measures are recorded in reference_diagnostic_results.json. Intervals are exploratory and not adjusted for multiple contrasts. Strength 0.99 uses 29 denoising steps versus 30 at strength 1.0.

Limitations: twelve detector-success identities, one synthetic eye mask per person, single seed, no poor-reference or inaccurate-mask strata, uncertain label/pretraining independence and no independent identity evaluator. Do not use this diagnostic to claim real-world identity recovery. See REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md and outputs/reference_diagnostics_v1/README.md.
