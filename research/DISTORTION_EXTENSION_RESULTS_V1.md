# Fresh development confirmation: distortion and preservation

Four additional development identities (386, 2087, 2289, 1485), six distortion types, medium central-face and severe half-face masks, seed 29. First reference recompressed at JPEG quality 20. These identities are now observed development data, never fresh final evaluation. 96 scheduled generations: 72 succeeded and 24 failed. Their 72 preservation outputs plus 48 unchanged controls yield 192 scored rows; all 240 scheduled rows retain a success/failure record.

Severity and mask shape vary together, so cross-stratum effects are not attributable solely to severity. Reference recompression is not an isolated reference-quality ablation. Each comparison holds inputs, references and seed fixed. The primary unit is identity, averaging ten partially degraded cases; removal is reported separately. Four Holm-corrected tests were specified before generation.

| Contrast | n | Delta | 95% identity bootstrap interval | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| standard_0.5_minus_standard_0.99/facenet_cosine | 3 | 0.037310132384300254 | [-0.006171724200248696, 0.0922944784164429] | 0.5 | 1 |
| standard_0.5_minus_standard_0.99/hole_mae | 3 | -0.02062681788357995 | [-0.030311813622773548, -0.007006856518137983] | 0.25 | 1 |
| preserve_0.99_minus_standard_0.99/facenet_cosine | 3 | 0.06892244219779968 | [0.006989544630050637, 0.12643153667449947] | 0.25 | 1 |
| preserve_0.99_minus_standard_0.99/hole_mae | 3 | -0.03298400423475754 | [-0.04058818389896687, -0.025696370647000283] | 0.25 | 1 |

All 60 condition/arm summaries, all metrics, coverage and unchanged-input safeguards are in distortion_extension_results_v1.json. Positive FaceNet differences and negative MAE differences favor the first arm. Four identities remain underpowered; this is confirmation of a development trend, not final evidence of superiority. Historical 0.8180 uses a different condition mixture and is not a comparable endpoint.

## Coverage and interpretation

Identity 386 failed generation in every condition because its compressed first reference had zero detected faces. The 24 failed generations and 24 dependent preservation rows remain in the ledger. No alternate reference or replacement identity was substituted. Primary complete-pair contrasts therefore contain three identities, not four. All four adjusted p-values are 1.0. Lower strength and preservation improve relative to aggressive generation descriptively, but unchanged-input comparisons still expose reduced identity similarity and increased hole error. LPIPS can improve at the same time; it must not be used alone to promote the method.
