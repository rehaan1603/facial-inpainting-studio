# Reference-proxy calibration — 23 September 2026

Frozen exploratory screen on four previously observed mixed-damage development cases. First supplied reference is synthetically damaged five ways and restored using only the remaining three references. Its known original supervises a nine-coefficient spatial blending rule and a global blend control; no restoration-backbone weights are trained. No actual target truth or withheld gallery enters generation or calibration. Proxy images, fitted coefficients and output photos remain local.

The predictor sees local input/output statistics, not a target degradation label. Proxy corruption severity is fixed medium, matching the development regime; this is not demonstrated blind real-world generalization. Reusing mask coordinates across aligned photos may misalign semantic regions. Mixing weights are not probabilities of correctness.

| Arm | FaceNet | Gallery FaceNet | Hole MAE | LPIPS |
|---|---:|---:|---:|---:|
| observed | 0.8960331380367279 | 0.48017145693302155 | 0.044574823425286965 | 0.05187846440821886 |
| all_reference | 0.8725358098745346 | 0.5250631868839264 | 0.050432011408518454 | 0.03613766096532345 |
| fixed_half | 0.9218363761901855 | 0.5146265501777331 | 0.041459238962708964 | 0.036034106742590666 |
| proxy_global | 0.9223961234092712 | 0.5044932241241137 | 0.04135864480597385 | 0.042822519317269325 |
| proxy_spatial | 0.9196640104055405 | 0.5060076639056206 | 0.04109474005166564 | 0.04132761713117361 |

| Primary contrast | Identities | Delta | 95% identity bootstrap CI | Holm p |
|---|---:|---:|---|---:|
| proxy_spatial_minus_fixed_half/facenet_cosine | 4 | -0.0021723657846450806 | [-0.007434055209159851, 0.0027372539043426514] | 1 |
| proxy_spatial_minus_fixed_half/hole_mae | 4 | -0.00036449891104332714 | [-0.001142951187890144, 0.000532165294359779] | 1 |
| proxy_spatial_minus_proxy_global/facenet_cosine | 4 | -0.002732113003730774 | [-0.007409200072288513, 0.0006716549396514893] | 1 |
| proxy_spatial_minus_proxy_global/hole_mae | 4 | -0.0002639047543082154 | [-0.0006087070854201709, 7.704257747943866e-05] | 1 |

The statistical unit is identity. Four primary tests were specified before generation. All other metrics and failure coverage are retained in JSON. The historical 0.8180 aggregate is not comparable to this four-mixed-case subset. Success requires benefit beyond both fixed/global blending without material fidelity or perceptual regression; a favorable single metric is insufficient. Novelty and unfamiliar-person generalization are not established by this screen.

## Decision and visual review

All four comparison rows were visually inspected. Spatial calibration retains blurred eye/mouth details and changes their sharpness unevenly; it does not recover the clean appearance exactly. Compared with fixed blending it slightly lowers average masked MAE (0.041095 vs 0.041459), but reduces target similarity (0.919664 vs 0.921836), lowers gallery similarity (0.506008 vs 0.514627), and worsens LPIPS (0.041328 vs 0.036034). All four Holm-adjusted p-values are 1.0. The confidence intervals include zero. The progression gate is not met; no fresh-identity expansion, generator training, or website promotion is justified on this result alone.

Simple fixed blending improves identity, masked MAE and LPIPS over the unchanged input on this mixed-damage subset. That does not make blending novel or establish a universal setting: the broader partial-damage study had different tradeoffs. The useful engineering baseline is retained; the incremental spatial-calibration contribution remains unsupported.
