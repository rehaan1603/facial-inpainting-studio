# Reference-disagreement diagnostic — 22 September 2026

Four already-observed mixed-damage cases. Four leave-one-reference-out runs per case (16 generations), same seed. All-reference restoration and two matched-coverage gates produce 12 scored images. Both gates preserve the highest-ranked 25% of masked pixels, using reference disagreement or edit magnitude. No clean target/gallery enters generation or gating. This is neither calibrated uncertainty nor unfamiliar-person validation.

| Case | Harm fraction | Disagreement AUC | Edit-magnitude AUC |
|---|---:|---:|---:|
| 1306_mixed | 0.5377038762973946 | 0.5326283594613175 | 0.5139485542349986 |
| 2790_mixed | 0.5616579127654716 | 0.5195509729617723 | 0.5173721755317096 |
| 1043_mixed | 0.6677825915714848 | 0.4171988305146494 | 0.4705702005376149 |
| 787_mixed | 0.5005271481286241 | 0.5251218875091875 | 0.5018431100384733 |

AUC describes within-image prediction of pixel-error increase. Pixels are not independent statistical units; there is no pixel-level significance claim.

| Arm | FaceNet | Hole MAE | LPIPS |
|---|---:|---:|---:|
| all_reference | 0.8725358098745346 | 0.050432011408518454 | 0.03613766096532345 |
| disagreement_gate | 0.8897046148777008 | 0.048292812313057064 | 0.04003068897873163 |
| edit_gate | 0.903730571269989 | 0.04792795913326924 | 0.053347292356193066 |

All metrics, coverage and exploratory identity-unit contrasts are in `reference_risk_results_v1.json`. No calibrated gate or website default is promoted. A gate may retain visible damage. Fresh identity-separated calibration/validation and simple-preservation controls remain necessary before a method claim.

## Decision and visual review

All four mixed-case comparison rows were visually reviewed. Both gates retain some blur and can introduce uneven texture in eyes/mouth; the edit gate more visibly retains damage in the first two cases. Disagreement is weakly predictive or worse than chance here and does not outperform edit magnitude on identity or masked error. Its lower LPIPS than the edit gate reflects a tradeoff, not overall superiority. The incremental reference-risk hypothesis is unsupported; do not label this a proven novelty or calibrated uncertainty mechanism. No generator/adapter training or website promotion follows.
