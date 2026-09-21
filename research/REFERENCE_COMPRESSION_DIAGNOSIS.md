# Compressed-reference detection diagnosis

21 September 2026. Fixed follow-up on development identity 386, reference 1 only. No target, gallery or reserved-final photographs were used. Detector settings stayed unchanged; the seven variants were specified before detection.

| Reference variant | Detected faces | Detection score |
|---|---:|---:|
| Original prepared PNG | 1 | 0.50692 |
| Historical JPEG quality 20 | 0 | Not available |
| JPEG quality 95 | 1 | 0.51125 |
| JPEG quality 75 | 1 | 0.50882 |
| JPEG quality 50 | 0 | Not available |
| JPEG quality 35 | 0 | Not available |
| JPEG quality 20 reproduced | 0 | Not available |

The reproduced quality-20 pixels have exactly the same decoded hash as the historical failed reference. This supports compression-sensitive detection as the immediate cause of the failed group. The detector's quality dependence is observed on one reference only; this is not a general quality threshold recommendation. These scores are detector confidence, not identity similarity.

The original protocol screened the reference before compression. A future protocol must record post-degradation reference validity before expensive generation, while retaining all failed identities in coverage statistics. If a method is explicitly designed to use fewer valid references, compare that separately against a matched reference-count control. Do not substitute a clearer image, reduce the detector threshold, or rerun the failed group as though its original failures never happened.

Evidence: `reference_detection_diagnostic_v1.json`. Reproduction: `scripts/check_reference_compression.py`, in the reference runtime. The script protects its existing diagnostic folder from overwrite. All seven variants were evaluated without tuning.
