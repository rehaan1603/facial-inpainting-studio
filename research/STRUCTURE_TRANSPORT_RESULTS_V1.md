# Structure-only reference transport: feasibility results

**Numerical progression gate: failed. Novelty is not established.**

This frozen screen transports coordinates from supplied references into fixed ResShift reconstructions. No reference texture is transferred and no new neural generation or training occurs. Four previously observed identities and seeds 17/29 are development diagnostics, not an independent or reserved-final test.

0/24 structural transforms completed; 40 control slots reused; 40/64 rows evaluated. All 40 completed outputs were separately checked for exact visible-pixel preservation. Every failed attempt and missing metric is retained. Visual review remains pending.

| Arm | FaceNet ↑ (n) | Gallery ↑ (n) | Hole MAE ↓ (n) | LPIPS ↓ (n) | Structure NME ↓ (n) |
|---|---:|---:|---:|---:|---:|
| candidate | NA (0) | NA (0) | NA (0) | NA (0) | NA (0) |
| single_reference | NA (0) | NA (0) | NA (0) | NA (0) | NA (0) |
| wrong_identity | NA (0) | NA (0) | NA (0) | NA (0) | NA (0) |
| scaffold | 0.741489 (4) | 0.423009 (4) | 0.062665 (4) | 0.025402 (4) | 0.020543 (4) |
| reference_context | 0.781441 (4) | 0.476390 (4) | 0.065988 (4) | 0.027110 (4) | 0.019812 (4) |
| reference | 0.780990 (4) | 0.480085 (4) | 0.068752 (4) | 0.028146 (4) | 0.018977 (4) |
| refface | 0.519579 (4) | 0.345113 (4) | 0.092307 (4) | 0.043791 (4) | 0.038886 (4) |
| observed | 0.480200 (3) | 0.183254 (3) | 0.197037 (4) | 0.098775 (4) | 0.069096 (2) |

Means require both seeds for each included identity. Every metric has its own explicitly reported denominator; progression checks require the same complete support of four identities. Full PSNR, SSIM, diagnostic ArcFace, NIQE, BRISQUE, visible error, coverage, and failure records are in the JSON/CSV.

| Prespecified contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |
|---|---:|---:|---|---:|
| candidate_minus_scaffold/facenet_cosine | 0 | NA | None | 1.000000 |
| candidate_minus_scaffold/hole_mae | 0 | NA | None | 1.000000 |
| candidate_minus_scaffold/structure_nme | 0 | NA | None | 1.000000 |
| candidate_minus_reference_context/facenet_cosine | 0 | NA | None | 1.000000 |
| candidate_minus_reference_context/hole_mae | 0 | NA | None | 1.000000 |
| candidate_minus_reference_context/structure_nme | 0 | NA | None | 1.000000 |
| candidate_minus_wrong_identity/facenet_cosine | 0 | NA | None | 1.000000 |
| candidate_minus_wrong_identity/hole_mae | 0 | NA | None | 1.000000 |
| candidate_minus_wrong_identity/structure_nme | 0 | NA | None | 1.000000 |

Nine primary contrasts use seed-averaged identity differences, exact two-sided sign flips and Holm correction. Four identities cannot establish superiority; the smallest attainable unadjusted exact p-value with four nonzero pairs is 0.125. Bootstrap intervals are unstable. Repeated deterministic RefFace/observation controls are not additional samples.

The structural metric uses separate InsightFace 106-point weights on the clean target and output. Only target landmarks whose rounded locations fall in the original missing mask contribute. Euclidean distances are normalized by the target detector’s eye separation; no Procrustes alignment is used. Both images must have exactly one detected face. This learned proxy shares a model vendor/detector family with the 68-point inference system and is not anatomical ground truth. Per-person coordinates and images remain local.

The machine-checked gate includes both-seed identity, gallery, pixel/perceptual, SSIM, PSNR and structure thresholds; personal-reference and single-reference ablations; complete reconstructed-arm detection/metric coverage; and exact preservation. A numerical pass alone does not authorize progression: every local comparison sheet needs separate visual review for eyes, spacing, nose, lips, teeth, expression and asymmetry. A later pass permits only a separately frozen independent confirmation plus further novelty review.

Reproduce in order after the frozen runner finishes all 64 terminal rows:

```powershell
& C:\Users\rehaa\.cache\facial-inpainting\evaluation_env_v1\Scripts\python.exe -X utf8 scripts/evaluate_structure_transport.py
& C:\Users\rehaa\.cache\facial-inpainting\evaluation_env_v1\Scripts\python.exe -X utf8 scripts/report_structure_transport.py
```

Outputs are immutable: do not rerun into an existing result directory. The scripts verify the before-generation source/model hashes, the complete comparison file, evaluation manifests, original image hashes and exact visible-pixel preservation. Eight reserved identities are excluded; no final model generation/evaluation is performed.
