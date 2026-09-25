# Context-support screen: visual and coverage review

Reviewed 25 September 2026 after generation, selection and metric receipts were locked. All eight local comparison sheets were inspected: four previously observed development identities, seeds 17 and 29. Clean targets were used only for this evaluation review. This is unblinded internal assessment, not a human-subject study.

| Development case | Findings across both seeds |
|---|---|
| 1306 removal | The target has an open, tooth-visible smile. Every displayed reconstruction closes the mouth; eye shape also changes. Context selection chooses radius 8 on both seeds and does not recover the missing expression. |
| 2790 removal | The target's oblique expression, lips and asymmetric eyes are altered across outputs. Radius 2 selection produces a plausible but mismatched central face. Averaging smooths features without establishing their correctness. |
| 1043 removal | Radius-zero results retain a tooth-visible smile but differ in eye/mouth geometry. Larger supports often suppress the teeth or change smile shape. Selected radius 4/8 does not match the actual expression. |
| 787 removal | Eyes, nose and lips change under every support choice. Larger supports and averages soften eye detail; plausibility does not demonstrate the hidden original. |

These are case descriptions, not blinded rankings or population conclusions. All native generations, calibration patches and exported comparisons remain local under `outputs/context_support_v1/`.

## Detection and statistical coverage clarification

All **72 reconstructed rows** have exactly one face detected by both evaluators, all three evaluation-gallery comparisons available, and unchanged outside-mask pixels. All 80 rows, including the eight damaged-input controls, complete numerical scoring. A completed score can contain an unavailable identity metric when a face detector fails; missing metrics are never zero-filled.

The damaged-input controls have six detector failures: ArcFace fails for 1306 on both seeds, and both detectors fail for 787 on both seeds. Consequently, the observed-control FaceNet mean covers **three identities**, while its ArcFace mean covers **two**; every reconstruction arm covers **four**. The observed aggregate should not be interpreted as a paired four-identity gain estimate. Primary candidate-versus-reconstruction contrasts all have four complete identity units.

The numerical report's `detections_complete` and `gallery_complete` flags check **all arms including damaged-input controls**, hence are false. Candidate/control reconstruction coverage itself is complete. The progression decision remains negative even with reconstruction-only coverage: identity and missing-region error fail the prespecified comparisons. Only one of the twelve seed/control combinations passes all four numerical margins; both seeds and all controls were required.

## Research decision

The held-out-patch score is misaligned with the task: five of eight within-case/seed rank correlations with actual hole error are negative. This is a descriptive diagnosis, not evidence that reversing the score would generalize. Do not invert the rule, change its radii, tune its score against these targets, and then reuse these same cases as independent confirmation.

The original-mask control is stronger in this screen. Its FaceNet is 0.741489 versus 0.611119 for selection; hole MAE is 0.062665 versus 0.076645. Even the stronger control visibly misses expressions, so this is not evidence that reconstruction accuracy is solved. Five-seed averaging reduces pixel error but also lowers identity similarity and softens details; it is not promoted as a universal improvement.

Further novelty work must supply a defensible source of missing structural information or change the claim to a rigorously evaluated uncertainty/limitation study. Mask expansion, held-out-context calibration, generic reference similarity and pixel averaging are not established contributions here. A future reference/observation-constrained mechanism needs its own prior-art comparison, versioned sampler implementation, matched controls and a frozen protocol before generation; it has not been implemented or validated by this screen. The adapter-training gate and eight reserved-final identities remain untouched.

## Reproduction

From the repository in the configured local environments, run the generator, the evaluator, then the reporter:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\run_context_support.py
& "$env:USERPROFILE\.cache\facial-inpainting\evaluation_env_v1\Scripts\python.exe" -X utf8 scripts\evaluate_context_support.py
.venv\Scripts\python.exe -X utf8 scripts\report_context_support.py
.venv\Scripts\python.exe -X utf8 -m unittest tests.test_context_support tests.test_context_statistics -v
```

The existing local run is immutable: completed-run guards prevent accidental overwrite. Restore the prerequisite local manifests using the project's dataset setup before reproducing in a fresh checkout/workspace; photographs and weights are not included in GitHub. Technical failures remain in the original ledger and require separately identified recovery runs rather than silent replacement.
