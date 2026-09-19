# First generalization cycle — completion status

Review date: 20 September 2026. The protocol and scorer were frozen on 19 September before generated outcomes were inspected.

## Implemented and exercised

| Work | Evidence |
|---|---|
| Architecture audit and preservation | `upgrade_baseline_audit_v1.json`; 202 existing source files and 24 project checkpoints unchanged; only the declared HTTP test file changed within that source audit |
| Local identity separation | `protocols/unseen_identity_protocol_v1.json`: four fresh validation groups, eight reserved final groups, separate target/four references/three gallery roles |
| Synthetic damage | `src/degradation`: eight distortion types, three severity presets, deterministic geometry and exact known-pixel preservation |
| Runtime reference scoring | `src/reference_selection`: coarse region demand, quality/pose/visibility proxies, identity compatibility, invalid-file diagnostics and six selection policies |
| Generator connection | `scripts/mask_aware_inpaint.py`, reusing the unchanged `reference_inpaint.py`; arbitrary one to four input paths, no identity-label lookup or clean-target argument |
| Real comparison | 110 unique GPU outputs representing 168 logical rows; seven policies, three conditions, two seeds, four validation identities |
| Evaluation | FaceNet target/gallery, conditioning-encoder ArcFace target/gallery, NIQE, BRISQUE, LPIPS, SSIM, whole/hole PSNR and MAE; identity-level paired analysis and Holm correction |
| Audit records | `generalization_selection_evidence_v1.json` contains numerical rows, selected-reference hashes, diagnostics, signatures and 21 verified pretrained-model files; images stay local |
| Local application | Separate experimental 1–4-reference mode, unchanged original modes, downloadable diagnostics and displayed disagreement warnings |

All 168 rows generated and evaluated successfully. Every row had valid target FaceNet/ArcFace measurements and NIQE/BRISQUE scores; known-region MAE is zero. These are execution/integrity facts, not a declaration that the reconstructed faces are correct.

## Measured result and failures

Mean target FaceNet cosine: mask-aware 0.8113; random 0.7959; identity-only 0.7896; quality-only 0.7938; all references 0.8180. Primary paired Holm-adjusted p-values: 0.50, 0.50, 0.75 and 0.75 respectively. All-reference gallery similarity is also higher (0.5304 versus 0.5015). There is no statistically supported superiority claim.

Visual failures include altered eyes/gaze, changed expressions and mouths, and overly smooth features, particularly with central-face mixed damage. The original LaMa facial-fidelity limitation is not solved by this work. Valid face detection and high whole-image similarity do not establish correct hidden details.

## Changed files

Created: `src/research_integrity.py`; the `src/degradation` and `src/reference_selection` packages; `configs/mask_aware_selection_v1.json`; preparation, inference, comparison, scoring, reporting and integrity-check scripts; two new test modules; protocol JSON/Markdown; upgrade plan; distortion, identity, selection, fusion-status and results documents; numerical summary/evidence JSON.

Modified: `PROJECT_REPORT.md`, `README.md`, `webapp/README.md`, `webapp/server.py`, `webapp/dist/app.js`, `webapp/dist/index.html`, `scripts/test_webapp.py`. The earlier local-folder export scripts are preserved. No existing trained checkpoint, frozen generator, frozen metric implementation or historical protocol was changed.

## Tests and scope

The pre-change Python script suite ran 45 tests with three environment-dependent skips. New tests cover all distortion presets, determinism, hidden-pixel independence, role/identity separation, pHash separation, mask-dependent ranking, malformed/duplicate references, disagreement warnings and identity-level statistics. The updated HTTP suite accepts one through four experimental references and rejects invalid counts while preserving original three/four-reference behavior. JavaScript syntax and framing checks pass. Actual CPU runtime checks exercise real detection on an arbitrary renamed photo and blank/corrupt/duplicate uploads, including a damaged-input face-detection fallback.

Final regression checks: 46 tests in the scripts suite (three environment-dependent skips) and 13 tests in the tests suite, with no failures. JavaScript syntax and mask-framing checks passed. A real browser run with one reference produced a 512 × 512 result, mask and HTTP-200 diagnostics download, preserving outside-mask pixels exactly. The explicit no-valid-reference CLI LaMa fallback also passed a real GPU smoke check with exact outside-mask preservation. See `generalization_webapp_verification_v1.json`.

The controlled comparison uses actual GPU outputs, not mocked generation. Only the three medium-severity conditions were GPU-benchmarked; other severity/type combinations have implementation tests. The final split remains unused. The LaMa fallback smoke check is separate from the controlled selection comparison.

## Remaining work and recommended next step

1. Plan a larger versioned development experiment with more identity groups, realistic masks, varying severity and resolution, and independent human assessment. Four selected identities cannot establish broad unknown-person generalization or demographic robustness.
2. Investigate spatial reference correspondence and actual occlusion/visibility estimation. Current selection passes a global FaceID embedding, so regional detail is not transferred. Quality normalization across reference resolutions and use of partly damaged input detail also need study.
3. Only then implement a justified regional fusion adapter with a training budget and equal-information-budget ablations. No learned regional fusion, automatic damage localization or candidate reranking is complete.
4. Compare appropriate external methods, freeze the method, evaluate reserved identities once, and prepare a venue-specific manuscript. Unknown pretraining overlap and dataset-label uncertainty remain explicit limitations.

The first engineering/measurement cycle is complete. Restoration superiority, a publishable novel contribution and overall publication readiness remain **inconclusive/incomplete**.
