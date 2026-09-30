# Structural transport: failure analysis and decision

Completed 29 September 2026. This review follows the immutable numerical report; its earlier 'visual review pending' text records the reporter's state before this review. All eight sheets have now been inspected. [Review receipt](structure_transport_review_v1.json), [numerical results](STRUCTURE_TRANSPORT_RESULTS_V1.md), [CSV](structure_transport_metrics_v1.csv), [generation audit](structure_transport_integrity_v1.json), [method](STRUCTURE_TRANSPORT_METHOD_V1.md), [protocol](protocols/structure_transport_v1.json).

## Result

**The frozen mechanism failed feasibility, before candidate-image scoring.** All 24 transformation attempts reached the displacement solver and were rejected for a finite negative inverse-map Jacobian. No candidate image was produced. This is not an identity/fidelity trade-off measurement: candidate quality is unmeasured because there are no valid candidate outputs. Forty reused control slots were evaluated, corresponding to 32 distinct prior image files. No neural generator was rerun or trained.

| Arm | Attempts | Accepted images | Admissible landmarks | Minimum Jacobian range | Maximum displacement range (pixels) |
|---|---:|---:|---:|---:|---:|
| Four personal references | 8 | 0 | 30–38 | −5.614 to −0.673 | 10.57–39.04 |
| First personal reference | 8 | 0 | 20–39 | −8.763 to −1.526 | 13.55–50.44 |
| Four references of another person | 8 | 0 | 34–39 | −7.360 to −0.837 | 19.42–33.60 |

All maximum displacements were below the prespecified 51.2-pixel bound; the negative-Jacobian rule alone rejected these attempts. None failed because of insufficient references, anchors, no admissible constraints, or a nonfinite linear solve. The negative values are well below zero, not tiny roundoff errors. The derivative check is a discrete numerical test, not a proof of continuous invertibility.

Eight intervention-off checks reproduced the scaffold exactly. Nineteen new synthetic unit tests pass, including inverse direction, preserved pixels, missing-RGB isolation, degenerate registration, disconnected holes, invalid fields, structural metric support, missing detections, JSON serialization and per-seed anti-regression decisions. An independent audit verifies 20 frozen source/model hashes, all 64 row receipts, all input/control mappings, and no clean-target/gallery/reserved inputs in the inference whitelist. The failure is therefore an observed limitation of this fixed numerical intervention, not evidence of a successful new reconstruction pipeline. It does not prove every geometric method must fail.

## What the measurements do and do not establish

The attempted construction asks independently registered 2D facial shapes to displace components inside small fixed holes while nearby pixels remain stationary. The resulting smooth least-squares field can reverse local orientation even when its landmark residual and maximum movement are modest. Two-dimensional registration also confounds identity shape, view and expression. The recorded evidence establishes invalid maps; it does not isolate which of registration, correspondence, expression variation or deformation capacity is the principal cause.

ResShift / reference-context / plain-reference control means are respectively FaceNet 0.741489 / 0.781441 / 0.780990, hole MAE 0.062665 / 0.065988 / 0.068752, and LPIPS 0.025402 / 0.027110 / 0.028146. The new 106-point structural proxy is 0.020543 / 0.019812 / 0.018977 (all n=4 seed-complete identities). Thus a slightly better landmark proxy can accompany worse pixel and perceptual fidelity. It cannot replace the anti-regression requirements or visual assessment. RefFace ref0 has structural proxy 0.038886 (n=4). The damaged observation has structural support n=2 and FaceNet n=3, explicitly not directly comparable as a common-support mean.

**All nine candidate contrasts have n=0.** Their deltas, confidence intervals and exact p-values are unavailable. The frozen statistical helper fills Holm bookkeeping with 1.0 for missing comparisons; those entries are not observed hypothesis-test results and are not evidence of equality. No superiority or significance test can be performed without candidate scores. The 98 failure/stage events in the numerical JSON include dependent unavailable scores and repeated detection events; they are not 98 independently failed reconstructions. There are 24 rejected structural attempts.

## Visual assessment

Every one of the eight identity/seed sheets contains explicit FAILED/NO OUTPUT panels for all three structural arms. No rejected warp is silently substituted by its scaffold or presented as a generated success. The remaining panels are verified pre-existing controls, not new images.

| Identity | Both-seed review of existing controls |
|---|---|
| 1306 | Scaffold/cascades close the target's teeth-visible mouth and alter eyelids/gaze; sharper reference texture does not recover the missing mouth state. RefFace shows ghosted eyes and smeared/enlarged lips. |
| 2790 | Eye shape, lip contour and mouth asymmetry differ from target, with hair crossing the face. RefFace adds blur and misaligned features. |
| 1043 | Open smile remains in scaffold/cascades, but eyes/gaze and tooth/lip geometry differ; seed29 mouth is wider. RefFace smears eye/nose/mouth structure. |
| 787 | Eye opening/gaze, nose shading and lip geometry differ; seed29 changes mouth shape. RefFace has a blurred nose/eyes and distorted puckered lips. |

These are internal assistant observations, not blinded human participant ratings. Candidate gaze, teeth, identity, expression, skin and asymmetry cannot be rated because candidate images do not exist. A coordinate warp would in any event be unable to create absent teeth or recover unknowable pupil direction. Neither realism nor a small landmark error establishes the true hidden face.

## Decision and next hypothesis

Progression is rejected. Do not weaken the Jacobian check, search interpolation/regularization strengths against these targets, start adapter training, expand to fresh identities, promote the mechanism to the studio, or open final data. Historical negative results and the dedicated external baseline remain unchanged.

The mechanistically different next question is whether **joint reference identity geometry, separated from per-photo pose and expression, can condition fresh synthesis of a missing component**. Its representation would estimate a shared canonical 3D identity shape from references while giving each reference independent pose/expression; only reliable observed target evidence could constrain target pose/expression. A structural renderer or a compatible generator would synthesize new pixels, rather than deform a fixed incorrect mouth/eye texture. Fully hidden expression would remain uncertain rather than be copied from the median reference.

This is a next hypothesis, not an implemented method or an established contribution. 3DFaceFill already separates shape/pose/albedo and ReSem-Face already predicts multi-reference semantics; combining these broad ideas would not suffice. Before another implementation, resolve whether a specific personal-shape estimator and target-observability constraint have an incremental distinction, whether the available generator can consume that structure, whether lawful models/data are available, and how to test structure accuracy independently of its own estimator. A separately specified input-only geometry feasibility check must precede expensive training. No such successor has yet earned a large-experiment gate. [Prior-art audit](STRUCTURAL_PRIOR_ART_20260928.md) records the closest overlaps.

## Reproduction and preservation

Run from the repository root in the existing evaluation environment, in this order, only in a separately preserved clean output workspace; the scripts refuse existing immutable output files:

```powershell
& 'C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe' -m unittest tests.test_structure_transport tests.test_structure_inference_scope tests.test_structure_evaluation -v
& 'C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe' -X utf8 scripts/run_structure_transport.py
& 'C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe' -X utf8 scripts/evaluate_structure_transport.py
& 'C:/Users/rehaa/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe' -X utf8 scripts/report_structure_transport.py
```

This continuation added no inference capability to the website. Main page, evidence page and session endpoint returned HTTP200 on29 September; the local studio was idle. These are readiness checks, not new website generation tests. Research photos, comparison sheets, landmark arrays and checkpoints remain local.

Engineering: structural code/evaluation/provenance work, with all attempted unsafe maps rejected. Science: this frozen structural intervention is infeasible on all eight known case/seed pairs; no candidate-quality comparison exists. Novelty: **unsupported**. Generalization: only already-observed labels1306/2790/1043/787, unknown pretraining overlap. Final test: **0/8 model evaluations**; historical automated duplicate/hash pixel decoding is disclosed in the [history audit](MECHANISM_HISTORY_AUDIT_20260928.md), and no new reserved access occurred.
