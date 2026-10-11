# Visible-feature registration diagnostic v2

11 October 2026. CPU-only experiment; no model training, app changes or new reconstructed outputs.

## Fixed experiment

[Protocol](VISIBLE_FEATURE_GEOMETRY_PREFLIGHT_V2.md) was written before implementation/execution. Same eight training identities, four real references each, original masks and source-image hashes. SIFT features used only sanitized targets, descriptor support away from the hole and mutual ratio-tested matches. Independent validation matches were excluded from RANSAC fitting. No development images, target truth, synthesized reference or generated scaffold was used for registration.

Four unit tests passed: hidden-pixel invariance, identity registration on synthetic texture, rejection of blank images, repeatability/input preservation. These tests establish mechanical behavior, not facial correspondence correctness.

## Actual outcomes

| Training identity | Passing references / 4 | Target gate (requires 2) |
|---|---:|---|
|6792|0|Fail|
|2250|0|Fail|
|2083|0|Fail|
|4164|0|Fail|
|674|0|Fail|
|476|0|Fail|
|1281|1|Fail|
|8748|0|Fail|

Of 32 reference pairs, **28 had fewer than 12 mutual matches**, three failed the fixed fit/held-out/coverage gate and one passed. Only one pair had substantial match support (111 pairs; held-out RMS 7.48 pixels). Overall **0/8 targets pass**. This is registration coverage, not facial reconstruction accuracy.

The observation supports rejecting sparse classical features as the current spatial prerequisite. It does not demonstrate that no correspondence exists: the fixed descriptor support margin removes features near the erased eye region, and differing expressions/pose/illumination can limit sparse matches. The experiment does not isolate those causes. No threshold sweep or excluded case was used to improve the result.

## Decision and next step

Do not integrate this mechanism or start adapter training. Preserve the v1 landmark failure and this separate v2 diagnostic. The app, CFG3 baseline, masks, original results and evaluator remain unchanged. Hole SSIM, LPIPS, FaceNet and ArcFace have no new candidate measurement.

Next: specify a **dense frozen-feature correspondence feasibility experiment**, using visible-only mutual matches, held-out geometric validation and explicit uncertainty for extrapolation into the missing region. First bound feature extraction memory and verify hidden-pixel invariance; only consider reference-attention training after reliable geometry is demonstrated. Dense features are a hypothesis, not a guaranteed fix. Keep the original eight cases and do not tune against development reconstruction scores.

Local detailed receipt: `outputs/spatial_reference_correspondence_v2/geometry/receipt.json`. [Photo-free public evidence](visible_feature_geometry_v2_results.json) records failures and source hashes without images or landmark coordinates. Scripts: `visible_feature_geometry_v2.py`, `audit_visible_feature_geometry_v2.py`, `test_visible_feature_geometry_v2.py` in `scripts/`.
