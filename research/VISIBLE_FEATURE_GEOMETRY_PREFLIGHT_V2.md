# Visible feature geometry v2 — CPU diagnostic

11 October 2026. Written before execution. Independent geometry feasibility test, not a revision of v1 results or a reconstruction-quality experiment.

Use the same eight training identities, masks and four real references from v1. Sanitize missing pixels to 128 before any feature extraction. No clean target, generated scaffold, target facial landmarks or development scores enter registration. Freeze application, model and evaluator.

Use OpenCV SIFT (maximum 1500 keypoints, contrast threshold 0.02). Restrict keypoint centres to the central 80% of each image to reduce background matches. Reject target keypoints whose descriptor support approaches the hole: distance must exceed max(8, 6 times keypoint size). The image has already been sanitized, so even rejected support cannot expose hidden truth. Reference descriptors use the same central support rule. Match L2 descriptors in both directions, Lowe ratio 0.75, retaining only mutual matches. This is a low-cost classical-feature diagnostic, not DIFT and not a novelty claim.

Sort mutual pairs by target position. Every third pair is validation; never include it in fitting. Require >=12 pairs, >=8 fit pairs and >=4 validation pairs. Fit a positive similarity transform with RANSAC threshold 4 pixels, seed 20261011, 2000 iterations, confidence 0.99. Require >=8 fit inliers and >=60% fit inliers. Held-out RMS <=12 pixels at 512 resolution; both fit point spans >=100 pixels horizontally and vertically; scale in [0.5,2], positive determinant. These independent pixel-based rules are not equivalent to v1's face-width normalized gates.

Each target needs >=2 passing references. This only establishes visible registration coverage; reference eye agreement and missing-region extrapolation remain unvalidated and must be tested before integrating geometry into attention. Record every failure. All eight must pass to justify further geometric validation. No training or generation follows automatically. Preserve the v1 receipt and write a separate output directory. If coverage fails, report the method unsuitable under these fixed rules, without threshold sweeps.
