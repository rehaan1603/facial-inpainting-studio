# Independent identity diagnostic v1

Frozen before FaceNet scoring on 14 September 2026. Development analysis only.

- Evaluate all 96 saved v3 reconstructions against their twelve clean targets. Verify image hashes against the original manifests/metrics before scoring. Do not regenerate or select outputs.
- Use official timesler/facenet-pytorch commit `787da06156087cd6b616fe6608213722bddc30cd`, VGGFace2 InceptionResnetV1 weights from official release v2.2.9. This evaluator differs from the InsightFace encoder used for conditioning. Unknown celebrity/pretraining overlap remains a limitation.
- CPU float32, eval mode; MTCNN independently detects each target and output: minimum face size 20, thresholds 0.6/0.7/0.7, factor 0.709. Require exactly one detected face. Standard bounding-box crop, image size 160, margin 0, standard postprocessing; no target-derived crop for outputs.
- Cosine similarity of normalized 512-dimensional embeddings, higher is better. Preserve failed/multiple-face cases and coverage; never replace failures with zero or silently drop them.
- Summarize each of eight configurations. For each strength/compositor, compare reference scale 0.8 minus 0 on jointly valid identities, paired identity bootstrap 10,000 resamples, seed 20260914, percentile 95% interval. These are descriptive, conditional-on-detection intervals, not multiplicity-adjusted confirmatory tests.
- No verifier ranking, threshold tuning, recognition accuracy, low-FAR claim or fresh final-test claim. Cosine similarity to a clean target is a diagnostic, not proof of correct real-world identity. No embeddings or face crops are published.

Source: https://github.com/timesler/facenet-pytorch
