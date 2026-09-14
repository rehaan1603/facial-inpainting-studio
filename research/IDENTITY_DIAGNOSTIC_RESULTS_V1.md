# Independent identity diagnostic — 14 September 2026

FaceNet provides preliminary evidence that reference conditioning improves facial similarity on these saved development examples. It does not establish a novel method or publication readiness.

All 96 original output hashes and all twelve target hashes were checked before scoring. The official pinned VGGFace2 FaceNet evaluator is separate from InsightFace used for conditioning. No generated candidates were replaced or selected using this score. No app dependencies were changed.

## Results

Each row has 11 valid identity pairs out of twelve. Higher cosine similarity is better; values are not recognition accuracy percentages.

| Strength | Composition | References off | References on | Paired change | Descriptive 95% interval |
|---|---|---:|---:|---:|---|
| 1.0 | Hard | 0.4257 | 0.6085 | +0.1827 | [0.0869, 0.3018] |
| 1.0 | Poisson | 0.4827 | 0.6596 | +0.1769 | [0.0873, 0.2866] |
| 0.99 | Hard | 0.4454 | 0.6378 | +0.1924 | [0.0780, 0.3087] |
| 0.99 | Poisson | 0.4942 | 0.6588 | +0.1646 | [0.0492, 0.2903] |

MTCNN returned two detections for target identity 4423 and multiple detections for its eight outputs. Those records remain in the JSON with null similarity, giving 88/96 valid scores. This describes detector behavior, not a verified count of real people. No threshold was changed to recover this case.

The reference-on benefit in this metric can coexist with the previous lack of consistent LPIPS/MAE improvement: the metrics measure different aspects. The experiment uses small eye masks, one generation seed, and existing development cases. Detection-conditioned intervals use 10,000 paired identity bootstrap resamples; they are not multiplicity-adjusted confirmatory tests. Celebrity overlap with VGGFace2 pretraining is unknown. Large-mask completion, independent identity galleries and fresh final testing remain necessary.

## Reproduction

From the repository directory, using the existing project environment:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/prepare_identity_evaluator.py
.venv/Scripts/python.exe -X utf8 scripts/evaluate_identity_diagnostics.py
```

The preparation script downloads pinned official code/weights and isolated dependencies into the configured external cache. Original local diagnostic images must be available. Code, model and dependency hashes are recorded in `identity_evaluator_provenance_v1.json`; per-output detections and scores are in `identity_diagnostic_results_v1.json`. Embeddings, model weights and face photographs are not committed.

Next: add secondary quality measurements, freeze larger-mask and gallery protocols, and implement controlled mask-aware reference-selection comparisons before attempting a learned fusion contribution.

Official evaluator source: https://github.com/timesler/facenet-pytorch
