Latest public photo-free presentation: [5 October 2026 results](Facial_Inpainting_Results_2026-10-05.pptx). This dated 20-slide snapshot covers architecture, training, corruption, v4/v5, metrics, official-input encoding and papers. Dataset photos are excluded.

## Latest local illustrated decks — 11 October 2026

The earlier overall and small-batch decks have been updated in fresh copies. The original cover, architecture, methods, photographs, historical tables and ten paper hyperlinks are preserved. Latest files in the original laptop's Downloads:

| Deck | Local filename | Slides |
|---|---|---:|
| Overall project | `Facial Inpainting Studio - Overall Results - Decoded Continuations - 2026-10-11.pptx` |42|
| Small batch | `Facial Inpainting - Small Batch - Decoded Continuations - 2026-10-11.pptx` |37|

Both include completed 64-update training and the two matched 16-update continuations, whole-image PSNR/SSIM/LPIPS, separate hole/eye scores, FaceNet/ArcFace and the actual scored outputs. All five methods stay together. RGB whole LPIPS improves slightly, but hole SSIM/FaceNet regress versus equally trained epsilon; numerical and visual gates fail and are retained. High whole-image SSIM includes unchanged visible pixels and is not an accuracy percentage. These are three development eye cases at seed17, separate from the earlier four-person whole-image ReF-LDM blur/noise studies; they are not pooled or described as full-dataset accuracy.

[Exact local filenames, SHA256, metric means and decisions](local_illustrated_update_20261011.json) · [paired decoded-loss results](../research/MASKED_SDXL_DECODED_CONTINUATION_RESULTS_V1.md) · [current project report](../PROJECT_REPORT.md).

The new illustrated files remain local because dataset-photo redistribution was not authorized. A clone includes this photo-free metadata and complete numeric/source evidence. Original 32-slide/20-slide source decks and the intermediate 38-slide/27-slide whole-image copies remain intact. The public decks below retain their dated historical scope.

# Research results presentation

[Download the results and comparison slides](Facial_Inpainting_Results_2026-10-02.pptx).

This 11-slide deck contains published-paper context and clickable sources, matched local experiments, identity/eye-region measurements, accuracy interpretation and future scope. It contains no dataset photographs or generated face examples. The full 35-slide illustrated review remains local under Downloads because dataset-photo redistribution has not been authorized.

Current evaluation covers four observed development identities, three inference seeds and two reference conditions. These identities were excluded from adapter training, but upstream pretrained-model exposure is unknown. There is no verified reconstruction-correctness percentage for arbitrary external client portraits. Cosine similarity is not accuracy, and successful generation/detection rates do not measure facial fidelity.

Numerical sources: [method report](../research/FINAL_METHOD_REPORT.md), [results](../research/identity_compatibility_v2_results.json), [CSV](../research/identity_compatibility_v2_metrics.csv), [execution provenance](../research/identity_compatibility_v2_provenance.json).
