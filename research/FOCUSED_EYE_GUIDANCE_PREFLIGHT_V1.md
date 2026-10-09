# Three-person eye fidelity: guidance diagnostic

9 October 2026. User reports incorrect iris colour and eyebrow shape in the current three-person outputs.

Inspecting the saved comparisons confirms visible discrepancies. Identity embeddings do not explicitly carry spatial iris/brow correspondence. Poisson blending can change colour, but this does not establish it as the sole cause. SSIM also responds to structure, contrast and luminance; it is not a dedicated iris-colour measure.

## One intervention

Keep persons263,6681,10149, eyes only, clean four-reference sets in their original order, seed17, DDIM30, FaceID scale0.8, strength1, model1024, output512 and existing Poisson finish. Change only classifier-free guidance5 to3. Lower guidance may reduce exaggerated colour/contrast while maintaining target context. This is a hypothesis, not an assumed improvement. Do not add an eye-colour prompt derived from the hidden target, substitute references, change the mask or tune settings per person.

The existing generator and website defaults remain unchanged. A research-only pipeline proxy overrides the single guidance argument. Replay person263 at guidance5 first, requiring raw and finished pixels to match the stored control exactly before candidate generation. Save the proxy/runner and generator hashes, case/reference hashes, timing, memory, raw/hard/final outputs and failures in a new directory.

## Evaluation and decision

Use the existing frozen scorer unchanged. Compare the three candidates with `focused_reference_finish_v1` on identical eye rows. Primary measures are hole SSIM, hole PSNR, hole/eye crop LPIPS, FaceNet and ArcFace with detection coverage. Inspect originals, both raw generations and finished outputs for iris/gaze/brow errors. Originals are evaluation-only. Visible pixels must remain exact.

Progress requires improved mean hole SSIM and no mean deterioration on the other measures, with per-person regressions disclosed. A setting that substantially lowers identity for a person is not a general replacement even if averages rise. This is repeated development testing, not population accuracy or an independent benchmark. No training or checkpoint changes. Further seeds/corrupted references follow only a promising result.
