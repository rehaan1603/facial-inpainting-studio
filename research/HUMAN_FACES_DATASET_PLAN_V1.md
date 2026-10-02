# Human Faces dataset: intake and controlled development plan

Date: 1 October 2026. Local-only data. Existing datasets, trained checkpoints and frozen experiments are unchanged.

## Intake

The supplied folder contains 5,000 files in `Real Images` (178 x 218) and 4,630 files in `Generated Images` (256 x 256). All 9,630 decode successfully. Exact file and decoded-RGB duplicates were absent within this folder. Comparison against 51,944 existing saved fingerprints found no exact matches. This does **not** exclude resized/cropped copies, same identities or pretrained-model exposure.

No identity annotations, genuine same-person groups, dataset documentation or source/license information were found. Category names are supplied folder labels, not independently verified provenance. Matching numeric filenames across the two folders do not indicate corresponding people.

Local audit receipts: `outputs/human_faces_intake_20261001/{summary,manifest,overlap_check}.json`. No source photos are committed to GitHub.

## Completed preparation

`scripts/prepare_human_faces_development.py` selects four Real Images entries in a fixed hash order and creates 12 cases: central missing rectangle, thin scratches, and local Gaussian blur plus noise for each photograph. Selection does not depend on model outputs. Missing masks use white=replace; evidence maps distinguish missing (0), partial (128) and reliable (255). Original visible pixels remain byte-exact outside each artificial damage mask.

The local pack is `Downloads/Human dataset/Ready to test - Human Faces v1`. Its Photo folders are **not person groups**. The source photos are aspect-preserving bicubic interpolations with gray padding on a 512-pixel canvas. This adds no genuine source detail, so they cannot support a high-resolution recovery claim. Separate inference and evaluation manifests keep clean comparison paths out of generator inputs.

`scripts/test_human_faces_baselines.py` evaluates existing LaMa and ResShift studio paths on the first two photographs with central missing and thin-scratch masks (eight planned generations). No clean target is read by that script until generation finishes. It saves every submitted attempt, outputs, masked MAE/PSNR, unchanged-input errors and exact visible-pixel checks. These limited pixel measures are functional diagnostics, not proof of identity preservation, novelty or generalization. Blur/noise cases are prepared but not evaluated by this missing-region test.

## Baseline results and visual review

All eight planned generations completed; all eight preserve outside-mask pixels exactly. Four side-by-side sheets were inspected and copied into the local pack's `Baseline comparisons` folder. The numerical receipt is [human_faces_baseline_v1.json](human_faces_baseline_v1.json).

| Development case | LaMa masked MAE | ResShift masked MAE |
|---|---:|---:|
| Photo 01, central missing | 0.09428 | 0.15777 |
| Photo 01, thin scratches | 0.02020 | 0.07748 |
| Photo 02, central missing | 0.10482 | 0.15317 |
| Photo 02, thin scratches | 0.02433 | 0.08429 |

MAE is on a 0–1 scale, lower is better. LaMa repairs thin scratches reasonably in these two photographs but fills the central eye-region holes with blurred skin-like content rather than faithful eyes. ResShift generates mismatched eye appearance/lighting with conspicuous rectangular seams in the central holes. **Lower MAE does not establish a better semantic reconstruction:** LaMa scores lower while failing to restore the eyes. Neither model has solved the large missing-region task here. No new model, adapter, restoration default or quality claim is promoted from this smoke test. Identity/perceptual metrics and blur/noise model outputs have not been measured on this pack.

## Synthetic-reference experiment

**2 October update:** a bounded four-photo diagnostic is now complete: 11 synthetic views, 12 matched reconstructions and 24 evaluation rows. It did not establish a consistent advantage over the same-backbone reference-off control. See [results and implemented controls](SYNTHETIC_REFERENCE_TRIAL_20261002.md). The broader plan below is not fully complete; no training has started.

Purpose: test whether synthesized views offer useful conditioning beyond the damaged input. They cannot substitute for independent photographs documenting the person's actual hidden appearance.

1. First obtain the original download-page link and confirm source, allowed use and possible overlap with existing research datasets. Do not call the new folder independent based on exact hashes alone. No formal identity-disjoint train/test split is possible without verified identity labels.
2. Use damaged input **only** for view generation. Never derive a synthetic test reference from the clean target, a restoration chosen using the target, or a sibling corruption of the same clean test photograph that reveals the hidden region. Do not feed clean target features into selection or conditioning.
3. Freeze a small development cohort, generation model/version, prompts, random seeds, number of views and failure rules before running the angle experiment. An externally generated illustration is a UI demonstration unless the synthesis process is sufficiently documented for reproduction.
4. Compare the same restoration backbone and sampling budget using: no references; repeated damaged input (reference-count control); one synthesized view repeated; multiple synthesized views. Where genuine same-person photos exist in the original labeled dataset, add a genuine-reference arm. Keep restoration and missing-region tasks separate.
5. Retain all attempts, invalid views and detector failures. Evaluate paired changes in masked error, perceptual similarity, independent identity similarity and visible-pixel preservation, plus blind visual checks. Report model calls and latency because view synthesis adds compute. Do not select samples or outputs using clean-target metrics.
6. Count original photographs/verified identities as statistical units, not synthetic views or mask variants. Group all derivatives with their source. Do not promote a method based on these four development photos. Broader validation requires provenance, identity/near-duplicate controls and a prespecified evaluation cohort.

## Training decision

No training has started on this folder. If provenance checks pass, test the existing checkpoint first, then consider a separately saved adaptation using real training photos. Keep generated photos in a separately labeled experiment. Reuse architecture where compatible, with explicit preprocessing and new manifests. Do not overwrite current weights or mix unverified images into the existing final-test protocol.

Next required user information: the original dataset download-page link, and any available identity mapping. Genuine additional photos are required for verified multi-reference supervision; generated angles do not provide that evidence.
