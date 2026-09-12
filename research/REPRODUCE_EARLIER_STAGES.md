# Facial inpainting research

Local research project investigating completion under inaccurate occlusion masks. The current implementation includes an audited data pipeline, expanded validation-only LaMa benchmark, and working ResShift face baseline smoke test. It is not a novel trained method or a publication-ready evaluation.

## Environment

Windows, Python 3.12, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0. GPU forward/backward has passed on this project's RTX 5070 Laptop GPU. Use the project environment, because the global Python contains CPU-only PyTorch.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The versions in requirements-lock.txt record this working installation. Copy configs/local.example.json to configs/local.json on a different machine and set local paths. The existing machine configuration is already populated. Datasets remain in their original download folders; models are cached outside OneDrive.

## Reproduce the first milestone

Run from this repository root in PowerShell. Check each command succeeds before proceeding.

```powershell
.\.venv\Scripts\python.exe scripts/check_gpu.py
.\.venv\Scripts\python.exe scripts/audit_data.py
.\.venv\Scripts\python.exe scripts/audit_annotations.py
.\.venv\Scripts\python.exe scripts/clean_manifests.py
.\.venv\Scripts\python.exe scripts/test_protocol.py
.\.venv\Scripts\python.exe scripts/download_baseline.py
.\.venv\Scripts\python.exe scripts/pilot.py --images 32
```

The image audit reads and fully decodes every source image, computes SHA256 file hashes, checks annotation coverage, and links HQ images to identity and partition metadata. The annotation audit decodes all binary masks and LaPa labels and checks label values and landmark syntax. RGB black/white HQ masks are valid; the source files are not rewritten.

Original official manifests are data/manifests/celebahq.csv and lapa.csv. The working versions end in `_clean.csv`. Every member of an exact-file duplicate group is excluded from working manifests, even within a split. This conservative quarantine changes official counts and must be reported when comparing published work. No original data is deleted or moved. Near-duplicate and cross-source identity contamination remain unaudited.

## Current outputs

- research/data_audit.json: image integrity, original split counts, identity overlap, exact duplicates.
- research/annotation_audit.json: segmentation and landmark validation.
- research/clean_manifest_report.json: quarantine policy, cleaned counts, and manifest hashes.
- research/gpu_check.json: actual CUDA environment test.
- research/baseline_provenance.json: model source and verified published checksum.
- research/pilot_results.json: cleaned-validation diagnostic summary and scope limitations.
- outputs/pilot_clean_v1/metrics.csv: all per-image/per-condition measurements.
- outputs/pilot_clean_v1/cases.json: fixed selected identities, seeds and source hashes.
- outputs/pilot_clean_v1/preview.png: first four preselected examples, including failures.

The earlier outputs/pilot directory is a superseded pre-quarantine engineering run. Do not use it as the current result. The canonical run is pilot_clean_v1. Rerunning the pilot overwrites that run; archive/version the run directory before changing its protocol or sample count.

## Baseline and protocol

LaMa is loaded from the [IOPaint/Sanster TorchScript export](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py), with published MD5 verification and locally recorded SHA256. The [original method](https://github.com/advimman/lama) is WACV 2022 LaMa. This export has not been independently checked for numerical equivalence to the original author checkpoint and is not a face-specific checkpoint. It is an engineering baseline only.

The pilot selects 32 distinct identities from cleaned validation using a fixed hash ordering. It uses 256-pixel RGB images, textured rectangular occlusion, accurate masks, 4-pixel erosion/dilation, and an 8-pixel translation. It compares unchanged input, supplied-mask LaMa, fixed dilation, and a privileged exact-mask diagnostic. The mask convention is 1 = replace. Clean target RGB is used only for simulation and scoring, never supplied to the model. The observer always sees corrupted pixels where the true occlusion exists, even if the supplied mask misses them.

Metrics separate true-hole error from changes in genuinely visible pixels and mark empty regions as unavailable. Images are scored as float arrays before saved-image quantization. GPU timings exclude CPU preprocessing and postprocessing. The test suite checks that hidden target pixels cannot leak through mask perturbation and that regional metrics distinguish changed visible and occluded regions.

Erosion by four followed by dilation by four restores these particular rectangles exactly. That control should match the exact-mask result; it does not establish a research contribution. This pilot has no semantic stratification, LPIPS, identity evaluator, tuned correction baseline, learned method, multi-seed statistics, or real occlusion benchmark. Final test images have not been used for model evaluation.

## Next research stage

Extend the radius sweep beyond 8, add feathering controls and area-matched occlusions, and evaluate ResShift on the same cases. Add held-out occluder assets. Only after a persistent gap remains should a trainable correction module be introduced. The publication audit and readiness checklist in research describe the wider requirements.

## Reproduce the second milestone

Run after the first milestone has generated clean manifests and pilot cases:

```powershell
.\.venv\Scripts\python.exe scripts/test_benchmark_v2.py
.\.venv\Scripts\python.exe scripts/benchmark_v2.py
.\.venv\Scripts\python.exe scripts/download_resshift.py
.\.venv\Scripts\python.exe scripts/resshift_smoke.py
.\.venv\Scripts\python.exe scripts/report_benchmark_v2.py
```

The benchmark uses configs/benchmark_v2.json: 96 distinct validation identities, 48 for tuning and 48 for assessment, three mask families, five error conditions, and four dilation radii. It writes 7,200 metric rows to outputs/benchmark_v2/metrics.csv. The assessment bootstrap resamples identities, not individual masks. The selected radius is at the largest tested value, so it is not yet an optimized control. LPIPS weights download on first use; all face processing stays local.

See research/MILESTONE_02.md, research/benchmark_v2_results.json and outputs/benchmark_v2/tradeoff.png. ResShift uses pinned official source and strict checkpoint loading, with provenance in research/resshift_provenance.json. Its four-image outputs/resshift_smoke/preview.png is only a compatibility check. Pretraining exposure is unresolved and the original S-Lab noncommercial license applies. Re-running scripts overwrites their outputs; version the run directory before changing settings.

Do not redistribute the face datasets or derived image packs. Follow original dataset, model and evaluator terms. Code, metadata, runs and citations need separate release checks before publication.

## Matched backbone comparison

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_v2.py --backbone resshift
.\.venv\Scripts\python.exe scripts/compare_backbones.py
```

The ResShift run uses the same v2 configuration and checks the selected cases against the completed LaMa run. It writes to outputs/benchmark_v2_resshift, preserving LaMa outputs. A metrics.partial.csv checkpoint is written after each identity for inspection; automatic resume is not implemented. Final metrics.csv and the results JSON are produced only after all identities finish.

The comparison script checks case metadata, configuration and manifest hashes, and every saved observed-image and mask byte. It generates research/MILESTONE_03.md, research/backbone_comparison.json and outputs/backbone_comparison/tradeoff.png. Each model chooses its own radius on tuning identities; paired assessment intervals resample identities. Diffusion uses one fixed seed per image/family shared across mask controls. It does not estimate sampling variation. See research/NEXT_CONTROL_PROTOCOL.md for prospective stronger controls.

## Expanded simple controls

```powershell
.\.venv\Scripts\python.exe scripts/test_control_sweep.py
.\.venv\Scripts\python.exe scripts/control_sweep.py --backbone lama
.\.venv\Scripts\python.exe scripts/report_control_sweep.py --backbone lama
```

The search in configs/control_sweep.json combines seven dilation radii with four inward feather widths. Each completion is generated using the expanded hard mask, then blended with observed RGB using only distance inside that mask. All 28 controls are scored on tuning identities; only the minimum-LPIPS combination proceeds to assessment. There are 20,160 tuning rows and 720 assessment rows per completed backbone. The control runner regenerates float inputs and verifies them against the earlier saved images and masks.

Completed case chunks are written atomically and loaded on rerun. A signature guards against changes to the runner, config, manifest, or case metadata; changed protocols require a new output directory. Results remain separate from v2. Use `--backbone resshift` for the same search on that checkpoint; support in the runner does not mean that experiment has completed. Read research/CONTROL_RESULTS_LAMA.md after the LaMa report command completes.

Both expanded searches are now complete. ResShift selects radius 12/feather 4; LaMa selects radius 8/feather 0. Read research/CONTROL_RESULTS_RESSHIFT.md for the reconstruction/preservation trade-off and condition breakdown. Run scripts/verify_control_run.py with `--backbone lama` or `--backbone resshift` to verify completed output counts, identity separation, signatures, and unchanged-control reproduction. Both verification reports are saved in research/control_verification_*.json.

## Area-matched diagnostic

```powershell
.\.venv\Scripts\python.exe scripts/test_area_matched_v3.py
.\.venv\Scripts\python.exe scripts/area_matched_v3.py
.\.venv\Scripts\python.exe scripts/verify_area_matched_v3.py
.\.venv\Scripts\python.exe scripts/evaluate_area_matched_v3.py
.\.venv\Scripts\python.exe scripts/report_area_matched_v3.py
```

Canonical outputs use area_matched_v3_margin12 and area_matched_v3_margin12_evaluation. The earlier directories without margin12 are superseded engineering artifacts and must not be used as results. The corrected generator requires enough image margin for all evaluated dilation controls. Do not change configs or generators during evaluation; evaluation signatures protect resumed runs from incompatible data or code.

True mask geometry, missing-pixel count, erroneous-mask geometry and FP/FN counts, and synthetic occluder RGB match under translation across three locations. Missing areas are approximately 3%, 6%, and 10%. A group is rejected consistently across all locations when semantic centers or margins are unavailable. The mask audit verifies every saved mask and matches occluder pixels across locations. One tuning identity was excluded for missing eye annotations; all 48 assessment identities remain.

Evaluation freezes earlier controls and uses assessment identities only. It does not tune new settings or access the final test set. The runner checkpoints each case. A complete run produces 5,184 metric rows, summarized in research/MILESTONE_04.md. Semantic centers do not guarantee full anatomical coverage, and synthetic texture occlusion does not establish real-occluder performance.

## Near-duplicate screening

```powershell
.\.venv\Scripts\python.exe scripts/test_near_duplicates.py
.\.venv\Scripts\python.exe scripts/audit_near_duplicates.py
.\.venv\Scripts\python.exe scripts/rank_duplicate_candidates.py
.\.venv\Scripts\python.exe scripts/save_duplicate_review.py
```

See research/MILESTONE_05.md and research/PRETRAINING_EXPOSURE_LEDGER.md. The screening reads all 51,944 images but is not an exhaustive transformed-duplicate or identity audit. Of 496 candidate pairs, 42 have visual decisions and 454 remain pending. The provisional *_reviewed_v1.csv manifests exclude 22 images from 11 visually reviewed near-duplicate pairs. Existing experiment manifests and source files remain unchanged; future training must explicitly select an appropriate reviewed manifest and finish the pending audit. The review-writing script pins the candidate-index hash so a changed ordering cannot silently reuse decisions.
