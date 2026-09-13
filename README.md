# Facial inpainting under inaccurate masks

For official dataset downloads and exact local restoration of the website/reference samples, see [Dataset downloads and samples](research/DATASET_DOWNLOADS_AND_SAMPLES.md). The photos remain local; the repository records their selections, roles and hashes.

A local research implementation with LaMa and ResShift inference, two matched trainable mask-refinement controls, data audits, controlled corruption benchmarks and a frozen object-composite test. It investigates reconstruction error versus changes to genuinely visible pixels. The weighted loss is a standard control, not a demonstrated novel method.

Read `research/FINAL_RESULTS.md` for generated measurements, `research/MANUSCRIPT.md` for the technical paper draft, and `research/PROJECT_STATUS.md` for completed engineering and remaining publication requirements. Earlier experiment commands and historical status are preserved in `research/REPRODUCE_EARLIER_STAGES.md`.

## Run local inpainting

The browser application is implemented. Double-click `Start Studio.cmd`, then open http://127.0.0.1:8765. Upload a face, paint or upload a mask, select LaMa, ResShift or Use 3–4 reference photos, reconstruct, compare and download the result. See `webapp/README.md` for controls, local storage and limitations. The website connects to the real local GPU engine. The launcher starts the server independently, so closing the launcher does not stop the app.

Use PowerShell in this project folder. The project's `.venv` contains CUDA PyTorch; the global Python installation is unsuitable. Inputs must have matching dimensions. White mask pixels mean replace; black means retain. Inference resizes both inputs to 256 × 256 and writes a PNG, the effective mask and JSON metadata. Crop and align faces before inference for comparable results.

```powershell
.venv/Scripts/python.exe scripts/inpaint.py --image "your_image.png" --mask "your_mask.png" --output "outputs/result.png" --backbone lama
```

Use `--backbone resshift` for the four-step face diffusion model. Both run locally; there are no image uploads. Without a refiner, pixels outside the supplied mask are preserved at the resized input resolution. Generated hidden content is a plausible reconstruction, not recovered evidence of the true face.

The trained controls are optional. For the predeclared seed-17 weighted control:

```powershell
.venv/Scripts/python.exe scripts/inpaint.py --image "your_image.png" --mask "your_mask.png" --output "outputs/refined.png" --backbone lama --refiner outputs/learned_refiner/preservation_weighted_17/best.pt --threshold 0.35
```

For seed-17 generic refinement use `outputs/learned_refiner/generic_17/best.pt` and threshold `0.5`. Other seeds have their own recorded thresholds in `outputs/refiner_evaluation/*_selection.json`. Refinement can miss occlusion or replace valid content. The weighted control is not an established quality upgrade: development measurements show less visible change accompanied by worse reconstruction.

A verified local example uses `outputs/benchmark_v2/cases/10371_brush/observed.png` and its `under.png` mask; output is in `outputs/demo/lama.png`.

## Reference-photo completion

Select **Use 3–4 reference photos** in the website. Add three or four distinct photographs of the same person, with one visible face per photograph. Mark the entire damaged target region and reconstruct. Uploading a new input clears previous references, so photos of another person cannot be reused accidentally. **Processing detail** offers standard (512 processing) and detailed (1024 processing, more GPU memory) modes. The optional **Match colour at edges** control reduces colour seams. The result is a 512 × 512 PNG; pixels outside the effective mask remain unchanged. **Load research sample** supplies a matching target, mask and four references when this mode is selected.

The reference mode uses existing SDXL inpainting, IP-Adapter FaceID Portrait and InsightFace weights in a separate local environment. See `research/REFERENCE_IMPLEMENTATION.md` and `research/reference-environment-v2-lock.txt`. This is a functional baseline, with broader quality evaluation still pending. The twelve-person diagnostic is recorded separately from the engineering example; see the current status and `research/QUALITY_FIX_STATUS.md`.

To install or repair the reference software after setting up the base CUDA environment and `configs/local.json`:

```powershell
.venv/Scripts/python.exe scripts/prepare_reference_runtime_v2.py
```

This uses the separate `reference_env_v2` cache directory and preserves the original experiment environments. Model weights still need to be restored from their pinned provenance sources. Restart the studio after an environment or server change.

## Environment and reproduction

Tested on Windows, Python 3.12, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0, RTX 5070 Laptop GPU (8 GB). Paths and the external cache are set in `configs/local.json`; use `configs/local.example.json` on another machine. Datasets stay in the original download locations. Do not include them in a code release.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m unittest discover -s scripts -p "test_*.py" -v
```

The historical reproduction guide covers downloading verified backbones, source audits and v2/v3 benchmark construction. After those stages, `scripts/run_remaining.ps1` prepares the training cache, resumes completed training/evaluations, prepares object assets, runs the frozen test and rebuilds the final tables. Invoke it from PowerShell with `& ./scripts/run_remaining.ps1`.

Signed runs refuse changed inputs. To study a changed configuration, create a new versioned output directory and protocol; do not delete signatures and silently reuse old rows. Repeated inference on the recorded test does not create a fresh test set. Selection used validation only, but these test results become observed evidence once evaluated.

## Artifact map

| Artifact | Purpose |
|---|---|
| `data/manifests/*_reviewed_v2.csv` | Current source paths, hashes, splits and quarantine outcome |
| `research/near_duplicate_review_v2.json` | All 496 candidate-pair review decisions |
| `configs/learned_refiner.json` | Matched training architecture, data, seeds and budget |
| `outputs/learned_refiner/*/best.pt` | Six fitted mask-refinement checkpoints |
| `outputs/refiner_evaluation/metrics.csv` | 13,824 learned-control development rows |
| `outputs/object_test/protocol.json` | Frozen sources, assets, checkpoints and thresholds |
| `outputs/object_test/metrics.csv` | Object-composite test with unchanged-input controls |
| `research/final_results.json` | Final tables, uncertainty and raw hashes |
| `research/DATA_AND_LICENSES.md` | Sources, exclusions, restrictions and attribution limits |

Neither the full generative backbones nor an identity-recognition model was trained locally. Pretraining membership is unresolved. The object test uses synthetic composites of photographed objects; LaPa uses annotation-derived face crops. Publication claims require the missing scientific comparisons and author review in the project status. IEEE Xplore itself is not a submission venue.

## GitHub checkpoint

This private repository contains code, reports, detailed experiment tables and settings, data-integrity manifests, and the project-trained mask-refiner checkpoints (including training-resume states). `research/REPOSITORY_ARTIFACT_INVENTORY.json` lists every uploaded research artifact and its hash. Photographs, downloaded model weights, environments and `configs/local.json` remain local. See `REPOSITORY_SCOPE.md` for restoration requirements and current limits.
