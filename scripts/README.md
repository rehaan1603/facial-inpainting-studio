# Implementation and reproduction

Run from the repository root using the prepared laptop environments and permitted cache/data. Inspect frozen protocols before GPU experiments. Versioned runners generally require fresh output directories; completed folders retain evidence.

## Application entry points

| File | Role |
|---|---|
| [launch_studio.py](launch_studio.py) | Start Studio; reuses matching loaded code or starts current code on a free local port |
| [server](../webapp/server.py) | HTTP validation, job dispatch and local outputs |
| [inpaint.py](inpaint.py) | LaMa / ResShift single-image inpainting |
| [reference_inpaint.py](reference_inpaint.py) | Frozen research SDXL + FaceID generator |
| [studio_reference_inpaint.py](studio_reference_inpaint.py) | Opaque-cover removal and optional gentler mode |
| [studio_reference_engine.py](studio_reference_engine.py) | Compact processing and optional CFG3 engine |
| [studio_match_edges.py](studio_match_edges.py) | Isolated optional colour matching |
| [finishing](../webapp/finishing.py) | Raw-result fallback for optional worker failure |
| [studio version](../webapp/version.py) | Code fingerprint recorded at server startup for freshness checks |
| [prepare_reference_runtime_v2.py](prepare_reference_runtime_v2.py) | Reference runtime setup |

The HTTP server uses project `.venv`; reference generation/colour matching uses `cache/reference_env_v2/Scripts/python.exe`. ReF-LDM uses a separate recorded runtime. Paths come from `configs/local.json`.

```powershell
.venv/Scripts/python.exe webapp/server.py --port 8766
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_webapp.py
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_studio_finishing.py
```

## Three-person masked studies

| Stage | Runner | Scoring / reporting |
|---|---|---|
| Erased regions | [audit](audit_masked_task_v1.py) | [report](report_masked_task_v1.py) |
| SDXL control | [runner](run_masked_sdxl_v1.py) | [masked scorer](score_masked_sdxl_v1.py) |
| Test sets | [prepare](prepare_three_reference_sets_v1.py) | [report](report_focused_reference_v1.py) |
| Finishing control | [runner](focused_reference_finish_v1.py) | Same frozen masked evaluator |
| Guidance 5 → 3 | [runner](run_focused_eye_guidance_v1.py) | [scorer](score_focused_eye_guidance_v1.py), [report](report_focused_eye_guidance_v1.py) |
| Three-seed guidance | [runner](run_focused_eye_guidance_replication_v2.py) | [scorer](score_focused_eye_guidance_replication_v2.py), [report](report_focused_eye_guidance_replication_v2.py), [verifier](verify_focused_eye_guidance_replication_v2.py) |
| Scheduler | [runner](run_focused_eye_scheduler_v1.py) | [scorer](score_focused_eye_scheduler_v1.py), [report](report_focused_eye_fidelity_v1.py) |
| Appearance | [pinned download](download_eye_appearance_models_v1.py), [runner](run_focused_eye_appearance_v1.py) | [scorer](score_focused_eye_appearance_v1.py), [report](report_focused_eye_appearance_v1.py) |
| Guidance/appearance | [runner](run_focused_eye_factorial_v1.py) | [scorer](score_focused_eye_factorial_v1.py), [report](report_focused_eye_factorial_v1.py), [verifier](verify_focused_eye_appearance_v1.py) |
| Rejected texture | [runner](run_focused_eye_texture_v1.py) | [scorer](score_focused_eye_texture_v1.py), [report/verifier](report_focused_eye_texture_v1.py) |
| FreeU decoder | [runner](run_focused_eye_freeu_v1.py) | [scorer](score_focused_eye_freeu_v1.py), [report](report_focused_eye_freeu_v1.py) |
| Query-local appearance | [runner](run_focused_eye_local_appearance_v1.py), [attention helper](local_appearance_processor_v1.py) | [scorer](score_focused_eye_local_appearance_v1.py), [report](report_focused_eye_local_appearance_v1.py), [attention tests](test_local_appearance_processor_v1.py) |
| Denoising 30→50 | [runner](run_focused_eye_steps_v1.py) | [scorer](score_focused_eye_steps_v1.py), [report](report_focused_eye_steps_v1.py) |
| SSIM round provenance | [verifier/export](verify_focused_eye_ssim_studies_v1.py) | Photo-free numeric records, original model pins, exact controls and visible-pixel checks |
| Eye/brow metrics | [diagnostic](diagnose_eye_brow_fidelity_v1.py) | Offline target annotations; never generator input |
| Live app replays | [GPU/API test](test_gentler_reference_website_v1.py) | [report](report_gentler_reference_website_v1.py) |

Generation uses the reference runtime. Scoring uses the prepared evaluation environment with pinned LPIPS/FaceNet/InsightFace packages and caches; read each scorer's subprocess invocation for the exact runtime. Geometry/texture also needs reference-runtime OpenCV. Plain report builders use Pillow/NumPy in project Python.

Order: restore data/runtime → read protocol → generate in a fresh versioned directory → score every attempt → verify hashes/coverage → report complete results. [Release export](export_release_evidence_20261010.py) extracts photo-free metrics from existing evidence without model inference.

Live replay scripts submit real GPU jobs to their specified idle server and refuse existing output folders. Unit-test discovery does not execute their `main()` functions. Saved verification is in [release evidence](../research/releases/2026-10-10/README.md).

## Earlier research/shared evaluation

| File | Role |
|---|---|
| [target_region_compatibility_v1.py](target_region_compatibility_v1.py) | 193-parameter regional compatibility module |
| [reference_intervention_core.py](reference_intervention_core.py) | Local frozen ReF-LDM integration |
| [spatial_reference_rejection_v1.py](spatial_reference_rejection_v1.py) | Controlled reference-token exclusion |
| [evaluate_spatial_reference_rejection_v1.py](evaluate_spatial_reference_rejection_v1.py) | Matched controls/interventions |
| [score_spatial_reference_rejection_v1.py](score_spatial_reference_rejection_v1.py) | Metrics and valid-pair coverage |
| [extended_evaluation_metrics.py](extended_evaluation_metrics.py) | Pinned metrics/provenance |
| [run_evaluation_suite.py](run_evaluation_suite.py) | Earlier reference-count/ablation suite |
| [run_remaining.ps1](run_remaining.ps1) | Historical training/object-test workflow |

Use [research navigation](../research/README.md) for losses, protocols and all negative findings. Whole-image restoration, masked inpainting and the separate Human Faces pilot have distinct evaluation tables. [Release verification](../research/releases/2026-10-10/verification.json) records the checked release scope.
