# Complete research record in this private repository

The source application is in webapp; its launchers are Start Studio.cmd and scripts/launch_studio.py. Model inference, data preparation, training, evaluation and report scripts are in scripts. Configuration templates and experiment settings are in configs; environment locks and readable reports are in research.

Detailed artifacts now include:

| Location | Contents |
|---|---|
| data/manifests | Original and reviewed dataset manifests; image-integrity records |
| outputs/pilot and outputs/pilot_clean_v1 | Original pilot settings and measurements |
| outputs/benchmark_v2 and outputs/benchmark_v2_resshift | Development case definitions and baseline records |
| outputs/control_sweep_* | Mask expansion and blending controls |
| outputs/area_matched_v3* | Original/revised area-matched protocols and measurements |
| outputs/learned_refiner | Six initial trained refiners: best/latest checkpoints and training histories |
| outputs/refiner_evaluation | Full first-stage learned-control evaluation |
| outputs/refiner_extension_v1 | Six extended runs: best/latest checkpoints and training histories |
| outputs/extension_evaluation_v1 | 27,648 extended-training/compositing evaluation rows |
| outputs/object_test | Frozen object-composite protocol and 5,632 measured rows |
| outputs/near_duplicate_audit | Fingerprints, candidate pairs and review metadata |
| outputs/osor_evaluation_v1 | Partial comparator records; not a completed comparison |
| outputs/reference_examples and outputs/reference_diagnostics_v1 | Reference selection, source hashes, exclusions and prepared diagnostic metadata |
| outputs/webapp_runs | Saved reconstruction settings and progress/error metadata; photographs excluded |

research/REPOSITORY_ARTIFACT_INVENTORY.json is the authoritative file list with SHA-256 hashes and byte sizes. It distinguishes locally trained checkpoints from text research records. Presence of a protocol or partial run is not evidence of completed evaluation. See research/PROJECT_STATUS.md and research/QUALITY_FIX_STATUS.md for current limitations.

Source, reference, uploaded and generated face photographs; downloaded third-party models; virtual environments; local secrets/settings; and ZIP backups remain on the laptop. They are not necessary to inspect the research record, but inference and reproducing the image-based experiments require restoring the permitted data/model cache and configs/local.json. All existing project files remain intact locally.
