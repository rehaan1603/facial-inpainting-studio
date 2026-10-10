# Research navigation

Updated 10 October 2026. Read the [current project report](../PROJECT_REPORT.md) first. Dated protocols preserve all experiment outcomes. [Release evidence](releases/2026-10-10/README.md) supplies photo-free per-output metrics.

## Current masked facial-region inpainting

This track tests missing eye/mouth regions. Its generator and supports differ from earlier whole-image ReF-LDM restoration.

| Stage | Protocol / audit | Results |
|---|---|---|
| Architecture, loss and metric audit | [8 October audit](MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md) | Frozen/trainable roles and experiment plan |
| Erased-region diagnostic | [masked protocol](MASKED_TASK_PREFLIGHT_V1.md) | [results](MASKED_TASK_RESULTS_V1.md), [JSON](masked_task_v1_results.json) |
| Mask-conditioned SDXL controls | [SDXL](MASKED_SDXL_PREFLIGHT_V1.md), [resolution](MASKED_SDXL_RESOLUTION_PREFLIGHT_V1.md) | [three-person results](FOCUSED_REFERENCE_RESULTS_V1.md), [JSON](focused_reference_v1_results.json) |
| Guidance 5 → 3 | [protocol](FOCUSED_EYE_GUIDANCE_PREFLIGHT_V1.md) | [seed 17 results](FOCUSED_EYE_GUIDANCE_RESULTS_V1.md) |
| Guidance seeds17/29/43 | [replication protocol](FOCUSED_EYE_GUIDANCE_REPLICATION_PREFLIGHT_V2.md) | [18-output results](FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md), [verification](focused_eye_guidance_replication_v2_verification.json) |
| DDIM versus Euler | [protocol](FOCUSED_EYE_SCHEDULER_PREFLIGHT_V1.md) | [comparison](FOCUSED_EYE_FIDELITY_RESULTS_V1.md); Euler loses mean identity |
| Extra visual-reference branch | [protocol](FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md) | [results](FOCUSED_EYE_APPEARANCE_RESULTS_V1.md), [verification](focused_eye_appearance_verification_v1.json); mean SSIM decreases |
| Guidance/appearance interaction | [protocol](FOCUSED_EYE_FACTORIAL_PREFLIGHT_V1.md) | [four arms](FOCUSED_EYE_FACTORIAL_RESULTS_V1.md); SSIM/ArcFace regress versus CFG3 |
| Registered eye/brow texture | [protocol](FOCUSED_EYE_TEXTURE_PREFLIGHT_V1.md) | [rejected results](FOCUSED_EYE_TEXTURE_RESULTS_V1.md), [verification](focused_eye_texture_verification_v1.json); all six means worsen |
| Fixed FreeU decoder | [audit/protocol](FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md) | [results](FOCUSED_EYE_FREEU_RESULTS_V1.md), [metrics](focused_eye_freeu_v1_metrics.json); SSIM/identity decrease |
| Hole-local appearance attention | [protocol](FOCUSED_EYE_LOCAL_APPEARANCE_PREFLIGHT_V1.md) | [results](FOCUSED_EYE_LOCAL_APPEARANCE_RESULTS_V1.md), [metrics](focused_eye_local_appearance_v1_metrics.json); two SSIM gains, combined gate fails |
| More denoising 30→50 | [protocol](FOCUSED_EYE_STEPS_PREFLIGHT_V1.md) | [results](FOCUSED_EYE_STEPS_RESULTS_V1.md), [metrics](focused_eye_steps_v1_metrics.json); SSIM/perceptual regression |
| Direct SDXL training feasibility | [architecture and losses](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V1.md), [backend recovery](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V2.md), [bounded 512](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V3.md) | [all three retained attempts](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_RESULTS_V1.md), [numeric evidence](masked_sdxl_compatibility_feasibility_v1.json) |
| Training-reference eligibility recovery | [fixed revision](MASKED_SDXL_REFERENCE_ELIGIBILITY_RECOVERY_V2.md), [v2 cache](MASKED_SDXL_COMPATIBILITY_DATA_PREFLIGHT_V2.md) | Original v1 cache failed; eight original people retain unchanged targets with four eligible references each; tests unchanged |
| 385-parameter masked SDXL training | [fixed 64-update pilot](MASKED_SDXL_COMPATIBILITY_PILOT_PREFLIGHT_V2.md), [enabled-zero replay](MASKED_SDXL_ZERO_STATE_REPLAY_PREFLIGHT_V1.md) | [complete results](MASKED_SDXL_COMPATIBILITY_PILOT_RESULTS_V2.md), [photo-free evidence](masked_sdxl_compatibility_pilot_v2_evidence.json); six mean gains over neutral local, fails current CFG3/visual gates |
| Anatomical diagnostics | Supplementary annotation supports | [eye/brow diagnostic](EYE_BROW_FIDELITY_DIAGNOSTIC_V1.md); sparse valid windows |
| App integration | Optional CFG3, default off | [four actual replays](GENTLER_REFERENCE_WEBSITE_V1.md) |

Three-person repeated guidance has mean hole SSIM **0.414146 → 0.433098**; person 10149 loses **0.013618 mean FaceNet cosine**. Iris/gaze/brow reconstruction remains imperfect. Positive averages and per-person failures both remain visible. These are development comparisons, not arbitrary unknown-client accuracy or a matched paper benchmark.

The 10 October inference round completes 13 GPU calls with four exact replays; none of the three recipes beats the current CFG3 combined gate. [Verification and pinned model assets](focused_eye_ssim_studies_v1_verification.json). Its seed-17 CFG3 mean SSIM 0.454540 is separate from the three-seed 0.433098 result above. The subsequent masked training pilot completes 64 GPU updates and five inference calls (two exact neutral replays plus three trained outputs). SSIM is 0.445016 versus 0.444884 neutral local and 0.454540 current CFG3; person 263 fails the identity guard. The trained candidate remains offline. A decoded masked-region objective needs a separately audited gradient/resource check before further training.

## Earlier ReF-LDM restoration and adapters

These experiments measure whole-image blur/noise restoration. The compact regional compatibility module has 193 trainable parameters; pretrained backbone/recognition models stay frozen. Each report specifies its own encoding, data, seeds, losses and valid-pair counts.

| Investigation | Evidence |
|---|---|
| Target-reference mechanism | [method report](FINAL_METHOD_REPORT.md), [papers](PAPERS_AND_IMPLEMENTATION_MAP.md) |
| Identity objective v2 | [per-row data](identity_compatibility_v2_results.json), method report above |
| Evaluation-aligned crop v3 | [results](IDENTITY_CROP_RESULTS_V3.md) |
| Balanced identity and gradient cap v4 | [results](IDENTITY_BALANCE_RESULTS_V4.md) |
| Timestep-scaled objective v5 | [results](IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md) |
| Official input/reference encoding | [one case](REFLDM_ENCODING_CHECK_V1.md), [expanded screen](REFLDM_ENCODING_SCREEN_V2_RESULTS.md) |
| Whole-reference oracle removal | [results](REFERENCE_REJECTION_PROBE_V1_RESULTS.md) |
| Spatial token exclusion | [protocol](SPATIAL_REFERENCE_REJECTION_PREFLIGHT_V1.md), [results](SPATIAL_REFERENCE_REJECTION_RESULTS_V1.md), [verification](spatial_reference_rejection_v1_verification.json) |

These studies retain identity/perceptual trade-offs. Oracle corruption locations are diagnostic inputs. Whole-image restoration gains cannot substitute for masked reconstruction evidence.

## Foundations, data and publication

| Subject | Start here |
|---|---|
| LaMa/ResShift and trained mask refinement | [initial results](FINAL_RESULTS.md), [earlier reproduction](REPRODUCE_EARLIER_STAGES.md) |
| NIQE, BRISQUE, identity, reference count | [expanded results](EXTENDED_ABLATION_RESULTS_V1.md), [protocol](EXTENDED_EVALUATION_PROTOCOL_V1.md) |
| Selection/routing history | [generalization](GENERALIZATION_RESULTS.md), [regional routing](REGIONAL_ROUTING_RESULTS_V2.md) |
| Unknown-client checks and failures | [client fixes](STUDIO_CLIENT_FIXES_20260924.md), [unfamiliar-image audit](UNFAMILIAR_STUDIO_ACCURACY_V2.md) |
| Dataset/sample restoration | [official sources](DATASET_DOWNLOADS_AND_SAMPLES.md) |
| Data/model restrictions | [terms](DATA_AND_LICENSES.md), [reference runtime lock](reference-environment-v2-lock.txt) |
| Paper comparisons | [paper/implementation map](PAPERS_AND_IMPLEMENTATION_MAP.md); retain original benchmark settings |
| Separate Human Faces pilot | [limited measurements](human_resshift_lora_v1.json); source rights/person overlap unresolved |
| Submission work | [remaining work](../PROJECT_REPORT.md#4-remaining-work), [manuscript draft](MANUSCRIPT.md) |

Historical decks/reports are dated snapshots. The root report identifies the current direction. Image pixels, reference features and downloaded weights remain local; hashes and metric records retain traceability.
