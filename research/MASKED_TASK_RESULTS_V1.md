# Erased facial-region diagnostic: results

8 October 2026. Completed 96 matched ReF-LDM outputs, eight separate native ResShift controls and one byte-exact restoration replay. No training or production changes.

## Main result

The frozen restoration baseline and its unchanged timestep-scaled identity adapter retain largely blank eye/mouth regions in the inspected seed-17 clean comparison sheets. The adapter does not solve missing-region reconstruction. ResShift constructs facial features, but eye shape, gaze, expression and boundary quality still differ from the original. This is a task-capability diagnosis, not a publication superiority claim.

## Matched ReF-LDM comparison

Four reused development identities × two locations × three seeds × two reference conditions × two arms. PSNR/SSIM below are measured inside the actual hole; LPIPS is its bounding crop with context. Identity similarity uses the composed full face. Each row has 12 pixel/FaceNet pairs; ArcFace has nine jointly valid pairs. These numbers are not comparable to earlier whole-image restoration results.

| Condition / hole | Arm | Hole PSNR ↑ | Hole SSIM ↑ | Hole crop LPIPS ↓ | FaceNet ↑ | ArcFace ↑ (n=9) |
|---|---|---:|---:|---:|---:|---:|
| clean/eyes | baseline | 12.974779 | 0.361417 | 0.703150 | 0.519036 | 0.757781 |
| clean/eyes | v5 adapter | 12.972395 | 0.361117 | 0.703034 | 0.523102 | 0.757050 |
| clean/mouth | baseline | 14.495379 | 0.411273 | 0.672465 | 0.858736 | 0.850165 |
| clean/mouth | v5 adapter | 14.493782 | 0.410910 | 0.671491 | 0.859938 | 0.848634 |
| wrong_patch/eyes | baseline | 12.994251 | 0.361560 | 0.702712 | 0.513400 | 0.758927 |
| wrong_patch/eyes | v5 adapter | 12.991900 | 0.361648 | 0.703136 | 0.512306 | 0.757706 |
| wrong_patch/mouth | baseline | 14.527211 | 0.411010 | 0.672619 | 0.860678 | 0.851364 |
| wrong_patch/mouth | v5 adapter | 14.520704 | 0.410758 | 0.672184 | 0.861109 | 0.850377 |

No joint success: hole PSNR declines in all four strata; corrupted-eye LPIPS and FaceNet worsen, and ArcFace declines in every stratum. Small aggregate changes do not compensate for missing facial structure.

## Native mask-conditioned control

Seed17 only, four identities per location. ResShift uses no references and native256 followed by512 compositing. ReF-LDM uses native512 and four clean references. These deliberately different backends are functional controls, not an equal-architecture or paper-reproduction comparison. The table restricts ReF-LDM to the same seed17 for clarity.

| Hole | Backend | Hole PSNR ↑ | Hole crop LPIPS ↓ | FaceNet ↑ |
|---|---|---:|---:|---:|
| eyes | ReF-LDM baseline, seed17 | 13.071687 | 0.695136 | 0.529802 |
| eyes | ResShift, seed17 | 18.180893 | 0.232105 | 0.719817 |
| mouth | ReF-LDM baseline, seed17 | 14.517904 | 0.693037 | 0.857712 |
| mouth | ResShift, seed17 | 18.549180 | 0.280991 | 0.861152 |

Native ArcFace support is only 3/4 per location. Its means are retained in JSON; no claim of improved ArcFace follows from this control.

## Verification and scope

- 104/104 composites preserve visible pixels exactly; all generated and raw image hashes verified.
- 8/8 erased inputs pass hidden-pixel invariance; diagnostics use only erased RGB and supplied references.
- Fresh restoration replay is byte-exact. The adapter checkpoint remains unchanged.
- Eight seed-17 clean comparison sheets were inspected across all identities/locations. This is descriptive inspection, not a blinded study of all 96 images.
- Four development identities, annotation-derived masks, unresolved upstream pretraining exposure and incomplete ArcFace coverage remain limitations.
- Measured ReF-LDM sampling/decoding time totals 430.7 seconds; maximum recorded allocated GPU memory 2687.7 MiB. Excludes loading, preparation, evaluation and native control. No evidence here attributes blank outputs to exhausted GPU memory.

## Revised next experiment

Do not begin the proposed ReF-LDM masked-loss pilot yet. The prerequisite functional screen failed. Keep ReF-LDM available for its separate blur/noise-restoration task. Next test the existing genuinely mask-conditioned, reference-guided SDXL path on these same frozen inputs and references, starting with the seed17 clean screen, then the identical corrupted-reference and multiseed grid. Hold selection, blending and model settings fixed. This identifies whether reference-conditioned inpainting provides a useful reconstruction baseline before one isolated training or reliability intervention.

No new dataset, synthetic target-derived references, checkpoint overwrite or website default change was introduced.

[Protocol](MASKED_TASK_PREFLIGHT_V1.md) · [Numerical results and verification](masked_task_v1_results.json) · [Earlier audit and conditional plan](MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md)

Local comparison images: `outputs/masked_task_v1/sheets/`. Dataset images remain local and are not included in the research JSON.
