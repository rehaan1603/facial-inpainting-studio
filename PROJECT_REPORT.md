# Facial Inpainting Project — Progress and Completion Report

**Report date:** 14 September 2026  
**Current platform:** Windows laptop, NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM  
**Current application:** http://127.0.0.1:8765/  
**Repository:** https://github.com/rehaan1603/facial-inpainting-studio  
**Status:** Working local research baseline; proposed multi-reference research method and final validation incomplete. Public hosting paused at the owner's request.

## 1. Project objective and scope

The application reconstructs a masked facial region from the observed image and a supplied mask. It supports single-image completion and a reference-assisted mode accepting three or four photographs of the same person.

The research objective is to determine whether selecting and combining reference evidence according to the missing facial region improves identity fidelity while maintaining reconstruction quality and preserving visible content. Candidate selection using identity and quality measurements is a possible subsequent component. These proposed mechanisms are not yet demonstrated contributions.

There are two distinct bodies of work:

1. **Earlier single-image research:** inaccurate masks, compact mask refinement, LaMa/ResShift comparisons and controlled synthetic occlusion experiments.
2. **Current multi-reference research:** a functional pretrained diffusion/identity-adapter baseline, a completed small development diagnostic and the proposed Retrieve–Fuse–Verify architecture.

The earlier results cannot be represented as experiments on a regional reference-fusion model that has not been implemented.

## 2. Completed work

### 2.1 Dataset preparation and traceability

- Decoded and hashed 52,168 supplied images: 30,000 CelebAMask-HQ images and 22,168 LaPa images.
- Linked HQ photographs to the supplied CelebA identity annotations and official partitions through the mapping file.
- Completed duplicate screening and review of 496 proposed near-duplicate pairs. Reviewed manifests retain 29,926 HQ and 21,994 LaPa images.
- Retained source hashes, selection records, exclusions and dataset/model provenance. These checks do not prove complete transformed-copy removal or unknown pretraining independence.
- Indexed 14,684 local output images at the recorded snapshot. This includes historical experiments, masks, inputs, results and previews; it is not a count of independent subjects or successful reconstructions.
- Implemented and verified local restoration of 91 exact PNG files from 65 original photos across 13 reference examples: one engineering example and twelve development cases.
- Kept dataset photos and derived examples local. GitHub contains official download links, selection records and restoration instructions, following the owner's chosen release approach.

### 2.2 Implemented models and local training

| Component | Completed implementation | What it establishes |
|---|---|---|
| LaMa | Local single-image inpainting | Functional pretrained baseline |
| ResShift | Local face-oriented diffusion inference | Functional pretrained baseline |
| Mask refiners | Six compact models, 119,057 parameters each, initially trained for 1,500 updates and extended to 6,000 total updates | Actual local training for mask prediction, not identity-reference fusion |
| SDXL inpainting | Local reference-assisted inference | Existing pretrained generative backbone |
| FaceID Portrait adapter | Three/four-reference website integration | Existing pretrained identity conditioning, not a newly trained project adapter |
| InsightFace buffalo_l | Face detection and normalized reference embeddings | Reference conditioning, not independent identity evaluation |
| Hard/Poisson composition | Known-pixel preservation and optional boundary harmonization | Explicit postprocessing; does not prove correct hidden features |

No new reference-conditioned generative backbone, regional identity-fusion adapter or LoRA has been trained locally.

### 2.3 Completed experiments

| Study | Recorded extent | Interpretation |
|---|---|---|
| Earlier mask-development study | 96 identity labels in the development protocol; 13,824 learned-control evaluation rows | Single-image mask research |
| Object-composite test | 128 images: 64 HQ identity labels and 64 LaPa images; 5,632 rows | Already observed synthetic test; not untouched future confirmation |
| Training-budget extension | Six refiners at 6,000 updates; 27,648 development rows | Reuses 48 assessment identity labels |
| Multi-reference development diagnostic | Twelve selected identity labels, 48 generated candidates, 96 scored rows | Reference scale × denoising strength, each with hard and Poisson composition |
| Local application checks | Real HTTP/GPU inference and browser downloads | Functional evidence on recorded examples |

Rows are correlated experimental measurements, not independent images or independent identities. The studies use different protocols and must not be pooled into one headline performance claim.

### 2.4 Working local website

The site supports image upload, crop/fit, mask painting/erasing/undo, mask upload, model selection, three/four-reference upload, standard or detailed reference processing, output comparison and result/mask downloads. A new input clears the previous person's reference photographs.

Local output sizes are 256 × 256 for LaMa/ResShift and 512 × 512 for reference mode. Detailed reference mode processes at 1024 and exports 512; it is not an established quality improvement. Images are saved locally with run metadata.

The reference runtime was repaired in a separate environment after Windows blocked native dependencies. Package pins and installation records were retained. No Windows protection was disabled. Progress-file retries prevent transient OneDrive locks from needlessly aborting reference inference.

On 14 September:

- Four-reference standard HTTP inference completed in 45.297 seconds including startup and verification, with 16.5 seconds of inference including offload. The output preserved all pixels outside the effective mask.
- LaMa passed exact, expanded and learned-mask checks; ResShift passed exact and expanded-mask checks.
- A browser-driven LaMa run displayed its output, and both result and mask downloads completed.

These timings describe individual checks, not a population latency benchmark. Functionality on these cases does not establish robustness across all user photographs.

### 2.5 Repository and hosting

Code, local refiner checkpoints, experiment records, provenance and reproduction instructions are in the repository. The current verified local checkpoint is commit 5a55eb6; this report is a subsequent documentation addition.

Temporary public sharing was implemented and an HTTPS GPU test completed. The in-app browser then exposed sign-in compatibility problems. Public sharing is now stopped, its homepage link removed, and further hosting work deferred. The local app requires no sign-in. The paused hosting investigation is not a blocker for local research.

## 3. What the results currently support

**Mask refinement has a tradeoff.** At the initial training budget, preservation weighting reduced changes to visible pixels but worsened aggregate reconstruction versus the generic refiner. At 6,000 updates, weighted refinement performed better than the generic 6,000-update control on aggregate reconstruction metrics, but the generic model itself deteriorated with longer training. This does not establish that more training universally improves completion or that the refiner is state of the art.

**Reference guidance has not shown a consistent benefit in the completed diagnostic.** At strength 0.99 with Poisson composition, mean LPIPS was 0.032674 with references and 0.032460 with reference conditioning disabled. The paired difference was +0.000214, with an exploratory 95% interval spanning zero. Hole MAE was also worse with references in this setting. Other settings differ; the complete table must be retained rather than selecting the most favorable contrast.

**Identity fidelity is still unmeasured independently.** Realistic output, exact preservation outside the mask and a successful download do not establish correct identity, gaze, expression or missing facial details.

The research is therefore not yet ready to claim the proposed method improves identity preservation or is publication-ready.

## 4. Remaining work

| Area | Missing work | Priority |
|---|---|---|
| Identity evaluation | Independent output embeddings, preprocessing, detection failures, paired identity-level analysis | Immediate |
| Quality evaluation | NIQE/BRISQUE as secondary measures; structural/landmark error; retain LPIPS and masked/visible errors | Immediate |
| Reference protocol | Disjoint conditioning and evaluation photos; stronger transformed-copy checks; adequate identity groups | Immediate |
| Severe damage | Defined face-region/image-area severity, locations, reference variation and failure cases | Before final testing |
| Reference scoring | Pose/quality/visibility measurements and mask-conditioned ranking | Core method development |
| Regional fusion | Actual regional features and a compact learned mask-aware fusion module | Conditional on baseline findings |
| Candidate selection | Multiple seeds, quality/identity selection and separate reporting evaluator | After scoring/fusion baseline |
| Ablations | A1–A7 or a justified reduced protocol with matched training and sampling costs | Required for claimed mechanisms |
| Close comparisons | Reproducible reference-based competitors; complete or explicitly delimit paused OSOR comparison | Required for comparative claims |
| Verification statistics | Genuine/impostor trials, calibrated thresholds, ROC/TAR only if data supports low-FAR claims | Conditional on dataset feasibility |
| Distributional quality | FID on an appropriately sized, controlled evaluation set | Later; not meaningful on the current small reference set |
| Text control/CLIP | Optional attribute experiments | Deferred |
| Final paper | Specific venue, corrected literature/architecture claims, figures, limitations, reproducibility and author review | After final evidence |
| Public hosting | Browser-compatible access and a suitable long-term deployment | Paused |

NIQE, BRISQUE, FID, SSIM, independent identity similarity, ROC/TAR and reconstructed-landmark error are not part of the completed reference metric set. Earlier pilot code does include hole PSNR. Dataset annotation syntax checks are not reconstructed-landmark accuracy measurements.

## 5. Planned execution and acceptance criteria

### Phase 1 — establish the measurement baseline

Add a separately specified identity evaluator without disrupting the working inference environment. Freeze checkpoint hashes, preprocessing, alignment and detector-failure treatment. Score existing development results with independent identity similarity and secondary no-reference/structural measures. Retain every condition and failed detection.

**Deliverable:** reproducible per-case metrics and paired summaries.  
**Acceptance criterion:** clean target information is used only for permitted scoring, and no selection encoder is mislabeled as an independent final evaluator.

### Phase 2 — prepare defensible reference splits

Count eligible multi-photo identity groups and reserve conditioning, calibration and final-evaluation photos/identities. Four input references, three withheld evaluation references and one target require eight suitable distinct photos per case. Freeze severity definitions and corruption/reference strata. Determine whether low-FAR verification claims are supported by the number of identities and trials.

**Deliverable:** versioned manifests and a frozen protocol.  
**Acceptance criterion:** no target/reference leakage or reuse of observed development cases as fresh final evidence; limitations recorded explicitly.

### Phase 3 — test simple reference selection

Compare random single-reference, all-reference and global-quality baselines against a mask-aware scorer while keeping the generator and candidate seeds fixed. Use only information available from the damaged input, supplied mask and input references. Diagnose whether poor results arise from visibility estimation, geometry or inadequate global features.

**Deliverable:** implemented selection rules and controlled comparison.  
**Acceptance criterion:** the proposed scoring mechanism has measurable support, or its failure is documented before adding complexity.

### Phase 4 — develop the smallest justified fusion method

If earlier results support the direction, implement regional reference features and a compact fusion adapter. Compare equal weighting, global attention and mask-aware fusion at matched capacity and training budgets. Profile actual laptop memory and speed before committing to a large training schedule. Preserve the working pretrained baseline and all frozen earlier results.

**Deliverable:** trained candidate module, checkpoints, loss histories and mechanism ablations.  
**Acceptance criterion:** reproducible improvement on declared measures, acceptable failure behavior and a defensible distinction from closest prior work. Implementation alone is not evidence of novelty.

### Phase 5 — evaluate candidate selection

Compare one, two and four sequential candidates first; expand to eight/sixteen only if worthwhile. Use the same candidate pool for random, identity-only, quality-only and constrained selection. Normalize scores with development data and respect their directions. Use a reporting evaluator not used to rank candidates. Include rejection rate and total added inference cost.

**Deliverable:** matched-budget reranking comparison.  
**Acceptance criterion:** improvement beyond the effect of extra sampling alone, with no hidden access to evaluation targets or gallery images.

### Phase 6 — final experiments and submission package

Freeze the final method before evaluating reserved identities. Run the chosen severe-mask protocol, relevant external comparisons and failure analysis. Keep real damaged photographs separate from synthetic cases with known ground truth. Revise the deck and manuscript to match the implemented system and supported results. Select an actual IEEE conference or journal with the project guide; IEEE Xplore itself is the library, not the submission destination.

**Deliverable:** final tables, figures, manuscript, corrected presentation and reproducible repository.  
**Acceptance criterion:** claims trace to saved evidence, comparisons are fair, data/model restrictions and pretraining uncertainty are disclosed, and the target venue's submission requirements are met.

No guaranteed completion date or acceptance probability is assigned. Training feasibility, eligible reference groups and development outcomes must be measured first. If the proposed method does not improve results, retain the negative evidence and revise the claim or research direction rather than manufacture a successful result.

## 6. Required corrections to the v6 proposal

- Slide 9 uses ArcFace-R100 for selection while slide 13 excludes it from selection. Define separate conditioning, ranking and final-evaluation roles.
- Replace the unsupported absolute novelty statement with a precise hypothesis and comparison against close prior work.
- Label Retrieve/Fuse/Verify and A1–A7 as planned until executed. Existing mask-refiner training is not regional identity-fusion training.
- Clarify the three-reference proposal versus four-reference current diagnostic, and the need for additional withheld gallery photos.
- Define face-region mask severity separately from whole-image coverage and conditional generation with the whole face missing.
- Correct the literature count: twelve table entries, despite headers implying sixteen. Verify exact bibliography and method limitations.

Detailed mapping: [v6 slide audit](research/SLIDE_V6_IMPLEMENTATION_AUDIT.md).

## 7. Running the local application

Double-click Start Studio.cmd, or run in PowerShell:

```powershell
cd "C:\Users\rehaa\OneDrive\Documents\ChatGPT\Facial inpainting Project"
& .\.venv\Scripts\python.exe scripts\launch_studio.py
```

Open http://127.0.0.1:8765/. Use matching references for the selected person, mark the complete damaged region and inspect the generated output. The laptop must be running; no public internet hosting is required.

## 8. Evidence index

- [Current local verification](research/LOCAL_WORK_STATUS_20260914.md)
- [Dated LaMa/ResShift checks](research/local_inference_checks_20260914.json)
- [Reference HTTP/GPU checks](research/reference_webapp_repair_checks.json)
- [Initial single-image results](research/FINAL_RESULTS.md)
- [Training extension results](research/TRAINING_EXTENSION_RESULTS.md)
- [Reference diagnostic results](research/REFERENCE_DIAGNOSTIC_RESULTS.md)
- [Reference diagnostic protocol](research/REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md)
- [Prior-method audit](research/REFERENCE_METHOD_AUDIT.md)
- [Dataset downloads and restoration](research/DATASET_DOWNLOADS_AND_SAMPLES.md)
- [Data and model provenance](research/DATA_AND_LICENSES.md)

This report documents existing evidence and planned work. Preparing it does not train a model, add a metric, establish novelty or complete the remaining experiments.
