# Slide v6 audit and submission-readiness plan

Checked 14 September 2026 against the 15 slides in mask_aware_facial_inpainting_v6.pptx, the current implementation and recorded experiments. The supplied PowerPoint was read without modification. Slide descriptions are proposals, not evidence of implemented or evaluated functionality.

## Main finding

The project has a working local inpainting baseline and substantial earlier mask-refinement experiments. It does not yet implement the deck's complete Retrieve → Fuse → Verify research system. The proposed mask-conditioned reference scorer, regional fusion module, candidate reranker and severe-occlusion biometric evaluation remain unimplemented. The deck cannot currently be presented as a completed-method or results presentation.

The earlier 96-identity mask study and the newer twelve-identity reference study answer different questions. Neither can be used as evidence that the proposed regional fusion system works. In particular, the trained 119,057-parameter mask refiners predict masks from observed RGB and a supplied mask; they are not trained identity-token fusion modules.

## Component-by-component audit

| Slides / item | What exists now | Why the difference matters | Next decision |
|---|---|---|---|
| 1–4: identity-preserving, mask-aware multi-reference system | Functional reference-assisted baseline, without demonstrated identity preservation | A runnable baseline does not establish the proposed contribution | Keep as proposed research until supported by independent measurements |
| 2–3: severe occlusion | Reference diagnostic uses small synthetic eye masks, roughly 4–9% of image area | This is not a >50% missing-face benchmark | Add declared face-region severity strata and report image-area coverage too |
| 3: no previous method scores references using mask geometry | Not established by current literature evidence | An absolute absence claim needs a much closer novelty comparison | Replace with a bounded research hypothesis and compare relevant methods |
| 4 O1: estimate yaw/pitch, sharpness, illumination and occlusion rate | Face detection, detection score, landmarks and file-duplicate checks exist | Detection confidence is not an image-quality or regional-visibility score | Implement explicit per-reference measurements with documented failure behavior |
| 4 O1: mask-conditioned overlap, regional relevance, top-K selection | Not implemented | Present CLI passes all supplied references; it does not rank them by the missing region | First compare fixed random, global-quality and mask-aware scoring rules |
| 4 O2: landmark-aligned patch identity features | Not implemented | Existing normalized face embeddings are global descriptors; projecting one into multiple tokens does not make them aligned image patches | Choose and validate a regional feature representation before fusion training |
| 4 O2: learned mask-conditioned fusion | Not implemented | Pretrained adapter tokens participate in existing attention, but there is no new trained mask-aware fusion module | Build a small versioned fusion adapter only after simple scoring baselines |
| 4/9: diffusion conditioning | Implemented using SDXL inpainting and the pretrained FaceID Portrait adapter | This is an existing model integration | Retain as baseline and keep its settings fixed in comparisons |
| 4/9: generate 8–16 candidates, rerank or regenerate | Not implemented | Current website produces one seeded candidate per request; diagnostic settings are experimental conditions, not an inference-time candidate search | Begin with 1/2/4 sequential candidates, then benchmark larger counts if worthwhile |
| 9: ArcFace-R100 reranker | Not implemented | Existing InsightFace buffalo_l w600k_r50 is used for conditioning, not an R100 candidate-selection system | Specify the selection encoder separately from the final evaluator |
| 4/9/13: NIQE and BRISQUE | Not implemented or measured | They measure no-reference image quality, not identity accuracy; adding them alone will not repair a face | Add as secondary diagnostics, with fixed implementation, crop, scale and failure handling |
| 4/13: structural/landmark reconstruction error | Not implemented | Landmarks used for dataset cropping or mask construction are not reconstruction-error measurements | Add normalized landmark error and detection failure rate; state that detector outputs are proxy labels |
| 9/10: quality-constrained selection | Not implemented | A weighted sum alone does not enforce a quality constraint; metrics have different scales and directions | Compare identity-only selection, quality-only selection and a validation-calibrated constrained policy |
| 10: four contributions C1–C4 | Research proposals | None has been validated as a novel contribution | Reduce to one primary contribution plus supporting experiments unless evidence justifies more |
| 11: Python, PyTorch, NumPy, OpenCV, Hugging Face diffusion stack, CUDA | Used | Software availability is not proof of methodological novelty | Keep pinned environments and measured laptop costs |
| 11: ArcFace / FaceNet biometric evaluation | ArcFace-style InsightFace conditioning only; no independent output identity evaluator | Conditioning and evaluation are different roles | Implement independent output similarity first; pin exact architecture/checkpoint/training source |
| 11: adapter/LoRA fine-tuning | Existing pretrained adapter loaded; no local identity adapter or LoRA training | Local mask-refiner training is a separate experiment | Test memory feasibility for a small trainable module before promising training on 8 GB |
| 11: VS Code/Jupyter, RAM and SSD specification | Local Python/scripts are verified; deck tools and hardware targets are not all individually verified here | Mention only tools actually used and measured hardware, not every planned tool | Report actual runtime and hardware in the final paper |
| 12: CelebA-HQ and CelebAMask-HQ | HQ photos and parsing annotations used; identity grouping comes through the HQ mapping and CelebA identity annotations | They overlap rather than being independent evaluation datasets | Document the mapping explicitly and avoid double-counting the same photos |
| 12: FFHQ | Not used as a locally audited project evaluation set; possible pretrained-model exposure is separately documented | A model trained on FFHQ is not an FFHQ evaluation experiment; original FFHQ is not automatically a multi-photo identity benchmark | Optional external single-image/generalization test, or a separately licensed and validated reference benchmark |
| 12: LaPa | Used in earlier single-image studies, although omitted from this deck | Those results remain relevant to that earlier scope | Include in history/method comparisons without claiming it supplies validated identity reference groups |
| 12: target plus up to three references | Current CLI accepts 1–4; website accepts 3–4; completed diagnostic uses four | Reference count affects comparability and inference cost | State the actual setting and freeze reference-count ablations |
| 13: held-out matcher cosine similarity | Not measured yet | No current numerical identity-fidelity conclusion is warranted | First evaluation priority |
| 13: TAR at FAR 0.1% and ROC AUC | Not implemented | Requires genuine/impostor trials, separate threshold calibration and enough independent identities | Add only after a sufficiently large verification protocol is feasible; report uncertainty and failures |
| 13: FID on at least 5,000 crops | Not computed | 48 reference candidates cannot support the proposed distributional study; 5,000 repeated outputs do not mean 5,000 independent subjects | Defer until sample size and compute justify it; freeze extractor, crops and equal sampling across methods |
| 13: CLIP score and text control | A fixed text prompt is used by the model; no controllable text experiment or CLIP evaluation | Using a text encoder internally is not measuring CLIP score | Defer optional text editing until the main identity question is resolved |
| 13: A1–A7 ablations | Not completed | Current 2×2 reference-scale/strength comparison plus compositing does not implement A1–A7 | Implement the actual variants and use matched seeds, references and compute budgets |
| 13: three withheld evaluation references per identity | Not prepared in the current twelve-person manifest | Current selected groups contain one target plus four conditioning photos | Audit larger groups and reserve disjoint evaluation photos before inference/ranking |
| 14–15: ten-month plan and three reviews | Proposed schedule, not completed milestones | Dates or module labels do not substitute for deliverables | Use evidence-based stage gates below; agree a real deadline and venue with the guide |

Current reference metrics are LPIPS, hole MAE, visible MAE and inner-boundary MAE. Earlier pilot code also computes hole PSNR. No implemented SSIM, FID, NIQE, BRISQUE, ROC/TAR or independent identity-similarity study was found in the audited project scripts. Annotation syntax checks named landmark_error are not geometric output-evaluation metrics.

## Why these were not already done

Implementation followed the initial single-image inaccurate-mask project: dataset auditing, small mask refiners, LaMa/ResShift comparisons and a local app. Multi-reference completion was added later using existing pretrained components. Subsequent work addressed Windows dependency failures, reference-photo mixing, progress-file failures and website functionality. Hosting was then attempted and paused.

This explains the sequence, but does not turn the missing research modules into completed work. There was no experimental finding that made NIQE, BRISQUE or independent identity measurement unnecessary; they simply were not implemented in the completed reference diagnostic. The earlier emphasis on infrastructure has left a material gap between the proposed presentation and the research evidence.

## Corrections needed before presenting the proposal

1. **Separate selection from final evaluation.** Slide 9 uses ArcFace-R100 to select candidates, while slide 13 says ArcFace-R100 is not used in A7 selection. Assign explicit roles to conditioning encoder, candidate selector and independent reporting encoder. A practical first configuration is existing InsightFace conditioning, optional selection with that existing encoder, and an independently specified FaceNet evaluator. If ArcFace-R100 remains the promised final matcher, keep it out of candidate selection and training and document the alternative selector. Different weights do not prove disjoint pretraining data.
2. **Specify what the matcher compares.** Candidate ranking may compare with permitted input references; it must never compare to the clean target or withheld evaluation photos. Final scoring may use clean targets and reserved gallery images after reconstruction. For four input references plus three withheld gallery photos and one target, each case requires eight suitable distinct photos; the three-reference setting requires seven.
3. **Define severity.** State whether 25/50/75% is measured over the image or a face-region mask. Full-face masking still may leave background, hair or head outline. Whole-image masking is a different conditional synthesis problem. Neither establishes recovery of historically true hidden expression or makeup.
4. **Fix score direction and constraints.** NIQE/BRISQUE conventionally reward lower scores; cosine similarity rewards higher scores. Raw addition is unjustified. Fit normalization and quality thresholds using development data only. A genuine constraint rejects candidates that fail it rather than allowing a high identity score to compensate without limit. Include a defined no-acceptable-candidate behavior and report rejection rate.
5. **Avoid circular reporting.** If NIQE selects outputs, NIQE improvement is partly expected by construction. Report independent quality evidence, errors, failure cases and blinded visual assessment as feasible. No-reference quality scores do not guarantee naturalness or identity.
6. **Correct literature language and coverage.** The four literature slides list three papers each: twelve rows, despite headers suggesting sixteen. Check each exact title, venue, year and claimed limitation against the original paper. Adjacent restoration work cannot simply be dismissed because its title does not say inpainting; test transfer or justify the task mismatch. Do not claim the current bounded source check verifies every literature entry.
7. **Correct metric claims.** PSNR/SSIM can penalize pixel/structure differences associated with a changed face; they are not explicit identity-verification measures. FID is distributional. The slide statement that these metrics do not penalize identity drift is too absolute.

## Concrete development sequence

### Gate 1 — establish a usable measurement baseline

Keep the local website working and public hosting paused. Add an evaluator in a separate environment or module so the functioning generator stack remains reproducible. Freeze preprocessing, checkpoint hashes and failure handling. Compute independent identity cosine similarity, NIQE/BRISQUE as secondary indicators, and normalized structural error on the already observed reference diagnostic. Keep existing reconstruction metrics. Publish all conditions and identity-level paired intervals, not only attractive examples. These remain development results.

Exit evidence: reproducible per-case scores, failures included, no reuse of the selection matcher as an independent final evaluator, and no clean-target information entering inference.

### Gate 2 — make the research protocol feasible

Audit available identity groups for conditioning/evaluation separation and transformed duplicates. Predeclare mask severity and location, reference number, poor/irrelevant reference strata, seeds and data exclusions. Use clean labels only for permitted benchmark construction and scoring. Freeze separate training, development/calibration and final identities. Assess genuine/impostor trial counts before promising FAR 0.1%. A thousand impostor trials has only 0.001 empirical resolution and is not sufficient evidence of a reliable low-FAR estimate. Correlated trials need identity-aware uncertainty.

Exit evidence: enough eligible photos/identities and a documented split, or an explicitly narrowed claim. Do not silently replace absent withheld references with conditioning photos.

### Gate 3 — implement reference scoring before a new fusion network

Compare fixed random reference, all references, global-quality selection and simple mask-aware scoring using the same frozen generator and matched inputs. Do not tune reference sets against clean-target likeness. If simple scoring does not help, inspect where geometry, occlusion visibility or the global embedding representation fails before adding a larger model.

Exit evidence: tested A1–A3-style controls and a clear, reproducible failure mechanism worth addressing.

### Gate 4 — test one compact regional-fusion contribution

Build a small adapter that receives real regional reference features and the supplied mask. Compare equal weighting, global attention and mask-aware fusion with matched capacity and training budgets. Freeze the large generator first. Measure peak VRAM, startup and inference cost; the proven 8 GB inference path does not prove that adapter training or multiple-model residency fits. Sequential processing and cached reference features are options to profile, not established results.

Exit evidence: a mechanism demonstrated by ablations, improvement on predeclared measures without unacceptable failures, and a documented distinction from close prior work. If it fails, report the failure or revise the hypothesis rather than calling it novel because it is implemented.

### Gate 5 — evaluate candidate selection fairly

Start at candidate counts 1/2/4, then consider 8/16. Generate sequentially on the laptop. Compare random selection, identity-only, quality-only and the proposed constrained policy using the same candidate pool and compute accounting. Calibrate the policy on development identities. Evaluate final output with an encoder not used for ranking and with unseen gallery photos where the protocol requires them. Report the added total latency and rejected cases.

Exit evidence: candidate selection gives a defensible benefit beyond simply spending more sampling compute. Repeated candidates must not be counted as independent identities.

### Gate 6 — final comparison and manuscript

Run the frozen final protocol once after development choices are fixed. Include closest feasible reference baselines, not only LaMa/ResShift. PATMAT and a suitable reference diffusion method are candidates subject to actual code/weight access, task compatibility and compute. Keep restoration-to-inpainting adaptations explicit. Complete or delimit the paused OSOR study; do not use six cases as a full comparison. Add larger distributional evaluation only if justified, plus varied damage, failures and real-photo demonstrations separated from synthetic ground-truth tests.

Choose the actual IEEE journal or conference with the guide; IEEE Xplore is the library. Revise the deck and manuscript around supported results, verify citations and release rights, document authorship and limitations, and release reproducible code/configuration/results with official dataset restoration links. A public website is not required for this research evidence. Neither a metric checklist nor implementing every slide guarantees acceptance.

## Sources and implementation evidence

- Existing implementation: scripts/reference_inpaint.py; scripts/refiner.py; scripts/evaluate_reference_diagnostics.py; research/REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md; research/REFERENCE_DIAGNOSTIC_RESULTS.md; research/LOCAL_WORK_STATUS_20260914.md.
- [PATMAT, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Motamed_PATMAT_Person_Aware_Tuning_of_Mask-Aware_Transformer_for_Face_Inpainting_ICCV_2023_paper.pdf): person-aware face inpainting already has precedent.
- [ReF-LDM](https://arxiv.org/abs/2412.05043): multiple-reference diffusion face restoration is an existing comparison family.
- [RIDFR](https://arxiv.org/abs/2507.10943): alignment learning addresses reference-related identity/semantic interference; inspect its exact inference setting before describing it as our proposed multi-reference fusion.
- [Reference Face Component Editing, IJCAI 2020](https://www.ijcai.org/proceedings/2020/0070.pdf): region/reference-conditioned face synthesis is adjacent prior work; this alone neither proves nor disproves the specific proposed novelty.
- [UT Austin LIVE quality methods](https://live.ece.utexas.edu/research/Quality/nrqa.htm): NIQE and BRISQUE are no-reference quality measures based on natural image statistics, not identity evaluators.
- [IEEE Access author guidance](https://ieeeaccess.ieee.org/authors/preparing-your-article/) and [reproducibility program](https://ieeeaccess.ieee.org/authors/reproducibility/): examples of venue-specific preparation and reproducibility resources, not a venue selection or acceptance promise.

No new model was trained, metric computed, or PowerPoint slide changed during this audit. Future items above are a proposed execution sequence; their completion must be established by subsequent artifacts and measurements.
