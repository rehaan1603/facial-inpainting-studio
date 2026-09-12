> Living research plan, status updated 12 September 2026. The original literature audit began on 10 September. See PROJECT_STATUS.md for current completion status, FINAL_RESULTS.md for the first-stage evidence, and TRAINING_EXTENSION_RESULTS.md for the completed longer-training study.

# Facial Inpainting Research and Publication Plan

**Recommendation:** continue the measured study toward a defensible submission, while testing rather than assuming the proposed method's novelty. A generic mask-reliability module substantially overlaps existing work. The most defensible candidate is a controlled study of the reconstruction–preservation trade-off under inaccurate facial occlusion masks, followed by a small method that improves that trade-off. An IEEE submission should follow positive evidence, not precede it as a promised outcome.

The original 10 September scope was single-image facial completion without identity reference photographs, developed locally on an RTX 5070 Laptop GPU. Its first-stage evidence includes six trained 119,057-parameter mask refiners, 13,824 learned-control development rows, and 5,632 object-composite test rows using frozen LaMa and ResShift backbones. All six refiners subsequently completed 6,000 updates and 27,648 longer-training/soft-compositing development rows. The generative backbones were pretrained externally and were not retrained locally.

**Scope expansion on 11 September 2026:** the user requested help from three or four other photographs of the same person. That reference-assisted branch now has functioning local CLI runs using the existing SDXL inpainting and FaceID Portrait models with InsightFace reference embeddings. It is a separate input setting and has no completed quality benchmark or locally trained novel reference model. Website acceptance testing is pending at this documentation checkpoint. See [REFERENCE_IMPLEMENTATION.md](REFERENCE_IMPLEMENTATION.md) for implementation evidence and [REFERENCE_METHOD_AUDIT.md](REFERENCE_METHOD_AUDIT.md) for the updated prior-work analysis.

The following original protocol rationale remains useful for interpreting the single-image study. Its proposed hypotheses are not automatically findings, and reference inputs must never be attached retrospectively to its results. Current evidence supersedes the original setup assumptions: longer training and simple soft compositing are complete; the OSOR comparison is partial and paused at six of 432 cases. The publication objective remains active, with neither novelty nor overall superiority established.

## 1. Publication destination

IEEE Xplore is a digital library containing journal articles, conference proceedings and other material. Manuscripts are submitted to a particular journal or conference, whose review process determines acceptance. Being intended for Xplore does not specify a technical standard or guarantee publication.[^1]

Two routes remain candidates; the specific venue has not yet been selected:

| Route | Suitable evidence package | Practical consideration |
|---|---|---|
| IEEE Access | A complete methodological or applied study with credible novelty, reproducible experiments, and explicit limitations | The 10 September 2026 audit recorded an APC of USD 2,160 plus applicable taxes; recheck the price and funding before submission.[^2] |
| A future IEEE ICIP edition | A focused image-processing contribution with a concise, strong experimental argument | ICIP is a relevant subject-area conference. The official society listing gives 4 February 2026 as the 2026 paper deadline, already past; it lists the 2027 deadline as forthcoming.[^3] |

These are candidate routes, not an assessment of acceptance probability. A narrow conference paper and a journal study demand different amounts of presentation space and supporting evidence. Select one once the hypothesis, results, funding, and timing are clear. Conference registration and attendance costs require separate verification for the chosen edition.

IEEE Access explicitly asks for original contribution, high technical standards, sufficient experimental detail, and conclusions supported by data.[^4] A sophisticated interface, a long list of modules, or the use of diffusion does not establish those properties.

## 2. Prior work that changes the original proposal

The search covered facial completion, identity preservation, blind inpainting, uncertain masks, refinement, confidence estimation, and preservation outside edited regions. It used original papers, author repositories, dataset maintainers, and IEEE sources. Some publisher material was available only as abstracts or previews; those sources support broad overlap findings, not detailed architectural exclusions.

| Work | What is already established | Consequence for this project |
|---|---|---|
| VCNet, 2020 | Predicts where to fill and reconstructs content; explicitly addresses mask prediction errors through its inpainting design.[^5] | Neither mask prediction nor robustness to prediction errors is a new claim. |
| Uncertainty-aware image inpainting with adaptive feedback network, 2024 | Predicts reconstruction uncertainty and uses lower-uncertainty content in iterative refinement.[^6] | Adding an uncertainty map or feedback loop alone is insufficient. Distinguish uncertainty about occlusion from uncertainty about generated content. |
| What's Behind the Mask, 2022 preprint | Uses masks and conformal calibration to identify reliable parts of image-to-image reconstructions, including completion.[^7] | Confidence-based acceptance and calibration are also established. Do not claim a new guarantee without a new result and valid assumptions. |
| PVA, WACV 2024 | Personalized face inpainting uses reference images to improve identity fidelity.[^8] | Identity preservation alone is occupied territory. Its reference-conditioned setting cannot be mixed with reference-free results as if inputs were equal. |
| ID-Inpainter, 2024 | Identity-guided recovery uses an identity sampling strategy and a fusing network.[^9] | An identity loss plus a fusion module is not a sufficient distinction. |
| Structure-Guided Diffusion for Portrait Shadow Removal, ICCV 2025 | Its paper reports robustness when shadow masks include excess non-shadow regions.[^10] | Portrait-specific mask robustness also has precedent; shadow removal remains a different observation model from opaque occlusion. |
| OSOR, June 2026 preprint | Learns a soft alpha editing region under incomplete masks for object removal.[^11] | This is particularly close to learned mask expansion and adaptive blending. Applying that pattern to faces alone is a weak novelty argument. |
| ReSem-Face, August 2026 preprint | Uses multiple references and semantic conditioning for face inpainting under large occlusions.[^12] | Recent facial work must be discussed, but this is an additional-information setting. |
| Semantic-guided face inpainting with subspace pyramid aggregation, 2025 | Uses face semantics and multi-scale feature aggregation; author code is linked.[^13] | Facial parsing, semantic guidance, and multi-scale blocks cannot be presented as new by themselves. |

The 2026 entries were treated as preprints in the initial audit. The OSOR author repository subsequently labelled the work ECCV 2026; independent publisher verification is still needed for final bibliographic status. ReSem-Face remains a preprint in this audit. Prior art can matter even before peer review. Authors' comparative claims are not imported as local results.

**Decision:** reject the broad title “novel uncertainty-aware identity-preserving facial inpainting.” Retain imperfect-mask completion as a candidate research setting, now supported by completed diagnostic studies but without an established new-method claim. The reference expansion needs its own comparison with PVA, ReSem-Face, reference selection, and adaptive conditioning methods described in REFERENCE_METHOD_AUDIT.md. A defensible distinction must specify and remedy a failure under fair controls.

Additional leads to examine before locking a contribution include MAD-paint, which describes mask-aware uncertainty-guided diffusion sampling, and recent blind-inpainting mutual-learning methods.[^14][^15] Their complete methods and code availability are not established by the available previews. The search cannot prove that a narrower idea has never appeared.

## 3. Candidate research question

**Working title:** “Facial Inpainting under Imperfect Occlusion Masks: Balancing Reconstruction and Visible-Feature Preservation.” This is a descriptive working title, not a novelty declaration.

The candidate question is: *At a matched budget of change to genuinely visible pixels, can a lightweight, image-conditioned mask correction method reduce reconstruction error in truly occluded regions better than dilation, ordinary segmentation refinement, and learned alpha blending?*

Inaccurate masks make two opposing mistakes. Under-coverage leaves some occluder pixels available as misleading context. Over-coverage discards valid observations and invites the model to regenerate information that was already present. A method that always enlarges the mask may improve removal while damaging visible eyes, brows, or lips. A method that never expands it may preserve the input while leaving the obstruction behind.

Three hypotheses should be registered before the final test:

1. **Failure characterization:** matched mask errors have different consequences depending on error direction and facial location, after controlling for true missing area and error size.
2. **Method benefit:** the proposed correction improves the reconstruction–preservation frontier over equally tuned simple corrections and a generic learned refiner.
3. **Transfer:** the effect persists on an unseen corruption family and a second compatible backbone; a claim of cross-dataset generalization additionally requires a second image source.

These are proposed hypotheses, not findings. Generic cost-sensitive learning and Pareto analysis are established tools. Their use is not itself a contribution. A paper requires a nontrivial result, a useful protocol, and/or an algorithmic distinction beyond these ingredients.

## 4. Precise task definition and leakage prevention

Let X be the complete RGB face, O a binary true occlusion map (1 means occluded), and C an occluder image. Construct the observed image as:

`Y = (1 - O) * X + O * C`

Construct a supplied mask independently as `M = perturb(O)`. The deployment input is `(Y, M)` only. The unoccluded target X, true map O, target identity embedding, and clean target parsing are restricted to training supervision or evaluation.

The order matters: perturbing M must never reveal X inside O. For example, if M misses an occluder edge, Y must still contain the occluder there. Conversely, Y legitimately retains valid pixels that an over-large M happens to cover. A correction module may inspect those pixels because they are present in the observed image, but competing correction baselines must have the same access.

Use opaque synthetic occlusion as the primary paired experiment. Randomly colored holes are a diagnostic condition; textured occluders and held-out objects provide a harder condition. Do not describe either as a real capture benchmark. Shadows, reflections, transparent glasses, and blur are separate observation models; do not silently fold them into the primary claim.

Clean target face parsing may define semantic evaluation strata and synthetic mask placement. It must not become a model input at test time. If a semantic branch is introduced, run it on Y, report its failures, and account for the parser's training data. Ground-truth parsing and exact-mask variants belong in clearly labelled diagnostic controls.

## 5. Minimal method worth testing

Start with a frozen inpainting backbone F and a compact mask predictor q(Y,M). A simple thresholded q is the generic learned-refiner baseline. Compare it with a soft alpha predictor before adding anything more elaborate. That establishes whether sophistication beyond segmentation is warranted.

Only then investigate a correction policy that estimates the relative reconstruction cost of retaining versus replacing a region, conditioned on observed evidence. Use validation data to choose operating points along the preservation trade-off. A possible implementation predicts an effective mask for F and a separate output blending map, but those components substantially overlap prior art and must not be claimed as novel on their own.

A candidate inference path is:

`observed RGB + approximate mask -> correction policy -> effective mask -> frozen inpainter -> constrained compositing`

Keep the untouched observed RGB available for compositing. A frozen network can still retain activations when gradients pass through it, so a frozen backbone is not a promise of negligible training memory. Initially train the predictor on corruption labels and use cached backbone outputs for cheap selection experiments. Distinguish this surrogate training from end-to-end reconstruction optimization.

If the final contribution is facial-context-dependent correction, compare uniform costs, learned generic costs, and facial conditioning with the same data and comparable parameter counts. If ordinary BCE segmentation with a tuned threshold performs equally well, use that result to revise or abandon the method claim.

Avoid adding adversarial loss, identity loss, a transformer, diffusion fine-tuning, and calibration simultaneously. Each extra element needs a reason and an ablation. Training a large generative backbone from scratch is outside the initial local plan.

## 6. Baselines and comparison fairness

| Baseline | Why it is needed | Status and limitation |
|---|---|---|
| Input unchanged | Reveals whether preservation is being rewarded despite failed reconstruction | Trivial to implement; never a competitive inpainting method |
| Frozen LaMa | Practical convolutional/Fourier baseline and first development target | Local engineering export is verified and evaluated; export lineage/pretraining limitations remain in the exposure ledger.[^16] |
| MAT, CelebA-HQ-256 | Face-trained transformer comparison | Authors provide a 256 option and note released weights were retrained; legacy dependencies require care.[^17] |
| ResShift face-inpainting configuration | A directly relevant diffusion comparison | Pinned official source and task checkpoint load successfully and have been evaluated at 256; complete pretraining membership remains unresolved.[^18] |
| 2025 semantic-guided face model | Recent face-specific comparison | Code and a Baidu pretrained link are documented; file retrieval, provenance, and compatibility remain unverified.[^13] |
| Dilation/erosion, feathering, threshold sweeps | Tests whether simple mask repair explains improvements | Tune every parameter on validation data, including mask radius |
| Generic learned refiner; generic alpha head | Closest low-cost architectural controls | Six refiners and a segmentation-score compositor are evaluated. A separately reconstruction-trained alpha head remains distinct and unimplemented |
| Corruption augmentation alone | Tests whether the training distribution explains gains | Give the baseline equivalent training exposure and report trainable parameters |
| Exact-mask variant of each backbone | Separates segmentation error from generative error | Diagnostic privileged input; not a deployable comparator or guaranteed optimum |

A dedicated Stable Diffusion inpainting checkpoint was initially an optional additional comparison.[^19] The subsequent reference implementation uses the separately recorded SDXL inpainting 0.1 checkpoint at 512, not the SD 1.5 checkpoint in that original reference. Neither substitutes for a recent face-specific baseline. Respect preprocessing and resolution; report resizing, its cost, and a separate common-resolution comparison where appropriate.

For the final method claim, include the nearest robust-mask baseline that is executable. If a full recent system cannot fit locally or lacks released weights, state that limitation and reproduce a clearly labelled mechanism-level control where legitimate. An independent implementation is not an official reproduction. Do not omit a close competitor merely because it might win.

Published numbers from different test splits, masks, checkpoints, or metric implementations should not be inserted into the same ranking table as locally measured results. Separate reference-based, video-based, and single-image methods into different input-information settings.

## 7. Data plan and permissions

**Primary candidate: CelebAMask-HQ.** It contains 30,000 face images with semantic annotations, and its maintainers direct identity-label requests to the CelebA team. Its research-use terms restrict redistribution.[^20] Obtain images from the documented source, retain the mapping to CelebA, and secure identity annotations before asserting identity-disjoint splits.

CelebA itself provides identity annotations through its research access process and specifies non-commercial use and redistribution restrictions.[^21] Release split identifiers and generation scripts where allowed, rather than repackaging face images onto Hugging Face. Verify permissions for example figures separately; download access is not automatically permission to republish images.

**External candidate: LaPa.** The official repository describes more than 22,000 facial images with landmarks and parsing labels and lists non-commercial terms.[^22] Use verified unobstructed examples for paired synthetic testing and separately label naturally occluded examples as unpaired. Identity-disjointness cannot be promised from parsing annotations alone; inspect available identifiers and document the limits.

**FFHQ is conditional, not the automatic default.** The maintainer states that it is not intended for development or improvement of facial-recognition technologies. It also provides image-level license metadata.[^23] It may support restoration-quality studies under appropriate terms, but do not use it as an unexamined recognition-improvement dataset. Whether a proposed identity evaluation is within scope must be resolved before using that evaluation on FFHQ.

Maintain a pretrained-data exposure ledger for every inpainter, parser, and evaluator. Separating adapter train and test identities does not establish that a pretrained backbone has never seen the test images. Report both “unseen to the adapter” and the known/unknown pretraining exposure. Check exact and near duplicates across data sources; do not claim a perfectly uncontaminated evaluation when provenance is unknown.

For actual photographs with real obstructions, use an explicitly permitted collection and separately annotated masks. Real paired photos rarely provide pixel-aligned hidden ground truth. Unpaired examples support qualitative realism and visible-region preservation, not pixel reconstruction accuracy of the hidden face.

## 8. Evaluation protocol

Generate masks on the fly during training, but freeze validation and test manifests with image identifiers, seeds, true masks, supplied masks, occluder identifiers, and all transformations. Hold out occluder assets, not just random seeds. Reserve an untouched final test after pilot decisions.

The initial diagnostic grid should cover under-coverage, over-coverage, translation, and boundary deformation at accurate, mild, moderate, and severe levels. Cross these with eyes/brows, lower face, and other face locations. Record true missing fraction, supplied mask fraction, false-positive area, false-negative area, IoU, and boundary distance. Match subsets by true missing area and error area to avoid attributing simple size differences to semantic location.

Do not run an enormous factorial grid immediately. A planning pilot of roughly 200–500 development images and 8–12 conditions per image should reveal basic failure modes. These counts are engineering starting points, not a statistical power calculation. Use pilot variance, image/identity clustering, and measured runtime to size the final evaluation.

### Measurements

| Quantity | Operational definition | Interpretation |
|---|---|---|
| True-hole MAE/PSNR | Pixel errors against X restricted to O, normalized by the number of selected RGB values | Measures paired completion; avoids dilution by unchanged background |
| Visible-pixel change | Mean absolute difference between output and Y on `(1-O)` | Measures unwanted edits in genuinely observed content |
| Missed-region error | Error on `O*(1-M)` | Tests whether under-covered occlusion is repaired |
| Over-covered-region error | Error on `(1-O)*M` | Tests whether valid pixels are unnecessarily regenerated |
| Perceptual quality | Full-face LPIPS plus a specified region/crop diagnostic | Report network version, normalization, and crop protocol; region variants are not standard LPIPS.[^24] |
| Structure | Landmark/parsing agreement and failure rate | Supplementary; evaluator mistakes and occluded landmarks need explicit handling |
| Mask correction | IoU, precision/recall, boundary quality | Better segmentation does not automatically mean better completion |
| Distributional quality | FID or KID on sufficiently sized, fixed sets | Secondary; fix implementation and sample size, and avoid tiny-subset rankings.[^25] |
| Identity consistency | Cosine similarity from independent, permitted evaluators | Supplementary proxy, not proof of true hidden identity |
| Efficiency | Batch-one median/p95 latency, peak allocated/reserved VRAM, model size, trainable parameters, total training time | Measure on the actual laptop and include all correction stages |

For masks with empty measured subsets, return “not applicable” and report counts; never convert them to zero error. Define PSNR data range, SSIM window behavior, and all normalization before evaluation. Test metric code on analytically controlled examples: identical images, a change only in the true hole, and a change only in visible pixels.

Plot true-hole error against visible-pixel change over validation-selected operating points. Compare Pareto curves as well as a selected deployment point. A method does not win simply by preserving everything or replacing the whole face. Do not average raw PSNR, LPIPS, identity similarity, and runtime into an arbitrary headline score.

Use paired image-level comparisons. Bootstrap at identity level where verified identities exist; otherwise state the image-level approximation. Multiple masks on one image are correlated and must not be counted as independent people. Repeat the final trainable variants with at least three seeds if feasible, and distinguish training seeds from diffusion sampling seeds. These are proposed rigor standards, not universal IEEE numeric requirements.

For stochastic models, predefine samples per case and report all fixed samples. “Best of N” using the clean target is privileged selection and belongs only in a labelled diagnostic. Face detection or alignment failures remain in the denominator and must be reported. Predefine qualitative examples, include failures, and avoid selecting only attractive completions.

If claiming human preference or realism, conduct a blinded, randomized study with an appropriate protocol and uncertainty analysis. If no such study is feasible, keep claims limited to measured automated metrics and qualitative examples.

## 9. Laptop feasibility and Hugging Face

Local inspection found an NVIDIA GeForce RTX 5070 Laptop GPU with 8,151 MiB VRAM, approximately 31.6 GB system RAM, Python 3.12.10, and approximately 455.8 GB free disk space. The active Python installation contains `torch 2.13.0+cpu`; CUDA availability is false, and a small CUDA convolution test fails because that build has no CUDA support. These are observations from this machine, not estimates.

Create an isolated project environment and install a compatible CUDA-enabled PyTorch build using official distribution instructions. PyTorch documents Blackwell support beginning with its 2.7 release and CUDA 12.8 wheels; official version tables include Windows options.[^26] Pin a tested modern combination rather than copying CUDA 10/11 commands from old baseline READMEs. Do not replace the global Python installation merely to satisfy one baseline.

Before any long run, test CUDA forward/backward, checkpoint loading, a small inference batch, and a short representative training loop. Measure peak VRAM after warm-up and sustained step time. Gradient accumulation changes effective batch size but does not reduce the memory of one sample; mixed precision and activation checkpointing need numerical checks. Run baselines sequentially so their weights do not compete for VRAM.

Hugging Face is useful for compatible checkpoint retrieval, version-pinned caching, and Diffusers. It does not require uploading local images or renting inference compute.[^27] Record repository revision and weight hashes. A mirror is not proof of original authorship, correct training provenance, or compatible licensing. Some important baselines are maintained in original repositories and do not use Diffusers.

Keep experiments at 256 initially where the selected architecture supports it. A full foundation-model retraining plan is not credible under this initial budget. A compact corrector and frozen-backbone experiments are plausible, but exact fit and runtime remain unmeasured. Leave practical VRAM headroom for Windows and temporary allocations.

Estimate compute from measurements: `total GPU hours = training steps * seconds per step / 3600 + evaluation cases * samples per case * seconds per sample / 3600`. Include failed runs and hyperparameter searches in the final compute ledger. Do not announce a completion date based only on the GPU model name.

Because the workspace is under OneDrive, place high-churn caches, datasets, and checkpoints in a deliberate local storage location when setting up experiments, while retaining source code and small reports in this project. This is an operational recommendation; no files have been moved.

## 10. Milestones and stopping criteria

| Stage | Required output | Continue only when |
|---|---|---|
| Setup | Locked environment and passing CUDA/checkpoint smoke tests | GPU execution works and the baseline output is sensible |
| Baseline validity | Fixed development set, clean-mask outputs, metrics, timing | There is no mask polarity, preprocessing, or metric bug |
| Failure pilot | Area-matched mask-error curves and simple correction sweeps | A meaningful unresolved trade-off remains after tuned simple corrections |
| Minimal method | Generic refiner, alpha control, proposed correction | Improvement survives comparable data, training budget, and parameter controls |
| Generalization | Unseen errors, second backbone, external source as applicable | Claims remain supported outside the easiest synthetic condition |
| Paper readiness | Repeated runs, uncertainty, failures, reproducible tables and citations | The written contribution is distinct and every main claim has evidence |

Stop or pivot if the gain disappears against dilation, generic refinement, matched compute, or accurate masks. Stop an identity claim if it appears only on the training evaluator. Narrow a real-world claim if gains exist only for solid-color rectangles. A modest significant difference is not automatically useful; choose a practical effect threshold on development data before the final test.

A planning sequence could allocate the first week to environment/data access and baseline reproduction, the next to the failure pilot, and several subsequent weeks to method and final experiments. This is a sequencing aid rather than a promised schedule. Data permissions, compatibility ports, experimental variance, and the selected venue can change it materially.

## 11. Paper and release package

The main manuscript should explain the exact input setting, why current methods fail, how the proposed method differs from its nearest predecessors, and which experiment supports each claim. Present the strongest counter-baseline fairly. A contribution statement should be rewritten after results rather than preserving an attractive initial promise.

Prepare a main comparison table, controlled mask-error plots, a component ablation, generalization results, efficiency measurements, and qualitative successes/failures. Document training exposure, preprocessing, masks, seeds, hyperparameters, checkpoint selection, and evaluator versions in supplementary material. Keep complete raw metrics so every table can be regenerated by a script.

Release source, environment locks, configurations, allowable split manifests, mask/occluder generation scripts, evaluation commands, and checkpoints if their licenses permit. Separate third-party licenses from the project's own code license. Include the compute ledger and known reproduction limitations. A polished demo is optional and should follow the evidence.

IEEE Access currently requires its submission template, matching source/PDF, relevant references, and disclosure of AI-generated article text in acknowledgements; it prohibits simultaneous submission elsewhere.[^28] Recheck the selected venue's current rules immediately before submission. Human authors must verify citations, results, and all claims. No acceptance, reviewer outcome, or publication date can be assured.

## 12. Current decision and unresolved items

The original generic novelty claim does not survive this audit. Environment setup, data audits, the first-stage study, and the 6,000-update extension with simple soft compositing are complete. Weighted refinement preserves visible pixels more closely but worsens aggregate completion at the first training budget. At the extended budget it beats the generic refiner on aggregate LPIPS and hole MAE while changing slightly more visible content. Longer training is not a uniform improvement; consult TRAINING_EXTENSION_RESULTS.md for the backbone-specific contrasts and paired intervals. The OSOR comparison is paused with only six of 432 cases saved and supports no complete comparative conclusion.

Working reference-conditioned CLI outputs now complement the single-image study. Multiple photographs, the existing FaceID Portrait adapter, and optional Poisson boundary blending are established techniques rather than a new contribution. Full website acceptance testing and a fresh reference-assisted evaluation protocol remain necessary. Earlier test cases are already observed evidence and cannot become a fresh confirmatory test.

Still unresolved are the precise new contribution, nearest-method comparisons, evidence supporting any convergence claim, independent data/identity checks, reference-target leakage, pretraining exposure, and final venue/author/permissions requirements. The main research decision is whether a distinct method or a sufficiently valuable benchmark result survives these controls. Functioning software does not settle that decision or guarantee publication.

## Sources

Sources below originated in the 10 September 2026 audit. Subsequent reading is identified where applicable; the 11 September reference-method audit has its own linked sources. Publication years are given where established; repository and policy pages are living documents. Abstract-only and preview limitations are indicated where material. This 12 September status update does not reverify every venue rule or fee.

[^1]: IEEE Author Center. [About the IEEE Xplore Digital Library](https://journals.ieeeauthorcenter.ieee.org/when-your-article-is-published/about-the-ieee-xplore-digital-library/). Official description.
[^2]: IEEE Access. [Article Processing Charges](https://ieeeaccess.ieee.org/about/article-processing-charges/). Current listed price; recheck before submission.
[^3]: IEEE Signal Processing Society. [ICIP event listings](https://signalprocessingsociety.org/event-names/icip); IEEE ICIP. [About ICIP](https://ieeeicip.org/about-icip/).
[^4]: IEEE Access. [Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/). Acceptance requirements.
[^5]: Yi Wang, Ying-Cong Chen, Xin Tao, Jiaya Jia. [VCNet: A Robust Approach to Blind Image Inpainting](https://arxiv.org/abs/2003.06816). 2020. Initial abstract-only review was followed by full method/training inspection; see CLOSEST_METHOD_UPDATE.md.
[^6]: Xin Ma et al. [Uncertainty-aware image inpainting with adaptive feedback network](https://maxin-cn.github.io/journal-article/Uncertainty-aware_image_inpainting_with_adaptive_feedback_network_Ma_et_al_2024.pdf). Expert Systems with Applications 235, 121148, 2024. Author-hosted full text; methodology inspected.
[^7]: Gilad Kutiel, Regev Cohen, Michael Elad, Daniel Freedman. [What's Behind the Mask: Estimating Uncertainty in Image-to-Image Problems](https://arxiv.org/abs/2211.15211). 2022 preprint. Abstract inspected; no formal guarantee is imported into this proposal.
[^8]: Jianjin Xu et al. [Personalized Face Inpainting With Diffusion Models by Parallel Visual Attention](https://openaccess.thecvf.com/content/WACV2024/html/Xu_Personalized_Face_Inpainting_With_Diffusion_Models_by_Parallel_Visual_Attention_WACV_2024_paper.html). WACV 2024. Initial indexed abstract/supplementary preview was followed by paper HTML and author-code review on 11 September; see REFERENCE_METHOD_AUDIT.md.
[^9]: ID-Inpainter authors. [Official implementation of Recovery-based Occluded Face Recognition by Identity-guided Inpainting](https://github.com/icaoyu/ID-Inpainter). Associated 2024 paper; README inspected.
[^10]: Wanchang Yu et al. [Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](https://arxiv.org/html/2507.04692v1). 2025; ICCV paper. Full text, inaccurate-mask analysis inspected.
[^11]: OSOR authors. [OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal](https://arxiv.org/html/2606.28094v1). June 2026 preprint. Soft-alpha mechanism, incomplete-mask experiments, and training appendix inspected.
[^12]: Feng Ding et al. [When Diffusion Models Forget Who You Are: Identity Preservation in Face Inpainting under Large Occlusions](https://arxiv.org/abs/2608.04820). August 2026 preprint. Initial abstract review was followed by full HTML method/training inspection on 11 September; see REFERENCE_METHOD_AUDIT.md.
[^13]: Yaqian Li, Xiumin Zhang, Cunjun Xiao. [Semantic-guided face inpainting with subspace pyramid aggregation](https://www.sciencedirect.com/science/article/pii/S1047320325000227). Journal of Visual Communication and Image Representation 108, 104408, 2025. Publisher preview; [author repository](https://github.com/xiumin123/Face_inpainting) inspected.
[^14]: [MAD-paint: Mask-Aware Diffusion Sampling for Image Inpainting](https://jingweiqu.github.io/publication/ICMR-2025.pdf). ICMR 2025 author-hosted PDF. Indexed excerpt only; full PDF retrieval failed.
[^15]: Haoru Zhao et al. [Context-aware mutual learning for blind image inpainting and beyond](https://www.sciencedirect.com/science/article/pii/S0957417424030914). Expert Systems with Applications 268, 126224, 2025. Publisher abstract/preview.
[^16]: LaMa authors. [Official LaMa repository](https://github.com/advimman/lama). WACV 2022 method; checkpoint, environment, and evaluation instructions inspected.
[^17]: Wenbo Li et al. [Official MAT repository](https://github.com/fenglinglwb/MAT). CVPR 2022. Model-release caveats and resolution instructions inspected.
[^18]: Zongsheng Yue, Jianyi Wang, Chen Change Loy. [Official ResShift repository, journal branch](https://github.com/zsyOAOA/ResShift). Face-inpainting application and evaluation instructions inspected; use final publisher metadata when building the paper bibliography.
[^19]: [Stable Diffusion inpainting checkpoint repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting). Hugging Face. Availability inspected; local weights and provenance not yet validated.
[^20]: Cheng-Han Lee et al. [CelebAMask-HQ official repository](https://github.com/switchablenorms/CelebAMask-HQ). Associated MaskGAN paper, CVPR 2020. Dataset description, identity access, and agreement.
[^21]: Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang. [CelebA official dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Associated ICCV 2015 paper; access and agreement inspected.
[^22]: Yinglu Liu et al. [LaPa official repository](https://github.com/jd-opensource/lapa-dataset); [author fork with dataset terms](https://github.com/lucia123/lapa-dataset). Associated AAAI 2020 paper. Validate complete downloaded terms before use.
[^23]: NVIDIA Research. [FFHQ official repository](https://github.com/NVlabs/ffhq-dataset). Dataset scope, image metadata, licensing, and recognition-use statement inspected.
[^24]: Richard Zhang et al. [LPIPS official implementation](https://github.com/richzhang/PerceptualSimilarity). Associated CVPR 2018 paper.
[^25]: Maximilian Seitzer. [pytorch-fid](https://github.com/mseitzer/pytorch-fid). Implementation and comparability notes.
[^26]: PyTorch Team. [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/), April 2025; [official version installation tables](https://pytorch.org/get-started/previous-versions/). Historical compatibility floor, not a claim that 2.7 is the latest release.
[^27]: Hugging Face. [Diffusers inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint) and [loading pipelines](https://huggingface.co/docs/diffusers/en/using-diffusers/loading). Local inference and loading workflows.
[^28]: IEEE Access. [Submission Guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/). Current author requirements.
