# Mechanism establishment: completed screens and research decision

Experiments: 25 September 2026. Integrity, coverage and data-availability review completed 27 September 2026.

**Neither tested mechanism establishes novel-method superiority.** These are completed implementations and measured results, not proposals awaiting execution. The useful finding is narrower: personal references improve the identity score of a completed face, but the tested restoration pipeline retains incorrect hidden facial structure/expression. A visible-context constraint reduces some error without solving that problem. Erasing scaffold conditioning instead causes severe failure.

## What was executed

- Reference/context factorial: 32 new ReF-LDM restorations, eight reused ResShift scaffolds and two full GPU sampler-equivalence calls. All 40 rows scored with complete FaceNet/ArcFace and three-image gallery coverage.
- Scaffold-conditioning intervention: 24 new restorations, 16 reused control rows and two equivalence calls. All 40 rows reached evaluation, but identity detection was incomplete. Two FaceNet and seven ArcFace failures occurred across seven output rows; those failures remain in the records.
- Combined: **56 new reconstructed images, four additional equivalence sampling calls and 80 scored comparison rows**. The 24 reused rows are not 24 new images. A separate initial v1 wrapper failure occurred before sampling and remains recorded.
- Both screens use the same four already observed development identities, removal cases and seeds 17/29. This is not an independent replication, an 80-person study, or a test of arbitrary client photographs. No generator weights were trained.
- All 80 comparison rows preserve original outside-mask pixels exactly. All 16 comparison sheets were inspected. Four new mechanism tests and eight existing context/statistical tests pass (12 total). Sources, protocols, input/output hashes, hook execution and evaluation receipts were independently checked; see `reference_context_audit_v1.json`.

## Screen A: references help identity; the context constraint does not establish a superior reconstruction

The fixed ResShift completion is supplied to the author's ReF-LDM, with true/zero references and a late visible-context correction on/off. The correction replaces only surviving original context in a decoded prediction, then adds the difference between its projected and unprojected encodings to the predicted clean latent. It acts before the next diffusion transition. The equation and frozen design are in `REFERENCE_CONTEXT_METHOD.md` and `protocols/reference_context_v2.json`.

All means below average two seeds within each of four identities. FaceNet is an independent identity evaluator; ArcFace remains diagnostic. Scores are not recognition accuracy percentages.

| Arm | FaceNet target ↑ | FaceNet withheld gallery ↑ | Missing-region MAE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| ResShift scaffold | 0.741489 | 0.423009 | 0.062665 | 0.025402 |
| Restoration, zero references | 0.758506 | 0.423336 | 0.066654 | 0.028055 |
| Restoration, real references | 0.780990 | 0.480085 | 0.068752 | 0.028146 |
| Zero references + context correction | 0.753154 | 0.430160 | 0.064710 | 0.027008 |
| Real references + context correction | 0.781441 | 0.476390 | 0.065988 | 0.027110 |

References contribute to identity similarity, but the proposed corrected cascade increases missing-region error by about 5.3% and LPIPS by about 6.7% over its original scaffold. Relative to plain reference restoration, correction reduces pixel error by about 4.0%, adds only 0.000451 target similarity, and reduces gallery similarity. The six prespecified primary comparisons have Holm-adjusted p-values of 0.75 or 1.0. With four identity units the smallest possible two-sided exact p-value is 0.125; bootstrap intervals are unstable. The progression gate fails.

Correction also adds four decoder and eight encoder passes. Mean recorded generation time is 4.40 seconds versus 3.28 seconds for plain reference restoration; these timings exclude model loading and conditioning. This is not equal total computation. No claim about broadly faster or more accurate reconstruction follows.

## Screen B: spatially dropping scaffold conditioning fails

The follow-up retains references and context correction but zeros low-quality latent cells overlapping missing pixels. It includes a reference-disabled arm and a global attenuation control that matches the mean multiplier, not the latent norm. Parameters and eight primary tests were frozen before this follow-up generated outputs. The hypothesis was informed by Screen A and remains exploratory.

| Arm | FaceNet target (identity n) ↑ | Missing-region MAE (n=4) ↓ | LPIPS (n=4) ↓ |
|---|---:|---:|---:|
| Scaffold | 0.741489 (4) | 0.062665 | 0.025402 |
| References + context correction | 0.781441 (4) | 0.065988 | 0.027110 |
| Spatially dropped conditioning + references | 0.432041 (3) | 0.137404 | 0.092064 |
| Spatially dropped conditioning, zero references | 0.445447 (3) | 0.140856 | 0.091680 |
| Global conditioning attenuation | 0.788347 (4) | 0.064979 | 0.028968 |

The masked arms' gallery FaceNet means also have n=3 and their ArcFace target/gallery means n=2. Control means have n=4. Do not subtract differently supported table means as paired effects. The actual paired FaceNet contrast against the scaffold uses three complete identity pairs and is -0.329600; the paired MAE contrast uses four and is +0.074739. Missing detections are not imputed as successful low scores. Both masked arms visibly retain grey, featureless eye/nose/mouth patches. Complete scoring execution does not imply complete identity measurements.

The global control has the largest descriptive identity mean, but it still has worse pixel and perceptual error than the scaffold. Selecting it after seeing these results would require a new prospective study; it is not a successful novel mechanism. The prespecified spatial policy fails all progression requirements. Its model was not trained on spatially dropped conditioning; this distribution change and the encoder's nonlocal behavior are plausible explanations, not causally proven explanations.

## Visual findings and technical integrity

All four people and both seeds were inspected in each screen:

| Development case | Main reconstruction limitation |
|---|---|
| 1306 | The target's visible teeth/open-mouth expression becomes closed lips; references/correction do not recover that expression. Dropped conditioning erases recognisable features. |
| 2790 | Lip shape, eyes and expression differ; sharpening does not fix the structural mismatch. The masked policy leaves featureless patches. |
| 1043 | Teeth are present in the scaffold/cascade, but eye geometry and expression still differ. Spatially dropped conditioning removes the recovered features. |
| 787 | Eyes, nose and lips differ despite plausible texture. Dropped conditioning produces obvious holes and detector failures. |

This is internal visual review, not blinded human validation or a claim about the subjects' identities. Local review images remain outside GitHub under the dataset restrictions.

The original v1 wrapper failed because blanket gradient-flag changes conflicted with author EMA registration. A second issue, missing empty hook options, was caught before rerunning. V2 keeps the author EMA flags and passes the corrector options explicitly; it changes no scientific settings. The initial statement that the author sampler did not forward the correction hook was wrong and has been corrected. Original failed source/protocol/receipt remain immutable.

In both completed screens, disabled correction produces bit-exact author-sampler latents, with 50 hook calls and zero maximum latent difference. Active correction executes at steps 46–49. Reference-disabled conditions are actually zero. Inference reads only damaged observations, masks, supplied references and prior scaffolds. Clean targets and withheld gallery photographs enter only the separate evaluator/review. The eight reserved-final identities remain unopened.

## Prior-art boundary

These sources rule out broad claims based merely on combining existing functions:

- [ReF-LDM](https://arxiv.org/abs/2412.05043) supplies the multi-reference restoration model; its learned reference attention is not our contribution.
- [Reference-Guided Large-Scale Face Inpainting](https://arxiv.org/abs/2303.07014) already uses identity and texture guidance to fill missing facial regions. It is an appropriate additional baseline, not a new idea to rename.
- [RePaint](https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf) and [ReSample](https://arxiv.org/abs/2307.08123) establish diffusion data-consistency ideas. Our encoding-difference intervention is not a reproduction of either full method.
- [PFStorer](https://openaccess.thecvf.com/content/CVPR2024/html/Varanka_PFStorer_Personalized_Face_Restoration_and_Super-Resolution_CVPR_2024_paper.html) addresses dependence on low-quality identity evidence; [CLR-Face](https://arxiv.org/abs/2402.06106) uses selective latent identity guidance. Masked or reliability-weighted identity conditioning alone is not a defensible first-in-literature claim.
- [InstantIR](https://arxiv.org/abs/2410.06551) already changes restoration conditioning during generation. Feedback or a changing scaffold would require a narrower distinction and supporting controls.

This is a targeted overlap audit, not proof that all literature has been exhausted. Implementation novelty, useful experimental benefit and publication acceptance are separate questions; none is established by giving a combination a new name.

## Data availability and next execution decision

The remaining local CelebA validation pool contains only **11 unused groups with at least eight photographs**, before quality/duplicate screening (`unused_development_pool_20260925.json`). They are an upper bound, not 11 verified usable new cases. The training partition and all previous attempted selections remain excluded. A larger confirmation cannot silently reuse already observed people or the reserved final set.

We also retrieved only the six official [FFHQ-Ref split/mapping text files](https://github.com/ChiWeiHsiao/ref-ldm/releases/tag/0.1.0), through checked HTTP byte ranges. No archive image entry or FFHQ pixel was fetched. The author validation mapping has 300 connected reference groups, but only **four groups with at least eight photos** (36 photos). Author test metadata has 18 such groups; their images remain unopened. Author training data cannot be treated as an independent ReF-LDM evaluation. These groups are derived from author-predicted ArcFace links, not manually verified identity labels; cross-dataset and pretraining overlap remain unknown. See `ffhq_ref_metadata_receipt_v1.json` and `ffhq_ref_pool_v1.json`.

**27 September continuation:** the published missing-region RefFaceInpainting baseline has now completed 16/16 predictions and 28/28 scores, using reference-only parsing and no target-truth inputs. All three author checkpoints loaded strictly and a synthetic check matches author native inference exactly. Its fixed-reference identity/pixel/perceptual outcomes lose to the existing scaffold, with blurred/misaligned features in all four comparison sheets. Read-only instrumentation reproduces every native prediction exactly and measures 46/163 eligible component/layer events with completely suppressed reference texture. See `REFFACE_BASELINE_RESULTS_V1.md` and `REFFACE_BASELINE_REVIEW_V1.md`. These additional 16 predictions are separate from the 56 mechanism predictions above.

The next bounded test is a prospective causal ablation of this component rule against unchanged, texture-disabled and reference-shuffled controls, before interpreting the suppression as the cause of poor features. Do not run a strength/radius search on these same four targets to manufacture a positive result, promote failed policies to the website, or start the gated fusion adapter. A simple rule change with no replicated benefit or prior-art distinction would still not establish novelty.

Before a publication claim, remaining work is: a beneficial mechanism against matched controls; a broader verified identity-disjoint development cohort; independent reference-count/pose/expression/damage ablations with all failures; blinded visual fidelity review; a frozen method followed by untouched final evaluation; and a manuscript with a specific venue and defensible scope. Nothing in these screens guarantees the hidden true expression when the input and references do not contain it. The working local website remains a reconstruction tool with these limits.
