# Facial Inpainting Project — Progress and Completion Report

**Report date:** 23 September 2026
**Current platform:** Windows laptop, NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM  
**Current application:** http://127.0.0.1:8765/  
**Repository:** https://github.com/rehaan1603/facial-inpainting-studio  
**Status:** Working local research baseline; proposed multi-reference research method and final validation incomplete. Public hosting paused at the owner's request.

## Latest status update

### 23 September: browser workflows verified; first uncertainty mechanism tested

- Local hosting is running at `http://127.0.0.1:8765/`. Both reference-guided evidence-map reconstruction and the separate experimental ReF-LDM blur/noise mode completed through the browser with downloadable outputs and no console errors. Actual result hashes and exact preservation outside the mask were checked; ReF-LDM's raw website result exactly matches the frozen baseline. Evidence: `research/website_browser_verification_20260922.json`. The browser-verification-pending statements below are superseded.
- ReF-LDM is available for testing as an explicitly experimental partial-damage option, not as a superior default. Requests marking completely missing pixels are rejected for this mode because it does not fill erased regions. The original missing-area reconstruction remains available. Fourteen HTTP tests, three upload-geometry tests and one matched-coverage gate test pass; JavaScript syntax passes. Instructions are in `LOCAL_TESTING.md`.
- Implemented and completed the reference-disagreement diagnostic: 16 leave-one-reference-out generations and 12/12 scored candidate/control images. On four observed mixed cases, FaceNet is **0.8725** for all-reference restoration, **0.8897** for disagreement gating and **0.9037** for edit-magnitude gating. Masked MAE is **0.05043 / 0.04829 / 0.04793**. Disagreement harm-prediction AUC ranges **0.417–0.533**, providing weak evidence. LPIPS worsens under both gates, particularly edit gating. The proposed reference-risk mechanism does not demonstrate incremental identity/pixel-error benefit over the simpler control; it is not calibrated and is not promoted to the website. See `REFERENCE_RISK_RESULTS_V1.md`.
- Two new local development identities, **1590 and 1529**, were selected with the earlier identities/attempts and all reserved-final identities excluded before pixel access. Initial inference completed 3/4 cases; one native process terminated without a Python traceback. Its unchanged-settings retry succeeded, with the initial failure retained separately. These are functionality checks on newly observed development identities, not pretraining-disjoint or final evaluation.
- Unfamiliar-case scoring is complete: **10/12 initial rows**, then **12/12 after the separately recorded runtime recovery**. On four cases from two identities, composed restoration / unchanged input FaceNet is **0.9635 / 0.9609**, LPIPS **0.01512 / 0.02857**, and masked MAE **0.04777 / 0.03991**. Identity/perceptual scores improve descriptively while pixel fidelity worsens. All four rows visually reviewed: blur is reduced, but skin texture and eye detail differ; native restoration also changes reliable context. No statistical generalization claim follows. See `research/UNFAMILIAR_SMOKE_RESULTS_V1.md`.
- No novel-method superiority is established. Remaining scientific work is a justified mechanism revision, larger identity-separated calibration/validation with simple controls, human fidelity review, then method freeze and reserved-final evaluation. Adapter training remains deferred. Publication readiness is incomplete.

### 22 September: restoration-specific baseline now runs; novelty remains unproven

- ReF-LDM completed a real 50-step GPU restoration on development case `1306_mixed`. The saved image was visually inspected and hashed (`research/refldm_smoke_v1.json`). This establishes execution only: no unfamiliar-identity success or measured quality advantage is claimed. The model is not yet integrated into the website.
- Windows dependency loading now succeeds after the owner allowed the blocked component. Compatibility changes are recorded: Lightning's rank-zero import location and explicit historical checkpoint loading for the verified author-release VAE. The older dependency-block statements below are historical, not the current runtime status.
- The confidence-map reconstruction endpoint completed real GPU generation with exact known-pixel preservation and four invalid-request checks (`research/confidence_web_integration_v2.json`). Final browser verification remains pending. Non-square confidence uploads now share an aspect-preserving image/mask/evidence transform; three geometry tests and twelve HTTP tests pass.
- Completed the frozen ReF-LDM comparison: 24/24 generations and 48/48 scored native/composited outputs. On 20 partially damaged cases, composited FaceNet is **0.9216**, versus **0.7134** for matched high-strength SDXL, but **0.9538** for unchanged observations. Masked MAE is **0.04603**, versus **0.09032** and **0.03367**, respectively. All eight primary Holm p-values are **1.0** with four identity units. This is an external baseline, not our novelty. See `research/REFLDM_RESULTS_V1.md` and `research/refldm_results_v1.json`.
- All four six-condition contact sheets were visually reviewed. ReF-LDM retains erased regions in all four removal cases; it is not a substitute for missing-region inpainting. Partial-damage outputs generally preserve facial structure better than the displayed SDXL control, but smoothing and eye/mouth differences remain. Native whole-image restoration alters reliable regions; composition preserves them. Review was not blinded.
- The narrower research hypothesis is whether calibrated reference uncertainty can prevent harmful replacement of surviving facial evidence. Manual blending, multiple references and spatial transfer alone are not new contributions. Any candidate must beat unchanged-input and simple-preservation controls on held-out development identities, with identity-level statistics and failure coverage.
- Still remaining: broader external controls, unfamiliar-person development tests, a beneficial and defensible mechanism, independent human fidelity review, conditional candidate/adapter experiments, method freeze, untouched final evaluation and manuscript. The concrete next hypothesis and rejection criteria are in `research/NOVELTY_NEXT_EXPERIMENT.md`. All eight reserved-final identities remain sealed. Latest integration and baseline work is local and not yet pushed.

The dated entries below retain the history of successful and blocked attempts. Their older “currently blocked” wording is superseded by this update.

### 21 September: website inference restored, GPU control passed

- Full imports and CUDA now pass. A real four-reference preservation CLI reconstruction and a localhost HTTP reconstruction at 512×512 completed. Output hashes and exact known-pixel preservation were verified. Earlier blocked attempts below are historical; no security policy was changed by the assistant.
- The actual GPU zero-gain control passed: disabling local-feature injection gives identical raw, hard-composited, final, input and effective-mask PNG hashes to the baseline on one development case. This confirms baseline equivalence, not restoration superiority.
- Reference 386/1 detection failure was reproduced at fixed JPEG levels: original and qualities 95/75 detect one face; 50/35/20 detect none. Reproduced quality-20 pixels match the historical failed reference. Failures remain in the original experiment; no replacement, threshold tuning or retrospective exclusion occurred.
- Evidence: `research/website_recovery_20260921.json`, `research/runtime_recovery_20260920.json`, `research/local_zero_gain_verification_v1.json`, and `research/REFERENCE_COMPRESSION_DIAGNOSIS.md`. All reserved-final identities remain untouched.

**Follow-up verification:** all 36 successful expanded cases visually reviewed across six degradation kinds and two strata. Low-strength removal leaves holes; severe mixed inputs can yield colored/cross-like artifacts; stronger generation changes eyes/lips/expression. See `research/DISTORTION_EXTENSION_VISUAL_REVIEW_V1.md`. A fresh unchanged-runtime retry passed a SciPy linalg import probe but failed full model loading on another blocked component (`_matching`); Windows Code Integrity events confirm the application-control block. No security policy or dependency version was changed. Fresh inference remains blocked.

**Latest verification:** local server restarted; 19 tests passed (7 preservation/correspondence/fusion and 12 HTTP tests). All 552 successful output hashes, three frozen generation-source signatures and 24 historical checkpoints verified. Editor image/confidence import and all three exported PNGs verified pixel-for-pixel. The new real CLI reconstruction attempt failed before generation because Windows Application Control blocked a SciPy DLL; no security policy was changed. Existing saved metrics remain valid, but fresh GPU inference is not currently verified. See `research/distortion_phase_verification_v1.json`.

### 20 September: distortion, preservation and true local-feature experiments completed

- **Initial distortion screen:** 144/144 generations completed; 312/312 rows scored, including preservation outputs and unchanged-input controls. Six separate degradation classes, strengths 0.50/0.75/0.99 and adapter scales 0.8/1.2 were compared on four previously observed development identities. Lower strength was not universally better: for removal, FaceNet fell from 0.7079 at 0.99 to 0.4684 at 0.50. For partially degraded inputs, preserving evidence often helped relative to aggressive generation, but the unchanged input retained better identity similarity. This is not a validated restoration improvement.
- **Actual local-feature fusion:** aligned spatial VAE reference features now enter the denoiser's masked-image context; regional AlexNet descriptors provide compatibility estimates. All six policies were evaluated over two seeds: 48/48 rows scored (32 new generations, 16 reused controls). Damage-conditioned fusion reached FaceNet **0.6846**, versus matched global **0.6894**; hole MAE worsened from **0.09054 to 0.09486**. The primary paired difference was **−0.00483**, Holm p **1.0**. All reported metrics had full coverage and known pixels remained unchanged. True local transfer is implemented, but superiority is not established.
- **Expanded development completed with failures retained:** four additional identities, six kinds, two severity/mask strata and compressed-reference variation. Of 96 planned generations, **72 succeeded and 24 failed**; 72 preservation outputs and 48 unchanged controls produced **192 scored rows out of 240 scheduled rows (80%)**. Identity 386's compressed first reference failed face detection, causing 24 generation failures and 24 corresponding unavailable preservation outputs. No replacement identity or easier reference was substituted. This leaves three complete identity units for the primary paired contrasts.
- **Expanded results:** lower strength versus aggressive generation improved FaceNet by **+0.03731** and reduced hole MAE by **0.02063**; preservation at high strength improved FaceNet by **+0.06892** and reduced hole MAE by **0.03298**. All four Holm-adjusted p-values were **1.0**. These are small-sample descriptive trends. Compared with unchanged-input controls, generation/preservation still reduced identity similarity and increased hole error on complete pairs, despite LPIPS improvements. Metric disagreement is explicitly retained.
- **Visual failures:** all eight local-fusion sheets inspected across both seeds: altered gaze/eye geometry, changed smiles and teeth, loss of original expressions and central-face smoothing. All 36 successful expanded cases across both strata are now visually reviewed; no blinded human study has been performed.
- **New local tool:** an experimental editable evidence map distinguishes missing, partially damaged and reliable pixels. Painting, suggestions, undo/reset were exercised in the browser; seven preservation/correspondence/fusion unit tests passed and JavaScript syntax passed. The editor exports inputs for an experimental CLI; it is not yet a fully verified end-to-end website reconstruction workflow. Heuristic suggestions are not validated blind degradation estimates.
- **Research decision:** no default generator is promoted. Conditional compact-adapter training is deferred because deterministic local fusion has not shown a convincing multi-metric benefit. Candidate reranking remains deferred until restoration is stable. The prior-work audit weakens a broad novelty claim: aligned components, spatial reference transfer and spatial strength control already exist.
- **Protection and publication status:** all eight reserved-final identities remain unused. No improvement over the historical **0.8180** all-reference aggregate is established; that aggregate covers a different condition mixture. No publication-ready superiority or acceptance is claimed. Photos, reference features and model weights remain local. This phase is prepared for the GitHub checkpoint; restricted photos and weights are excluded.

Evidence: [distortion screen](research/DISTORTION_AWARE_GENERATION_RESULTS.md), [expanded confirmation](research/DISTORTION_EXTENSION_RESULTS_V1.md), [local-fusion results](research/LOCAL_LATENT_RESULTS_V1.md), [correspondence and visual limitations](research/LOCAL_FEATURE_CORRESPONDENCE.md), and [novelty audit](research/NOVELTY_GAP_ANALYSIS.md).

### Completion percentages — explicit denominators

These percentages describe finite tasks or recorded rows, not overall scientific quality or probability of publication.

| Deliverable | Completion | Meaning |
|---|---:|---|
| Initial distortion evaluation | 100% (312/312) | All planned screen rows scored |
| Local-feature diagnostic evaluation | 100% (48/48) | All six policies and two seeds scored |
| Expanded run accounting | 100% (240/240) | Every scheduled row has a success/failure record |
| Expanded usable evaluation | 80% (192/240) | Remaining 48 rows unavailable after reference detection failure |
| Combined new-phase usable evaluation | 92% (552/600) | Includes controls and reused rows; not 600 independent samples |
| Local-fusion visual sheet review | 100% (8/8) | Both seeds reviewed; not a blinded study |
| Expanded successful-case visual review | 100% (36/36) | Failed identity has no generated outputs; blinded review still pending |
| Reserved final evaluation | 0% (0/8 identities) | Deliberately untouched until method freeze |
| ReF-LDM development generation | 100% (24/24) | External baseline, existing development identities |
| ReF-LDM development scoring | 100% (48/48) | Native and composed outputs, not 48 independent identities |
| ReF-LDM visual sheet review | 100% (4/4) | Six conditions per identity; not blinded |
| Reference-risk diagnostic generation | 100% (16/16) | Leave-one-reference-out, four observed mixed cases |
| Reference-risk diagnostic scoring | 100% (12/12) | All-reference and two matched-coverage gates |
| Unfamiliar development initial generation | 75% (3/4) | One native process failure retained |
| Unfamiliar development after runtime retry | 100% (4/4) | One separately recorded successful retry |
| Unfamiliar development scoring after retry | 100% (12/12) | Two identities, controls and native/composed outputs |
| Demonstrated advantage over historical 0.8180 | Not established | No defensible percentage applies |
| Publication readiness | Incomplete | External controls, human review, method freeze and final evidence remain |

### 20 September: regional-routing comparison completed

- **Implemented and tested:** deterministic regional weighting of global FaceID reference descriptors, six matched conditioning policies, GPU-resident routing masks, integrity verification and identity-level statistical reporting. Six routing/statistics tests passed. This has zero trained parameters and does not transfer local reference patches.
- **Completed:** 144 comparison rows (120 new GPU generations plus 24 hash-verified existing concatenation controls), all scored successfully with FaceNet, ArcFace, NIQE, BRISQUE and reconstruction metrics. Known pixels remained unchanged. All 144 output hashes and frozen source signatures verified; all 24 historical checkpoints remained unchanged.
- **Outcome:** regional target FaceNet cosine averaged **0.8137**, versus **0.8180** for original all-reference concatenation. Gallery similarity was **0.5237** versus **0.5304**. All five Holm-adjusted primary p-values were **1.0**. No improvement over the existing baseline is established.
- **Visual limitation:** all eight mixed-damage sheets were reviewed across both seeds. Altered gaze, eye shapes, expressions and mouth/teeth details remain. Regional routing stays a research option in scripts; it is not promoted to a better website mode.
- **Scope and recovery:** four already-observed development identities, three medium-severity conditions, two seeds; eight reserved final identities remain unused. An interrupted attempt was archived intact and resumed with unchanged settings. No quality-based output rejection occurred. No generator or fusion module was trained.
- **Evidence:** `research/REGIONAL_ROUTING_RESULTS_V2.md`, `research/regional_routing_evidence_v2.json`, `research/REGIONAL_ROUTING_VISUAL_REVIEW_V2.md`, and `research/REGIONAL_ROUTING_REPRODUCTION.md`. The local website was restarted and its page/session endpoints verified.

### 19 September: first generalization implementation cycle

- **Implemented:** identity-disjoint local validation/final reservation; deterministic synthetic removal/blur/noise/JPEG/resolution/mixed/illumination damage; mask-aware reference diagnostics and top-one selection; first/random/identity-only/quality-only/all-reference controls; a new 1–4-reference experimental website mode with downloadable diagnostics.
- **Protocol:** four fresh local validation identities, three conditions and two seeds, with eight separate identities reserved for final evaluation. All reviewed training identities and previously inspected experiment groups are excluded. Pretraining overlap remains unknown.
- **Experiment completed:** 168 comparison rows from 110 distinct GPU generations, with all requested metrics evaluated. All output pixels outside the effective mask were preserved. No generation/evaluation or identity-detection failures occurred in this selected sample. See `research/GENERALIZATION_RESULTS.md` and the numerical evidence ledger.
- **Result is mixed:** mean target FaceNet cosine was 0.8113 for mask-aware selection, 0.7959 random, 0.7896 identity-only, 0.7938 quality-only and 0.8180 all-reference conditioning. All four primary Holm-adjusted p-values were at least 0.50. Gallery FaceNet also favored all references (0.5304 versus 0.5015). Four identities are insufficient to establish a general quality advantage; visible facial distortions remain. Regional fusion was unimplemented at the end of that first cycle.
- **Dataset folders corrected:** `Downloads/Celeb TEST data` contains 18 same-person folders and 116 images: 13 earlier prepared sets plus five additional training/practice sets. These are convenience samples, not evidence of training the generator or a fresh final test. Dataset photos remain local.
- **Preservation:** historical generators, metrics, protocols and checkpoints remain unchanged. The baseline audit is `research/upgrade_baseline_audit_v1.json`. The original local website modes remain available; public hosting remains paused.
- **Final verification:** 59 Python tests ran with no failures and three environment-dependent skips; JavaScript checks passed. A real one-reference browser reconstruction and the explicit CLI LaMa fallback passed. The local server was restarted successfully on 20 September. Detailed completion status and blockers are in `research/GENERALIZATION_CYCLE_STATUS.md`.

The completed baseline and remaining research work are summarized below.

- **Completed:** local website, reference-guided baseline, six locally trained mask-refinement controls, expanded 144-output evaluation, ArcFace/FaceNet/NIQE/BRISQUE/SSIM/PSNR/LPIPS measurements, and initial reference-count and configuration ablations.
- **Quality remains unresolved:** the user still finds LaMa facial completion unsatisfactory. The 512-pixel website processing and mask-framing fixes address preprocessing, not its tendency to blur or invent missing facial features. A successful run is not evidence of a correct face.
- **Current priority:** improve facial fidelity and test the proposed reference-selection/fusion mechanisms. Adding more metric names alone will not complete the research contribution.
- **Dataset usability request completed:** 18 same-person sets containing 116 images are available in `Downloads/Celeb TEST data`, with source/split records. These local convenience photos must not be confused with the newly reserved final-test identities.
- **Training clarification:** “local dataset” does not mean all supplied images trained the current generator. Actual local training covered the compact mask refiners. LaMa, ResShift, SDXL, FaceID and recognition encoders use pretrained weights; no new regional identity-fusion module has been trained.
- **Release status:** the LaMa repair checkpoint was pushed as `8303b97`. This report update follows that checkpoint. Dataset photographs and downloaded pretrained weights remain local.

### Next work in order

1. Maintain and broaden the verified browser workflow: confidence reconstruction and experimental blur/noise restoration now work. Investigate the isolated native-process termination; add broader real-photo and non-square end-to-end coverage beyond the passing geometry tests.
2. Use the completed compression diagnosis to design separately versioned post-degradation validity handling with failure coverage and matched reference-count controls.
3. Revise the evidence-preservation mechanism using the completed external comparison and negative reference-disagreement diagnostic. Current local latent injection and disagreement gating do not justify fusion-adapter training or a novelty claim. Expand appropriate external controls before claiming an incremental contribution.
4. Expand pose/expression correspondence, independently varied severity/masks/reference quality and identity groups. Conduct blinded human fidelity assessment; confidence intervals from three or four identities are insufficient.
5. Once a method is stable, perform the prescribed 1/2/4-candidate comparison with an independent reporting evaluator. Then freeze the method, evaluate the reserved identities once, and prepare the manuscript for a specific conference or journal.

Publication readiness remains incomplete. No acceptance, novel-method superiority or reliable recovery of hidden facial features is claimed.

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
| Reference-count extension | 24 additional candidates; expanded total 72 candidates and 144 scored outputs across the same twelve identities | Fixed nested one/two/four-reference subsets; ten metrics and 18 paired contrasts |
| Local application checks | Real HTTP/GPU inference and browser downloads | Functional evidence on recorded examples |

Rows are correlated experimental measurements, not independent images or independent identities. The studies use different protocols and must not be pooled into one headline performance claim.

### 2.4 Working local website

The site supports image upload, crop/fit, mask painting/erasing/undo, mask upload, model selection, three/four-reference upload, standard or detailed reference processing, output comparison and result/mask downloads. A new input clears the previous person's reference photographs.

Website output sizes are now 512 × 512 for LaMa/reference mode and 256 × 256 for ResShift. LaMa's unnecessary downsampling and nonsquare uploaded-mask alignment were repaired on 14 September. Its missing-eye/large-feature reconstruction remains unreliable; retaining resolution is not a semantic-quality fix. Historical command-line/research settings remain unchanged. Detailed reference mode processes at 1024 and exports 512; it is not an established quality improvement. Images are saved locally with run metadata.

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

**Reference guidance has not shown a consistent reconstruction-quality benefit in the completed diagnostic.** At strength 0.99 with Poisson composition, mean LPIPS was 0.032674 with references and 0.032460 with reference conditioning disabled. The paired difference was +0.000214, with an exploratory 95% interval spanning zero. Hole MAE was also worse with references in this setting. Other settings differ; the complete table must be retained rather than selecting the most favorable contrast.

**Independent identity diagnostic completed on 14 September:** FaceNet scored the frozen reference outputs, with 88/96 valid scores across 11/12 identities. At strength 0.99 with Poisson composition, reference conditioning increased mean cosine similarity from 0.4942 to 0.6588 (paired difference +0.1646, descriptive bootstrap 95% interval [0.0492, 0.2903]). One identity returned multiple detections in its target and all outputs and remains unscorable under the frozen protocol. This small development result does not establish final generalization, correct gaze/expression or a novel contribution. See `research/IDENTITY_DIAGNOSTIC_RESULTS_V1.md`.

**Expanded evaluation completed:** ArcFace/InsightFace, FaceNet, NIQE, BRISQUE, RGB SSIM, whole-image/hole PSNR, LPIPS and hole/visible MAE now cover 144 outputs. NIQE/BRISQUE/structural scores are valid for all 144; FaceNet has 132 valid scores and ArcFace has 140. ArcFace uses the same w600k_r50 checkpoint as conditioning and is not an independent verifier. FaceNet reproduces the prior scores exactly. All image hashes and all 24 new candidates' matched settings were verified.

**Reference-count ablation:** at strength 0.99 with Poisson composition, FaceNet means are 0.6125, 0.6482 and 0.6588 for one, two and four references. Four minus one is +0.0463, descriptive 95% interval [-0.0150, 0.1184], on eleven jointly scorable identities. This does not establish that more references reliably help. All 180 metric/contrast tests receive Holm correction; no FaceNet contrast meets corrected p < 0.05. See [expanded results](research/EXTENDED_ABLATION_RESULTS_V1.md).

The measurement baseline is substantially stronger, but the research is not yet ready to claim the proposed method improves identity preservation or is publication-ready.

## 4. Remaining work

**Reconciled with completed work on 23 September 2026.** Completed implementation is distinguished from unresolved research claims below.

| Area | Current evidence and remaining work | Status |
|---|---|---|
| Identity/quality measurement | FaceNet target/gallery, diagnostic ArcFace, LPIPS, SSIM, PSNR, hole/visible MAE, NIQE and BRISQUE executed; final evaluation remains sealed | Development implemented |
| Distortion/preservation | Six kinds, strength/scale screen and fresh development extension completed; unchanged-input controls still expose fidelity loss | Diagnostic complete; method unresolved |
| Reference robustness | Compression-sensitive failure diagnosed at fixed JPEG levels; retain failed cases and test robust handling in a new protocol | Immediate |
| Local correspondence | Five-point similarity and eight regional features implemented; 22/24 initial cases detected; 3D pose/expression/occlusion unresolved | Prototype complete |
| Local fusion | Six-policy, two-seed 48-row comparison complete; no multi-metric advantage established | Negative/mixed result |
| Learned adapter | Train only after deterministic local fusion shows convincing benefit; do not train full SDXL | Conditional gate not met |
| Confidence UI | Integration complete: reference-guided evidence-map reconstruction and experimental ReF-LDM restoration both passed real browser generation/download checks; known pixels and output hashes verified. Broader real-photo/non-square end-to-end coverage remains | Implemented and browser verified |
| Runtime reliability | One unfamiliar-case native process termination recovered on unchanged-settings retry; original failure retained. Root cause remains unresolved; investigate recurrence | Follow-up required |
| Visual assessment | Eight local-fusion sheets, 36 successful expanded cases, four ReF-LDM six-condition sheets, four risk-diagnostic rows and four unfamiliar-case rows reviewed; blinded human evaluation remains | Internal review complete for these runs; human study pending |
| Candidate selection | Matched 1/2/4 candidate pools, random/identity/quality/combined selection and separate evaluator | Deferred until stable restoration |
| External comparisons | ReF-LDM completed 24/24 generations and 48/48 native/composed evaluations; partial-damage results are stronger than matched SDXL but fail erased-region completion. Additional appropriate external controls and broader validation remain; paused OSOR results remain explicitly delimited | First external baseline complete |
| Reference-risk mechanism | Implemented 16 leave-one-reference-out generations and 12/12 scored controls. Disagreement does not beat edit magnitude on identity or masked error; AUC 0.417–0.533. Revise the mechanism before calibration/training or novelty claims | Diagnostic complete; incremental benefit unsupported |
| Unfamiliar-person checks | Two new development identities, four cases: 3/4 initial generations, 4/4 after a separate runtime retry; 12/12 rows scored after recovery. Mixed metrics; these identities are now observed. Larger identity-separated validation remains | Functional smoke complete; generalization unproven |
| Statistical power | More identity units, independent severity/mask/reference factors, additional seeds | Required before method freeze |
| Optional metrics | Reconstructed-landmark error; FID only at adequate sample size; ROC/TAR only with adequate verification trials | Unimplemented; scope-dependent |
| Final evaluation | Freeze method/parameters first; eight reserved identities remain unused | 0/8, intentionally |
| Manuscript and presentation | Correct novelty/architecture/results, choose actual venue, prepare figures, limitations and reproducibility | Incomplete |
| GitHub release | Verified application, experiment scripts, reports and numerical evidence pushed in checkpoint `43ac507` on the repository's default branch. Restricted photos and weights excluded; keep later changes synchronized | Checkpoint pushed; ongoing |
| Public hosting | Continue loopback-only use on the laptop | Paused by user |

No single overall project-completion percentage is assigned because successful research findings are not predictable implementation tasks. The explicit measured percentages above separate finished experiments from unsolved quality and publication requirements.

## 5. Planned execution and acceptance criteria

### Phase 1 — establish the measurement baseline

Completed for the expanded development set: separately specified identity and quality evaluators, model/source hashes, preprocessing, detector-failure handling, all 144 output records and paired summaries. The app's inference environment remains unchanged. Broader masks, seeds and held-out evaluation belong to subsequent phases.

**Deliverable:** reproducible per-case metrics and paired summaries.  
**Acceptance criterion:** clean target information is used only for permitted scoring, and no selection encoder is mislabeled as an independent final evaluator.

### Phase 2 — prepare defensible reference splits

Count eligible multi-photo identity groups and reserve conditioning, calibration and final-evaluation photos/identities. Four input references, three withheld evaluation references and one target require eight suitable distinct photos per case. Freeze severity definitions and corruption/reference strata. Determine whether low-FAR verification claims are supported by the number of identities and trials.

**Deliverable:** versioned manifests and a frozen protocol.  
**Acceptance criterion:** no target/reference leakage or reuse of observed development cases as fresh final evidence; limitations recorded explicitly.

### Phase 3 — test simple reference selection (completed diagnostic; no superiority established)

Compare random single-reference, all-reference and global-quality baselines against a mask-aware scorer while keeping the generator and candidate seeds fixed. Use only information available from the damaged input, supplied mask and input references. Diagnose whether poor results arise from visibility estimation, geometry or inadequate global features.

**Deliverable:** implemented selection rules and controlled comparison.  
**Acceptance criterion:** the proposed scoring mechanism has measurable support, or its failure is documented before adding complexity.

### Phase 4 — develop the smallest justified fusion method (deterministic prototype completed; training gate not met)

Regional reference features and deterministic latent fusion are implemented and evaluated. Their negative/mixed results do not yet justify a compact learned adapter. If future evidence supports the direction, train a compact fusion adapter. Compare equal weighting, global attention and mask-aware fusion at matched capacity and training budgets. Profile actual laptop memory and speed before committing to a large training schedule. Preserve the working pretrained baseline and all frozen earlier results.

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
- [LaMa quality findings and repairs](research/LAMA_QUALITY_CHECK_20260914.md)
- [Expanded metrics and ablations](research/EXTENDED_ABLATION_RESULTS_V1.md)
- [Expanded evaluation protocol](research/EXTENDED_EVALUATION_PROTOCOL_V1.md)
- [Integrity and numerical verification](research/extended_evaluation_verification_v1.json)
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
