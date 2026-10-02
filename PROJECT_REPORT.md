# Facial Inpainting Project — Progress and Completion Report

**Report date:** 2 October 2026
**Current platform:** Windows laptop, NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM  
**Current application:** http://127.0.0.1:8765/  
**Repository:** https://github.com/rehaan1603/facial-inpainting-studio  
**Status:** Working local research baseline; proposed multi-reference research method and final validation incomplete. Public hosting paused at the owner's request.

## Latest status update

### 2 October: identity-aware objective audit and three-seed experiment

Reproduced all 12 prior clean rows plus three FaceNet-failure rows with **zero numeric difference**, including eye crop LPIPS; verified 96 image and two adapter hashes. Added a frozen FaceNet decoded-loss feasibility test and two matched **64-update GPU training arms** with the existing 193-parameter architecture. Completed **96/96 new generations and scores**, four development identities, seeds 17/29/43, clean/wrong-person-patch references, with **16 byte-exact old-output replays**. Eye crop LPIPS improves on average, but the identity-supervised arm worsens whole-image LPIPS and FaceNet against baseline in both conditions; it does not replace the working model. ArcFace has 8/12 valid pairs per condition, with failures retained. Full metrics, gradient calibration, memory costs, remaining ablations and next experiment are in [the current method investigation report](research/FINAL_METHOD_REPORT.md), [audit](research/IDENTITY_COMPATIBILITY_AUDIT_V2.md) and [CSV](research/identity_compatibility_v2_metrics.csv). Final identities and Human Faces work remain untouched by this experiment.

### 2 October: target-conditioned reference compatibility trained and tested

Completed two **193-parameter adapter training runs, 128 GPU updates each**, on 16 existing-dataset identities. Generated and scored **96 outputs** across four observed development identities, three methods and eight reference conditions; eight archived baseline replays are byte-exact. Added official facial-component metrics and six passing mechanism/metric checks. Corrupted-reference LPIPS improves only **0.000128 (0.047%)**, PSNR by **0.0058 dB**; eye-region bbox LPIPS slightly worsens. Three FaceNet detections fail, leaving incomplete identity support and an **INCONCLUSIVE joint gate**; the perceptual improvement threshold is not met independently of that missing support. **Do not promote or expand this candidate.** Four inspected sheets show no consistent fidelity gain. This is full-image restoration, not erased-eye inpainting. Novelty and publication readiness remain unresolved. See [results, loss audit and limitations](research/TARGET_COMPATIBILITY_RESULTS_V1.md) and the [ten-paper literature/protocol matrix](research/MATCHED_COMPARISON_AND_LITERATURE_20261002.md). New Human Faces work remains set aside; final identities were not used in this experiment.

### 2 October: synthetic multi-reference idea generated and tested

Created **11 synthetic reference views from 12 requests** using only the four damaged development portraits; one view was blocked by the image tool and retained as a failure. Completed **12 matched SDXL/FaceID reconstructions** (reference influence off, first generated view, all generated views) and scored **24 rows** including saved baselines/input controls. All-view outputs improve eye-region LPIPS over ResShift in 4/4, but worsen hole MAE against the same SDXL reference-off control in **4/4**, with FaceNet improvements in only 2/4. Visual review still shows wrong gaze, closed eyes and exaggerated features. **Synthetic references are not promoted as the default; accurate eye recovery and novelty remain unproven.** The main website now accepts 1–4 reference photos; a real two-reference run reproduces the trial PNG exactly and 21 HTTP checks pass. Sets, prompts and comparison images are saved locally. See [full trial, controls and failures](research/SYNTHETIC_REFERENCE_TRIAL_20261002.md).

### 1 October: single-image edge matching implemented and checked

Added optional colour-boundary matching to LaMa/ResShift in the local studio, with raw reconstruction retained as **Before edge matching**. **18 paired model runs** across four new-dataset photographs and two existing development cases completed; every output preserves outside-mask pixels exactly and eight repeated raw PNGs match previous outputs byte-for-byte. ResShift masked MAE improves in **8/10** cases (mean 0.09946 to 0.07392); LaMa improves in 5/8 and keeps matching off by default. All sheets were reviewed; large-eye-region geometry/gaze errors and a glasses-like hallucination remain. This is a boundary-finishing improvement, not identity recovery, dataset-equivalent accuracy, retraining or novel research. **23 Python checks and three JavaScript polling checks pass.** The local browser shows finished and raw-result downloads. See [details, regressions and limitations](research/SINGLE_IMAGE_EDGE_MATCHING_20261001.md).

### 1 October: additional Human Faces dataset intake and baseline tests

Audited the newly supplied folder: **5,000 Real Images and 4,630 Generated Images**, all readable, with no exact file/RGB duplicates or exact matches against 51,944 existing saved fingerprints. Source/license, near-duplicate overlap and person identities remain unverified; this is not yet an independent validation dataset. Prepared **12 local damage cases from four photographs**, with separate inference/evaluation manifests. Completed **eight LaMa/ResShift generations on two photographs**; all preserve visible pixels exactly. Visual review found reasonable LaMa thin-scratch repair but failed large eye-region completion; ResShift produces mismatched eyes and seams. No training or synthetic-angle generation has started, and no existing dataset/checkpoint was changed. The [dataset plan, measured results and synthetic-reference protocol](research/HUMAN_FACES_DATASET_PLAN_V1.md) specify source verification, leakage controls and remaining experiments. Local practice files and comparison sheets are in `Downloads/Human dataset/Ready to test - Human Faces v1`.

### 1 October: real adapter training and comparison completed

Implemented and tested a 545-parameter attention-logit adapter, then ran three 64-step training arms on 16 identities with four separate adapter-check identities. Completed **36 generated images and 40 scored image rows**, including damaged-input controls. The initial intervention worsened corrupted-reference FaceNet similarity (0.90060 baseline to 0.89208). A training-log-scaled follow-up reached 0.90082 but worsened LPIPS (0.26503 to 0.26872) and clean-reference identity. These four identities now constitute development data. **Neither candidate is promoted; novelty remains unestablished.** See [pilot results and remaining work](research/REFERENCE_INTERVENTION_PILOT_RESULTS_V1.md). Four attention integrity tests pass, and paired training fits in roughly 3.5 GiB of allocated GPU memory. Reserved final pixels were not used. A local 24-slide presentation now covers the architecture, algorithms, reconstructed outputs, results and remaining milestones.

### 30 September: adapter backward feasibility passed; prior-art overlap tightened

The [reference-intervention preflight](research/REFERENCE_INTERVENTION_PREFLIGHT_20260930.md) records three real GPU probes through the frozen pretrained ReF-LDM UNet. A 273-parameter synthetic-latent adapter completed an optimizer update with four references at approximately 2975 MiB peak tensor memory; its neutral initialization reproduced the baseline exactly and the backbone received no gradients. This excludes image encoding and full paired training losses, so full training feasibility and quality remain unverified. RefSTAR's full text already covers supervised reference-region selection; the broad reliability proposal is not novel. A narrower paired-intervention objective remains a hypothesis requiring further overlap checks and matched ablations. No dataset images or reserved identities were used in these probes; no production defaults changed.

### 30 September: base-paper improvement feasibility checked

The [ReF-LDM comparison and experiment decision](research/REFLDM_IMPROVEMENT_FEASIBILITY_20260930.md) distinguishes inherited methods from possible contributions. Multiple references, identity/structure pathways and reference selection already have prior art. A training-time reference-intervention hypothesis is proposed for further overlap review and feasibility testing; it has not been implemented, trained or validated. Matched baseline reproduction, equal-capacity ablations and independent identity-level evaluation are required before claiming improvement. Missing-region inpainting remains a separate task. Novelty is not established.

### 29 September: structural mechanism tested and rejected before image production

- Completed the requested [mechanism-history audit](research/MECHANISM_HISTORY_AUDIT_20260928.md) and [structural prior-art review](research/STRUCTURAL_PRIOR_ART_20260928.md). The published missing-region baseline was already complete; it was not rerun or modified. Shape transfer, visible-anchor fitting and landmark warping have substantial prior art and are not claimed as new.
- Specified one geometry-only diagnostic, then froze its code, controls, evaluators, models and anti-regression gate before attempts. It registers personal reference landmarks to surviving target anchors and transports the fixed ResShift geometry inside the original mask. This differs from prior blending/routing experiments but is not established novelty.
- **All 24 structural attempts were rejected for negative inverse-map Jacobians; 0 new images were produced.** Candidate, single-reference and wrong-person arms each fail 8/8. Eight intervention-off checks are byte-exact and 19 synthetic tests pass. Do not mistake these method-feasibility failures for completed but poor candidate reconstructions.
- **40 reused control slots evaluated; 64 total terminal rows retained.** Candidate metrics/paired contrasts have n=0 and are unavailable. The reporter's Holm 1.0 missing-comparison placeholders are not statistical results. The separate 106-landmark metric shows that slightly improved landmark alignment can still accompany worse pixel/perceptual fidelity in existing reference controls. All eight visual sheets were reviewed; candidate panels explicitly show no output.
- Numerical results, failure analysis, source integrity, per-seed coverage and the next mechanistically different hypothesis are in [structural review](research/STRUCTURE_TRANSPORT_REVIEW_V1.md), [results](research/STRUCTURE_TRANSPORT_RESULTS_V1.md), [CSV](research/structure_transport_metrics_v1.csv), [integrity receipt](research/structure_transport_integrity_v1.json) and [visual review receipt](research/structure_transport_review_v1.json). No parameter search, training, website promotion or independent-cohort expansion follows this failed gate.
- **Reserved-data correction:** recorded final model generation/evaluation remains 0/8. Earlier automated duplicate/hash preparation decoded reserved images, so historical wording that they were never opened was too broad. No reserved image was accessed in this continuation. This clarification preserves all historical results.
- Local studio restored and main/evidence/session routes checked: HTTP 200, idle. Public hosting remains paused. The dataset browsing library and existing practice sets are in Downloads/All Face Datasets; licensed photos remain local.

### 27 September: two further mechanisms and a published missing-region baseline tested

- Completed **56 new reconstructed images**, four additional full GPU sampler-equivalence calls and **80 scored comparison rows** across two frozen screens. Reused controls are counted separately. These runs used the same four already observed development identities and two seeds, not a fresh validation cohort. No generator training or final-test use occurred.
- Personal references plus visible-context correction raise FaceNet similarity **0.7415 → 0.7814**, but missing-region MAE worsens **0.06266 → 0.06599** and LPIPS **0.02540 → 0.02711** relative to the original ResShift completion. Context correction helps relative to the plain reference cascade but does not satisfy the prespecified multi-metric gate. A follow-up that drops generated-region conditioning produces featureless patches and also fails. **Neither mechanism establishes novelty or is promoted to the website.**
- All 80 rows preserve visible pixels exactly; all **16 visual sheets** were inspected. The first screen has complete identity measurements; the second has two FaceNet and seven ArcFace detector failures across seven output rows. Its masked-arm FaceNet means use three complete identities and ArcFace means two, with missing scores retained. Twelve relevant mechanism/statistical tests pass. The initial wrapper failure and its technical recovery remain separate.
- Checked the next data source without downloading image pixels: local unused validation has at most **11 groups with eight photographs**; official FFHQ-Ref validation has only **four** author-predicted reference groups of that size. These counts are before duplicate/quality/identity screening. Training data and reserved final images cannot substitute for independent validation.
- The published **RefFaceInpainting missing-region baseline now runs locally**: all three author checkpoints strict-loaded, native inference matches the author on synthetic tensors, **16/16 new predictions and 28/28 evaluation rows** completed. Its fixed first-reference FaceNet **0.5196**, hole MAE **0.09231** and LPIPS **0.04379** are worse than matched seed17 ResShift **0.7265**, **0.06320**, **0.02515**. All four reference choices and four visual sheets are retained; it is not promoted to the website. See [baseline results](research/REFFACE_BASELINE_RESULTS_V1.md) and [failure review](research/REFFACE_BASELINE_REVIEW_V1.md).
- Read-only instrumentation reproduced all 16 native outputs exactly and found the author's component rule fully suppresses reference texture in **46/163 eligible component/layer events** when part of that component remains visible. This is a causal hypothesis to test, not proof of the source of blur or a novel improvement. Two additional preprocessing checks pass. Across this continuation there are **72 new reconstruction images and 108 scored rows**, with reused controls and repeat diagnostics explicitly separated. Publication still requires a beneficial mechanism, independent validation and human fidelity assessment.
- Full results, failure counts, visual findings, prior-art boundaries and current decisions: [mechanism establishment](research/NOVELTY_ESTABLISHMENT_20260925.md), [reference/context results](research/REFERENCE_CONTEXT_RESULTS_V2.md), [scaffold-conditioning results](research/SCAFFOLD_CONDITIONING_RESULTS_V1.md) and [integrity audit](research/reference_context_audit_v1.json). The local main/evidence pages respond successfully; public hosting remains paused.

### 25 September: new mask-support mechanism implemented and rejected by its controls

- Implemented a frozen-generator policy that hides reliable context patches, measures their recovery, and selects how far to expand the reconstruction mask. This addresses fully missing regions; it does not read reference photos, clean target pixels or withheld galleries during inference. A protocol and source signatures were frozen before generation. Existing research generators and website defaults are unchanged.
- Completed **104/104 generator calls and 80/80 scored candidate/control rows**, across four already observed development identities and two seeds. Controls include four fixed mask radii, random support, support averaging and equal-cost five-seed averaging. **Eight new unit/statistical checks pass**; all eight visual comparison sheets were inspected. All 72 reconstructed rows have successful FaceNet/ArcFace detections and exact visible-pixel preservation. Damaged-input detector failures remain counted separately.
- The candidate loses to the original-mask control: FaceNet **0.6111 versus 0.7415**, gallery FaceNet **0.3550 versus 0.4230**, missing-region MAE **0.07665 versus 0.06266**, and LPIPS **0.02962 versus 0.02540**. Probe error and actual missing-region error have negative rank correlation in five of eight case/seed comparisons. The available visible-patch signal does not support this selection rule. All eight primary Holm p-values are 1.0; four identity units cannot establish superiority.
- **Progression gate failed.** Do not promote this mechanism, expand its data collection, train an adapter on this premise, or describe it as proven novelty. Novelty review found substantial overlap with self-supervised calibration, internal-image adaptation and mask perturbation. Details, primary sources, immutable numerical receipts and visual findings are in `research/CONTEXT_SUPPORT_METHOD.md`, `research/CONTEXT_SUPPORT_RESULTS_V1.md` and `research/CONTEXT_SUPPORT_REVIEW_V1.md`.
- Remaining main research work is a mechanism that improves missing-feature identity/expression over matched controls, followed by independent development confirmation and final evaluation. Small visible-patch errors, reference agreement and visual sharpness have all proved insufficient here. Broader client accuracy and publication readiness remain unresolved. The local studio was restarted and its readiness checked at `http://127.0.0.1:8765/`; public hosting stays paused.

### 24 September: final client-input repair verification

- Real browser uploads completed missing-area reconstruction, repeated partial restoration and a 256/512/1024 selected-reference comparison. Six generated outputs have verified matching input/reference pixels, HTTP 200 downloads, hashes and exact outside-mask preservation. The main comparison uses 512 processing. Changing targets clears old-person references; editing the map clears the previous result with an explanation. No console errors were observed.
- Corrected ResShift mask coverage improved FaceNet **0.7481 → 0.8630**, original-damage MAE **0.08800 → 0.07163**, and LPIPS **0.02982 → 0.02037** on the same two diagnostic identities. Both cases improved on these metrics, but this small before/after check does not establish arbitrary-client accuracy.
- Visual inspection exposed learned refinement dropping user-painted pixels. The studio now preserves every requested pixel and allows refinement only to add coverage. Two subsequent LaMa/ResShift runs and scores passed, with **zero** requested pixels dropped. LaMa still produces poor large-feature reconstructions; it remains a small-repair option. References, larger sizes and confidence blending do not guarantee the hidden face.
- Final validation: **41 Python checks and three Node suites passed**. The initial four expanded/refined score failures were traced to the caller passing an original mask into an effective-mask preservation check; a separate corrected receipt scores all four saved outputs without changing the evaluator or images. All runtime and evaluator failures remain recorded. See `research/STUDIO_CLIENT_FIXES_20260924.md` and `research/studio_browser_verification_v2.json`.

### 24 September: unfamiliar-image audit and client reconstruction fixes

- Audited all offered generation variants on two newly selected local development identities: **25/26 generations completed and 29/30 rows scored**, including four unchanged-input controls. One OpenCV runtime failure remains in the original audit. All four comparison sheets were reviewed: 256-pixel reference output can leave coloured hole artifacts, gentle 0.5 reconstruction can retain erased damage, and large missing features still produce identity/expression errors. These identities are now observed; pretrained exposure is unknown. See [the accuracy audit](research/UNFAMILIAR_STUDIO_ACCURACY_V2.md).
- Fixed client-input handling: narrow ResShift masks survive resolution reduction; image/mask/evidence transforms preserve aspect ratio; reduced evidence maps conservatively retain missing/partial marks. Unsupported thin SDXL components are rejected with actionable guidance rather than silently losing mask support. This guard does not make every fine boundary reconstructible. Successful 512 processing is now the main result in size comparisons.
- The evidence editor clears stale references/results on target changes, guards asynchronous uploads and running jobs, preserves explicit missing/reliable labels during suggestions, rejects transparent evidence maps, and prevents gentle reconstruction of completely missing areas. ReF-LDM now validates each reference before GPU work and frames rectangular references without stretching. This is studio preprocessing, not a change to the frozen research models.
- Separate post-fix verification passed **8/8 functional checks**, including the previously failed 1024 run, a byte-identical 512 default output, thin-mask ResShift, portrait uploads, and expanded/learned masks. A further client-input check had two successful ResShift runs and two ReF-LDM wrapper failures caused by an unnecessary OpenCV import in its separate environment. That wrapper fault was fixed; **both separately recorded recovery runs passed**. Square-reference ReF-LDM output exactly matches its pre-fix result. Original failures are retained in their receipts.
- **41 Python checks (19 HTTP and 22 geometry/preparation/runtime checks) and three Node suites pass.** The studio worker enforces its OpenCV thread cap through nested calls; the ReF-LDM child avoids the unneeded OpenCV import. This mitigates the observed failures but does not establish a universal native-runtime root cause. **Final browser verification of the current main-page and evidence-page flows is complete**; the six outputs and state checks are recorded in `research/studio_browser_verification_v2.json`.
- Frozen research generators, reserved final identities and local-only photo/weight handling remain unchanged. Facial accuracy on arbitrary client images, a beneficial novel method and publication readiness are still unresolved. Fix details and evidence: [client reconstruction fixes](research/STUDIO_CLIENT_FIXES_20260924.md).

### 24 September: user-requested processing-size comparison

- Added 256 / 512 / 1024 reference-processing choices and a three-run comparison with shared image, mask, photos and seed. Each result has its own image, timing, settings and download; failures are displayed without discarding successful sizes. Exports stay at 512 pixels for visual comparison. Default processing remains 512, with obstruction-colour neutralization off.
- Real browser comparison completed at all three actual model resolutions, with matching inputs/settings, HTTP 200 downloads and exact outside-mask preservation. All three images loaded; no browser console errors. **23 automated tests pass**. The 256-pixel result had a severe coloured eye-region artifact and is explicitly experimental, not a quality upgrade.
- The 256-pixel path is a studio-only extension; frozen research generators are unchanged. This feature does not establish that a larger or smaller size is more accurate. It is separate from the preceding 36-generation audit.

### 24 September: full studio-mode accuracy audit and default correction

- Attempted **36 generations across four development identities** and eight unchanged controls: **34/36 generations completed, 42/44 rows scored**. All 34 completed reconstructions preserve outside-mask pixels exactly. Two runtime failures remain counted; separate recovery attempts are documented independently.
- The recent neutralized 512-pixel reference path regressed identity in **3/3 completed historical comparisons**. Restored the previous reference path as the default; obstruction-colour removal is now an unchecked experimental option. This supersedes the earlier default-preprocessing claim below. ResShift's visible-detail preservation fix remains.
- On four mixed-damage cases, ReF-LDM plus the fixed evidence blend improved identity, masked MAE and LPIPS over unchanged input in **4/4** cases; SDXL evidence blending reduced identity in **4/4** cases. This does not establish performance on arbitrary unknown images. All eight target/output sheets were inspected; gaze, expression, blur and mouth/eye errors remain.
- **21 automated tests pass** after the default correction; JavaScript syntax passes. See `research/STUDIO_ACCURACY_V1.md`, `research/STUDIO_ACCURACY_REVIEW_20260924.md` and numerical receipts. Accuracy inside missing regions and novelty remain unproven; reserved final identities remain untouched.


### 23 September: main website generation repair

- Investigated the user-upload failure across LaMa, ResShift and reference modes. Five fresh HTTP/GPU runs completed; all preserve pixels outside the effective mask exactly. Reference missing-area inference now neutralizes masked RGB and uses full denoising, with original-image composition. The tested 1024-pixel case generates eyes instead of sunglasses, but identity/expression accuracy is not established.
- ResShift still infers at 256 pixels but now preserves visible detail in a 512-pixel output. Mask expansion is consistently 8 pixels on the display canvas. Higher-resolution reference inference is explicitly experimental. LaMa blur and ResShift eye errors remain on the large missing-eye case; generation quality is not solved.
- Twenty automated tests pass (14 HTTP, 6 geometry/composition); JavaScript syntax passes. Local hosting restarted. Frozen research generators and final identities are untouched. See `research/STUDIO_GENERATION_REPAIR_20260923.md` for evidence and remaining limitations.


### 23 September continuation: reference-proxy mechanism implemented and tested

- Replaced disagreement ranking with a fitted reference-proxy preservation rule. The first supplied reference is synthetically damaged and restored using only the other three references; its original supervises a nine-coefficient blending rule. Actual target truth and withheld gallery are evaluation-only. The generator and reference adapter remain frozen. This is an implemented postprocessing mechanism, not trained local-feature fusion or calibrated correctness probability.
- Frozen protocol completed **20/20 proxy generations and 20/20 scored candidate/control images** on four already-observed mixed-damage cases. Three new unit tests pass; all output hashes and frozen mechanism source verified. All four visual comparison rows inspected. Evidence: `research/REFERENCE_PROXY_METHOD.md`, `research/PROXY_CALIBRATION_RESULTS_V1.md`, protocol and numerical receipts.
- Spatial calibration / fixed-half blending: FaceNet **0.9197 / 0.9218**, gallery FaceNet **0.5060 / 0.5146**, masked MAE **0.041095 / 0.041459**, LPIPS **0.041328 / 0.036034**. Slightly lower pixel error does not compensate for worse identity and perceptual metrics. All four primary Holm p-values are **1.0**; progression gate not met. No fresh-identity expansion, fusion training or website promotion is justified by this experiment.
- Novelty remains unproven. Prior-art review also covers self-supervised reference restoration and test-time adaptation; a new name or fitted blending rule is insufficient. Future work must address the structural/identity errors of the restorer and demonstrate incremental benefit over fixed blending, rather than expanding these unsupported gates. Reserved-final identities remain untouched and the local testing site is unchanged.

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
| Reference-proxy calibration generation | 100% (20/20) | Five proxy degradations per observed identity |
| Reference-proxy candidate scoring | 100% (20/20) | Four identities and five arms; progression gate not met |
| Unfamiliar development initial generation | 75% (3/4) | One native process failure retained |
| Unfamiliar development after runtime retry | 100% (4/4) | One separately recorded successful retry |
| Unfamiliar development scoring after retry | 100% (12/12) | Two identities, controls and native/composed outputs |
| New studio audit initial generation | 96.2% (25/26) | Two additional development identities; one original runtime failure retained |
| New studio audit scoring | 96.7% (29/30) | Includes four unchanged controls; not 30 independent people |
| New studio visual sheet review | 100% (4/4) | Missing/mixed conditions on two identities; internal review, not blinded |
| Separate studio repair verification | 100% (8/8) | Functional checks on now-observed images, not an accuracy benchmark |
| Further client-input check / recovery | 2/4 initially; 2/2 recovery | Two ReF-LDM wrapper failures retained separately from successful reruns |
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

1. Broaden real client-photo and varied framing/reference coverage; the current main/evidence browser flows are verified. Retain native-runtime failures and monitor recurrence; passing repairs do not establish universal reconstruction accuracy.
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

The generative backbones remain pretrained and frozen. Actual local training now includes the 545-parameter reference-intervention pilot and the 193-parameter target-conditioned regional compatibility pilot. These development adapters are not promoted to the website and do not establish improved identity recovery. Separate Human Faces LoRA work is set aside and is not evidence for the existing-dataset study.

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

The site supports image upload, crop/fit, mask painting/erasing/undo, mask upload, model selection, reference upload, 256/512/1024 reference processing and comparison, and result/mask downloads. Ordinary reference mode uses three or four photos; experimental selection accepts one to four and selects one. A new input clears the previous person's reference photographs on both upload pages.

Website exports are 512 × 512 for every method. ResShift operates at 256 internally and composites into the 512-pixel input to preserve visible detail. Reference models offer 256, 512 and 1024 processing; comparison defaults to a successful 512 result. LaMa's unnecessary downsampling and nonsquare uploaded-mask alignment were repaired on 14 September. The 24 September client fixes additionally preserve thin ResShift mask coverage and aspect ratio, validate ReF-LDM references, and reject unsupported thin reference marks. Large missing-feature reconstruction remains unreliable; these repairs do not establish semantic accuracy. Historical command-line/research settings remain unchanged. Images are saved locally with run metadata.

The reference runtime was repaired in a separate environment after Windows blocked native dependencies. Package pins and installation records were retained. No Windows protection was disabled. Progress-file retries prevent transient OneDrive locks from needlessly aborting reference inference.

On 14 September:

- Four-reference standard HTTP inference completed in 45.297 seconds including startup and verification, with 16.5 seconds of inference including offload. The output preserved all pixels outside the effective mask.
- LaMa passed exact, expanded and learned-mask checks; ResShift passed exact and expanded-mask checks.
- A browser-driven LaMa run displayed its output, and both result and mask downloads completed.

These timings describe individual checks, not a population latency benchmark. Functionality on these cases does not establish robustness across all user photographs.

### 2.5 Repository and hosting

Code, local refiner checkpoints, experiment records, provenance and reproduction instructions are in the repository. The client-input repair release follows `d712706` on `codex/local-studio-checkpoint` and includes the new audit, runtime/geometry/UI fixes, mask-coverage corrections and numerical verification receipts. Restricted photographs and pretrained weights remain local.

Temporary public sharing was implemented and an HTTPS GPU test completed. The in-app browser then exposed sign-in compatibility problems. Public sharing is now stopped, its homepage link removed, and further hosting work deferred. The local app requires no sign-in. The paused hosting investigation is not a blocker for local research.

## 3. What the results currently support

**Mask refinement has a tradeoff.** At the initial training budget, preservation weighting reduced changes to visible pixels but worsened aggregate reconstruction versus the generic refiner. At 6,000 updates, weighted refinement performed better than the generic 6,000-update control on aggregate reconstruction metrics, but the generic model itself deteriorated with longer training. This does not establish that more training universally improves completion or that the refiner is state of the art.

**Reference guidance has not shown a consistent reconstruction-quality benefit in the completed diagnostic.** At strength 0.99 with Poisson composition, mean LPIPS was 0.032674 with references and 0.032460 with reference conditioning disabled. The paired difference was +0.000214, with an exploratory 95% interval spanning zero. Hole MAE was also worse with references in this setting. Other settings differ; the complete table must be retained rather than selecting the most favorable contrast.

**Independent identity diagnostic completed on 14 September:** FaceNet scored the frozen reference outputs, with 88/96 valid scores across 11/12 identities. At strength 0.99 with Poisson composition, reference conditioning increased mean cosine similarity from 0.4942 to 0.6588 (paired difference +0.1646, descriptive bootstrap 95% interval [0.0492, 0.2903]). One identity returned multiple detections in its target and all outputs and remains unscorable under the frozen protocol. This small development result does not establish final generalization, correct gaze/expression or a novel contribution. See `research/IDENTITY_DIAGNOSTIC_RESULTS_V1.md`.

**Expanded evaluation completed:** ArcFace/InsightFace, FaceNet, NIQE, BRISQUE, RGB SSIM, whole-image/hole PSNR, LPIPS and hole/visible MAE now cover 144 outputs. NIQE/BRISQUE/structural scores are valid for all 144; FaceNet has 132 valid scores and ArcFace has 140. ArcFace uses the same w600k_r50 checkpoint as conditioning and is not an independent verifier. FaceNet reproduces the prior scores exactly. All image hashes and all 24 new candidates' matched settings were verified.

**Reference-count ablation:** at strength 0.99 with Poisson composition, FaceNet means are 0.6125, 0.6482 and 0.6588 for one, two and four references. Four minus one is +0.0463, descriptive 95% interval [-0.0150, 0.1184], on eleven jointly scorable identities. This does not establish that more references reliably help. All 180 metric/contrast tests receive Holm correction; no FaceNet contrast meets corrected p < 0.05. See [expanded results](research/EXTENDED_ABLATION_RESULTS_V1.md).

The measurement baseline is substantially stronger, but the research is not yet ready to claim the proposed method improves identity preservation or is publication-ready.

## 4. Remaining work

**Reconciled with completed work on 2 October 2026.** Completed implementation is distinguished from unresolved research claims below. The target-conditioned regional pilot is in `research/TARGET_COMPATIBILITY_RESULTS_V1.md`; the earlier trained intervention is in `research/REFERENCE_INTERVENTION_PILOT_RESULTS_V1.md`. Rejected hypotheses remain history, not unexecuted promises. The additional Human Faces dataset work is set aside.

| Area | Current evidence and remaining work | Status |
|---|---|---|
| Identity/quality measurement | FaceNet target/gallery, diagnostic ArcFace, LPIPS, SSIM, PSNR, hole/visible MAE, NIQE and BRISQUE executed; final evaluation remains sealed | Development implemented |
| Distortion/preservation | Six kinds, strength/scale screen and fresh development extension completed; unchanged-input controls still expose fidelity loss | Diagnostic complete; method unresolved |
| Reference robustness | Compression failure diagnosed; studio ReF-LDM now validates one face per reference and frames rectangular uploads without distortion. Broader poor-quality, pose and expression handling still needs matched validation with failures retained | Basic client checks implemented; broader validation pending |
| Local correspondence | Five-point similarity and eight regional features implemented; 22/24 initial cases detected; 3D pose/expression/occlusion unresolved | Prototype complete |
| Local fusion | Six-policy, two-seed 48-row comparison complete; no multi-metric advantage established | Negative/mixed result |
| Learned adapter | Old 545-parameter intervention and new 193-parameter target-conditioned pilot trained on the GPU; regional scores and equal-capacity control recorded. Balanced timestep/corruption sampling, image-space losses, complete ablations and independent confirmation remain | Trained development candidates; no promotion or novelty claim |
| Confidence UI | Prior generation/download checks passed. Current update fixes stale state, upload races, evidence suggestions/transparency, and missing-area strength selection; new HTTP/GPU checks pass. Current main/evidence browser flows are verified with real generated images, matching references, downloads and state invalidation | Implemented and browser verified |
| Runtime reliability | Original native/OpenCV failures retained; fixed-thread studio retry passed. Two wrapper failures from an unnecessary OpenCV import were fixed and both recovery runs passed. Continue recurrence checks; no universal root-cause claim | Mitigations and recoveries verified |
| Visual assessment | Prior reviewed studies retained; all four new studio audit sheets reviewed, with coloured holes, retained erasures and facial errors documented. Blinded human evaluation remains | Internal review complete for these runs; human study pending |
| Candidate selection | Matched 1/2/4 candidate pools, random/identity/quality/combined selection and separate evaluator | Deferred until stable restoration |
| External comparisons | ReF-LDM completed 24/24 generations and 48/48 native/composed evaluations; new RefFaceInpainting completed 16/16 predictions and 28/28 comparison rows, but fails these missing-feature fidelity controls. Additional relevant comparisons and broader validation remain; paused OSOR results remain delimited | Two external author baselines executed; broad superiority unsupported |
| Reference-risk mechanism | Implemented 16 leave-one-reference-out generations and 12/12 scored controls. Disagreement does not beat edit magnitude on identity or masked error; AUC 0.417–0.533. Revise the mechanism before calibration/training or novelty claims | Diagnostic complete; incremental benefit unsupported |
| Reference-proxy mechanism | Implemented a reference-only fitted spatial blend; 20 proxy generations and 20 scored images complete. Slight MAE gain versus fixed blending comes with worse identity/gallery/LPIPS. Three new tests pass. Address structural fidelity or develop a justified alternative before further expansion | Diagnostic complete; progression gate not met |
| Context-support mechanism | New input-only mask-support calibration completed 104 generations, 80 scores, two seeds and eight visual sheets. It loses to fixed-mask and equal-budget controls; small visible probes do not reliably predict missing-feature fidelity. No expansion or website promotion is justified | Implemented and tested; progression gate not met |
| Reference/context mechanism | Completed 32 new restorations, eight reused scaffolds, two equivalence calls and 40 scored rows. References improve identity; correction reduces some cascade error but still loses to the scaffold on pixel/perceptual error. All eight visual sheets reviewed | Implemented and tested; progression gate not met |
| Scaffold-conditioning mechanism | Completed 24 new restorations, 16 reused controls, two equivalence calls and 40 scored rows. Spatial conditioning dropout causes featureless regions and nine detector-failure events; all failures and eight visual sheets retained | Implemented and rejected; no website promotion |
| Missing-region external baseline | Official RefFaceInpainting and parser pinned, all three checkpoints strict-loaded, synthetic author equivalence passed. Sixteen predictions, 28 scores and all four visual sheets completed; fixed first-reference result loses to ResShift. Component gate measured with 16 bit-exact repeat forwards; its causal role remains untested | Reproduced locally; not suitable for website promotion on these results |
| Structure-only reference transport | Frozen personal/single/wrong-person geometry screen: 24/24 inverse fields rejected for negative Jacobians; 0 new images, 40 reused control slots evaluated, 8 visual sheets reviewed. 19 tests and 8 zero-displacement checks pass. No candidate quality scores exist. Next hypothesis must separate identity geometry from pose/expression and synthesize missing structure; its novelty/feasibility remain unresolved | Implemented; feasibility gate failed; no promotion or expansion |
| Unfamiliar-person checks | Earlier two-identity smoke retained. Two additional identities now have 25/26 initial studio generations and 29/30 scores; separate repair/recovery checks preserve original failures. All are now observed development cases, with unknown pretraining exposure. Larger independent validation remains | Expanded diagnostic complete; generalization unproven |
| Client geometry and mask handling | Thin ResShift masks, paired evidence downsampling, nonsquare API framing and primary 512 comparison fixed. SDXL latent-unsupported components are rejected; fine-boundary fidelity and broad client-photo accuracy remain unresolved | Concrete bugs fixed; model limits remain |
| Statistical power and data availability | More identity units and independent severity/mask/reference factors needed. Remaining local validation has at most 11 groups with eight photos; official FFHQ-Ref validation only four such predicted groups. Metadata checked without opening new/final pixels. Larger studies need a verified additional cohort or a separately frozen different reference/gallery design | Required before method freeze |
| Optional metrics | Reconstructed-landmark error; FID only at adequate sample size; ROC/TAR only with adequate verification trials | Unimplemented; scope-dependent |
| Final evaluation | Require method/parameters/evaluator freeze, independent confirmation, ablations and baseline comparison first. No reserved model generation/evaluation; historical automated duplicate/hash preparation did decode files, disclosed in the history audit | 0/8 model evaluations; no new reserved access |
| Manuscript and presentation | Correct novelty/architecture/results, choose actual venue, prepare figures, limitations and reproducibility | Incomplete |
| GitHub release | Current research checkpoint adds two frozen screens, corrected technical recovery, numerical receipts, coverage/integrity audit, metadata-only data census and updated report after context-support commit `7c73cce`. Restricted photos, per-person feature arrays and weights remain local | Included in this research checkpoint |
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

### Phase 4 — develop the smallest justified fusion method (small adapters trained; useful contribution unresolved)

Regional reference features, deterministic fusion and two compact learned-adapter investigations are implemented. The old intervention did not satisfy its joint quality gate. The new pilot compares target-conditioned regional compatibility with a matched-capacity feature ablation and the frozen baseline under eight reference conditions. Its results are development evidence only. Complete the missing loss investigations and regional/global ablations before any independent expansion, and require a passing predeclared gate. Preserve the working pretrained baseline and all frozen earlier results.

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

- [Complete mechanism history and reserved-data clarification](research/MECHANISM_HISTORY_AUDIT_20260928.md)
- [New structural method review and next decision](research/STRUCTURE_TRANSPORT_REVIEW_V1.md)
- [Frozen structural protocol](research/protocols/structure_transport_v1.json) and [numerical results](research/structure_transport_results_v1.json)

- [Current client reconstruction fixes and remaining limitations](research/STUDIO_CLIENT_FIXES_20260924.md)
- [New unfamiliar-identity studio accuracy audit](research/UNFAMILIAR_STUDIO_ACCURACY_V2.md)
- [Separate repair verification](research/studio_fixes_verification_v2.json)
- [Client-input failures retained](research/studio_client_inputs_verification_v2.json) and [separate recovery](research/studio_client_inputs_verification_v2_recovery.json)
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


### 30 September 2026 — restoration display correction

Inspection of local job 12b33b5ab5a745969c81ca4b46a930da found that evidence blending visibly reintroduced noise into the restored facial regions. The evidence page now defaults to the unblended, mask-composited prediction for ReF-LDM and provides a selector for the evidence-blended alternative. Downloads follow the selected image. Outside-mask pixels were verified unchanged in the saved prediction. Synthetic interface regression checks and live HTTP asset checks passed. No new model inference or accuracy measurement was performed for this display correction. Facial detail and boundary imperfections remain; this is not evidence of improved model accuracy or established novelty. Frozen research results are unchanged.


### 30 September 2026 - full restoration inspection

Visual inspection of studio job d85fb5d619344c879498075514f5c5b1 found spatial disagreement between the whole-image ReF-LDM prediction and the original photo: composing small facial patches produces visible seams. Added an explicit full-image restoration comparison/download option for ReF-LDM. It is optional because it modifies unmarked regions and can alter eyes and expression. No model accuracy improvement is claimed. Interface regression checks passed; the local server was restarted while idle and the served raw PNG checksum matched existing metadata. No new model run or research evaluation was performed. Better spatial fidelity remains unresolved.


### 30 September 2026 - tested restoration finishing

Added studio-only gradient-domain patch lighting correction and made its separate output the default ReF-LDM preview. Original, full-image and evidence-blended outputs remain selectable. On 20 previously observed development cases (four identities; blur, noise, JPEG, downsample and mixed), mean masked MAE decreased from 11.7368 to 9.6629 on a 0-255 scale; 19/20 improved. These are reused model predictions, a limited engineering screen, not independent confirmation or an identity improvement claim. A fresh GPU wrapper run on the uploaded example completed. Synthetic tests cover outside-mask preservation, lighting continuity, border/empty masks and invalid geometry. Noise outside the selected mask and inaccurate facial structure remain possible. Frozen research sources and outputs were not modified.


### 30 September 2026 - noise-boundary defect

The previous smoothing step retained a noisy ring because its solve boundary lay within the damage. Added a studio heuristic: median RGB residual against a 3x3 median filter above 6 at the inner mask edge triggers an 8px wider solve domain. Only originally marked pixels are written back. Four observed noise cases improved; the other 16 development cases retain prior processing. This rule was chosen using development diagnostics and is not independently validated. Uniform expansion worsened overall MAE and was rejected. A fresh ReF-LDM GPU trial with pre-denoising still distorted features and was rejected. Synthetic ring-removal/outside-preservation and UI regressions passed. Blur detail and identity fidelity remain unresolved model limitations; no novelty or universal accuracy claim.


### 30 September 2026 - unknown portrait eye failure

Inspected the user portrait eye-erasure size comparison. Input/mask alignment was correct, but generated eye shape and gaze were wrong. A fresh 1024px trial with adapter scale 1.2, 50 steps and full denoising exaggerated the eyes and was rejected. Production settings were not changed. Scores are recorded in research/uploaded_portrait_eye_failure_20260930.json; this is a single diagnostic using synthetic references derived from the target photo, not valid independent generalization evidence. Erased-eye identity fidelity remains unresolved.


### 30 September 2026 - replacement model screening on user portrait

Completed four RefFaceInpainting predictions (one per supplied synthetic reference), one ResShift scaffold and one ReF-LDM refinement of that scaffold. All six saved predictions were scored after inference; reference1/reference4 and scaffold/cascade were visually inspected. Dedicated model outputs show severe artifacts; the cascade retains incorrect eye geometry. None is promoted to the application. All six preserve observed pixels outside the mask. This one-photo diagnostic does not support independent generalization or improved identity accuracy. Production remains unchanged; reliable unknown-image reconstruction remains an open core requirement. See research/unknown_portrait_candidates_20260930.json.


### 30 September 2026 - visible-landmark alignment diagnostic

Tested eight reference-transfer outputs: four supplied synthetic references aligned using only unmasked nose/mouth landmarks, each with ordinary or collar-aware Poisson composition. No clean target entered inference. Collar-aware composition reduced the grey colour artifact, but incorrect eye placement and pose artifacts remained on visual review. These candidates are not promoted. This does not establish that alignment fixes unknown-image reconstruction. A real near-frontal independent reference is needed to test whether reliable reference geometry helps this case; success is not guaranteed. See research/unknown_alignment_diagnostic_20260930.json.


### 30 September 2026 - single-image seed and framing screen

Generated 30 fresh ResShift predictions: seeds 17/42/12345 and vertical shifts -32/-64/+32, on four existing development identities plus the uploaded portrait. Inference used no reference photos or clean targets. Every scored output preserves unmarked pixels. Seed changes did not repair the uploaded eye geometry. The -32px shift reduced uploaded masked MAE from 41.53 to 27.40 but worsened the four-identity development mean from 18.31 to 25.47; gaze remained wrong. No offset or seed was promoted as a general fix. This development diagnostic is not independent confirmation. See research/single_image_diagnostic_20260930.json. Reliable framing normalization and unknown-image fidelity remain unresolved.


### 30 September 2026 - paper map and RAD compatibility investigation

Added research/PAPERS_AND_IMPLEMENTATION_MAP.md separating implemented generators, evaluation metrics, and related work. The single-image foundation is the extended ResShift restoration paper; blur/noise restoration uses ReF-LDM; reference-guided missing-region inference combines SDXL and FaceID Portrait. There is no validated single novel base method.

Downloaded the official RAD FFHQ checkpoint, recorded its hash and source revision, and created an isolated runtime without changing the website environment. Full checkpoint tensors load strictly. Both 100-step and 1000-step smoke runs completed but produced unusable noise in the missing region. Treat this as an unresolved reproduction/compatibility problem, not a demonstrated failure of the published method. RAD is not deployed. The ignored attention_legacy_order config and checkpoint/inference compatibility need investigation. See research/rad_preflight_20260930.json and scripts/rad_diagnostic.py.
