# Facial Inpainting Project — Progress and Completion Report

**Report date:** 25 September 2026
**Current platform:** Windows laptop, NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM  
**Current application:** http://127.0.0.1:8765/  
**Repository:** https://github.com/rehaan1603/facial-inpainting-studio  
**Status:** Working local research baseline; proposed multi-reference research method and final validation incomplete. Public hosting paused at the owner's request.

## Latest status update

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

**Reconciled with completed work on 25 September 2026.** Completed implementation is distinguished from unresolved research claims below.

| Area | Current evidence and remaining work | Status |
|---|---|---|
| Identity/quality measurement | FaceNet target/gallery, diagnostic ArcFace, LPIPS, SSIM, PSNR, hole/visible MAE, NIQE and BRISQUE executed; final evaluation remains sealed | Development implemented |
| Distortion/preservation | Six kinds, strength/scale screen and fresh development extension completed; unchanged-input controls still expose fidelity loss | Diagnostic complete; method unresolved |
| Reference robustness | Compression failure diagnosed; studio ReF-LDM now validates one face per reference and frames rectangular uploads without distortion. Broader poor-quality, pose and expression handling still needs matched validation with failures retained | Basic client checks implemented; broader validation pending |
| Local correspondence | Five-point similarity and eight regional features implemented; 22/24 initial cases detected; 3D pose/expression/occlusion unresolved | Prototype complete |
| Local fusion | Six-policy, two-seed 48-row comparison complete; no multi-metric advantage established | Negative/mixed result |
| Learned adapter | Train only after deterministic local fusion shows convincing benefit; do not train full SDXL | Conditional gate not met |
| Confidence UI | Prior generation/download checks passed. Current update fixes stale state, upload races, evidence suggestions/transparency, and missing-area strength selection; new HTTP/GPU checks pass. Current main/evidence browser flows are verified with real generated images, matching references, downloads and state invalidation | Implemented and browser verified |
| Runtime reliability | Original native/OpenCV failures retained; fixed-thread studio retry passed. Two wrapper failures from an unnecessary OpenCV import were fixed and both recovery runs passed. Continue recurrence checks; no universal root-cause claim | Mitigations and recoveries verified |
| Visual assessment | Prior reviewed studies retained; all four new studio audit sheets reviewed, with coloured holes, retained erasures and facial errors documented. Blinded human evaluation remains | Internal review complete for these runs; human study pending |
| Candidate selection | Matched 1/2/4 candidate pools, random/identity/quality/combined selection and separate evaluator | Deferred until stable restoration |
| External comparisons | ReF-LDM completed 24/24 generations and 48/48 native/composed evaluations; partial-damage results are stronger than matched SDXL but fail erased-region completion. Additional appropriate external controls and broader validation remain; paused OSOR results remain explicitly delimited | First external baseline complete |
| Reference-risk mechanism | Implemented 16 leave-one-reference-out generations and 12/12 scored controls. Disagreement does not beat edit magnitude on identity or masked error; AUC 0.417–0.533. Revise the mechanism before calibration/training or novelty claims | Diagnostic complete; incremental benefit unsupported |
| Reference-proxy mechanism | Implemented a reference-only fitted spatial blend; 20 proxy generations and 20 scored images complete. Slight MAE gain versus fixed blending comes with worse identity/gallery/LPIPS. Three new tests pass. Address structural fidelity or develop a justified alternative before further expansion | Diagnostic complete; progression gate not met |
| Context-support mechanism | New input-only mask-support calibration completed 104 generations, 80 scores, two seeds and eight visual sheets. It loses to fixed-mask and equal-budget controls; small visible probes do not reliably predict missing-feature fidelity. No expansion or website promotion is justified | Implemented and tested; progression gate not met |
| Unfamiliar-person checks | Earlier two-identity smoke retained. Two additional identities now have 25/26 initial studio generations and 29/30 scores; separate repair/recovery checks preserve original failures. All are now observed development cases, with unknown pretraining exposure. Larger independent validation remains | Expanded diagnostic complete; generalization unproven |
| Client geometry and mask handling | Thin ResShift masks, paired evidence downsampling, nonsquare API framing and primary 512 comparison fixed. SDXL latent-unsupported components are rejected; fine-boundary fidelity and broad client-photo accuracy remain unresolved | Concrete bugs fixed; model limits remain |
| Statistical power | More identity units, independent severity/mask/reference factors, additional seeds | Required before method freeze |
| Optional metrics | Reconstructed-landmark error; FID only at adequate sample size; ROC/TAR only with adequate verification trials | Unimplemented; scope-dependent |
| Final evaluation | Freeze method/parameters first; eight reserved identities remain unused | 0/8, intentionally |
| Manuscript and presentation | Correct novelty/architecture/results, choose actual venue, prepare figures, limitations and reproducibility | Incomplete |
| GitHub release | This research checkpoint follows client-repair commit `c1bc5e3`; includes context-support source/tests, frozen protocol, numerical receipts, prior-art boundaries and updated report. Restricted photos and weights remain excluded | Included in the context-support research release |
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
