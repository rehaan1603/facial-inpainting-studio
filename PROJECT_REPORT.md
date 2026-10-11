# Facial Inpainting Project — Progress and Completion Report

**Report date:** 12 October 2026
**Current platform:** Windows laptop, NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM  
**Current application:** http://127.0.0.1:8765/ (current code; Start Studio verifies loaded version)
**Repository:** https://github.com/rehaan1603/facial-inpainting-studio  
**Status:** Working local research baseline; proposed multi-reference research method and final validation incomplete. Public hosting paused at the owner's request.

## Latest status update

### 11–12 October: dense correspondence and GPU controls completed

Frozen CLIP and DINOv2-small feature extraction completed on the same eight training people, with original masks, four real references, wrong-person controls and hidden-pixel invariance. No development reconstruction scores were used. DINO at1036 improved the known8px-shift gate from **0/8 to8/8**, with1.08px fitted-transform error, but reduced32px-control acceptance from7/8 at518 to3/8. **All three dense feature settings still have0/32 eligible real reference pairs and0/8 usable targets.** These controlled localization improvements are not reconstruction-quality gains.

Peak feature-extraction GPU allocation was1.30GB for CLIP,71.6MB for DINO518 and121MB for DINO1036. The encoder is not exhausting the laptop GPU; increasing compute alone has not resolved correspondence ambiguity. These costs are not SDXL training memory. Thirteen new unit tests passed (5 CLIP,4 DINO518,4 DINO1036), plus real known-warp and invariance controls. The original v3a settings-serialization assertion failure is retained alongside the corrected retry.

The fixed sequence is complete and fails its real-reference gate. No adapter training, new checkpoint, baseline replacement or new SSIM/LPIPS/identity claim follows. Next direction: separately specified pose-dependent anatomical correspondence/confidence, with training-only supervision if justified, rather than additional resolution sweeps. Current app, old model checkpoints and evaluation remain preserved.

[Complete controlled comparison and decision](research/DENSE_DINO_RESOLUTION_RESULTS_V5.md) · [DINO518 results](research/DENSE_DINO_GEOMETRY_RESULTS_V4.md) · [Known-warp diagnosis](research/DENSE_EQUIVARIANCE_RESULTS_V3A.md) · [CLIP results](research/DENSE_FEATURE_GEOMETRY_RESULTS_V3.md).

### 11 October: detector-independent sparse-feature diagnostic completed

A separately preregistered CPU-only SIFT registration test used the same eight training people and 32 real reference pairs. Four unit tests passed, but only **1/32 reference pairs** and **0/8 target cases** passed the fixed geometric gates; 28 pairs had insufficient mutual matches. No thresholds were tuned after execution. This method will not be used for training. The next candidate is dense frozen-feature correspondence with held-out visible-region validation. No new reconstruction scores or GPU training resulted; the application and baseline remain unchanged.

[Protocol](research/VISIBLE_FEATURE_GEOMETRY_PREFLIGHT_V2.md) · [Results and next step](research/VISIBLE_FEATURE_GEOMETRY_RESULTS_V2.md).

### 11 October: spatial correspondence audit and prerequisite test

The SDXL reference path was audited before changes. Its 64 FaceID tokens have no explicit anatomical coordinates. An isolated spatial-reference attention prototype now has **20 trainable head gates**, with the existing UNet/FaceID weights frozen. **18 CPU tests pass**, including zero-effect equality, finite gate gradients and hidden-pixel invariance. The application and current baseline remain unchanged.

The fixed geometry prerequisite passed for **only 2/8 training people**: four erased targets were not detected and two lacked reliable reference alignment. All failures are preserved. Per the preflight, **no GPU feature extraction, gradient/VRAM probe, training or new reconstruction evaluation was started**. Candidate quality/VRAM are unmeasured. Next is a separately specified detector-independent visible-feature correspondence test, before any training.

[Audit and ranked mechanisms](research/SPATIAL_CORRESPONDENCE_AUDIT_20261011.md) · [Preflight](research/SPATIAL_REFERENCE_CORRESPONDENCE_PREFLIGHT_V1.md) · [Results and failure diagnosis](research/SPATIAL_REFERENCE_CORRESPONDENCE_RESULTS_V1.md).

The main local PPT is corrected to preserve the original ReF-LDM architecture, methods and reconstruction slides, with two essential current result tables (18 slides total). The separate small-batch deck remains 10 slides. See [presentation index](presentations/README.md).

### 10–11 October: decoded pixel-loss comparison and whole-image PPT updates completed

The direct decoded missing-region gradient/resource audit passed. Two fresh **16-update GPU continuations** then completed from the same preserved 64-update checkpoint, with identical reset optimizers, eight training people, timesteps/noises and **385 trainable parameters**. Only the added **0.1 decoded-hole RGB L1 loss** differs. Initial checkpoint bytes and first-forward epsilon losses match exactly. Frozen-weight guards, evaluation exclusions and source/config/trace pins pass.

Seven generation calls complete: one exact initial-state replay and six new reconstructions. Nine scored controls are reused, giving **15/15 complete records**, valid FaceNet/ArcFace, matched supports and unchanged final visible pixels. Both arms' raw/finished outputs were inspected for all three people. The RGB candidate **fails the combined numerical and visual gates**; it stays offline.

| Same three people / seed 17 | Whole PSNR ↑ | Whole SSIM ↑ | Whole LPIPS ↓ | Hole SSIM ↑ | FaceNet ↑ | ArcFace ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Current CFG3 | 29.387906 | **0.961286** | 0.022612 | **0.454540** | **0.784486** | **0.816156** |
| Neutral local appearance | 29.397466 | 0.960728 | 0.022462 | 0.444884 | 0.782640 | 0.805353 |
| Prior 64-update scorer | **29.401143** | 0.960739 | 0.022468 | 0.445016 | 0.784304 | 0.805454 |
| Epsilon-only +16 updates | 29.399513 | 0.960732 | 0.022481 | 0.444992 | 0.782793 | 0.804817 |
| Decoded RGB +16 updates | 29.399369 | 0.960727 | **0.022452** | 0.444860 | 0.781355 | 0.805360 |

Against equally trained epsilon, RGB worsens hole PSNR by **0.000144**, hole SSIM by **0.000132**, hole LPIPS by **0.000025** and FaceNet by **0.001437**. Eye-crop LPIPS improves **0.152312 → 0.151757** and ArcFace improves **0.000543**. Whole LPIPS improves while missing-region measures regress. High whole SSIM includes preserved visible pixels and is **not an accuracy percentage**. Wrong gaze, cyan/pale iris colours and inaccurate eye/brow geometry remain. All 17 failed numerical checks and the negative visual decision are retained.

Epsilon worker: **81.625 seconds**, peak allocated GPU **7,290,791,936 bytes**. RGB worker: **322.094 seconds**, peak allocated GPU **7,290,989,056 bytes**, with differentiable CPU fp32 VAE decoding. Both finish within fixed 900-second limits. The roughly 3.95× cost explains compute expense, not the cause of fidelity errors. Short training, 512 training/1024 evaluation and statistical-token representation are additional limitations; RAM/VRAM alone is not an established explanation.

The overall and small-batch illustrated PPTs now show whole-image and separate region tables, actual scored outputs, architecture/methods and preserved historical values/paper links. Original decks remain intact. Photos stay local; code and full numeric evidence can be uploaded. Each existing person folder now includes **Experimental - Decoded RGB 16 updates** with five arms' raw/final images, comparisons and all values. Current website: **http://127.0.0.1:8765/**; its loaded source fingerprint matches disk.

[All results and failed gates](research/MASKED_SDXL_DECODED_CONTINUATION_RESULTS_V1.md) · [photo-free evidence](research/masked_sdxl_decoded_continuation_v1_evidence.json) · [fixed paired protocol](research/MASKED_SDXL_DECODED_CONTINUATION_PREFLIGHT_V1.md) · [gradient/resource audit](research/MASKED_SDXL_DECODED_LOSS_PROBE_RESULTS_V1.md) · [PPT locations](presentations/README.md).

**Next:** stop this exact RGB recipe and audit spatial attention/reference correspondence on training-only cases before selecting one capacity/representation change. Preserve masks, controls, evaluation and identity guards. Corrupted references, unfamiliar people, multiple seeds and reserved final validation remain after a promising controlled candidate. Research validation is work in progress.

### 10 October: masked SDXL GPU training and matched reconstruction completed

Completed a **64-update real GPU pilot** on eight existing training identities, training only **385 shared compatibility parameters** while freezing all **3,366,130,756 loaded backbone/adapter parameters**. Persons 263, 6681, 10149 and 7400 stayed outside training. The initial sixteen-case input cache failed on profile-face detection; a separate, preregistered training-only reference eligibility revision retained eight original targets with four distinct encodable same-person references each. Existing datasets, manifests, checkpoints and test inputs/references remain intact. Cache preparation, all 64 finite gradient updates, source/model hashes and the two exact initial-state inference replays pass.

| Fixed three people / seed 17 / erased eyes | Hole SSIM ↑ | Hole PSNR ↑ | Hole LPIPS ↓ | Eye LPIPS ↓ | FaceNet ↑ | ArcFace ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Current CFG3 | **0.454540** | 17.894556 | 0.178454 | 0.152351 | 0.784486 | **0.816156** |
| Neutral local appearance | 0.444884 | 17.904116 | 0.176645 | 0.152241 | 0.782640 | 0.805353 |
| Trained compatibility /64 updates | 0.445016 | **17.907794** | **0.176358** | **0.152193** | 0.784304 | 0.805454 |

Training improves all six batch means over its own neutral local architecture, with small gains. It **does not pass the stronger current CFG3 combined gate**: mean SSIM/ArcFace remain lower, FaceNet is slightly lower, and person 263 loses 0.014004 FaceNet/0.038068 ArcFace. Person10149 also has small LPIPS/ArcFace regressions against neutral local. All nine records have valid metrics/detections, matched supports and exact final visible RGB, but raw/finished visual review still finds incorrect gaze, iris saturation and eyebrow/lid shape. No new website mode or extra-seed replication is promoted from this result.

The worker ran 420.828 seconds (438.125 supervisor wall), at 512 training resolution, with 7.291 GB peak allocated GPU memory. Evaluation stayed 1024/512 export. **Five inference calls** comprise two exact neutral controls and three trained candidates; six control scores are reused. Eight attention/receipt tests pass. All earlier processes and current local service remain available. Each existing person folder has **Experimental - Trained compatibility 64 updates** with raw/final outputs, both controls, comparison sheets and complete values.

[Full results and every-person failures](research/MASKED_SDXL_COMPATIBILITY_PILOT_RESULTS_V2.md), [photo-free numeric/manifest evidence](research/masked_sdxl_compatibility_pilot_v2_evidence.json), [architecture/resource checks](research/MASKED_SDXL_COMPATIBILITY_FEASIBILITY_RESULTS_V1.md), [training-reference eligibility revision](research/MASKED_SDXL_REFERENCE_ELIGIBILITY_RECOVERY_V2.md).

**Follow-up completed:** the decoded gradient audit and matched 16-update comparison are reported above; the added objective fails the combined gate. Current epsilon MSE does not directly supervise decoded eye colour/gaze/brow geometry; token compatibility cannot establish anatomical correspondence or abstain from an entirely misleading fixed-strength appearance branch. These are hypotheses to test, not evidence that another loss or more GPU power will resolve the errors. The completed objective comparison kept 385 parameters, data, optimizer and evaluation fixed; retain FaceNet/ArcFace and per-person guards, and add no simultaneous capacity/gating/perceptual changes. Independent-person, corruption and multiple-seed validation remain necessary after a candidate passes.

### 10 October: SSIM-focused GPU round completed

Completed **13 GPU calls**: nine new candidates and four neutral control replays across three isolated experiments on persons 263, 6681 and 10149 at seed 17. The GPU reached 99% utilization near its rated 115 W limit; query-local appearance reached **7.394 GB peak allocated tensor memory**. No training, new weights, dataset changes or production-setting changes occurred.

| Same three people / seed 17 / erased eyes | Hole SSIM ↑ | Hole PSNR ↑ | Hole crop LPIPS ↓ | FaceNet ↑ | ArcFace ↑ |
|---|---:|---:|---:|---:|---:|
| Original CFG5 / DDIM30 | 0.428353 | 17.019171 | 0.193782 | 0.764280 | 0.787376 |
| Current CFG3 / DDIM30 | **0.454540** | 17.894556 | 0.178454 | 0.784486 | 0.816156 |
| Fixed Diffusers FreeU at CFG3 | 0.399476 | 16.219278 | 0.202284 | 0.743002 | 0.770318 |
| Hole-local reference appearance 0.2 at CFG3 | 0.444884 | 17.904116 | 0.176645 | 0.782640 | 0.805353 |
| CFG3 / DDIM50 | 0.448495 | 17.743326 | 0.184688 | 0.784870 | 0.812475 |

None passes the predefined combined gate against current CFG3, so no candidate is deployed or expanded to more seeds. Local appearance improves all six metrics versus CFG3 for person 6681 and slightly improves person 10149's SSIM, but person 263 regresses enough to lower the batch SSIM/identity. More steps improve mean FaceNet slightly while worsening SSIM, PSNR and both local LPIPS measures. All nine candidate comparisons were inspected; eye colour, gaze and brow fidelity remain unresolved. The **0.454540 seed-17** mean above must not be mixed with the preserved **0.433098 three-seed** CFG3 mean.

Four raw/finished control replays are byte-exact. The three studies retain **30 scored records** comprising nine new candidates and 21 reused control records; these are not 30 independent images or people. Every record preserves visible pixels exactly and has FaceNet/ArcFace measurements. Five real-attention tests pass in the reference runtime; source/input/reference pins and 22 pretrained SDXL asset files/aliases verify against original download hashes. [Verification](research/focused_eye_ssim_studies_v1_verification.json) and complete photo-free per-output exports retain all values.

Reports: [FreeU audit/results](research/FOCUSED_EYE_FREEU_RESULTS_V1.md), [query-local appearance](research/FOCUSED_EYE_LOCAL_APPEARANCE_RESULTS_V1.md), [50-step check](research/FOCUSED_EYE_STEPS_RESULTS_V1.md). Each existing local person folder now includes **Experimental - FreeU decoder**, **Experimental - Local appearance** and **Experimental - 50 steps**, with control/candidate PNGs and comparison sheets. These are labelled offline experiments; Start Studio retains its working methods and default-off gentler option. Current local service: **http://127.0.0.1:8765/**.

**Follow-up:** the separately audited 64-update masked SDXL pilot is complete, as reported above. It produces small neutral-control gains but fails the current CFG3 combined gate. The decoded supervision audit and matched comparison are complete as reported above; spatial correspondence and independent validation remain work in progress.

### 10 October: Start Studio freshness fix

The launcher previously reused any server on port 8765 with a valid session, so an older backend could be opened after files had been updated. The server now records a startup fingerprint of studio source; Start Studio reuses only a matching process, otherwise starting current code on an available loopback port. It preserves existing jobs and can reuse a current server on fallback ports. Eight launcher tests and all 22 HTTP boundary tests pass. A fresh server is running at **http://127.0.0.1:8765/**; live checks confirm its revision matches disk, the gentler control is served, and a second launch reuses the same process. [Verification](research/releases/2026-10-10/launcher_verification.json). This launch fix changes no model settings or experimental scores.

### 10 October: organized repository release and verification

The current GitHub release on `codex/local-studio-checkpoint` includes the completed masked-task audit, three-person reference studies, repeated guidance comparison, unsuccessful appearance/texture experiments, optional app control and the earlier spatial-rejection diagnostic. New [research](research/README.md), [script](scripts/README.md) and [review](review/README.md) indexes provide direct navigation. [Release evidence](research/releases/2026-10-10/README.md) contains photo-free per-output metrics, failures/nulls, hashes and four live website replay receipts.

Release checks pass **41 targeted tests**, parse **358 Python files**, and verify **81 completed image records** across nine experiment folders (including reused outputs). Guidance and appearance provenance checks pass; the frozen research generator hash is unchanged. These are release checks rather than new quality measurements. Optional single-image edge matching now retains the raw reconstruction with an explicit warning if its worker fails, times out or returns no receipt; five finishing tests include the actual worker and failure paths.

The launcher fix above supersedes the earlier separate port 8766 testing address. Restricted photographs, personal uploads, reference arrays, downloaded weights and environments remain local. Existing checkpoints and experimental outputs are retained. Public PPTs keep their dates; the illustrated three-person deck remains local. This release publishes current implementation/evidence while quality and validation work continues.

### 9–10 October: eye SSIM replication and appearance-conditioning investigation

Completed two isolated seed-17 tests on the same three people: lower classifier-free guidance (5 → 3) and DDIM → Euler at guidance 5. Lower guidance improves all six matched mean measures; Euler reduces mean identity and is not selected. The neutral guidance replay matches both original raw and finished images byte-for-byte. Original inputs, reference order, masks, checkpoints, evaluator and website defaults are preserved.

Then repeated the guidance-only comparison across the fixed seeds **17, 29 and 43**: **18/18 outputs and scores**, including 12 new GPU reconstructions and six hash-verified reused images. All 18 have valid FaceNet and ArcFace measurements, no visible-pixel changes, and reused metrics reproduce with zero difference. These are nine paired outputs from **three development people**, not nine independent identities or unknown-client accuracy.

| Matched eye-hole metric | CFG 5 control | CFG 3 candidate |
|---|---:|---:|
| Hole PSNR ↑ | 17.01785 | 17.82148 |
| Hole SSIM ↑ | 0.41415 | 0.43310 |
| Hole contextual LPIPS ↓ | 0.20144 | 0.19185 |
| Anatomical-eye contextual LPIPS ↓ | 0.15958 | 0.14498 |
| FaceNet cosine ↑ | 0.76920 | 0.79887 |
| ArcFace cosine ↑ | 0.76072 | 0.78508 |

SSIM improves in **8/9** pairs and PSNR in **9/9**. All six three-seed means improve for persons 263 and 6681. Person **10149** still loses **0.01362 mean FaceNet cosine**; iris colour, gaze and eyebrow discrepancies remain visible. Therefore no universal website default has been changed. SSIM and identity cosine are not correctness percentages. Supplementary anatomical-eye SSIM uses small annotated regions with limited valid windows and must not be confused with the larger eye-hole SSIM above.

Local three-person folders now include **Experimental - Eye fidelity** and **Experimental - CFG 3 repeated seeds**, with every output and comparison labelled. See [three-seed results](research/FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md), [verification](research/focused_eye_guidance_replication_v2_verification.json) and [sampler comparison](research/FOCUSED_EYE_FIDELITY_RESULTS_V1.md).

**Appearance investigation completed:** added a fixed 0.2 pretrained visual-reference branch alongside the unchanged 0.8 FaceID branch, using the same four real photos. At CFG 5, three-person seed-17 hole SSIM decreases **0.42835 → 0.42350** despite improved mean LPIPS and identity. Completing the fixed four-arm comparison at CFG 3 improves mean LPIPS further, but loses SSIM and ArcFace relative to CFG 3 alone. Neither appearance arm is deployed. Both zero-appearance raw/final replays match their controls byte-for-byte; all six candidates have valid identity measures and exact visible pixels. The initial CPU encoder dtype failure is retained with its log and runner snapshot. [Appearance results](research/FOCUSED_EYE_APPEARANCE_RESULTS_V1.md) · [four-arm results](research/FOCUSED_EYE_FACTORIAL_RESULTS_V1.md) · [verification](research/focused_eye_appearance_verification_v1.json).

**Spatial texture prototype rejected:** visible-anchor registration and fixed eye/brow texture transfer complete for all three people, but cause local misalignment, double-detail or eyelid-expression errors. Against the CFG 3 scaffold, hole SSIM decreases **0.45454 → 0.40762**, and all six mean measures worsen. Zero-transfer bitmap equivalence, exact regeneration and visible-pixel preservation pass; these checks establish implementation correctness, not improved quality. Labelled rejected outputs remain in the local sets and [full results](research/FOCUSED_EYE_TEXTURE_RESULTS_V1.md). Global recognition/appearance cues and visible-anchor fits do not establish spatial correspondence in hidden eyes.

**Optional local app comparison verified:** the updated build at **http://127.0.0.1:8766/** exposes an unchecked **Gentler eye details** option only for 1024 reference inference with opaque-cover removal. It changes CFG 5 to 3; other methods and defaults are preserved. Four real HTTP/GPU reconstructions pass raw/final byte-exact replay and exact visible-pixel checks; 22 boundary tests pass. Browser checks confirm default-off and reset/hide behavior outside the supported mode. The three actual website outputs and instructions are in **Website - Gentler details** inside each existing test set. [App verification and instructions](research/GENTLER_REFERENCE_WEBSITE_V1.md). Automatic approval review rejected stopping/restarting 8765 with "blocked by policy"; that original server remains running.

**Remaining quality work:** establish trustworthy local pose/expression correspondence and reference reliability before transferring eye/brow detail; keep a safe fallback. Iris colour, gaze and eyebrow fidelity are still unresolved, and person 10149's guidance-only identity regression remains. Further corrupted-reference checks, unseen-person validation and reserved final evaluation follow a promising fixed development result. No new training was performed in the focused continuation, no checkpoints were replaced, and no percentage accuracy or matched paper superiority is inferred from these three-person tests. The 10 October repository release includes this evidence; the local illustrated PPT retains its 9 October snapshot.

### 9 October: focused reference reconstruction on three people

Created three local ready-to-test folders for identities263, 6681, 10149, each containing four real reference photos, an erased-eye test, an erased-mouth test, masks, an evaluation-only original and labelled comparisons. Source images and checkpoints remain unchanged.

Completed the fixed SDXL512 diagnostic: **36 outputs from48 attempts**, with12 reference-detection failures retained for identity7400. The1024 comparison completed6 of8 attempts with the same reference failure. On the three jointly usable identities at seed 17, the tested **1024 + boundary blending** settings improve average hole PSNR/SSIM/LPIPS and identity over512 hard composition. Eye FaceNet rises **0.70956→0.76428**, and eye-hole crop LPIPS falls **0.45963→0.19378**. Mouth FaceNet rises **0.87103→0.91241**. These are development similarities, not accuracy percentages; gaze and expression remain imperfect.

All **three actual website eye reconstructions** completed and were byte-exact matches to the offline candidate, with visible pixels preserved. The local server is running at http://127.0.0.1:8765/. Existing global defaults have not been changed. A separate scale0.4 probe improved average metrics but reduced FaceNet for person 10149 by0.06276; it remains an experimental alternative rather than a promoted setting. Read-only integration checks found no reference-shape or conditioning-order bug.

[Focused results and test instructions](research/FOCUSED_REFERENCE_RESULTS_V1.md) · [Fixed strength probe](research/FOCUSED_REFERENCE_STRENGTH_PREFLIGHT_V1.md). Test packs are in `Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09/`.

### 8 October: actual erased-region capability diagnostic completed

Completed **96 matched ReF-LDM reconstructions**, **8 native ResShift controls** and **one byte-exact restoration replay**. The test uses four existing development identities, erased eye/mouth regions, three seeds and clean/corrupted references. All 104 composites preserve visible pixels exactly; eight inputs pass hidden-pixel invariance. Existing checkpoints and production behavior are unchanged.

The restoration baseline and unchanged v5 adapter largely retain blank regions in the inspected comparisons. Clean erased-eye hole LPIPS is **0.703150 → 0.703034**, while FaceNet is **0.519036 → 0.523102**; these small changes do not establish functioning reconstruction. With corrupted eye references, both measures worsen. Native mask-conditioned ResShift generates facial features but retains identity/gaze/expression errors. Its different resolution and reference-free setup make it a functional control, not a matched paper benchmark.

**Next:** evaluate the existing mask-conditioned reference-guided SDXL path on the frozen cases before selecting an isolated training intervention. The proposed ReF-LDM masked-loss pilot remains conditional and has not started because its functional prerequisite failed. Whole-image restoration scores must not be presented as missing-region reconstruction results.

[Full masked-task results](research/MASKED_TASK_RESULTS_V1.md) · [Protocol](research/MASKED_TASK_PREFLIGHT_V1.md) · [Audit and experiment plan](research/MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md).

### 6 October: spatial reference-rejection diagnostic completed

Completed **48 new GPU reconstructions**, **eight byte-exact neutral replays**, and **72 scored rows**, including 24 hash-verified reused baseline controls. Four mechanism tests and two paired-statistics tests pass. Inspected all eight comparison sheets. No training or production changes occurred.

The fixed intervention rejects the central quarter of the first reference's attention tokens, using the known corrupted-patch location. An equal-area seeded corner mask provides a spatial control; both are tested on four observed development identities, three inference seeds and clean/corrupted references.

- With corrupted references, paired FaceNet rises **0.90932 → 0.91303** over 11 valid pairs, but only **4/11 individual pairs improve**. LPIPS worsens **0.27769 → 0.27860**, and eye LPIPS worsens **0.12800 → 0.13333**. PSNR/SSIM increase slightly. ArcFace has 8/12 valid pairs.
- With clean references, FaceNet falls **0.91497 → 0.91022** and LPIPS worsens **0.27725 → 0.27855**. Oracle-region rejection also has worse corrupted-reference eye LPIPS than the random-corner control (**0.13333 vs 0.12623**).
- Two intervention FaceNet detections are unavailable and remain explicit. Comparison means use matched available pairs. The known region is an oracle input, not an automatic unknown-image detector; encoder receptive fields can spread corruption beyond the masked token window.

**Decision:** hard regional exclusion does not meet the combined quality requirements and is not deployed. Before training a localization gate, investigate query-local or softer attenuation with a fixed control protocol and account for useful eye information lost by rejection. Broader validation remains contingent on a promising development result. See [results and interpretation](research/SPATIAL_REFERENCE_REJECTION_RESULTS_V1.md), [fixed protocol](research/SPATIAL_REFERENCE_REJECTION_PREFLIGHT_V1.md) and [verification](research/spatial_reference_rejection_v1_verification.json).

The latest local illustrated PPT has **32 slides** with checked values and historical results kept under their original protocols. Complete review notes and 42 faculty questions are available in `review/REVIEW_NOTES_2026-10-06.md`. These research changes are local and have not been pushed in this continuation.

### 5 October: expanded encoding and reference-rejection diagnostics completed

Completed **62 GPU calls**: 30 official-input encoding runs, 24 rejection interventions and eight baseline replays. Two encoding runs also replay earlier smoke outputs. All ten replay checks are byte-exact. Scored 35 encoding/input rows and 48 rejection/control rows, with 24 of the latter reusing verified baseline metrics. Three new summary tests pass and all nine comparison sheets were inspected.

- **Encoding:** four multi-reference targets across three seeds belong to only one observed identity; a second identity supplies the single-reference equivalence control. Joint RGB encoding improves mean FaceNet **0.89963 to 0.90372** but worsens LPIPS **0.28170 to 0.28295**, SSIM and eye LPIPS. It does not replace separate encoding. See [expanded results](research/REFLDM_ENCODING_SCREEN_V2_RESULTS.md).
- **Oracle reference removal:** four observed identities and three seeds. Removing the deliberately corrupted first reference improves mean FaceNet **0.90545 to 0.91154**, but LPIPS worsens **0.27769 to 0.28373** in 11/12 pairs, and mean eye LPIPS slightly worsens. Zeroing all reference latents degrades identity further. This uses known corruption location, so it is a feasibility diagnostic rather than a deployable selector. See [rejection results](research/REFERENCE_REJECTION_PROBE_V1_RESULTS.md).
- **Measurement correction:** v5 misleading-reference FaceNet has 11 valid candidate pairs; the correctly paired baseline is **0.90932**, compared with candidate **0.90414**. The old full-baseline mean **0.90545** used 12 images. Missing detections remain disclosed.
- **Presentation:** the latest local illustrated deck has 27 slides, the restored old cover, reference photos and input/output comparisons. Slides 20–23 show rounded structural similarity percentages (SSIM × 100), not overall accuracy. The 20-slide GitHub deck remains photo-free.

The rejection summary initially failed JSON serialization after scoring. A count-type fix and regression test recovered the completed metrics without rerunning or changing them. [Verification](research/encoding_rejection_verification_20261005.json) records hashes, comparisons and checks. The local application and checkpoints remain unchanged.

**Next:** test spatial rejection of only the corrupted reference region, retaining useful reference detail, with an oracle-location feasibility arm, equal-size random-region control and neutral equivalence. Only then consider learning localization from degraded target/reference evidence. Whole-reference removal alone did not meet the combined requirements. Broader validation, independent training seeds, unfamiliar-client testing and final manuscript work remain.

### 5 October: official-input encoding diagnostic and GitHub release

Completed two matched float32 runs on one already-observed development identity using an official degraded input, five references and 100 DDIM steps. Author-style joint RGB reference encoding gives FaceNet **0.89641 vs 0.87150** for separate encoding, LPIPS **0.24070 vs 0.24114**, PSNR **22.94182 vs 22.87994**, and SSIM **0.55095 vs 0.55037**. Peak tensor memory is **7.09 vs 2.38 GiB**. This single case motivates broader encoding checks, not a deployment or paper-performance claim. ArcFace target detection is unavailable. See [diagnostic](research/REFLDM_ENCODING_CHECK_V1.md) and [measurements](research/refldm_encoding_smoke_v1.json).

The official benchmark archive is verified. The JSON history scan distinguishes metadata references from explicit completed records and does not certify the entire benchmark as fresh. The latest [public presentation](presentations/Facial_Inpainting_Results_2026-10-05.pptx) includes v4, v5 and the encoding check. Dataset photos and checkpoints remain local.

### 4 October: timestep-loss comparison completed

Completed 64 GPU updates, 24 new reconstructions and all 96 metric rows. All 72 control hashes and metrics match their originals. The only change against v4 fixed-weight training is the author's sqrt_alphas_cumprod[t] multiplier, while FaceNet and the small adapter remain local choices.

Clean-reference FaceNet improves to **0.91623 vs 0.91497** baseline, but LPIPS worsens to **0.27743 vs 0.27725** and eye LPIPS to **0.13018 vs 0.12901**. With misleading references, eye LPIPS improves to **0.12647 vs 0.12800** and LPIPS worsens to **0.27823 vs 0.27769**. FaceNet has 11/12 valid candidate detections: on the same 11 pairs, it declines to **0.90414 vs 0.90932** baseline. The previously listed baseline mean **0.90545** included all 12 outputs and was not a matched-pair estimate. Neither condition satisfies all requirements. Production remains unchanged. See [v5 results and fixed-weight comparison](research/IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md) and [verification receipt](research/identity_timestep_v5_verification.json).

Official CelebA-Test-Ref benchmark download and CRC validation are complete. Initial metadata checks found training overlap in 13 rows, development overlap in 5, and identity-label disagreements in 111. Counts may overlap. A broader history audit and preprocessing alignment remain before inference. No benchmark image pixels have been decoded. [Source audit and protocol differences](research/AUTHOR_PROTOCOL_AND_TIMESTEP_PREFLIGHT_V5.md) distinguish published settings from the released demo. No official benchmark reproduction is claimed.

### 3 October: balanced identity and gradient-cap experiment completed

Completed **two 64-update GPU runs**, with four identity-supervised updates for every one of the 16 training identities. Generated and scored **48 new reconstructions** across four development identities, three inference seeds and two reference conditions. All 72 reused controls match their original image hashes and metrics in both evaluations. No generation or scoring failures occurred. The cap reduced the maximum weighted identity/reconstruction gradient ratio from **55.21 to 1.00**.

The capped candidate improves misleading-reference FaceNet **0.90700 vs 0.90545** and ArcFace **0.83808 vs 0.83625** (8/12 valid pairs), but worsens LPIPS **0.27800 vs 0.27769** and eye LPIPS **0.12841 vs 0.12800**. With clean references, LPIPS improves **0.27707 vs 0.27725**, while FaceNet falls **0.91184 vs 0.91497**. Neither candidate passes the combined requirements; production remains unchanged. See [complete v4 comparison](research/IDENTITY_BALANCE_RESULTS_V4.md), [verification receipt](research/identity_balance_v4_provenance.json) and [per-image values](research/identity_balance_v4_metrics.csv).

**Next investigation:** evaluate how the current relative reference weights behave when every reference is unreliable, and design an explicit reference-rejection control before another training experiment. More GPU training alone has not resolved the identity/perceptual trade-off. Broader corruption tests, training-seed replication and reserved final evaluation remain pending; do not spend final-test data on candidate selection.

### 3 October: fixed face-crop supervision tested

Completed the next controlled experiment using MTCNN face crops for identity supervision, with the previous architecture, 64-update schedule and loss weight unchanged. All 16 training crops are valid; crop embedding agreement with evaluation is at least **0.99996**. Generated **24 fresh images** across four development identities, three seeds and two reference conditions, and reused **72 hash-verified controls** whose rescored metrics match exactly. The cropped candidate still loses to baseline: misleading-reference FaceNet **0.90384 vs 0.90545**, LPIPS **0.27830 vs 0.27769**, and eye crop LPIPS **0.12855 vs 0.12800**. It is not deployed. Two crop tests pass. Gradient checks expose a **0.26–56.5x** weighted identity/diffusion gradient ratio and opposing directions in 4/16 checks; periodic identity supervision covers only 12/16 training identities. **Next:** balance identity-supervision coverage and test bounded gradient contributions before any architecture or dataset expansion. See [v3 results](research/IDENTITY_CROP_RESULTS_V3.md), [measurements](research/identity_crop_v3_results.json), [CSV](research/identity_crop_v3_metrics.csv) and [provenance](research/identity_crop_v3_provenance.json). Production, reserved final identities and Human Faces work are unchanged.

### Current results, accuracy interpretation and presentation

The published **identity compatibility v2** investigation includes two 64-update GPU training runs, 96/96 reconstructed outputs and 96/96 metric rows, four development identities, three inference seeds and two reference conditions. Its table and presentation below remain a versioned snapshot; the new v3 crop ablation is recorded above. The 16 adapter-training identities are separate from these four evaluation identities. Upstream pretrained-model exposure is unknown. The deployed local baseline remains unchanged while fidelity improvement is work in progress.

| Method / reference condition | PSNR | SSIM | LPIPS | FaceNet cosine | Eye crop LPIPS |
|---|---:|---:|---:|---:|---:|
| Local ReF-LDM / clean | 23.86081 | 0.63481 | 0.27725 | 0.91497 | 0.12901 |
| Identity-aware adapter / clean | 23.84613 | 0.63458 | 0.27750 | 0.91048 | 0.12757 |
| Local ReF-LDM / misleading patch | 23.84680 | 0.63377 | 0.27769 | 0.90545 | 0.12800 |
| Identity-aware adapter / misleading patch | 23.86112 | 0.63401 | 0.27791 | 0.90433 | 0.12609 |

These are matched three-seed means, with 12 outputs per method/condition, for whole-image blur/noise restoration. Higher PSNR/SSIM/identity cosine and lower LPIPS are preferable. Eye crop LPIPS includes context around the annotated region. Eye scores improve on average, but overall LPIPS and FaceNet do not reach baseline in either condition. Earlier one-seed and missing-region studies remain separate evidence.

**Accuracy:** 96/96 jobs completed and 96/96 outputs had one FaceNet detection. These are execution/detection rates, not reconstruction correctness. FaceNet cosine is not an accuracy percentage. ArcFace has 8/12 valid target-output pairs per method/condition, with detector failures retained. FaceNet supervises the new training arm, so its evaluation is not independent of the training objective. Arbitrary external-client correctness has no verified percentage yet; independent portraits, valid references and suitable ground truth remain future evaluation.

**Presentations and release:** the full illustrated local review now has **35 slides**, including results, accuracy interpretation and future scope. The [11-slide GitHub results presentation](presentations/Facial_Inpainting_Results_2026-10-02.pptx) includes paper links and numerical comparisons without restricted dataset photos. Code, reports, CSV/JSON scores, loss traces and execution provenance were pushed in commit `fcb8426`. This project-report update follows that release; unrelated local UI and Human Faces changes are not included in that research commit.

**Next controlled experiment:** the face-crop change has now been tested. Balance identity-supervision coverage and bound gradient contributions, using matched controls before increasing training or dataset size. Independent reliability gating and broader regional/consistency objectives follow as separate ablations. Keep reserved final identities out of tuning.

Sources: [method report](research/FINAL_METHOD_REPORT.md), [all results](research/identity_compatibility_v2_results.json), [CSV](research/identity_compatibility_v2_metrics.csv), [execution provenance](research/identity_compatibility_v2_provenance.json), [audit](research/IDENTITY_COMPATIBILITY_AUDIT_V2.md). The dated entries below preserve earlier experiment history.

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
- The published **RefFaceInpainting missing-region baseline now runs locally**: all three author checkpoints strict-loaded, native inference matches the author on synthetic tensors, **16/16 new predictions and 28/28 evaluation rows** completed. Its fixed first-reference FaceNet **0.5196**, hole MAE **0.09231** and LPIPS **0.04379** are worse than matched seed 17 ResShift **0.7265**, **0.06320**, **0.02515**. All four reference choices and four visual sheets are retained; it is not promoted to the website. See [baseline results](research/REFFACE_BASELINE_RESULTS_V1.md) and [failure review](research/REFFACE_BASELINE_REVIEW_V1.md).
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
- **Training clarification:** “local dataset” does not mean all supplied images trained the current generator. Local training includes compact mask refiners, the 545-parameter intervention and 193-parameter regional/identity-aware adapters. The generative backbones and recognition encoders remain pretrained and frozen in these adapter experiments. Human Faces work is separate and set aside.
- **Release status:** latest research code/results and the photo-free results presentation were pushed as `fcb8426`. Dataset photographs and downloaded pretrained weights remain local.

### Next work in order

1. Broaden real client-photo and varied framing/reference coverage; the current main/evidence browser flows are verified. Retain native-runtime failures and monitor recurrence; passing repairs do not establish universal reconstruction accuracy.
2. Use the completed compression diagnosis to design separately versioned post-degradation validity handling with failure coverage and matched reference-count controls.
3. Use the completed regional and identity-aware training experiments to correct training/evaluation crop mismatch and inspect loss gradients. Run matched objective and reliability-gating ablations before increasing training or evaluating a larger cohort.
4. Expand pose/expression correspondence, independently varied severity/masks/reference quality and identity groups. Conduct blinded human fidelity assessment; confidence intervals from three or four identities are insufficient.
5. Once a method is stable, perform the prescribed 1/2/4-candidate comparison with an independent reporting evaluator. Then freeze the method, evaluate the reserved identities once, and prepare the manuscript for a specific conference or journal.

Publication readiness remains incomplete. No acceptance, novel-method superiority or reliable recovery of hidden facial features is claimed.

## 1. Project objective and scope

The application reconstructs a masked facial region from the observed image and a supplied mask. It supports single-image completion and a reference-assisted mode accepting one to four photographs of the same person. Whole-image restoration is a separate task and experimental protocol.

The research objective is to determine whether selecting and combining reference evidence according to the missing facial region improves identity fidelity while maintaining reconstruction quality and preserving visible content. Candidate selection using identity and quality measurements is a possible subsequent component. These proposed mechanisms are not yet demonstrated contributions.

There are two distinct bodies of work:

1. **Earlier single-image research:** inaccurate masks, compact mask refinement, LaMa/ResShift comparisons and controlled synthetic occlusion experiments.
2. **Current multi-reference research:** functional pretrained baselines plus implemented regional compatibility and identity-aware training experiments. Retrieve–Fuse–Verify remains a broader proposed architecture, not a completed validated method.

The earlier mask-refinement results are separate from the later regional-adapter experiments and cannot substitute for their evaluation.

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
| FaceID Portrait adapter | One-to-four-reference website integration | Existing pretrained identity conditioning, not a newly trained project adapter |
| ReF-LDM | Frozen local restoration backbone with cached reference attention | Matched baseline for the recent adapter experiments |
| RefFaceInpainting | Local author-model comparison, 16 predictions and 28 comparison rows | External missing-region baseline; no production promotion |
| Regional compatibility and identity supervision | 193-parameter regional module; v1 128-update arms and v2 matched 64-update arms with frozen decoded FaceNet supervision | Actual GPU training; average eye improvements with remaining overall identity/perceptual trade-offs |
| Masked SDXL token compatibility | 385 shared parameters across 70 attention processors; 64 initial updates plus two matched 16-update continuations on eight retained training people | Initial small gains over neutral local; decoded RGB trades eye/whole LPIPS for lower hole SSIM/FaceNet; all candidates retained offline |
| InsightFace buffalo_l | Face detection and normalized reference embeddings | Reference conditioning, not independent identity evaluation |
| Hard/Poisson composition | Known-pixel preservation and optional boundary harmonization | Explicit postprocessing; does not prove correct hidden features |

The generative backbones remain pretrained and frozen. Actual local training includes the 545-parameter reference-intervention pilot, the 193-parameter target-conditioned regional compatibility pilot and the 385-parameter masked SDXL compatibility pilot. These development adapters are not promoted to the website and do not establish improved identity recovery. The decoded-hole RGB gradient probe completed one disposable in-memory update without producing a checkpoint or reconstruction score. Two subsequent matched 16-update continuations produced separate checkpoints and 15 scored comparison records; the RGB candidate fails the combined gate. Separate Human Faces LoRA work is set aside and is not evidence for the existing-dataset study.

### 2.3 Completed experiments

| Study | Recorded extent | Interpretation |
|---|---|---|
| Earlier mask-development study | 96 identity labels in the development protocol; 13,824 learned-control evaluation rows | Single-image mask research |
| Object-composite test | 128 images: 64 HQ identity labels and 64 LaPa images; 5,632 rows | Already observed synthetic test; not untouched future confirmation |
| Training-budget extension | Six refiners at 6,000 updates; 27,648 development rows | Reuses 48 assessment identity labels |
| Multi-reference development diagnostic | Twelve selected identity labels, 48 generated candidates, 96 scored rows | Reference scale × denoising strength, each with hard and Poisson composition |
| Reference-count extension | 24 additional candidates; expanded total 72 candidates and 144 scored outputs across the same twelve identities | Fixed nested one/two/four-reference subsets; ten metrics and 18 paired contrasts |
| Local application checks | Real HTTP/GPU inference and browser downloads | Functional evidence on recorded examples |
| Target compatibility v1 | 96 generations over four development identities and eight reference conditions | One inference seed; regional training pilot |
| Identity compatibility v2 | 96 generations over four development identities, three inference seeds, two conditions and four methods | Matched identity-objective ablation; baseline remains deployed |

Rows are correlated experimental measurements, not independent images or independent identities. The studies use different protocols and must not be pooled into one headline performance claim.

### 2.4 Working local website

The site supports image upload, crop/fit, mask painting/erasing/undo, mask upload, model selection, reference upload, 256/512/1024 reference processing and comparison, and result/mask downloads. Ordinary reference mode accepts one to four photos; experimental selection accepts one to four and selects one. A new input clears the previous person's reference photographs on both upload pages.

Website exports are 512 × 512 for every method. ResShift operates at 256 internally and composites into the 512-pixel input to preserve visible detail. Reference models offer 256, 512 and 1024 processing; comparison defaults to a successful 512 result. LaMa's unnecessary downsampling and nonsquare uploaded-mask alignment were repaired on 14 September. The 24 September client fixes additionally preserve thin ResShift mask coverage and aspect ratio, validate ReF-LDM references, and reject unsupported thin reference marks. Large missing-feature reconstruction remains unreliable; these repairs do not establish semantic accuracy. Historical command-line/research settings remain unchanged. Images are saved locally with run metadata.

The reference runtime was repaired in a separate environment after Windows blocked native dependencies. Package pins and installation records were retained. No Windows protection was disabled. Progress-file retries prevent transient OneDrive locks from needlessly aborting reference inference.

On 14 September:

- Four-reference standard HTTP inference completed in 45.297 seconds including startup and verification, with 16.5 seconds of inference including offload. The output preserved all pixels outside the effective mask.
- LaMa passed exact, expanded and learned-mask checks; ResShift passed exact and expanded-mask checks.
- A browser-driven LaMa run displayed its output, and both result and mask downloads completed.

These timings describe individual checks, not a population latency benchmark. Functionality on these cases does not establish robustness across all user photographs.

### 2.5 Repository and hosting

Code, existing project-trained artifacts, experiment records, provenance and reproduction instructions are in the repository. The 10 October release on `codex/local-studio-checkpoint` includes the current application updates, masked SDXL studies, eye-fidelity experiments and explicit release verification. Earlier research releases remain in Git history. Restricted photographs, reference features and downloaded weights remain local. The separate Human Faces pilot code/results are retained and labelled outside the main CelebA findings.

Temporary public sharing was implemented and an HTTPS GPU test completed. The in-app browser then exposed sign-in compatibility problems. Public sharing is now stopped, its homepage link removed, and further hosting work deferred. The local app requires no sign-in. The paused hosting investigation is not a blocker for local research.

## 3. What the results currently support

**Mask refinement has a tradeoff.** At the initial training budget, preservation weighting reduced changes to visible pixels but worsened aggregate reconstruction versus the generic refiner. At 6,000 updates, weighted refinement performed better than the generic 6,000-update control on aggregate reconstruction metrics, but the generic model itself deteriorated with longer training. This does not establish that more training universally improves completion or that the refiner is state of the art.

**Reference guidance has not shown a consistent reconstruction-quality benefit in the completed diagnostic.** At strength 0.99 with Poisson composition, mean LPIPS was 0.032674 with references and 0.032460 with reference conditioning disabled. The paired difference was +0.000214, with an exploratory 95% interval spanning zero. Hole MAE was also worse with references in this setting. Other settings differ; the complete table must be retained rather than selecting the most favorable contrast.

**Independent identity diagnostic completed on 14 September:** FaceNet scored the frozen reference outputs, with 88/96 valid scores across 11/12 identities. At strength 0.99 with Poisson composition, reference conditioning increased mean cosine similarity from 0.4942 to 0.6588 (paired difference +0.1646, descriptive bootstrap 95% interval [0.0492, 0.2903]). One identity returned multiple detections in its target and all outputs and remains unscorable under the frozen protocol. This small development result does not establish final generalization, correct gaze/expression or a novel contribution. See `research/IDENTITY_DIAGNOSTIC_RESULTS_V1.md`.

**Expanded evaluation completed:** ArcFace/InsightFace, FaceNet, NIQE, BRISQUE, RGB SSIM, whole-image/hole PSNR, LPIPS and hole/visible MAE now cover 144 outputs. NIQE/BRISQUE/structural scores are valid for all 144; FaceNet has 132 valid scores and ArcFace has 140. ArcFace uses the same w600k_r50 checkpoint as conditioning and is not an independent verifier. FaceNet reproduces the prior scores exactly. All image hashes and all 24 new candidates' matched settings were verified.

**Reference-count ablation:** at strength 0.99 with Poisson composition, FaceNet means are 0.6125, 0.6482 and 0.6588 for one, two and four references. Four minus one is +0.0463, descriptive 95% interval [-0.0150, 0.1184], on eleven jointly scorable identities. This does not establish that more references reliably help. All 180 metric/contrast tests receive Holm correction; no FaceNet contrast meets corrected p < 0.05. See [expanded results](research/EXTENDED_ABLATION_RESULTS_V1.md).

The measurement baseline is substantially stronger, but the research is not yet ready to claim the proposed method improves identity preservation or is publication-ready.

## 4. Remaining work

**Reconciled on 10 October 2026 with the completed masked-task diagnostic, focused three-person reference comparisons, guidance replication, appearance/texture results, FreeU/local-appearance/50-step screens, the64-update masked SDXL pilot and application integration.** Earlier v2–v5 restoration and rejection studies remain under their original protocols. Research improvement is work in progress. The additional Human Faces dataset work remains separate.

| Area | Current evidence and remaining work | Status |
|---|---|---|
| Identity/quality measurement | FaceNet target/gallery, diagnostic ArcFace, LPIPS, SSIM, PSNR, hole/visible MAE, NIQE and BRISQUE executed; final evaluation remains sealed | Development implemented |
| Distortion/preservation | Six kinds, strength/scale screen and fresh development extension completed; unchanged-input controls still expose fidelity loss | Diagnostic complete; method unresolved |
| Reference robustness | Compression failure diagnosed; studio ReF-LDM now validates one face per reference and frames rectangular uploads without distortion. Broader poor-quality, pose and expression handling still needs matched validation with failures retained | Basic client checks implemented; broader validation pending |
| Local correspondence | Five-point similarity and eight regional features implemented; 22/24 initial cases detected; 3D pose/expression/occlusion unresolved | Prototype complete |
| Local fusion | Six-policy, two-seed 48-row comparison complete; no multi-metric advantage established | Negative/mixed result |
| Learned adapter | Earlier v2–v5 restoration studies remain preserved. Masked SDXL has a preserved 64-update pilot and two matched 16-update continuations. The decoded gradient/resource audit passes, but RGB lowers hole SSIM/FaceNet versus equal epsilon despite eye/whole LPIPS gains. All 15 comparison records and negative visual decisions remain. Audit spatial representation and correspondence before selecting one further intervention | Real masked training implemented; consistent improvement over current control remains |
| Confidence UI | Prior generation/download checks passed. Current update fixes stale state, upload races, evidence suggestions/transparency, and missing-area strength selection; new HTTP/GPU checks pass. Current main/evidence browser flows are verified with real generated images, matching references, downloads and state invalidation | Implemented and browser verified |
| Runtime reliability | Original native/OpenCV failures retained; fixed-thread studio retry passed. Two wrapper failures from an unnecessary OpenCV import were fixed and both recovery runs passed. Continue recurrence checks; no universal root-cause claim | Mitigations and recoveries verified |
| Visual assessment | Prior reviewed studies retained; all four new studio audit sheets reviewed, with coloured holes, retained erasures and facial errors documented. Blinded human evaluation remains | Internal review complete for these runs; human study pending |
| Candidate selection | Matched 1/2/4 candidate pools, random/identity/quality/combined selection and separate evaluator | Deferred until stable restoration |
| External comparisons | ReF-LDM completed 24/24 generations and 48/48 native/composed evaluations; new RefFaceInpainting completed 16/16 predictions and 28/28 comparison rows, but fails these missing-feature fidelity controls. Additional relevant comparisons and broader validation remain; paused OSOR results remain delimited | Two external author baselines executed; broad superiority unsupported |
| Reference encoding | Thirty official-input generations and 35 scored rows complete; one multi-reference identity plus a second single-reference control. Joint encoding trades identity gains for worse perceptual/eye quality, so production is unchanged. Broader author-protocol alignment remains | Diagnostic complete; no default change |
| Reference rejection | Whole-reference diagnostic: 24 interventions and eight exact replays. Spatial diagnostic: 48 interventions, eight exact replays, 72 scored/control rows and eight inspected sheets. Hard spatial rejection worsens LPIPS and eye quality despite higher mean corrupted-reference FaceNet. No learned gate or website change | Both diagnostics complete; improved attenuation mechanism pending |
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
| Optional metrics | A limited 106-landmark diagnostic is recorded in the structure study. Broader landmark evaluation, FID at adequate sample size and ROC/TAR with adequate verification trials remain | Limited landmark diagnostic complete; broader scope pending |
| Final evaluation | Require method/parameters/evaluator freeze, independent confirmation, ablations and baseline comparison first. No reserved model generation/evaluation; historical automated duplicate/hash preparation did decode files, disclosed in the history audit | 0/8 model evaluations; no new reserved access |
| Manuscript and presentation | Public photo-free deck retains the 5 October snapshot; the illustrated 32-slide review remains local. A separate local 20-slide small-batch deck dated 9 October includes architecture, methods and three-person outputs. Later guidance/appearance/texture results are in dated reports and the release tables. Venue choice, manuscript and broader validation remain | Dated presentations retained; manuscript and final validation in progress |
| GitHub release | The 10 October release contains current app fixes, spatial and masked research, all focused pilot outcomes, review notes, navigation indexes and photo-free metric/hash evidence. Existing private/restricted image and model assets remain local | Organized current code/documentation/evidence release; scientific work remains open |
| Public hosting | Continue loopback-only use on the laptop | Paused by user |

No single overall project-completion percentage is assigned because successful research findings are not predictable implementation tasks. The explicit measured percentages above separate finished experiments from unsolved quality and publication requirements.

## 5. Planned execution and acceptance criteria

### Phase 1 — establish the measurement baseline

Completed for the expanded development set: specified identity/quality evaluators, model/source hashes, preprocessing, detector-failure handling and paired summaries. The latest v2 experiment adds three inference seeds and 96 rows; multiple training seeds, broader masks and independent evaluation remain. FaceNet now supervises the identity-aware arm, so evaluator independence must be stated separately for each study. The application remains unchanged by these research runs.

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

Regional reference features, deterministic fusion and compact learned-adapter investigations are implemented. The regional v1 pilot and v2–v5 identity objectives concern whole-image ReF-LDM restoration and retain quality trade-offs. Whole-reference and hard spatial rejection diagnostics are complete and failed the combined requirements. Actual erased-region tests now use mask-conditioned SDXL. Guidance, global appearance, registered texture, FreeU, query-local appearance and denoising-budget checks are recorded under their own protocols. The shared 385-parameter masked SDXL compatibility module now completed 64 GPU updates on eight eligible existing training people, followed by three matched candidate outputs and two exact initial replays. It improves the six neutral local means slightly, while failing the stronger current CFG3 SSIM/identity and visual gates. The decoded gradient/resource audit and two matched 16-update continuations are complete; added RGB loss still fails the combined gate. Next audit spatial representation/reference correspondence before choosing one further change. Declare each training loss/ablation separately, preserve all controls and complete regional/global and consistency comparisons before independent expansion. Preserve the working pretrained baseline and all earlier results.

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
