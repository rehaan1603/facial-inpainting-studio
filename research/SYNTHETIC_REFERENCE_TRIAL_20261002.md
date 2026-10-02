# Synthetic references from damaged portraits: completed diagnostic

Date: 2 October 2026. Four existing development photos, one seed. No training, final-test use or novelty claim.

## What was created

The built-in image-generation tool received only each portrait's `central_missing/input.png`, with the eye-region rectangle already erased. Three separate requests per photo asked for frontal, approximately 20-degree left, and approximately 20-degree right views. **11 of 12 requests produced images.** Photo 01's third view was blocked by the image tool's output safety system and was not retried or replaced. Photo 01 therefore has two references; Photos 02–04 each have three. All successful outputs were retained without target-based selection.

These are invented portraits conditioned on surviving pixels, not independently observed photographs of the person. Some requested left/right angles were not faithfully followed. Exact synthesis model revision and seed are not exposed by the tool, so the synthesis stage is not fully reproducible from prompts alone. Generated pixels, input hashes and prompts are retained locally. No clean target was supplied to synthesis. The operators had previously seen these development originals; this is not a blinded or held-out study. Dataset/pretraining overlap is unresolved.

Local delivery: `Downloads/Human dataset/Ready to test - Human Faces v1/Synthetic references - experimental 2026-10-02`. Project assets and exact prompts: `outputs/synthetic_references_v1/asset_sources.json` and `references/`. Photos remain excluded from GitHub.

Prompt template: one photorealistic head-and-shoulders view using the supplied damaged image as the only reference; preserve visible hair/lower face/clothing; imagine plausible occluded anatomy; request one of the three fixed angles; no occluder, glasses, montage or text. Exact prompts, including failures, are in the delivered `PROMPTS AND GENERATION RECORD.json`.

## Matched reconstruction

Completed **12/12 local SDXL + FaceID Portrait generations**:

- `reference_off`: available synthetic embeddings are computed but adapter scale is zero, eliminating their attention contribution.
- `first_synthetic`: only the fixed frontal-request output, adapter scale 0.8.
- `all_synthetic`: all available generated views, adapter scale 0.8.

All arms use the same pretrained pipeline, seed 17, 30 DDIM steps, strength 1, guidance 5, 512 resolution, masked RGB neutralized to 127, and the existing Poisson compositor. The reference-off arm retains the same reference count as the all-view arm. Clean targets are accessed only by the separate evaluator. Historical LaMa and edge-matched ResShift outputs are contextual baselines; comparing different generators cannot isolate the value of synthetic references.

The earlier plan also contemplated repeated-input/reference-count controls and genuine-reference arms. Those are not implemented here. This bounded trial uses a zero-scale control and one-versus-all generated references; there are no verified genuine reference groups in this dataset.

## Measured results

All **24 image rows** were scored: twelve new reconstructions plus four damaged inputs, four saved LaMa outputs and four saved ResShift outputs. No evaluation rows were dropped. FaceNet fails detection on all four masked inputs and two LaMa outputs; all three SDXL arms and ResShift have four valid comparisons. ArcFace is also the conditioning encoder family, so its metric is not independent. FaceNet provides a separate identity measure.

| Method | Hole MAE, lower | Eye-region LPIPS, lower | FaceNet cosine, higher |
|---|---:|---:|---:|
| ResShift, edge matched | 0.10511 | 0.33706 | 0.41311 |
| SDXL, reference off | **0.08263** | 0.28210 | 0.50294 |
| SDXL, first synthetic | 0.09209 | **0.26724** | 0.56137 |
| SDXL, all synthetic | 0.09361 | 0.27560 | **0.56988** |

Each table entry has n=4. Eye-region LPIPS uses the prespecified missing rectangle, not a target-selected crop. MAE uses 0–1 pixels. Cosines are not accuracy percentages. NIQE, BRISQUE, full-image LPIPS/SSIM/PSNR, detection status and common-support paired changes are retained in [the receipt](synthetic_reference_trial_v1.json).

Compared with ResShift, all synthetic references improve eye-region LPIPS in 4/4 and hole MAE in 3/4, but FaceNet in only 2/4. Compared with **the same SDXL model with reference influence off**, all references worsen hole MAE in **4/4**, and improve eye-region LPIPS and FaceNet in only **2/4**. All views improve FaceNet over the first view in 3/4 but worsen eye-region LPIPS in 3/4. More views are not a consistent quality improvement.

## Visual findings and decision

All four comparison sheets, including enlarged missing regions, were inspected:

- Photo 01: synthetic conditioning produces more defined eyes than ResShift/reference-off, but changes eyebrow shape, iris appearance and gaze.
- Photo 02: all SDXL arms generate lowered/closed eyes rather than the actual visible-eye expression. Synthetic views do not solve this.
- Photo 03: synthetic conditioning strengthens the eyebrow/eye detail but changes expression, iris appearance and gaze. Single-view conditioning is perceptually better than all views here.
- Photo 04: the ResShift glasses-like artifact disappears in SDXL, including with references off. Synthetic references exaggerate eyes and worsen resemblance compared with the no-reference control.

**Do not promote synthetic references as the default or claim accurate missing-eye recovery.** The change of backbone accounts for part of the improvement over ResShift; generated views do not provide missing factual observations. These four low-resolution development examples cannot establish cross-dataset generalization, statistical superiority or novelty. No checkpoint or research model was updated.

## Working local delivery

The main-page reference mode now accepts **1–4 photos**, matching the already supported model interface. Experimental best-reference selection remains separate; blur/noise ReF-LDM still requires three to four at the API boundary. Twenty-one HTTP checks pass, including one/two-photo dispatch and invalid counts; main-page JavaScript syntax passes.

A real two-reference website run completed and produced a PNG **byte-identical** to Photo 01's `all_synthetic` trial output. This additional functional reproduction is separate from the 12 research generations. The studio is running locally. Delivered instructions specify exact-mask, 512, ignore opaque cover colour and edge matching to reproduce the tested path. Each Photo folder contains input/mask, synthetic references, all three outputs and a comparison sheet.

The evaluator's first invocation stopped at its completeness guard while generation was still running; it accessed no targets before that guard. A later invocation scored all 24 rows successfully. Both logs remain saved. No failed synthesis or model attempt was silently substituted.
