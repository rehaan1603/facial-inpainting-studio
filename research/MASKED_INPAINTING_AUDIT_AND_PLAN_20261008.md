# Masked facial inpainting: audit and experiment plan

8 October 2026. Audit only: no training, checkpoint replacement, dataset change or production-code change in this review.

## Main finding

The recent ReF-LDM compatibility experiments optimize **whole-image blur/noise restoration**, not reconstruction of an erased facial region. Their preparer explicitly labels this distinction. The evaluator passes an all-ones mask. The 193-parameter adapter receives pooled target/reference descriptors but no actual hole mask. Its diffusion loss emphasizes a fixed face rectangle, not the requested missing region.

This is the strongest directly evidenced mismatch with the current objective. It is not proof that correcting it alone will recover accurate eyes. Frozen generator capacity, reference alignment, global identity conditioning and training/inference mismatch also limit reconstruction.

**Recommendation: first run a no-training masked-task baseline and information-leakage diagnostic.** Then, if its input and evaluation contracts pass, compare the current diffusion objective against a hole-weighted objective on exactly the same erased-target training stream. Do not enlarge the adapter or add another identity loss simultaneously.

## Architecture and information flow

| Path | Frozen / locally trained | Reference and mask flow | Limitation |
|---|---|---|---|
| ReF-LDM research compatibility | Frozen pretrained generator and first-stage encoder/decoder; **193** trainable adapter parameters | Target/reference latent descriptors produce five regional reference priors; installed in 16 attention layers. References separately encoded in these studies. | No explicit hole-mask input; regionwise reference bias cannot reconstruct a new spatial correspondence. |
| ReF-LDM website restoration | Pretrained inference; no local optimization in runner | Joint RGB reference conditioning; mask used in final compositing, not learned conditioning. | Must not be described as an explicitly mask-conditioned generator. Research and website reference encoding differ. |
| Reference-guided SDXL inpainting | Pretrained SDXL and FaceID adapter; no local training in this runner | One normalized InsightFace embedding per reference, projected to tokens. Image and mask supplied to inpainting pipeline. Cached UNet config has **9 input channels**. | Identity embeddings do not retain all reference spatial detail, expression or gaze. |
| ResShift face inpainting | Frozen generator and VQ model in original backend | Erased RGB and mask supplied through `lq` and `mask`; native 256 resolution, four diffusion steps. | Original backend has no reference conditioning; native resolution limits detail. |
| LaMa | Frozen pretrained model | RGB and binary mask; generated region composited with visible input | General inpainting prior does not guarantee facial identity or eye geometry. |
| MaskRefiner | **119,057** trainable parameters in historical training | RGB plus supplied mask; predicts an improved mask for a frozen generator | Trains mask estimation, not facial texture synthesis. |

The current compatibility network is Linear(10,16), SiLU, Linear(16,1): 176 + 17 = **193 registered trainable parameters**, independently counted from the instantiated class during this audit. Earlier reference-only bias used 545 parameters. Frozen backbone parameter counts are not represented as locally trained parameters.

Compatibility features include local/global cosine, reference consensus, norm discrepancy and six diagnostics. Five fixed boxes approximate eyes, nose, mouth and whole face. Relative log-softmax priors redistribute reference preference; they are not an independent all-references rejection gate. Every key belonging to a reference receives the same bias for a given target region. There is no learned keywise alignment. The shared final scalar bias cancels in the reference log-softmax, so registered count is not a count of independent effective degrees of freedom.

The studio missing-area wrapper replaces masked RGB with 127 before reference generation and uses full denoising, then composites against the actual observation. Its 256 option uses a separate engine; it is not the same execution path as the 512/1024 runner. Preserve these distinctions in comparisons.

Source: [compatibility](../scripts/target_region_compatibility_v1.py), [research core](../scripts/reference_intervention_core.py), [restoration runner](../scripts/refldm_restore.py), [SDXL reference runner](../scripts/reference_inpaint.py), [studio wrapper](../scripts/studio_reference_inpaint.py), [ResShift](../scripts/resshift_adapter.py), [mask refiner](../scripts/refiner.py).

## Loss and training audit

Current diffusion objective is `0.25 * mean(epsilon_error²) + 0.75 * mean(epsilon_error²[:,:,10:58,12:52])`. The second term is a fixed latent face ROI, **not a per-example occlusion mask**. There is no explicit decoded masked RGB reconstruction, masked perceptual or boundary loss in this objective.

Identity variants add frozen FaceNet cosine loss on a decoded one-step predicted clean image. V2 resizes the entire image to 160; v3 uses fixed target-detected face crops. V4 supervises every update with fixed or gradient-capped weight; v5 scales the fixed weight by the diffusion signal coefficient. These are separate experiments. V5 is not the combination of all earlier changes. FaceNet is consequently a supervised metric in these variants; ArcFace is separate from that loss. SDXL FaceID uses ArcFace-family embeddings for conditioning, so independence must be described per backend.

The v2–v5 pilots use 16 training identities, 64 updates per fit, batch one, learning rate 0.001, AdamW, gradient clipping at 1, training seed 20261002. Clean/blur/noise/wrong-patch reference conditions are balanced. Training timesteps are 100/200/300/400. Four evaluation identities and three inference seeds do not constitute multiple training seeds. Short adapter fits and one-step training versus 50-step sampling are plausible additional bottlenecks, not established explanations for every error.

Historical MaskRefiner training used region-balanced binary cross-entropy with visible-region penalty, not an identity-generating loss. Six runs were extended from 1,500 to 6,000 updates each. More updates did not consistently improve reconstructed faces.

Sources: [v2 trainer](../scripts/train_identity_compatibility_v2.py), [v4 report](IDENTITY_BALANCE_RESULTS_V4.md), [v5 report](IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md), [training extension](TRAINING_EXTENSION_RESULTS.md).

## Matched baseline and reproduction status

Recent restoration protocol: four development identities (263, 6681, 7400, 10149), seeds 17/29/43, 512 pixels, four references encoded separately, DDIM 50, guidance 1.5, bfloat16, trailing uniform schedule. Targets are downsampled 512→64→512 and given Gaussian noise; wrong-patch condition replaces the central 256-square region of the first reference with another identity. These are restoration conditions, not true holes. Repeated development identities cannot establish generalization to arbitrary photographs or exclusion from upstream pretraining.

I independently recomputed these means from the 12 clean baseline rows in the v5 results JSON during this audit:

| Metric | Baseline | Valid rows |
|---|---:|---:|
| PSNR | 23.8608088739 | 12 |
| SSIM | 0.6348084536 | 12 |
| LPIPS | 0.2772496069 | 12 |
| FaceNet cosine | 0.9149657587 | 12 |
| ArcFace cosine | 0.8396019861 | 8 |
| Eye crop LPIPS | 0.1290107413 | 12 |

The historical metric replay receipt contains 15/15 passing rows within 1e-6, including missing-detection checks. The October 6 diagnostic records eight byte-exact neutral inference replays. **This audit rechecked the saved aggregates and receipts; it did not rerun full inference or recompute embeddings from every image.** Full new replay remains a prerequisite of the proposed masked experiment. The earlier single-seed 24.04671 PSNR baseline is a different aggregate and must not replace the three-seed baseline selectively.

## Metric audit

- PSNR uses normalized RGB squared error and `-10 log10(MSE)`. Whole and masked values must be reported separately. Perfect zero error has an explicit infinite-status marker, not a fabricated finite score.
- SSIM uses Gaussian weighting, sigma 1.5, an 11-pixel window and data range 1. Regional metrics average only fully contained windows after region erosion. Regional whole-image map averages and another library's scalar SSIM must not be mixed silently.
- LPIPS uses AlexNet, RGB scaled to [-1,1]. Full-image LPIPS can dilute hole errors with unchanged visible pixels. Region LPIPS is a bounding-box crop with 10-pixel context and no resizing; it is **not pixel-masked LPIPS**. Tiny crops return unavailable.
- FaceNet uses MTCNN face detection, 160-pixel standardized crops, normalized VGGFace2 embeddings and cosine similarity. ArcFace uses InsightFace buffalo_l, 640 detection size and 0.5 threshold. Detection failures remain missing, with coverage reported; they are not zeros and are not silently excluded from one side only.
- Eye components come from evaluation annotations and are intersected with the damaged mask. On restoration's all-ones mask this becomes the entire eye region. It is not an erased-eye result.
- Pair candidate and control on identical valid rows; report detector failures and paired denominators. Cluster uncertainty by identity, keeping its seeds together. Twelve outputs from four identities are not twelve independent identities.
- No cosine score is a calibrated reconstruction “accuracy percentage.” NIQE/BRISQUE are supplementary naturalness diagnostics, not proof of identity recovery.

Sources: [base metrics](../scripts/extended_evaluation_metrics.py), [regional metrics](../scripts/region_metrics_v1.py), [paired scoring](../scripts/score_identity_compatibility_v2.py).

## Previous changes: what helped and hurt

| Change | Helpful observation | Cost / limitation |
|---|---|---|
| V4 gradient-capped identity | Clean LPIPS 0.277072 versus 0.277250; PSNR 23.865872 versus 23.860809 | Clean FaceNet 0.911838 versus 0.914966; SSIM also lower |
| V5 timestep-scaled identity | Clean FaceNet 0.916227 and SSIM 0.635006 | Clean LPIPS 0.277431 and eye LPIPS 0.130181 worsen; corrupted-reference results remain mixed |
| Hard spatial reference rejection | Wrong-patch FaceNet 0.909319→0.913034 on 11 matched pairs | Only 4 pairs improve, 7 worsen; LPIPS and eye LPIPS worsen. Clean identity also worsens |
| Mask-aware reference selection, actual inpainting | Better mean identity than some single-reference choices | FaceNet 0.8113 versus all-reference 0.8180; did not beat simpler all-reference control |
| Regional reference routing | Slight hole-MAE improvement, 0.0899 versus 0.0900 | FaceNet 0.8137 versus concatenation 0.8180 |
| Longer mask-refiner training | Some visible-pixel preservation gains; preservation-weighted ResShift hole MAE improves | Generic ResShift hole MAE 0.089615→0.158915; more training alone is not a solution |

These rows span different protocols and must not be pooled into a leaderboard. Sources: reports linked above, [spatial rejection](SPATIAL_REFERENCE_REJECTION_RESULTS_V1.md), [selection](GENERALIZATION_RESULTS.md), [routing](REGIONAL_ROUTING_RESULTS_V2.md).

## Ranked experiments — one change at a time

All proposed new files below are **planned, not implemented**. Use new output directories with fail-if-exists behavior and hash source checkpoints/manifests. Preserve current datasets and identity splits. Do not select masks or remove difficult cases after looking at results.

### 1. Masked-task contract and baseline diagnostic — perform first

Hypothesis: a substantial part of apparent progress on restoration does not transfer to genuinely erased eyes/mouth; input leakage or task mismatch must be ruled out before training.

Files: new `scripts/audit_masked_task_v1.py` and `configs/masked_task_v1.json`; call existing backend and metric functions without modifying them. Freeze a development-only subset from existing masked manifests before inference, retaining eye and mouth cases and their original masks. Keep original restoration protocol intact as a separate task.

Changes: no learned parameters. Compare unchanged baseline and current compatibility adapter under identical erased input, references, seed, sampling and compositing. Include native mask-conditioned SDXL/ResShift as separately labeled controls, not direct reproductions of ReF-LDM. Primary ReF-LDM contrast must retain its own encoding method. Check that changing RGB beneath the same mask cannot change the effective generator input. Never give clean target pixels or generated target-derived references to the inference path.

Budget: proposed four existing development identities × two existing mask locations × three seeds × clean/wrong-patch = 48 outputs per reference-conditioned arm. Baseline and adapter require 96 outputs; add a small native-backend smoke screen first. GPU inference only; measure seconds/image and peak VRAM before estimating full duration. Checkpoint: none, immutable existing checkpoints. Outputs: `outputs/masked_task_v1/`.

Success: mask/input contract and replay checks pass; raw and composited images are saved; visible pixels remain exact for hard composition; metric coverage and matched denominators are explicit. This experiment establishes a valid baseline, not a guaranteed quality gain. If ReF-LDM cannot handle erased input acceptably, stop adapter-loss expansion and report the native mask-conditioned backbone comparison before choosing an architecture change.

### 2. Explicit masked-region diffusion reconstruction objective — first training experiment after step 1

Hypothesis: allocating reconstruction gradients to actual missing pixels is more useful than weighting a fixed face rectangle.

Files: new `scripts/train_masked_compatibility_v1.py`, new config and scorer wrapper; reuse `target_region_compatibility_v1.py` unchanged. Both control and candidate train on the **same** erased-target stream and same neutral initial 193-parameter state. This shared task-alignment setup is separate from the loss contrast; also retain the untouched pretrained baseline.

Single changed variable: replace the fixed ROI term with normalized hole-weighted epsilon error, using area-downsampled masks at latent resolution: `0.25*global_MSE + 0.75*sum(mask*error²)/(channels*sum(mask))`. Reject empty masks. This is latent reconstruction supervision, not a claim of decoded RGB loss. Keep the architecture, optimizer, 64-update pilot budget, training identities and timestep schedule equal. Do not add identity/perceptual loss in this first contrast. A decoded RGB loss is a separate follow-up if latent weighting is ineffective.

Checkpoint: `outputs/masked_compatibility_v1/fixed_roi.pt` and `masked_roi.pt`. Cost: two 64-update fits with frozen backbone; broadly comparable to existing diffusion-only pilot training, but measured timing is required. Full 50-step evaluation will dominate. Evaluate both against the unchanged baseline and each other on the frozen step-1 protocol. No identity-count expansion or budget increase within this contrast.

Success: lower hole crop LPIPS and no worsening of hole PSNR/SSIM, FaceNet, ArcFace or damaged-eye LPIPS, separately for clean and corrupted references; visible pixels unchanged. Keep whole-image metrics too. Numerical 1e-6 tolerance is a reproducibility check, not a clinically or perceptually meaningful margin. Do not introduce a looser identity margin after results. Repeat a promising result with additional training seeds before broader claims.

### 3. Add masked reconstruction plus identity supervision

Hypothesis: identity regularization can constrain a task-aligned reconstruction objective more effectively than it did on the restoration pilot. Files: a new trainer derived from step 2; reuse existing fixed target-crop FaceNet implementation. Single variable: add its frozen-encoder cosine term; retain prior weight-calibration rule without post-hoc searching. Keep 193 parameters, masked objective, data and schedule fixed. Checkpoint: `outputs/masked_identity_v1/adapter.pt`. Cost: decoded-image backpropagation and recognition increase memory/time; use existing checkpointed decoder approach. Require paired gains without eye or perceptual regression, including independent ArcFace coverage. FaceNet-only improvement is insufficient.

### 4. Increase adapter capacity

Hypothesis: after task alignment, the small compatibility scorer underfits reliability patterns. Files: new compatibility class and matched trainer config. Only change hidden width 16→64: **769** registered parameters (640+64+64+1). Keep features, layer placement, losses, data and updates fixed; neutral final-layer initialization. Checkpoint: `outputs/masked_capacity_v1/adapter.pt`. Cost: negligible extra adapter memory relative to backbone; similar inference budget. Require the same joint metric gate. Larger scalar gates still do not learn tokenwise correspondence; failure should not trigger arbitrary width sweeps.

### 5. Damaged-region perceptual supervision

Hypothesis: pixel/epsilon supervision alone does not constrain eye texture and structure. Files: new loss helper and one new trainer configuration derived from the selected preceding control. Add only a frozen LPIPS term on fixed mask-derived crops, identical predicted/target bounds; do not select crops using generated quality. Checkpoint: `outputs/masked_perceptual_v1/adapter.pt`. Cost: decoder and LPIPS backpropagation; benchmark memory first. Fix its weight from a preregistered training-only gradient calibration. Require identity and regional structural metrics to remain intact; full-image LPIPS alone is not success.

### 6. Boundary consistency

Hypothesis: the generator and visible image disagree at mask boundaries. Files: one new boundary-loss helper and config; add only an inner-boundary RGB/gradient term using a fixed mask-derived band. Checkpoint: `outputs/masked_boundary_v1/adapter.pt`. Cost: low incremental loss cost if decoding is already active. Evaluate seam error, hole metrics and identity with hard compositing fixed. Do not change Poisson blending at the same time. This may improve seams without recovering missing anatomy.

### 7. Reference reliability / soft rejection

Hypothesis: selective soft attenuation preserves useful reference detail better than global hard exclusion. Files: new research attention intervention using existing spatial-rejection harness, no production change. First test a single fixed attenuation versus neutral and equal-area controls, keeping query scope fixed. No training checkpoint for that diagnostic. Cost: additional matched inference arms. Only after improvement without clean-reference harm should a learned reliability model be trained. Oracle corruption locations are diagnostic controls, never inputs to a claimed automatic client-image method.

## Publication and completion boundary

The code audit and saved-number reproduction are complete at the scope described above. A fresh end-to-end replay and the new masked-task contract screen remain to be executed; therefore large training should not start yet. No candidate has been promoted by this report.

After a successful development experiment, repeat multiple training seeds, evaluate broader predeclared corruption severity and mask sizes, then run reserved identities once after choices are frozen. Audit original/reference duplication and synthetic-reference leakage. Report raw and composited outputs, detection failures, identity-cluster intervals, runtime, memory, ablations and negative results. Hardware limits constrain experiment scale; the inspected evidence does not establish GPU or RAM as the cause of inaccurate eyes.
