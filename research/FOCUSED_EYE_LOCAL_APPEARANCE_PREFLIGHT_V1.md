# Hole-localized reference appearance: frozen preflight

10 October 2026. One isolated inference intervention on the existing three-person erased-eye batch. Preserve current working methods, all checkpoints, original controls, datasets and experiment records. No training, new weights, changed references, parameter search or website default change. Freeze this document and candidate configuration before generation.

## Evidence and hypothesis

The existing CFG 3 reference control improved nine-pair mean hole SSIM to 0.433098 from CFG 5's 0.414146, but gaze/iris/brow fidelity remains incorrect and person 10149's mean FaceNet decreased. Whole-image SSIM around 0.96 mainly reflects exactly preserved pixels. Anatomical-eye SSIM is much lower; the existing eye-window counts and metric limitations in [the preceding audit](FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md) remain applicable.

Adding the fixed pretrained Plus Face appearance branch globally at scale 0.2 and CFG 3 improved mean contextual LPIPS on seed 17, but reduced hole SSIM from 0.454540 to 0.441567 and ArcFace from 0.816156 to 0.810635. Visible-anchor RGB eye/brow transfer also failed, producing local geometric errors. These negative experiments are retained. The fixed FreeU screen has now failed on all three seed-17 cases: hole SSIM 0.454540 to 0.399476, with FaceNet and ArcFace worse for every person. Do not replicate FreeU or combine it with this experiment.

**Hypothesis:** restricting the existing appearance-attention contribution to queries inside the supplied reconstruction hole may reduce unwanted global appearance/context interference while retaining useful reference detail in the damaged region. This is an output-attention localization test, not a new encoder, RGB texture paste or learned reference selector. It does not establish local target-reference correspondence.

The primary comparison isolates localization: **global appearance 0.2 versus hole-localized appearance 0.2**, both at CFG 3. The stronger CFG 3 **appearance OFF** control remains a required second comparison. Retain the original CFG 5 control for identity context; a gain against a weaker intervening baseline alone is insufficient.

## Fixed architecture and attention contract

Use the existing nine-channel SDXL inpainting backbone and all four real references in unchanged order. Keep FaceID Portrait conditioning globally active at scale **0.8**. Keep the same pretrained Plus Face visual-reference projection, saved CLIP penultimate hidden-state feature banks, unconditional features, reference tokens and appearance scale **0.2** used in [the previous factorial comparison](FOCUSED_EYE_FACTORIAL_RESULTS_V1.md). Frozen inference components and zero locally optimized parameters remain unchanged. No backward pass, optimizer, new model download or training checkpoint.

The four references must continue to participate in the **original joint multi-reference attention**, with the same key/value concatenation and softmax normalization. The built-in per-image IP-Adapter masking path must not be substituted: it computes reference-image contributions separately and changes the original joint normalization. Such a change would confound the localization comparison.

Change only the appearance branch's already-computed attention contribution, conceptually:

```text
unchanged text attention
  + 0.8 * unchanged global FaceID attention
  + 0.2 * hole_query_mask * unchanged joint-reference appearance attention
```

Implementation delegates the unchanged processor twice: the full text/FaceID/appearance output, and the same output with only appearance scale set to zero. It then uses `torch.where(hole_query_mask, full_output, appearance_off_output)` after the unchanged per-query output projection/residual. The projection does not mix query positions; this preserves the original joint-reference computation and uses exact branch outputs rather than fp16 subtraction. Full-mask and appearance-zero fast paths directly return the original processor output. FaceID, text attention, keys, values, scaling, original attention calculation and common projection remain unchanged. Do not mask RGB references, replace their crops, select reference tokens, change softmax, or transfer reference pixels.

The query mask is derived **only from the supplied binary hole mask**, never a clean original, semantic evaluation annotation, generated-eye detector or reference-corruption oracle. Resize the 512-pixel effective binary hole by nearest neighbor to each square attention-query grid and broadcast over attention batches/channels. No dilation, feather, per-reference masks, anatomical-eye submask or extra scaling is added. Record query sizes and mask coverage so spatial ordering can be audited. Reject an unrecognized non-square query layout rather than inventing geometry.

A full-ones query mask must reproduce global appearance output exactly. Appearance scale zero must reproduce the original CFG 3 appearance-OFF output exactly. These neutral contracts prevent a hidden attention-normalization or ordering change from being mistaken for a quality improvement.

## Inputs and generation settings remain unchanged

- Identities: **263, 6681, 10149**; their existing erased-eye input and exact masks in `outputs/masked_task_v1/cases.json`.
- Same four real clean reference photographs per identity, identical order and hashes; same saved appearance feature banks. The evaluation target must never become a reference.
- CFG **3**, DDIM **30** steps, FaceID **0.8**, appearance **0.2**, strength **1**, inference **1024**, output **512**, same prompts, precision, existing model CPU offload, VAE tiling and **Poisson** finish.
- Seed **17** for the initial screen. Seeds **29 and 43** are conditional replication only, with every identity included and the same frozen factors.
- FreeU **disabled**. No scheduler, reference feature, face alignment, preprocessing, mask or compositing change accompanies the attention localization.

Generation may open damaged inputs, supplied masks, references, saved appearance feature banks and existing inference outputs. Clean originals and evaluation component masks may only be opened by the frozen evaluator and after-generation diagnostic comparison, never candidate inference or automatic selection. Do not optimize localization against hidden target RGB.

## Execution and cost

Use a new fail-if-exists output directory `outputs/focused_eye_local_appearance_v1/`. Suggested isolated files are `scripts/run_focused_eye_local_appearance_v1.py`, its attention helper, scorer/report/verifier wrappers and a small attention-contract test. The original generator and installed attention processor stay available and unchanged. Save hashes of this protocol, wrappers, installed attention implementation, feature banks, model/adapter assets, manifest and all inputs/references; record library versions and the entire scheduler configuration.

1. Inspect and unit-check the attention modification: unchanged joint multi-reference computation, correctly shaped query-mask multiplication, no effect on the FaceID branch, all-ones equivalence, and scale-zero equivalence. No GPU image generation is needed for these small synthetic contract checks.
2. On person 263 / seed 17, replay appearance 0.2 with the full query mask against the saved global-appearance raw and finished images. Replay appearance zero against saved CFG 3 raw and finished images. Require pixel equality for **both** replays; stop before candidates on mismatch.
3. Generate the same fixed hole-localized candidate for **all three** seed-17 people. Save raw, finished, effective mask, damaged input, attention wiring/mask receipts, runtime, peak allocation, attempted failures and file hashes. Do not alter a setting after a difficult person fails.
4. Release GPU inference memory and use the existing CPU evaluation environment. Compare with both preserved controls on identical valid pairs. Inspect equal-scale original/control/candidate contact sheets for eye/gaze/iris/brow geometry and seams.
5. Only if the seed-17 gates below pass, generate **all six** fixed seed-29/43 identity combinations without changed parameters. If any initial gate fails, retain the initial screen and stop replication.

Initial budget: **two replay calls plus three candidates**, five GPU calls. Conditional replication: six more. Existing 30-step Portrait-only generation took about 30 seconds per warmed-up image and used about 6.09 GiB peak allocated tensor memory; the second appearance branch and two delegated attention calls add work and memory. Measure this experiment's actual costs. Existing saved appearance features avoid repeating the CPU CLIP encoder and no optimizer/backpropagation state is added. Execute GPU calls serially without competing training jobs; preserve the local website session and original defaults.

Generation uses the existing `cache/reference_env_v2/Scripts/python.exe`; scoring uses `cache/evaluation_env_v1/Scripts/python.exe`, resolved through `configs/local.json`. No environment upgrade is part of this experiment.

## Frozen metric and success gates

Reuse [the masked scorer](../scripts/score_masked_sdxl_v1.py), [regional definitions](../scripts/region_metrics_v1.py) and [identity/LPIPS definitions](../scripts/extended_evaluation_metrics.py) unchanged. Hole SSIM uses fully contained 11-by-11 Gaussian RGB SSIM windows; hole/eye LPIPS are contextual bounding-box crops, not pixel-masked LPIPS. FaceNet remains separate from generation conditioning; ArcFace is conditioning-family similarity. Detector failures remain null with explicit coverage. Do not expand masks, alter crop bounds, resize small components or change scoring to improve values.

The seed-17 gate requires **all three attempts completed**, zero changed visible pixels, and three matched valid identity measurements for both controls. Missing measurements or failures do not produce a favorable smaller aggregate.

Against **each** of the global-appearance and appearance-OFF CFG 3 controls:

- Mean hole SSIM must strictly improve.
- Mean hole PSNR, hole contextual LPIPS, eye contextual LPIPS, FaceNet and ArcFace must not worsen. Use only 1e-6 numerical tolerance for the non-worsening checks.
- Every person's FaceNet and ArcFace cosine must be at least the matched control minus **0.005**. This fixed pragmatic development-screen allowance is not a validated perceptual threshold; record every regression and retain the stricter non-worsening mean requirements.
- Visual inspection must show no new duplicated/displaced/clipped eye or brow geometry, obvious worsened iris/gaze/colour mismatch or seam artifact versus either control. A plausible but different person's eye is not correct reconstruction.

If all initial gates pass, apply the same mean gates to the nine fixed matched pairs and the 0.005 per-person identity guard to three-seed means after conditional replication. Report each person, each seed and all individual deltas; no best-seed presentation. Original CFG 5 identity results also stay visible, particularly person 10149's existing CFG 3 FaceNet loss. Passing a localized/global appearance contrast does not by itself remove that historical regression.

Keep supplementary anatomical-eye SSIM, per-eye diagnostics, whole-image scores, valid-window counts and detector records. Person 6681's left-eye SSIM has only four valid centres, so it must not carry a broad eye-fidelity claim. Three development identities with repeated seeds support descriptive findings; seeds are not independent people for statistical significance. None of these cosines or SSIM values is a calibrated reconstruction accuracy percentage.

## Scope and follow-up

Query-output masking is local at that attention operation, but later UNet convolutions, self-attention, lower-resolution mixing and denoising can propagate its influence beyond that location. Exact final RGB preservation outside the hole remains a separate compositing guarantee. The mechanism still uses global reference features and does not establish eye-to-eye, iris or eyebrow correspondence, reference reliability, pose/expression compatibility or automatic rejection of a misleading reference.

If the candidate fails, retain it and its failure explanation without changing website defaults. If it passes, corrupted-reference tests and independently reserved identities remain required future work after choices are frozen. The original clean controls, old processes and all negative experiments remain available. No new scientific-contribution or unknown-client-accuracy claim follows from this three-person screen.
