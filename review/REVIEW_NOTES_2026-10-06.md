# Facial Inpainting Studio — Review Preparation Notes

**For your review on 6 October 2026. Verified against local code, training receipts and saved results on 5 October 2026.**

Read sections 1–7 first to understand the project. Read sections 8–10 for results. Use section 12 to demonstrate the code, and section 15 to practise faculty questions. The final section is a quick revision sheet.

These notes describe the existing CelebA/CelebAMask-HQ research. The separate Human Faces dataset work is outside the main results discussed here. The latest illustrated presentation has 32 slides.

## 1. Your opening explanation

> “Our project is a local facial reconstruction application and research pipeline. It supports missing-region inpainting and restoration of blurred or noisy faces. We first integrated pretrained reconstruction models and trained small mask-refinement controls. Our current research uses a frozen ReF-LDM backbone with multiple photographs of the same person. We train a lightweight target-reference compatibility adapter to change how the model uses references, especially when a reference contains misleading facial information. We evaluate reconstruction, identity similarity and eye-region quality together. The application and experiments run locally on an RTX 5070 Laptop GPU. Some candidates improve individual measurements, while achieving consistent improvement across all quality criteria is still work in progress.”

**The main code to show:** [scripts/target_region_compatibility_v1.py](../scripts/target_region_compatibility_v1.py#L37). This implements the current research mechanism. It is different from the website launcher and from the pretrained generator itself.

**The base paper for the current research:** [ReF-LDM](https://arxiv.org/html/2412.05043v1). LaMa and ResShift are other implemented reconstruction backends and controls, not the backbone of the current 193-parameter adapter experiment.

## 2. What problem are we solving?

There are two related tasks. Be precise about which one each experiment tests.

| Task | Input | Expected output | Main difficulty |
|---|---|---|---|
| Missing-region inpainting | A face photograph and a mask marking missing/damaged pixels | Plausible content inside the mask, with visible content preserved | Missing eyes or other features may have many plausible appearances |
| Blur/noise restoration | A degraded photograph that still contains facial evidence, optionally with references | A clearer facial image consistent with the person | Detail has been lost, and a sharp-looking output may still have incorrect identity or gaze |

The website supports both. **The latest matched ReF-LDM adapter tables measure whole-image blur/noise restoration, not erased-eye inpainting.** Do not describe those scores as proof that every website mode reconstructs missing eyes accurately.

Three objectives guide the work:

1. Improve perceptual quality and local facial detail.
2. Preserve identity and trustworthy observed pixels where applicable.
3. Reduce the effect of corrupted or misleading reference photographs.

The output is a predicted reconstruction. Even with reference photographs, the exact expression, gaze or hidden details of a missing region cannot always be recovered.

## 3. What we built, and what we actually trained

| Component | What it does | Locally trained? | Status |
|---|---|---|---|
| Browser application | Upload, mask/evidence controls, select method, run GPU inference, compare/download | No | Implemented locally |
| LaMa / Big-LaMa | Single-image masked reconstruction | No backbone retraining | Implemented pretrained control |
| ResShift | Diffusion-based reconstruction through the local adapter | No backbone retraining in this existing-dataset study | Implemented pretrained control |
| SDXL inpainting + FaceID Portrait | Missing-region generation conditioned on reference photographs | Pretrained components | Implemented reference mode |
| ReF-LDM | Restoration conditioned on a degraded target and reference photographs | Backbone frozen | Current main research backbone |
| MaskRefiner | Predicts which pixels should be considered damaged | Yes, 119,057 parameters | Earlier generic/weighted control study |
| Reference-bias pilot | Modifies reference attention using reference features | Yes, 545 parameters | Earlier pilot and follow-up |
| Regional compatibility adapter | Uses degraded-target and reference evidence to bias reference attention | Yes, 193 parameters | Current v1–v5 research family |
| FaceNet / ArcFace | Produce recognition embeddings for similarity evaluation | Recognition networks not trained locally | FaceNet also supplies a frozen training loss in some adapter runs |
| Evidence maps / blending | Express how much observed evidence to retain | User-provided/control logic | A control mechanism, not a calibrated probability of correctness |

**Important naming correction:** ResShift is not ResNet. FaceNet's implementation uses an InceptionResnetV1 recognition network; it evaluates identity features and is not the image generator. Our small mask refiner is a U-Net-like convolutional model, not a face-generating ResNet.

The pretrained backbones supply most of the generative knowledge. Our locally trained modules are much smaller. We did not train a foundation-scale facial generator from scratch on the laptop.

### Earlier method investigations you may be asked about

| Investigation | What was tested | How to describe its outcome |
|---|---|---|
| Mask-aware reference selection | Choose a reference using the damaged target and supplied mask | 168 logical evaluation rows; no training in that cycle. It did not consistently outperform all-reference conditioning. [research/GENERALIZATION_RESULTS.md](../research/GENERALIZATION_RESULTS.md) |
| Regional routing | Use different reference information for different regions | 144 logical rows, including 120 new generations; no training. It did not improve mean target identity over the original all-reference control. [research/REGIONAL_ROUTING_RESULTS_V2.md](../research/REGIONAL_ROUTING_RESULTS_V2.md) |
| Aligned local-feature fusion | Inject aligned reference VAE features into masked-image conditioning | A 48-row development diagnostic with no trainable parameters. It remains separate from the later ReF-LDM adapter. [research/LOCAL_LATENT_RESULTS_V1.md](../research/LOCAL_LATENT_RESULTS_V1.md) |
| Scaffold/context conditioning | Change how prior reconstruction/context enters missing-region generation | A separate controlled inference study; exact visible-pixel preservation did not establish correct hidden features. [research/SCAFFOLD_CONDITIONING_RESULTS_V1.md](../research/SCAFFOLD_CONDITIONING_RESULTS_V1.md) |
| Structure-only transport | Transfer reference geometry into existing reconstructions without transferring texture | All 24 structural-transform attempts failed in that recorded screen. The 40 completed rows were controls, not successful new transforms. [research/STRUCTURE_TRANSPORT_RESULTS_V1.md](../research/STRUCTURE_TRANSPORT_RESULTS_V1.md) |
| RefFaceInpainting author baseline | Run an external published reference-guided model locally | 16 new predictions and 12 reused controls, 28 scored rows. An implemented comparison, not our proposed mechanism. [research/REFFACE_BASELINE_RESULTS_V1.md](../research/REFFACE_BASELINE_RESULTS_V1.md) |

These studies explain how the project moved from selection and inference controls toward training a compact attention adapter. They are different protocols; their row counts and metrics must not be pooled as one experiment. Failed attempts remain recorded.

## 4. Datasets and separation of experiments

### 4.1 Sources and their roles

| Source | Project use |
|---|---|
| CelebA annotations and partitions | Identity mapping and split information |
| CelebAMask-HQ / HQ images and mapping | Face images, association with CelebA identities, facial-component annotations |
| LaPa | An additional source used in the earlier object-composite evaluation; face framing used annotation-derived crops |
| Official ReF-LDM evaluation inputs | Recent limited reference-encoding diagnostic, not a complete author-benchmark reproduction |
| Separate Human Faces folder | Separate investigation; do not mix its examples or training claims into this review's main results |

Dataset photographs and downloaded backbone weights remain local. GitHub contains code, reports and permitted experiment artifacts. The dataset restoration instructions use official sources rather than redistributing restricted photographs.

### 4.2 Two different local training populations

**Earlier mask-refiner study:** the recorded training cache contains **24,133 images from 5,007 identity labels**, plus **48 tuning images from 48 separate identity labels**. These counts were checked from the local training-cache metadata. This is the mask-refiner dataset, not the training population for the current attention adapter.

**Current attention-adapter study:** **20 selected identities**, each with **one target and four references**. That gives 100 selected original images: 80 for the 16 adapter-training identities and 20 for the four adapter-development identities. Corrupted versions and generated outputs do not count as additional independent people.

The four repeatedly evaluated development labels are **263, 6681, 7400 and 10149**. They are separate from the 16 identities used to fit these adapters. They came from the original dataset training partition and have now been repeatedly inspected. They are development data, not a fresh final test. Earlier project or upstream backbone exposure is not ruled out.

The project uses manifests, identity labels, exact image hashes and perceptual-hash checks to reduce overlap. These controls do not prove that every near-duplicate or upstream pretraining overlap is absent.

**Code:** [scripts/prepare_reference_intervention_pilot.py](../scripts/prepare_reference_intervention_pilot.py#L11). **Recorded cases:** [outputs/reference_intervention_pilot_v1/manifest.json](../outputs/reference_intervention_pilot_v1/manifest.json).

## 5. Current main architecture, explained simply

```text
Degraded target photograph ──> frozen VQ encoder ──> target latent
                                                           │
Reference photographs ──────> frozen VQ encoder             │
              │                         │                  │
              │                  reference K/V cache       │
              │                         │                  ▼
              └─> target/reference descriptors ──> small compatibility MLP
                                                     │
                                    regional reference attention biases
                                                     │
                         frozen ReF-LDM diffusion denoiser
                                                     │
                             50 DDIM sampling steps
                                                     │
                                  frozen VQ decoder
                                                     │
                                  reconstructed RGB face
```

The VQ encoder compresses a 512 × 512 RGB face into an **8 × 64 × 64 latent representation** in the matched research pipeline. The diffusion model operates on this compact representation. The decoder converts the final latent back to an image.

Reference images contribute cached attention keys and values. Think of attention as deciding which information to consult when predicting the next representation. The adapter changes the preference for different references in different target regions. It does not directly copy pixels or independently synthesize a whole face.

### 5.1 The five regions

The implementation uses fixed rectangles for left eye, right eye, nose, mouth and whole-face/context information. These are approximate aligned-face regions. They are **not learned segmentation**, a 3D face model or geometric warping. Large pose changes can therefore misalign the regions.

### 5.2 The ten adapter features

For each region/reference pair, the feature vector contains:

1. Regional degraded-target/reference cosine similarity.
2. Whole-image/context cosine similarity.
3. Agreement with the median reference descriptor.
4. Normalized difference in descriptor magnitude.
5. A five-landmark pose-proxy discrepancy.
6. Reference sharpness proxy.
7. Reference high-frequency/noise-edge proxy.
8. Target/reference brightness difference.
9. Minimum target/reference face-detection confidence.
10. Observed-evidence coverage slot, currently fixed to **1** in this whole-image experiment.

The current code uses cosine, magnitude and quality features. It does **not** implement every proposed concatenation, learned correspondence, segmentation or uncertainty idea. Consensus is fallible: several references can share the same incorrect information.

### 5.3 Why exactly 193 trainable parameters?

The shared MLP is `Linear(10,16) → SiLU → Linear(16,1)`.

| Layer | Weights | Biases | Parameters |
|---|---:|---:|---:|
| 10 inputs to 16 hidden units | 160 | 16 | 176 |
| 16 hidden units to 1 output | 16 | 1 | 17 |
| **Total registered trainable parameters** | **176** | **17** | **193** |

The same adapter is used at **16 attention layers**. It is shared, so the count is **193 total**, not 193 × 16. This is the parameter count of the adapter only, not of the complete ReF-LDM system.

### 5.4 How the attention bias works

For region `r` and reference `i`:

```text
score(r,i) = MLP(features(r,i))
bias(r,i) = log_softmax_over_references(score(r,i)) + log(number_of_references)
attention = softmax(QKᵀ / sqrt(d) + bias) V
```

The code clamps scores and uses a temperature. The final layer starts at zero. Equal zero scores make the normalized bias zero, preserving the baseline at neutral initialization. Reference biases affect the conditional branch; self-attention receives zero added bias and the null CFG branch is preserved.

**Current limitation:** relative normalization redistributes reference preference, but does not provide an independent knob to reject all unreliable references together. That motivates the next spatial rejection experiment.

**Code walkthrough:** [scripts/target_region_compatibility_v1.py](../scripts/target_region_compatibility_v1.py#L21) → [scripts/target_region_compatibility_v1.py](../scripts/target_region_compatibility_v1.py#L37) → [scripts/target_region_compatibility_v1.py](../scripts/target_region_compatibility_v1.py#L64). **Frozen model loader and reference cache:** [scripts/reference_intervention_core.py](../scripts/reference_intervention_core.py#L19) and [scripts/reference_intervention_core.py](../scripts/reference_intervention_core.py#L86).

## 6. Training: what the optimizer learns

### 6.1 Frozen versus trainable

The optimizer receives only adapter parameters. The ReF-LDM backbone, VQ encoder/decoder and FaceNet recognition weights stay frozen. The training code checks that frozen networks have no parameter gradients.

Frozen does not mean that gradients cannot pass through a network. For identity supervision, the decoded image passes through frozen FaceNet, and its loss gradient flows back to the adapter. The decoder/recognizer computations consume memory even though their weights are not updated.

### 6.2 Diffusion reconstruction loss

```text
Ldiff = 0.25 × full-latent noise-prediction MSE
      + 0.75 × face-rectangle noise-prediction MSE
```

The model predicts the noise added to a clean training latent at a sampled diffusion timestep. The face rectangle gives more emphasis to the central facial area. It is a latent rectangle, not the evaluation parsing mask.

### 6.3 Identity loss

```text
Lid = 1 − cosine(FaceNet(predicted clean image), FaceNet(clean training target))
Ltotal = Ldiff + λ × Lid
```

The predicted clean latent `x0` is estimated during a training step and decoded. This makes identity supervision possible without running the full 50-step sampler during every optimizer update. Its limitation is that one-step predicted `x0` is not identical to the final multi-step reconstruction.

The calibrated base identity weight is **0.5408350358795648**. In v2 it was derived from the first supervised-step gradient-norm ratio, capped at 10. It was not selected by searching for the best development score.

### 6.4 Changes across versions

- **v1:** target/reference compatibility, diffusion reconstruction loss only; includes a no-target-feature ablation.
- **v2:** same 193-parameter architecture; matched diffusion-only versus diffusion-plus-FaceNet loss. Identity supervision occurs every fourth update.
- **v3:** use fixed training-target MTCNN face bounds for both predicted and target crops, instead of resizing the whole aligned image. The every-fourth-step schedule remains.
- **v4 fixed:** supervise identity at every update, giving four identity-supervised updates to each of the 16 identities.
- **v4 capped:** same coverage, but cap the weighted identity-gradient norm relative to the reconstruction gradient. Effective weight is `min(base_weight, ||gdiff|| / max(||gid||, 1e-12))`.
- **v5:** same coverage and crops as v4 fixed, but multiply the base identity weight by `sqrt_alphas_cumprod[t]`. This scales the loss by the retained signal at the diffusion timestep. It is a separate arm, not v4 capped plus another change.

**LPIPS is an evaluation metric in these current runs, not a trained loss term.** The code has not silently trained all proposed perceptual, consistency and regional losses. The older reference-bias pilot did include a separate consistency intervention.

## 7. Exactly how many times did we train?

### 7.1 Current 193-parameter adapter family

| Version / arm | Independent training runs | Updates per run | Recorded loop time | Purpose |
|---|---:|---:|---:|---|
| v1 no-target | 1 | 128 | 57.83 s | Remove target-dependent inputs |
| v1 full regional | 1 | 128 | 50.61 s | Target/reference compatibility |
| v2 diffusion-only | 1 | 64 | 23.61 s | Equal-budget loss control |
| v2 identity | 1 | 64 | 71.42 s | Add frozen FaceNet supervision |
| v3 fixed face crop | 1 | 64 | 93.56 s | Align the identity training crop |
| v4 balanced fixed | 1 | 64 | 291.78 s | Identity loss on every update |
| v4 balanced capped | 1 | 64 | 288.66 s | Limit identity-gradient dominance |
| v5 timestep-scaled | 1 | 64 | 294.45 s | Test timestep-dependent identity weighting |
| **Total for this family** | **8** | **640 combined** | **About 19.5 minutes combined** | **Different arms, not 640 updates of one model** |

These are small, controlled adapter experiments using prepared data/latents. Loop times exclude downloads, source audits, caching, model startup and evaluation. They are not total project duration.

Common v2–v5 settings:

| Setting | Recorded value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Gradient clipping | Global adapter norm 1.0 |
| Training seed | 20261002 |
| Target batch per update | One target with four references |
| Training identities | 16 |
| Schedule | 64 updates; four occurrences per identity |
| Reference conditions | Clean, blur, noise, wrong-person patch; 16 scheduled updates each |
| Training diffusion timesteps | 100, 200, 300, 400, shuffled separately from the conditions |
| Precision | bfloat16 autocast for main model computations; not every operation is bfloat16 |
| Selection | Fixed-budget experiments; no best evaluation-seed selection |

v1 differs: 128 updates, eight reference conditions, training timesteps 100/300/500/700. Do not apply the v2 schedule to v1.

In v2 identity and v3, 16 identity-supervised updates covered only **12 of 16 identities**, despite diffusion training covering all 16. v4/v5 fix that coverage imbalance with **64 supervised updates**, four per identity.

Each new arm is a controlled fit from its recorded initialization; the version numbers do not mean one checkpoint was continually trained through every version.

### 7.2 Earlier training, kept separate

**545-parameter pilot:** two initial 64-update arms plus one 64-update scaled follow-up = **3 runs and 192 updates**. The initial pair compared reconstruction loss and an intervention consistency term. These are not the 193-parameter architecture.

**119,057-parameter mask refiner:** two loss variants × three seeds (17, 29, 43) = **6 runs**. Each first trained for **1,500 updates**, then was extended to **6,000 total updates**. That is 36,000 final-budget updates across six runs, not 45,000; the extension adds 4,500 to each existing run. Batch size was 16, resolution 128, learning rate 0.0003. Generic versus preservation-weighted loss uses negative-pixel costs 1 versus 4. Best retained checkpoints were at update 5,500 or 6,000, depending on the run.

The mask refiner predicts a mask, not a reconstructed face. A lower mask loss is therefore not automatically a better eye reconstruction.

### 7.3 Four counts you must not confuse

| Term | Meaning in this project |
|---|---|
| Training update | One optimizer step changing learned weights |
| Epoch | A full pass through a defined dataset; do not call 64 updates “64 epochs” |
| DDIM inference step | One denoising step during generation; no learning occurs |
| Inference seed | A controlled sampling-noise choice; three inference seeds are not three independently trained models |

Receipts: [outputs/target_compatibility_v1/training/receipt.json](../outputs/target_compatibility_v1/training/receipt.json), [outputs/identity_compatibility_v2/training.json](../outputs/identity_compatibility_v2/training.json), [outputs/identity_crop_v3/training.json](../outputs/identity_crop_v3/training.json), [outputs/identity_balance_v4/balanced_fixed/training.json](../outputs/identity_balance_v4/balanced_fixed/training.json), [outputs/identity_balance_v4/balanced_capped/training.json](../outputs/identity_balance_v4/balanced_capped/training.json), [outputs/identity_timestep_v5/timestep_scaled/training.json](../outputs/identity_timestep_v5/timestep_scaled/training.json). Earlier refiner settings: [configs/learned_refiner.json](../configs/learned_refiner.json) and [research/TRAINING_EXTENSION_RESULTS.md](../research/TRAINING_EXTENSION_RESULTS.md).

## 8. Evaluation: how we checked outputs fairly

### 8.1 Main matched adapter protocol

Four development identities × three inference seeds (17, 29, 43) × two conditions = **24 outputs per method**, or **12 per reference condition**. Common generation settings: 512 RGB output, four separately encoded references, 50 DDIM steps, guidance 1.5, bfloat16, `uniform_trailing` sampling schedule and the same frozen backbone.

The target is degraded by bicubic resizing **512 → 64 → 512**, followed by Gaussian RGB noise with standard deviation **8 on the 0–255 scale**. Clean-reference condition means the references are clean; the target is still degraded.

In the misleading condition, the central **256 × 256** patch of reference 1 is replaced with another identity's content. The other three references stay unchanged. The learned adapter receives degraded-target/reference features, not the clean evaluation target or a label identifying which reference was corrupted.

### 8.2 Output counts by experiment

| Experiment | New generated outputs | Meaning |
|---|---:|---|
| v1 broad reference-corruption screen | 96 | Four identities × three methods × eight conditions × one seed |
| v2 matched objective screen | 96 | Four identities × four methods × two conditions × three seeds |
| v3 crop candidate | 24 | Reuses 72 previously verified control outputs for a 96-row panel |
| v4 fixed and capped candidates | 48 total | 24 new per arm; the same 72 controls are reused for each panel |
| v5 timestep candidate | 24 | Reuses 72 controls |
| Expanded encoding diagnostic | 30 | Five targets × two encoding methods × three seeds; plus five input-control scores |
| Whole-reference rejection diagnostic | 24 interventions + 8 replays | Uses 24 archived baseline controls, yielding 48 scored rows |

Do not add reused controls as though they were new independent generations. Some replay calls reproduce existing cases to check reproducibility.

### 8.3 Metrics and what they mean

| Metric | Measures | Preferred direction | What it cannot establish alone |
|---|---|---|---|
| PSNR, in dB | Pixel mean-squared error relative to the clean target | Higher | Natural-looking detail or identity correctness |
| SSIM | Local structural similarity | Higher | Exact identity recovery |
| LPIPS, AlexNet implementation | Learned perceptual distance to the target | Lower | Identity preservation or true hidden content |
| FaceNet cosine | Similarity between recognition embeddings | Higher | Percentage of correct reconstructed pixels |
| ArcFace cosine | A second recognition-embedding comparison | Higher | A universal independent identity check across every pipeline |
| Eye bounding-box LPIPS | Perceptual error in the annotated eye crop plus context | Lower | Error exclusively inside eye pixels |
| NIQE / BRISQUE | No-reference image naturalness diagnostics | Usually lower | Correct identity, faithful gaze, or fidelity to a known target |
| Visible-pixel change / error | Whether trusted pixels were modified | Lower; exact preservation where required | Correctness of the newly generated region |

PSNR uses normalized RGB error. LPIPS inputs are scaled to −1…1. SSIM uses RGB, an 11 × 11 Gaussian window, sigma 1.5 and population covariance. Regional SSIM only includes fully contained windows. Eye LPIPS uses a bounding crop with **10-pixel context**, not a strictly pixel-masked LPIPS calculation.

FaceNet uses MTCNN face detection/cropping and a VGGFace2-pretrained InceptionResnetV1 encoder. ArcFace uses InsightFace `buffalo_l`. A missing/multiple-face detection is recorded as unavailable, not filled with zero.

FaceNet contributes to the identity-supervised training objective, so its evaluation is not independent of that loss. ArcFace is separate from the ReF-LDM adapter training objective, but the SDXL FaceID path uses an ArcFace-family conditioning encoder. The independence claim must be specific to the experiment.

Paired comparisons use the same identity, condition and seed. Statistical summaries use identity-level aggregation and 2,000 bootstrap resamples. Four identities provide limited evidence; 12 seeded outputs do not become 12 independent people.

**Code:** [scripts/extended_evaluation_metrics.py](../scripts/extended_evaluation_metrics.py#L24), [scripts/region_metrics_v1.py](../scripts/region_metrics_v1.py#L17), [scripts/score_identity_compatibility_v2.py](../scripts/score_identity_compatibility_v2.py#L13).

## 9. Results to understand and explain

### 9.1 The matched baseline and selected recent candidates

**Clean references, 12 outputs per row:**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FaceNet ↑ | Eye LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Local ReF-LDM | 23.86081 | 0.63481 | 0.27725 | 0.91497 | 0.12901 |
| v4 capped | 23.86587 | 0.63465 | 0.27707 | 0.91184 | 0.12843 |
| v5 timestep-scaled | 23.85949 | 0.63501 | 0.27743 | 0.91623 | 0.13018 |

v4 capped slightly improves PSNR/LPIPS but loses identity similarity. v5 improves FaceNet and SSIM but loses LPIPS and eye quality. **No row is best on every important criterion.**

**Misleading references:**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FaceNet ↑ | Eye LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Local ReF-LDM, all 12 | 23.84680 | 0.63377 | 0.27769 | 0.90545 | 0.12800 |
| v4 capped, all 12 | 23.84916 | 0.63390 | 0.27800 | 0.90700 | 0.12841 |
| v5 timestep-scaled | 23.84852 | 0.63387 | 0.27823 | 0.90414* | 0.12647 |

**For v5 FaceNet, use the same 11 valid pairs: baseline 0.90932 versus candidate 0.90414.** The other v5 metrics above use 12 outputs. Comparing its 11-output FaceNet mean directly with the full 12-output baseline would be misleading. The PPT now makes this correction.

For the original matched panel, ArcFace generally has **8/12 valid pairs** per condition. Missing detections remain part of the report.

### 9.2 Best earlier values retained in the PPT

The earlier **seed-17-only**, four-identity clean-reference comparison was:

| Method | PSNR | SSIM | LPIPS | FaceNet |
|---|---:|---:|---:|---:|
| Baseline | 24.04671 | 0.63723 | 0.26544 | 0.91158 |
| Regional v1 | 24.05182 | 0.63730 | 0.26477 | 0.90670 |

These values belong to one seed. Do not replace the three-seed means with them or compare them against a different cohort as evidence of improvement.

Across the matched three-seed clean-reference methods, the best recorded PSNR is **23.86587** and LPIPS **0.27707** from v4 capped; best SSIM **0.63501** and FaceNet **0.91623** come from v5. They are **different fitted models**, not one model combining all four values.

### 9.3 What the latest diagnostics taught us

**Encoding:** four multi-reference targets from one identity, across three seeds. Joint RGB encoding raises mean FaceNet from **0.89963 to 0.90372**, but LPIPS worsens from **0.28170 to 0.28295**. The full experiment also contains a second identity as a single-reference control. Therefore it is not a four-identity multi-reference validation.

**Removing the known bad reference:** across four identities and three seeds, FaceNet rises from **0.90545 to 0.91153**, but LPIPS worsens from **0.27769 to 0.28373**, worsening in **11/12 cases**. Removing an entire reference can discard useful information along with its corrupted region.

This removal uses knowledge of which reference we deliberately corrupted. It is an **oracle feasibility diagnostic**, not a learned detector ready for arbitrary client photos. Zeroing every reference degraded identity further.

**Research direction:** preserve useful reference content while suppressing only incompatible regions. Test a known-region intervention first, with equal-area random-region and neutral controls, before training a selector.

### 9.4 What does “70% similar” on an image mean?

In the PPT image examples, that label is **SSIM × 100 rounded to a whole percentage against the original target**. It is a displayed structural similarity score. It does not mean 70% probability of being the correct person, 70% reconstruction accuracy or 70% of pixels exactly recovered.

We do not have a validated universal accuracy percentage for arbitrary unknown client photographs. Finishing 96/96 inference jobs is an execution success rate, not reconstruction accuracy.

## 10. What is working, and what remains in progress?

**Established engineering work:** local model inference and UI, controlled damage generation, trained compact modules, separate configs/checkpoints, reconstruction comparisons, reproducible metrics, hash-verified controls, explicit failure reporting and paper/protocol comparisons.

**Measured research findings:** identity supervision, crop alignment, gradient capping, timestep scaling and reference encoding each affect the quality trade-off. The gradient cap reduced the maximum weighted identity/reconstruction gradient ratio from **55.21 to 1.00**, but that optimization success did not make every reconstruction metric improve.

**Still in progress:** a candidate that improves perceptual fidelity while preserving identity and eye detail consistently, broader corruption coverage, more independent identities, independent training seeds, arbitrary client-image evaluation and final held-out validation.

The proposed contribution is the investigation of compact target/reference compatibility and reliability under misleading references. Prior work already contains reference attention, identity losses and region selection. Publication preparation requires a specific defensible mechanism and fair evidence for its benefit. We should present the current improvements and trade-offs honestly rather than claim a universal advantage.

## 11. Laptop hardware and runtime

The development laptop uses an **RTX 5070 Laptop GPU with about 8 GB dedicated VRAM**, and about **32 GB system RAM**. Main code is Python/PyTorch with CUDA; the browser UI communicates with a local Python service. Separate environments isolate model dependencies.

Small diffusion-only adapter training used about 3.6–3.8 GiB of allocated tensors. Decoded identity-loss runs recorded about **8,489–8,491 MiB**, roughly 8.29 GiB. They completed, but that does not prove comfortable dedicated-VRAM headroom or separately quantify shared-memory use.

The encoding diagnostic recorded approximately **7.09 GiB** for joint RGB encoding versus **2.39 GiB** for separate encoding. Its measured generation times excluded model startup.

A larger GPU can enable larger batches, more experiments or larger models. It does not automatically resolve wrong gaze, objective imbalance, reference mismatch or limited validation. There is no measured evidence that system RAM directly caused the metric limits. Avoid saying “our score is low only because of the GPU.”

**Hugging Face's role:** model distribution and libraries such as Diffusers support the SDXL path. Using these tools is not the proposed research contribution. The reference inference code sets offline flags after local setup; core local inference does not require uploading the user's face to a cloud API.

## 12. Clickable code map and demonstration order

Open the files in this order. You do not need to read the entire repository during the review.

| Priority | File / link | What to show |
|---|---|---|
| **1 — Main proposed mechanism** | [scripts/target_region_compatibility_v1.py](../scripts/target_region_compatibility_v1.py#L37) | Ten features, 10→16→1 MLP, regional priors and attention installation |
| **2 — Latest training** | [scripts/train_identity_timestep_v5.py](../scripts/train_identity_timestep_v5.py#L17) | Frozen backbone/FaceNet, losses, timestep weight, optimizer and checkpoint |
| **3 — Baseline integration** | [scripts/reference_intervention_core.py](../scripts/reference_intervention_core.py#L19) | Verified author checkpoint loading, EMA weights and frozen model |
| **4 — Matched generation** | [scripts/evaluate_identity_compatibility_v2.py](../scripts/evaluate_identity_compatibility_v2.py#L11) | Identical seeds, references, conditions, backbone and sampling settings |
| **5 — Scoring and statistics** | [scripts/score_identity_compatibility_v2.py](../scripts/score_identity_compatibility_v2.py#L13) | Valid pairs, metrics, summaries and progression checks |
| Supporting metric definitions | [scripts/extended_evaluation_metrics.py](../scripts/extended_evaluation_metrics.py#L24) / [scripts/region_metrics_v1.py](../scripts/region_metrics_v1.py#L17) | Face detection, LPIPS/SSIM/PSNR and eye-crop calculations |
| Earlier loss-control comparison | [scripts/train_identity_balance_v4.py](../scripts/train_identity_balance_v4.py#L17) | Fixed versus gradient-capped identity supervision |
| Data selection and damage | [scripts/prepare_reference_intervention_pilot.py](../scripts/prepare_reference_intervention_pilot.py#L11) / [scripts/cache_reference_intervention_pilot.py](../scripts/cache_reference_intervention_pilot.py#L10) | Identity separation, source hashes, target degradation and wrong-reference patch |
| Previous mask-refinement model | [scripts/refiner.py](../scripts/refiner.py#L12) / [scripts/train_refiner.py](../scripts/train_refiner.py#L38) | U-Net-like mask predictor and generic/weighted loss |
| LaMa / ResShift dispatch | [scripts/inpaint.py](../scripts/inpaint.py#L27) | GPU reconstruction and visible-pixel composition |
| SDXL reference inpainting | [scripts/reference_inpaint.py](../scripts/reference_inpaint.py#L25) | FaceID conditioning and missing-region generation |
| Website ReF-LDM restoration | [scripts/studio_refldm_restore.py](../scripts/studio_refldm_restore.py#L21) / [scripts/refldm_restore.py](../scripts/refldm_restore.py#L11) | Reference preparation and partial-damage restoration |
| Website entry point | [webapp/server.py](../webapp/server.py#L109) / [scripts/launch_studio.py](../scripts/launch_studio.py) | Input validation, background job dispatch and local startup |
| Latest diagnostic | [scripts/evaluate_reference_rejection_v1.py](../scripts/evaluate_reference_rejection_v1.py#L15) | Known-reference removal and zero-reference controls |

**Main-method walkthrough in about three minutes:**

1. In `features()`, show that descriptors use the degraded target and supplied references. The clean target is absent at inference.
2. In `Compatibility`, explain the 193-parameter MLP and normalized log-priors.
3. In `install()`, show where the bias enters attention logits and how the null/self branch is preserved.
4. In the training script, show `params = list(adapter.parameters())` and the frozen backbone checks.
5. Show `training.json` for completed steps and checkpoint hash, then `scores.json` for real measured outputs.

**Important website/research distinction:** the website's ReF-LDM runner currently uses the author-style joint RGB conditioning path. The matched adapter studies use separately encoded cached latents. The research candidate is not automatically deployed in the website. Do not call a website screenshot a v5 adapter result unless its run record actually says so.

The local notes link directly to source files. In the printable HTML companion, “GitHub copy” links are also included for tracked code; local edits may differ from the remote version.

## 13. Papers to cite and why

| Paper / project | How we use it |
|---|---|
| [ReF-LDM](https://arxiv.org/html/2412.05043v1) | Main pretrained restoration backbone and reference-attention foundation |
| [LaMa](https://arxiv.org/html/2109.07161v1) | Implemented single-image inpainting control |
| [Extended ResShift](https://arxiv.org/html/2403.07319v1) | Relevant diffusion restoration/inpainting implementation family |
| [RefFaceInpainting](https://arxiv.org/html/2303.07014v1) | Local reference baseline experiments and related work |
| [ReFine](https://openaccess.thecvf.com/content/WACV2025/papers/Chong_Copy_or_Not_Reference-Based_Face_Image_Restoration_with_Fine_Details_WACV_2025_paper.pdf) | Related work on reference-guided detail restoration, not locally reproduced |
| [RefSTAR](https://arxiv.org/html/2507.10470v1) | Related work on reference selection/transfer, not locally reproduced |

The broader comparison also includes GFRNet, 3DFaceFill, DMDNet and ReSem-Face. See [research/PAPER_VS_PROJECT_COMPARISON_20261002.md](../research/PAPER_VS_PROJECT_COMPARISON_20261002.md) for selected values, links and exact protocols.

**Why published scores may be higher:** datasets, degradation severity, masks, resolutions, crops, training scale, references and metric implementations differ. Published numbers and local development numbers are context unless those conditions match. Our strongest causal comparison is the same local ReF-LDM protocol with and without the adapter. A numerically higher score on a different task does not establish a better method.

IEEE Xplore is a digital library. A submission must target a specific conference or journal with an appropriate scope and evaluation standard.

## 14. Demo plan for tomorrow

1. Have the 32-slide PPT and these notes available locally before the review.
2. Start the existing application using `Start Studio.cmd`, or the command below.
3. Explain the task before selecting the method: a missing region needs inpainting; blur/noise restoration still requires observed evidence.
4. Use a prepared case with known ground truth first. Show the degraded input, the method, the references if applicable and the reconstructed output.
5. Show a saved matched research comparison for the adapter. Explain its recorded seed, method and metrics.
6. Demonstrate the main code and receipt rather than starting a training experiment during the review.
7. If a live run fails, show the saved run and error honestly. Keep a saved comparison ready in case the GPU or model startup takes longer than expected.

```powershell
Set-Location 'C:\Users\rehaa\OneDrive\Documents\ChatGPT\Facial inpainting Project'
.\.venv\Scripts\python.exe .\scripts\launch_studio.py --no-browser
```

Then open [the local application](http://127.0.0.1:8765/). Its [evidence-map page](http://127.0.0.1:8765/confidence.html) expresses user-provided preservation confidence; it is not a calibrated prediction-confidence map. Check the laptop/server before the review; this document is not a guarantee of live uptime.

## 15. Likely faculty questions and answers

### A. Problem and approach

**1. What is the main aim?**
Improve facial reconstruction quality while preserving identity, especially when supplied reference photos contain misleading information. We evaluate perceptual, structural and local facial quality together.

**2. What is the main method you developed?**
A small target-reference compatibility adapter attached to frozen ReF-LDM attention. It computes regional reference preferences using degraded-target and reference features. The main file is `target_region_compatibility_v1.py`.

**3. Is the whole system your own neural network?**
No. It combines pretrained generative and recognition models with locally developed preprocessing, interface, evaluation and small trained modules. The current adapter has 193 trainable parameters; the backbone stays frozen.

**4. How is restoration different from inpainting?**
Restoration improves degraded but present information. Inpainting fills a missing region. The website supports both, but the latest adapter experiment measures whole-image restoration.

**5. Why use reference images?**
They provide additional information about a person's facial appearance. They can also introduce pose, lighting or identity errors, which is why compatibility and reliability matter.

**6. Does more reference imagery always help?**
No. Duplicate, low-quality or misleading references can bias the output. Our whole-reference removal diagnostic shows that discarding a bad reference can improve identity while also losing useful detail.

**7. What happens with an unknown person?**
Pretrained models can run inference on a new upload without per-person retraining. Reliable recovery for arbitrary people is not established by that fact; pose, damage severity and reference quality still matter. Broader external testing is future work.

**8. Can it recreate exactly what was hidden?**
Not reliably in every case. Missing content can have multiple plausible solutions. Reference photos help constrain identity but do not necessarily reveal the exact target gaze or expression.

**9. Why not make synthetic references?**
That was explored separately, but generated references can repeat or invent errors. They must be labelled synthetic and cannot count as independent real identity evidence. Current matched adapter experiments use real dataset references.

### B. Training and architecture

**10. How many parameters did you train?**
193 in the current adapter, shared across 16 attention layers. Earlier experiments used a 545-parameter attention adapter and a separate 119,057-parameter mask refiner.

**11. How many training runs?**
Eight runs in the current 193-parameter v1–v5 family, totalling 640 optimizer updates across separate arms. Earlier: three 64-update pilot runs and six mask-refiner runs extended to 6,000 updates each. These are different studies.

**12. Did you train on the GPU?**
Yes. The completed receipts record CUDA training, durations, allocated memory and checkpoint hashes. The full generator was not retrained.

**13. How many epochs did you train?**
The current protocol specifies optimizer updates. v2–v5 use 64 updates with four scheduled appearances per identity, not 64 epochs. We report the recorded budget rather than invent an epoch count.

**14. Why are there only 16 adapter-training identities?**
This is a controlled mechanism study designed to fit the local resource budget and expose failure modes before expansion. It is too small to establish broad generalization. The earlier mask-refiner study used a much larger training pool.

**15. Why freeze the backbone?**
To retain its pretrained reconstruction ability, reduce trainable parameters and isolate the effect of the compatibility mechanism. It also lowers optimization costs, though gradient flow through frozen computations can remain expensive.

**16. What do the 193 parameters calculate?**
A scalar compatibility score for each region/reference pair from ten features. The same MLP is reused, then scores become normalized attention biases.

**17. How does the model preserve identity?**
Through reference conditioning and, in some training arms, a frozen FaceNet cosine loss. That loss encourages identity similarity but does not guarantee it; final images must still be evaluated.

**18. Why did adding identity loss not solve everything?**
The identity gradient can dominate or conflict with reconstruction, the training crop can differ from evaluation, and a one-step prediction differs from the final sampled output. Our crop, gradient-cap and timestep studies investigate those issues separately.

**19. Are you training on LPIPS?**
Not in the current v1–v5 objective. LPIPS is a perceptual evaluation metric. The current trained objectives are noise-prediction reconstruction and, for selected arms, identity cosine loss.

**20. Why fixed face rectangles?**
They keep the initial mechanism compact and auditable. Their limitation is imperfect alignment for large pose changes. Learned alignment or segmentation would require a separate controlled experiment.

**21. What is an ablation?**
A controlled comparison that removes or changes one component to test its effect. Examples are no-target features, diffusion-only versus identity loss, fixed versus capped identity weighting, and timestep scaling versus fixed weighting.

**22. Why not just train for longer?**
More updates can overfit or strengthen a poor trade-off. The earlier refiner extension tested a longer budget explicitly. For the current adapter we first need a mechanism that passes the quality checks before scaling training.

### C. Results and scientific validity

**23. What is your accuracy?**
There is no single validated classification-style accuracy for this reconstruction task. We report PSNR, SSIM, LPIPS, identity similarity and local-region quality with their protocols. PPT percentages are rounded SSIM-based similarity, not correctness probabilities.

**24. Why use many metrics?**
An image can be pixel-close but blurry, or visually sharp but the wrong face. Different metrics reveal different failure modes. We also inspect outputs visually.

**25. Why FaceNet and ArcFace?**
They offer complementary recognition-embedding comparisons. FaceNet supplies training supervision in identity-loss arms; ArcFace is separate from that loss. Detector coverage and conditioning overlap are disclosed.

**26. What are NIQE and BRISQUE?**
No-reference image-quality diagnostics. They assess naturalness without requiring a clean target, but they cannot certify correct identity or missing facial content. They are implemented and used as complementary diagnostics.

**27. Which is the best method?**
No trained candidate improves all required metrics consistently. The matched pretrained ReF-LDM baseline remains the reference point. We report the best individual measurements by method, rather than combine different models into one fictitious best row.

**28. What is your strongest positive finding?**
We have a reproducible local evaluation pipeline and measurable insights into the trade-off. For example, v5 improves clean-reference FaceNet from 0.91497 to 0.91623, while the gradient cap bounds loss dominance. Both still require perceptual/local-quality improvements.

**29. Why did the FaceNet baseline value change in one PPT row?**
v5 had one unavailable face detection. Its comparison must use the same 11 valid cases: baseline 0.90932 versus v5 0.90414. The full 12-case baseline is 0.90545, but it is not the correct pair for that comparison.

**30. Are the results statistically significant?**
We do not make that claim for the current small repeated development cohort. The statistics include paired identity-level summaries and bootstrap intervals, but four identities limit the strength of inference.

**31. Why three seeds?**
To check sensitivity to generation noise. These are inference seeds. Independent training seeds are a separate robustness requirement still to expand for the current adapter family.

**32. How did you prevent leakage?**
Separate adapter-training and development identities, recorded manifests, image hashes and near-duplicate screens, with reserved final data excluded from current tuning. Upstream pretraining exposure remains unknown, and previously inspected development cases are not claimed as fresh final tests.

**33. Is the clean target used while generating a test output?**
No. It supplies training supervision or post-generation evaluation. At inference the adapter uses the degraded target and supplied references. The oracle diagnostic deliberately has extra corruption-location knowledge and is labelled separately.

**34. Did you beat the base paper?**
We have not reproduced its full matched benchmark. We compare against a matched local ReF-LDM baseline. Published and local scores should not be ranked without matching data, degradation and implementation.

**35. What is the proposed research contribution?**
Compact regional compatibility and reference reliability under corrupted evidence, with identity/perceptual safeguards. Establishing a consistent advantage and distinguishing it from related mechanisms are current research work and future validation priorities.

### D. Application and next steps

**36. Why do eyes sometimes look wrong?**
They are sensitive to alignment, missing evidence and reference pose. A generative prior may produce plausible eyes with incorrect gaze or shape. Sharpening and edge blending cannot restore evidence that is absent.

**37. Why not increase RAM/GPU to solve it?**
More memory enables larger experiments but does not fix a mismatched objective or misleading reference. Hardware limits experiment scale; they do not by themselves explain the measured quality trade-off.

**38. What does the evidence map mean?**
It tells the application how much trust/preservation to assign to observed pixels. It is user-provided guidance, not a learned calibrated confidence that a generated eye is correct.

**39. Is the latest adapter running behind every website button?**
No. The website has multiple pretrained reconstruction paths; research adapters remain separate until their validation passes. A saved research image must be identified by its actual run record.

**40. What is your next experiment?**
Suppress only the known corrupted reference region while keeping valid information, compare against random-region and neutral controls, then investigate a learned input-based gate if that feasibility test supports it. After that, expand identities, training seeds, damage types and final validation.

**41. What parts are your implementation work?**
The local integration/UI, data/provenance checks, mask-refiner controls, compatibility adapter, training modifications, controlled interventions and evaluation/reporting. Pretrained backbones and recognition encoders are attributed to their authors.

**42. Can I reproduce the results?**
The recorded configs, seeds, source/weight hashes, saved checkpoints, generated-output hashes and scoring code support local reproduction. Required photographs, pretrained weights and environments must be restored separately. Some archived outputs have been reproduced byte-for-byte.

## 16. Last-minute revision sheet

| Remember | Answer |
|---|---|
| Main base model | Frozen ReF-LDM |
| Main proposed code | `scripts/target_region_compatibility_v1.py` |
| Main adapter | 10→16→1 MLP, 193 parameters, shared across 16 attention layers |
| Regions | Left eye, right eye, nose, mouth, context |
| Adapter dataset | 16 training identities + 4 observed development identities |
| References | Four per identity in the matched study |
| v2–v5 optimizer/budget | AdamW, learning rate 0.001, 64 updates per arm |
| Training seed | 20261002 |
| Inference seeds | 17, 29, 43 |
| Inference | 512 pixels, 50 DDIM steps, guidance 1.5 |
| Current-family training count | 8 runs, 640 combined updates across distinct arms |
| Latest loss | Diffusion MSE + timestep-scaled cropped FaceNet loss |
| Baseline clean FaceNet / LPIPS | 0.91497 / 0.27725 |
| v5 clean FaceNet / LPIPS | 0.91623 / 0.27743 |
| v5 misleading FaceNet | 0.90414 versus paired baseline 0.90932; 11 pairs |
| What is frozen? | Backbone, VQ encoder/decoder and recognition weights |
| What remains? | Consistent joint quality gains, regional rejection, broader independent validation |

**A safe closing statement:**

> “We have completed a working local application and a controlled series of training and evaluation experiments. The results show that identity and perceptual quality must be assessed together. Our next work focuses on rejecting only misleading reference regions, followed by broader validation. The code, checkpoints and measured results make each step traceable.”

## Evidence used for these notes

[research/FINAL_METHOD_REPORT.md](../research/FINAL_METHOD_REPORT.md) · [research/IDENTITY_BALANCE_RESULTS_V4.md](../research/IDENTITY_BALANCE_RESULTS_V4.md) · [research/IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md](../research/IDENTITY_TIMESTEP_V5_TIMESTEP_SCALED.md) · [research/REFLDM_ENCODING_SCREEN_V2_RESULTS.md](../research/REFLDM_ENCODING_SCREEN_V2_RESULTS.md) · [research/REFERENCE_REJECTION_PROBE_V1_RESULTS.md](../research/REFERENCE_REJECTION_PROBE_V1_RESULTS.md) · [research/FINAL_RESULTS.md](../research/FINAL_RESULTS.md) · [research/PAPER_VS_PROJECT_COMPARISON_20261002.md](../research/PAPER_VS_PROJECT_COMPARISON_20261002.md)

Use dated receipts and result files for exact experiment counts. Older paragraphs in the cumulative project report describe earlier milestones and should not be combined into a single current model or lifetime experiment count.
