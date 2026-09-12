# Project status

Updated 13 September 2026. The project contains measured single-image experiments and a working local reference-assisted inference baseline. On 11 September, the user expanded the requested functionality to three or four reference photographs. This expands the implementation scope; it does not retroactively change the input setting of earlier experiments. The four-reference website flow has now generated an image and passed result/mask download checks; research/reference_webapp_check.json records exact outside-mask preservation. The user requested a private GitHub checkpoint and local hosting at this stage.

## Implemented and measured

| Stage | Verified evidence | Interpretation |
|---|---|---|
| Data and environment | CUDA on the RTX 5070 Laptop GPU; all 52,168 supplied face images decoded and hashed; 496 proposed near-duplicate pairs reviewed; current manifests retain 29,926 HQ and 21,994 LaPa images | Integrity and photograph-similarity screening, with remaining identity/pretraining limitations |
| Initial single-image study | Pilot, 96-identity development protocol, simple mask controls, area-matched assessment; six 119,057-parameter refiners trained for 1,500 updates each; 13,824 learned-control development rows | Frozen LaMa and ResShift generators; no generative backbone training |
| Initial object-composite test | 5,632 rows from 128 images: 64 HQ identity labels and 64 LaPa images, using 12 photographed object cutouts | Completed, already observed synthetic-composite test; not a fresh test for subsequent methods |
| Training and compositing extension | All six refiners completed 6,000 total updates; 27,648 development evaluation rows; hard and segmentation-score compositing | Completed follow-up on the same 48 assessment identity labels; not new test evidence |
| OSOR comparator | Verified downloads and a functional inference check; six of 432 comparison cases saved, containing 24 metric rows | Partial comparison paused; no complete ranking or comparative conclusion is available |
| Reference-assisted CLI | Existing SDXL inpainting + FaceID Portrait adapter + InsightFace; saved 512×512 outputs from three- and four-reference runs | Functional baseline integration, with no local training of the reference generator or adapter and no evaluated novelty claim |
| Reference development diagnostic | All 48 candidates and 96 scored rows completed on twelve preselected development identity labels; 2 × 2 scale/strength comparison, each with hard and Poisson composition | Reference benefits are inconsistent; no independent identity-fidelity measurement or final-test claim |
| Local website | LaMa/ResShift HTTP/GPU checks plus actual four-reference upload, generation, optional blending and download checks | Functional website checkpoint; reconstruction quality is not a verified identity-fidelity claim |

The reference output files were checked on 12 September: both modify the intended region and preserve every pixel outside the effective mask at the processed 512×512 resolution. This is a compositing property, not evidence that hidden features are correct. Details, artifacts, timings, and limitations are in [REFERENCE_IMPLEMENTATION.md](REFERENCE_IMPLEMENTATION.md).

## What the evidence supports

The first-stage weighted refiner reduces visible-region change while worsening aggregate hole reconstruction and LPIPS versus the generic refiner. The original object-composite test repeats that direction on both datasets and backbones. Those results remain intact in [FINAL_RESULTS.md](FINAL_RESULTS.md); they describe the 1,500-update protocol.

The extension changes the interpretation. At 6,000 updates, weighted refinement has lower aggregate LPIPS and hole MAE than generic refinement on both backbones, but slightly greater visible-region change. Compared with its own 1,500-update version, the extended weighted refiner worsens LaMa reconstruction, while the predefined ResShift seed improves all three reported means. Generic refinement worsens completion after extension on both backbones. The blanket first-stage conclusion that weighted refinement always worsens completion must therefore not be applied to every training budget.

Segmentation-score compositing reduces visible-region change but increases hole MAE in every reported extension setting; whole-image LPIPS moves in different directions across backbones. It is an evaluated simple control, not a reconstruction-trained alpha model. Full paired intervals and training histories are in [TRAINING_EXTENSION_RESULTS.md](TRAINING_EXTENSION_RESULTS.md). Lower tuning loss alone did not establish better reconstruction or convergence.

Reference-assisted runs establish that real inference can execute with three or four photographs on this laptop. They do not establish improved identity fidelity, robustness, a novel method, or publication readiness. The selected validation examples are functional demonstrations. Their same-person grouping follows supplied identity labels and their pretraining overlap is unresolved. Optional Poisson boundary harmonization is established image processing and must be reported as a separate postprocessing choice.

## Remaining work

1. Broaden website failure-case and real-photo quality evaluation beyond the recorded successful flows. The current four-reference reconstruction, downloads and exact outside-mask checks are complete.
2. Finish or explicitly delimit the paused OSOR comparison. Do not treat six completed cases as the planned 432-case study.
3. Define and test a contribution against the closest relevant methods. Multi-reference inpainting, reference selection, reliability gating, identity losses, and Poisson blending already have precedents. [REFERENCE_METHOD_AUDIT.md](REFERENCE_METHOD_AUDIT.md) proposes falsifiable experiments, not established originality.
4. The twelve-identity development diagnostic is complete under the frozen v3 protocol; all source signatures and 96 output hashes were verified. A separate final protocol still needs reserved identities and stronger transformed-copy checks. Earlier observed test cases cannot become untouched confirmation.
5. Measure reference benefits and failures with matched no-reference and reference baselines, ablations, paired identity-level uncertainty, independent identity evaluation where permitted, and wider corruption/real-capture evidence. Two functioning examples are insufficient.
6. Resolve or disclose pretraining exposure, data permissions, attribution, and release limitations. Select a specific venue, human authorship/contributions, funding where needed, bibliography, and submission format. IEEE Xplore is the library rather than the submission venue.

The local implementation and evidence base have advanced beyond the original planning document. No novel reference model has been trained, the completed reference development diagnostic does not establish consistent improvement or identity fidelity, and no article has been submitted or accepted. The original technical manuscript remains historical until it is revised around the final supported contribution.

## Latest repair and diagnostic interpretation

The website uses the repaired reference_env_v2, with package pins and a reproducible setup script. Windows security settings were not changed. New input uploads clear old reference photographs. Standard 512 processing and optional 1024 processing both export 512 PNGs; 1024 is an engineering option, not a validated identity-quality upgrade. Progress-file locks now retry and cannot abort a valid reconstruction merely because progress text cannot be saved. Three-photo standard and four-photo detailed website runs completed; the detailed result and mask passed browser download checks.

At the website strength 0.99 with Poisson composition, mean LPIPS is 0.032674 with references versus 0.032460 with reference conditioning disabled. The paired difference is +0.000214 (exploratory 95% identity bootstrap interval -0.004225 to +0.003962). Hole MAE is 0.091549 versus 0.084491. These results do not support claiming a consistent reference benefit. Hard-composed strength 1.0 has a small favourable reference LPIPS contrast, but it does not establish identity preservation. Poisson composition improves the measured reconstruction/boundary errors over hard paste in these cases. Full results: REFERENCE_DIAGNOSTIC_RESULTS.md.
