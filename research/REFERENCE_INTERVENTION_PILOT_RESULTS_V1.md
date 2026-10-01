# Reference intervention: implemented pilot and results

Updated 1 October 2026. This is development evidence, not publication-level superiority.

## Completed

- Selected 16 training identities and four separate adapter-check identities from the reviewed training partition. Metadata and existing fingerprints excluded protected identity groups and near-duplicate matches before loading selected images. Reserved final image pixels were not accessed.
- Encoded one target and four independent dataset reference photos per identity. Degraded targets use downsampling plus noise. A controlled central patch from another identity corrupts one reference. References are independently encoded, which differs from the author's concatenated-width encoding and prevents calling these results an exact published-score reproduction.
- Implemented a 545-parameter shared attention-logit adapter across 16 attention layers. The pretrained EMA backbone remains frozen. Four tests pass: neutral and disabled equivalence, reference removal, nonzero adapter gradients, and active bias effects.
- Trained reconstruction-only and intervention arms for 64 steps each. Peak allocated GPU memory was 3546 and 3543 MiB. These are real cached-latent diffusion training steps, without image-space or identity losses.
- Generated 28 images on four check identities and scored all outputs. Preserved missing identity scores and reported paired common-support comparisons.
- Diagnosed an initially weak consistency term: its weighted contribution was a median 0.0184% of the reconstruction term. One follow-up set the weight to 54.4592 using training logs to target a median 10% loss contribution. It ran 64 further steps and generated eight images. The reused check identities are now development data.
- Scored the eight follow-up outputs and four damaged-input controls. Total: 192 optimizer steps, 36 generated images, 40 scored image rows.

## Main results

Four identities, one generation seed, full-image metrics. FaceNet is cosine similarity, not recovery accuracy. All rows below have FaceNet n=4.

| Reference condition | Method | FaceNet higher | LPIPS lower | MAE lower |
|---|---|---:|---:|---:|
| Clean | Baseline | 0.91158 | 0.26544 | 0.04127 |
| Clean | Reconstruction-only training | 0.91557 | 0.26471 | 0.04123 |
| Clean | Initial intervention | 0.91283 | 0.26424 | 0.04113 |
| Clean | Scaled intervention | 0.90869 | 0.26733 | 0.04118 |
| Corrupted | Baseline | 0.90060 | 0.26503 | 0.04157 |
| Corrupted | Initial intervention | 0.89208 | 0.26432 | 0.04152 |
| Corrupted | Scaled intervention | 0.90082 | 0.26872 | 0.04142 |

The reconstruction-only corrupted arm has FaceNet n=3, so comparing its available-case mean against a four-identity mean is invalid. The recorded paired comparison with the initial intervention uses the same three identities. ArcFace has only two valid target identities for these restored comparisons. All detection events remain in the score receipts.

The scaled objective recovers the initial corrupted-reference FaceNet regression, but worsens LPIPS and clean-reference identity. The tiny corrupted-reference identity increase does not establish significance. Neither intervention version justifies promotion to the website. The original baseline improves LPIPS over the damaged input (0.58763 to 0.26544), but pixel MAE worsens (0.04024 to 0.04127), demonstrating a fidelity/naturalness trade-off.

## Interpretation and next decision

The training and evaluation implementation now works locally. The research mechanism has not established a consistent benefit. This small adapter only reads reference value features, so it cannot explicitly compare reference identity or expression against the damaged target. This is a design limitation, not a proven explanation of every failure. More training on the same architecture is not automatically the solution.

Next: specify a target-conditioned compatibility mechanism, check its exact overlap with RefSTAR and newer work, and compare it against a matched ordinary-training control. Broad reference selection remains prior art. Preserve this failed pilot rather than renaming it as a success. Larger cohorts, multiple seeds, varied corruptions, independent evaluation and manuscript work remain incomplete. No overall 90% publication-completion claim is justified.

## Reproduction and evidence

Run the new prepare, cache, train, evaluate, score and analyze scripts with `reference_intervention` in their names in that order. Model/training scripts use the isolated `refldm_env_v1`, while score scripts use `evaluation_env_v1`. Existing output directories deliberately prevent accidental overwrites. The scaled follow-up runs after the initial pilot, followed by its separate scorer. Licensed source images and checkpoints must be restored locally from official sources first.

Public evidence: `reference_intervention_pilot_training_v1.json`, `reference_intervention_pilot_scores_v1.json`, `reference_intervention_pilot_analysis_v1.json`, and `reference_intervention_scaled_scores_v1.json`. Local manifests, checkpoints, image outputs and training traces remain under `outputs/reference_intervention_pilot_v1`. The 24-slide project review in Downloads includes editable diagrams and actual reconstruction examples. The deck remains local because it embeds dataset and user photographs.
