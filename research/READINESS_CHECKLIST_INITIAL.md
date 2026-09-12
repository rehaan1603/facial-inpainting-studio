# Publication Readiness Checklist

This checklist accompanies PUBLICATION_AUDIT.md. Unchecked items represent work still required. Numeric experiment choices are proposed research standards, not automatic IEEE acceptance criteria.

## Audit completed

- [x] Inspect laptop GPU, RAM, active Python, and disk capacity.
- [x] Test active PyTorch CUDA support: failed because installed build is CPU-only.
- [x] Establish that the initial generic reliability-module claim overlaps prior work.
- [x] Identify close comparisons: VCNet, uncertainty feedback, soft-alpha mask correction, portrait-mask robustness.
- [x] Distinguish IEEE Xplore from the actual submission venue.
- [x] Record current IEEE Access APC and candidate ICIP timing.
- [x] Write a conditional research question, experiment plan, and rejection criteria.

## Before substantial training

- [ ] Choose venue route, available publication funding, and practical deadline.
- [ ] Read the remaining nearest full papers, including MAD-paint and recent blind-inpainting work; document precise differences.
- [x] Create an isolated CUDA-enabled environment compatible with the GPU.
- [x] Pass CUDA forward/backward and real baseline checkpoint tests (LaMa engineering export).
- [ ] Verify dataset source, terms, access, and allowable figure/release use.
- [x] Obtain CelebA identity labels and check HQ split overlap; LaPa identity validation remains open.
- [x] Create a pretraining exposure ledger for backbones, parsers, and evaluators; training membership remains unresolved as documented in PRETRAINING_EXPOSURE_LEDGER.md.
- [x] Preserve official splits and create hashed working manifests excluding all exact duplicate groups; near-duplicate checks remain open.
- [x] Verify corruption is generated independently from the supplied erroneous mask in the pilot.
- [x] Ensure pilot correction baselines have equal access to observed RGB.
- [x] Verify no target pixels, target embeddings, or clean parsing enter pilot inference.
- [x] Run pilot mask-polarity, hidden-pixel leakage, regional metric and RGB-mask tests; landmark alignment evaluation remains open.
- [ ] Measure representative training step time and controlled end-to-end latency; pilot VRAM and GPU forward time recorded.

## Pilot decision

- [x] Run accurate masks and controlled under/over-coverage conditions on 32 cleaned validation identities.
- [x] Match true missing area and mask-error area across placement locations (v3 translated geometry, matched textures, and correction margins verified).
- [ ] Tune dilation, feathering, and threshold controls on validation only. Both LaMa and ResShift dilation/feathering grids are complete; learned-score threshold controls remain open.
- [x] Plot completion error against visible-region change (expanded LaMa validation benchmark).
- [ ] Choose practical effect and non-inferiority tolerances before final testing.
- [ ] Continue only if a meaningful gap survives the simple controls.

## Method evidence

- [ ] Compare ordinary segmentation refinement and a generic learned alpha head.
- [ ] Compare corruption augmentation without the proposed method.
- [ ] Match data, trainable capacity where applicable, and tuning budget.
- [ ] Run each claimed component ablation.
- [x] Include a recent executable face-specific or diffusion baseline (ResShift matched development comparison; exposure caveats remain).
- [ ] Address the closest robust-mask method; record unavailable code honestly.
- [ ] Confirm no material accurate-mask regression within the preset tolerance.
- [ ] Test unseen mask errors and unseen occluder assets.
- [ ] Test a second backbone before claiming backbone transfer.
- [ ] Test an external image source before claiming cross-dataset generalization.
- [ ] Report multiple training seeds and paired confidence intervals.
- [ ] Report detector failures, empty metric strata, and difficult examples.
- [ ] If using identity loss, evaluate with independent permitted recognition models.
- [ ] If claiming human preference, run an appropriate blinded evaluation.

## Release and manuscript

- [ ] Regenerate every table and plot from stored raw outputs and manifests.
- [ ] Record model hashes, commits, environment locks, masks, and seeds.
- [ ] Include total compute, memory, latency, and omitted comparisons.
- [ ] Release allowable code, configs, splits, and masks without restricted face-image redistribution.
- [ ] Verify every reference and distinguish preprints from accepted papers.
- [ ] Tie each contribution sentence to a specific experiment and nearest prior work.
- [ ] State synthetic/real, paired/unpaired, and pretraining-exposure limitations.
- [ ] Confirm author contributions, image permissions, venue template, and applicable disclosure rules.
- [ ] Recheck venue fees and submission requirements; submit to one venue at a time.

## Current next action

The area-matched assessment is complete in MILESTONE_04.md: 48 assessment identities and 5,184 model/control rows, after verifying 4,275 source masks. The v2-selected ResShift radius-12/feather-4 control is worse than radius-8/hard blending on all three aggregate v3 metrics. The protocol changes several factors, so this is not causal proof about area matching alone. Next add held-out realistic occluders and finish the pretraining/near-duplicate audit before defining and testing a learned contribution. No novel method or publication claim has been validated.
