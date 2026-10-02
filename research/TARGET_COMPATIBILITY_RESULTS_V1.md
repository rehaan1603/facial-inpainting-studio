# Target-conditioned regional compatibility: development pilot

Date: 2 October 2026. Existing CelebA-HQ research only; new Human Faces work is set aside.

**Task: full-image blur/noise restoration, not erased-region inpainting.** Four already-observed development identities, one seed. No independent or reserved-final evaluation. This is not publication-level superiority evidence.

## Completed implementation and execution

- Frozen ReF-LDM with a 193-parameter regional attention-log-prior network in 16 attention layers. Features combine observed degraded-target/reference latent similarity, pose and quality proxies, and reference consensus.
- Two arms trained on the same 16 identities for 128 updates each; backbone weights frozen. Neutral initialization reproduced baseline predictions exactly.
- 96 generations: four identities, three methods, eight reference conditions. Eight archived clean/wrong-patch baseline replays are byte-exact.
- Conditions: clean, blur, noise, JPEG, downsampling, occlusion, lighting and wrong-person patch. Genuine pose mismatch remains untested as a controlled intervention.
- All rows and failures retained. The `no_target` arm ablates explicit target similarity/pose; minimum target/reference detection confidence remains, so it is not strictly target-free.

## Declared engineering gate

{'baseline': 'INCONCLUSIVE', 'no_target': 'INCONCLUSIVE'}

**Decision: do not promote or expand this candidate.** LPIPS improvement is below the declared threshold. Three FaceNet measurements are unavailable (263: baseline/JPEG, full/noise, full/JPEG); partial-support identity deltas below are descriptive, not complete matched identity evidence. All 96 images were generated and scored for available metrics; failed identity detections are retained.

Requires corrupted-reference LPIPS delta <= -0.005, PSNR delta >= -0.1 dB, FaceNet delta >= -0.005, at least three of four identities improving LPIPS, and complete paired support. Passing would only justify further development.

| Contrast / stratum | LPIPS delta | Relative LPIPS reduction | PSNR delta (dB) | FaceNet cosine delta | LPIPS improved/worsened |
|---|---:|---:|---:|---:|---:|
| full minus baseline/clean | -0.000670 | +0.252% | +0.0051 | -0.004883 | 3/1 |
| full minus baseline/corrupted | -0.000128 | +0.047% | +0.0058 | -0.005302 | 3/1 |
| full minus no_target/clean | -0.001765 | +0.662% | +0.0157 | -0.005040 | 3/1 |
| full minus no_target/corrupted | -0.000124 | +0.046% | +0.0013 | -0.002587 | 2/2 |

Negative LPIPS delta is better; positive PSNR/identity delta is better. Conditions are averaged within each person before the four-person comparison. Identity-cluster bootstrap intervals, medians, standard deviations and paired support are in the JSON; they are descriptive with only four observed people.

## Per-condition whole-image means

| Method / condition | LPIPS | PSNR | SSIM | FaceNet | ArcFace diagnostic | NIQE | BRISQUE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline/clean | 0.26544 | 24.04671 | 0.63723 | 0.91158 | 0.84569 | 4.96262 | -8.07684 |
| baseline/blur | 0.26500 | 24.16013 | 0.64533 | 0.90991 | 0.84326 | 5.04556 | -7.73578 |
| baseline/noise | 0.29025 | 23.97733 | 0.62428 | 0.89417 | 0.84565 | 5.18339 | -8.40009 |
| baseline/jpeg | 0.27260 | 24.01741 | 0.63354 | 0.91372 | 0.84326 | 4.89575 | -8.03314 |
| baseline/downsample | 0.26786 | 24.12649 | 0.64325 | 0.91100 | 0.85000 | 5.07104 | -8.32404 |
| baseline/occlusion | 0.26742 | 24.01935 | 0.63607 | 0.91167 | 0.85159 | 4.92911 | -8.05951 |
| baseline/lighting | 0.26418 | 23.95851 | 0.63592 | 0.91025 | 0.84601 | 4.95408 | -8.33185 |
| baseline/wrong_patch | 0.26503 | 24.00866 | 0.63551 | 0.90060 | 0.83343 | 4.87340 | -7.90302 |
| no_target/clean | 0.26653 | 24.03612 | 0.63665 | 0.91174 | 0.84662 | 4.93114 | -8.14508 |
| no_target/blur | 0.26465 | 24.16124 | 0.64559 | 0.90834 | 0.83904 | 5.01287 | -7.57684 |
| no_target/noise | 0.29038 | 23.97938 | 0.62493 | 0.89837 | 0.85763 | 5.11429 | -8.21503 |
| no_target/jpeg | 0.27210 | 24.02147 | 0.63391 | 0.89549 | 0.84935 | 4.89956 | -7.84210 |
| no_target/downsample | 0.26796 | 24.12653 | 0.64318 | 0.91092 | 0.83896 | 5.08612 | -8.21039 |
| no_target/occlusion | 0.26674 | 24.03033 | 0.63644 | 0.90441 | 0.84456 | 4.94268 | -8.28351 |
| no_target/lighting | 0.26486 | 23.95471 | 0.63655 | 0.90993 | 0.84290 | 4.95235 | -8.42487 |
| no_target/wrong_patch | 0.26562 | 24.02552 | 0.63618 | 0.89410 | 0.83520 | 4.90846 | -7.79816 |
| full/clean | 0.26477 | 24.05182 | 0.63730 | 0.90670 | 0.85386 | 4.93491 | -8.08893 |
| full/blur | 0.26490 | 24.16485 | 0.64570 | 0.89843 | 0.83916 | 5.10370 | -7.83588 |
| full/noise | 0.28974 | 23.98482 | 0.62510 | 0.90322 | 0.84252 | 5.20463 | -8.07452 |
| full/jpeg | 0.27322 | 24.01404 | 0.63305 | 0.90660 | 0.84881 | 4.92091 | -7.92963 |
| full/downsample | 0.26802 | 24.12216 | 0.64262 | 0.90625 | 0.83476 | 5.05340 | -8.20380 |
| full/occlusion | 0.26614 | 24.04378 | 0.63644 | 0.91361 | 0.84735 | 4.94200 | -8.03912 |
| full/lighting | 0.26438 | 23.96062 | 0.63628 | 0.90916 | 0.84505 | 4.91339 | -8.16644 |
| full/wrong_patch | 0.26503 | 24.01822 | 0.63581 | 0.89518 | 0.84221 | 4.87119 | -7.70245 |

## Facial-component checks (all seven corrupted conditions)

| Method | Region | MAE | PSNR | Bbox LPIPS | Scored rows |
|---|---|---:|---:|---:|---:|
| baseline | eyes | 0.113743 | 16.284245 | 0.117812 | 28 |
| baseline | nose | 0.034850 | 26.420421 | 0.146716 | 28 |
| baseline | mouth | 0.061302 | 22.062114 | 0.197231 | 28 |
| baseline | other | 0.040947 | 24.162880 | 0.270333 | 28 |
| no_target | eyes | 0.113194 | 16.335973 | 0.117388 | 28 |
| no_target | nose | 0.034872 | 26.436068 | 0.145760 | 28 |
| no_target | mouth | 0.061178 | 22.090219 | 0.197627 | 28 |
| no_target | other | 0.040921 | 24.165153 | 0.270330 | 28 |
| full | eyes | 0.113064 | 16.344839 | 0.118043 | 28 |
| full | nose | 0.034820 | 26.436577 | 0.145819 | 28 |
| full | mouth | 0.061036 | 22.104337 | 0.196847 | 28 |
| full | other | 0.040919 | 24.166326 | 0.270206 | 28 |

Components use official target annotations only during offline scoring. Bbox LPIPS includes ten-pixel context and is **not strictly masked LPIPS**. Fully contained 11x11 SSIM windows are required; thin regions may have no valid SSIM. All pixels are degraded here, so the visible region is empty and masked equals whole-image.

## Loss-gradient audit

| Arm | Loss | Weight | Mean raw value at audit steps | Mean weighted gradient norm |
|---|---|---:|---:|---:|
| no_target | reconstruction | 0.25 | 0.106327 | 0.00000492 |
| no_target | face_roi | 0.75 | 0.105494 | 0.00002612 |
| full | reconstruction | 0.25 | 0.106332 | 0.00000703 |
| full | face_roi | 0.75 | 0.105501 | 0.00003718 |

These are latent noise-prediction losses, not image-space reconstruction/identity losses. Eight gradient audits per arm record raw/weighted terms and gradients. Image perceptual, identity, visible-preservation and reference-consistency training losses remain unimplemented in this pilot; no claim that all requested losses have been tested.

## Visual inspection

Inspected 263/wrong_patch, 6681/noise, 7400/occlusion and 10149/blur comparison sheets. Candidate/control differences are subtle; smoothing and altered eye, mouth or fine facial details remain relative to the target. No consistent visual fidelity gain was apparent. All 32 sheets are saved, but this does not claim all 32 received human review.

## Limitations and next decision

- Corruption type and diffusion timestep blocks are coupled during training; the follow-up must balance them independently.
- Adapter regions are canonical rectangles, not pose-aligned anatomical regions. Pose features are 2D proxies, not measured 3D pose.
- Full, target-feature-ablated and baseline runs are complete. Global, target-only, reliability-only and other component ablations are not complete.
- Reference selection already has close prior art; see [ten-paper literature and protocol matrix](MATCHED_COMPARISON_AND_LITERATURE_20261002.md). This module is not established novelty.
- If either gate fails, keep the baseline deployed and diagnose on development data. Do not expand to reserved final identities or represent a failed pilot as an improvement.
- Still required: a justified image-space objective, fully target-free control, matched inpainting panel, balanced training, multi-seed independent validation after a passing pilot, and final manuscript/human assessment.

Numerical source: [all scores and paired statistics](target_compatibility_results_v1.json). Local comparison images are in `outputs/target_compatibility_v1/comparisons`; dataset photos are not distributed in the repository.
