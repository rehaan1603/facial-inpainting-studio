# Identity-preserving compatibility audit

2 October 2026. Existing dataset only. Production website and previous experiments remain unchanged.

## Reproduction result

PASS for saved-image metric reproduction: all 12 clean-reference rows (four identities, three arms), plus all three missing FaceNet rows, reproduce within 1e-6; actual maximum numeric difference is zero. Eye-region crop LPIPS also reproduces. All 96 output hashes and both adapter checkpoint hashes match. Receipt: `outputs/compatibility_audit_v2/receipt.json`. This audit rescored saved images, not a fresh 96-image generation. The existing generation receipt separately records eight byte-exact baseline replays.

## Exact configuration

| Item | Setting |
|---|---|
| Backbone | Frozen ReF-LDM author revision af6690c19fdc6421802fd7996510bcfef259bfd1; strict checkpoint loading, EMA copied; model hashes checked against refldm_downloads_v1.json |
| Inference | 512 RGB; 8x64x64 latent; DDIM 50 steps, uniform_trailing, guidance 1.5, bfloat16, seed 17 |
| References | Four separately VQ-encoded images concatenated as latents; differs from author concatenated-width RGB encoding |
| Data | 16 training identities; four observed development identities 263, 6681, 7400, 10149; no reserved final evaluation |
| Damaged target | 512 to 64 to 512 bicubic, Gaussian noise sigma 8; NumPy seed 20260930 with sequential manifest order |
| Original intervention | 545 trainable parameters, reference-value-only bias, 16 attention layers, 64 updates per arm |
| Regional adapter | 193 trainable parameters, 10-to-16-to-1 MLP, 16 layers; degraded-target/reference cosine and magnitude features, consensus, pose/quality proxies |
| Regional training | 128 updates per arm, seed 20261002, AdamW lr .001; .25 global + .75 face-rectangle latent epsilon MSE; zero final layer initialization |
| Corruptions | First of four references: clean, blur radius 4, noise sigma 25, JPEG 15, downsample 32, central gray occlusion, color shift, different-person central patch; three other references unchanged |

## Metrics and detector findings

RGB PSNR/MAE use [0,1] values. LPIPS uses frozen AlexNet and [-1,1] tensors. SSIM uses Gaussian sigma 1.5, 11x11 windows, population covariance and RGB channel averaging. Region SSIM averages only fully contained windows. Official target parsing masks are used only offline; eye LPIPS is a bounding-box crop with ten-pixel context, not strictly pixel-masked LPIPS. Whole-image degradation means visible region is empty.

FaceNet uses frozen VGGFace2 InceptionResnetV1 and MTCNN; ArcFace uses buffalo_l. Three unavailable FaceNet scores all reproduce **multiple-face detections**, not missing files or failed generator calls: identity 263 baseline/JPEG, full/noise, full/JPEG. ArcFace detects one face in each. Do not silently select a face or lower thresholds for these rows. Any alternative aligned-crop diagnostic must remain a separate metric. FaceNet/ArcFace cosine is not accuracy.

## Findings requiring correction in the next experiment

1. Existing regional features already depend on the degraded target. Merely adding target cosine is not a new change.
2. Training has no decoded identity or perceptual objective. Latent epsilon improvement does not guarantee identity preservation.
3. Corruption and timestep are coupled in fixed blocks; balance them independently.
4. `no_target` retains minimum target/reference detection confidence. Label it feature ablation, not completely target-free.
5. Canonical rectangles do not align with individual eye/pose geometry. The whole-context descriptor also overlaps the smaller regions.
6. Prior gate permits -0.1 dB PSNR and -0.005 FaceNet and lacks SSIM/eye safeguards. It does not implement the user's new success requirement.
7. Mean identity scores with different valid supports must not be compared naively. Report paired support and detection rates explicitly.
8. Current mechanism normalizes relative reference weights; it cannot independently reject all references while retaining the baseline self/reference balance.
9. Only one generation seed and four observed identities were tested. Numerical determinism is not evidence of generalization or a meaningful non-inferiority margin.

## Next controlled step

First test differentiable decoded identity supervision on training identities only. Freeze both backbone and recognition encoder. Record memory, loss gradients and neutral equivalence. If feasible, compare the same small adapter with/without identity loss before changing architecture. Use a balanced corruption/timestep schedule, preserve the frozen old protocol, and evaluate identical images across at least three seeds. FaceNet training supervision would make FaceNet a supervised metric; retain ArcFace as a separate diagnostic and disclose this dependence. Broader identity expansion follows a passing development screen. No large training is authorized by an audit pass alone.
