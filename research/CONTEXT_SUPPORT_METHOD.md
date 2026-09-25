# Visible-context mask-support calibration — 25 September 2026

This is an implemented, falsifiable mechanism candidate for fully missing regions. It does not establish novelty or identity recovery. The preceding reference-disagreement and reference-proxy blending mechanisms did not pass their controls. This experiment changes the generator's conditioning mask, rather than blending its output with erased pixels. Generator weights stay frozen.

**Completed outcome:** all 104 calls and 80 scores completed, but the proposed selection lost to the original-mask and equal-cost controls. The progression gate failed. See `CONTEXT_SUPPORT_RESULTS_V1.md` and the visual/coverage interpretation in `CONTEXT_SUPPORT_REVIEW_V1.md`. The protocol below describes the pre-generation design; its negative result does not authorize training or website promotion.

## Mechanism

Given the damaged observation O and a binary missing mask M, form four native-resolution support masks by dilating M by 0, 2, 4 or 8 pixels. Select eight small, dispersed patches of visible context from a common annulus outside the largest support. Placement uses only mask geometry, with a fixed random seed; it cannot see clean targets, gallery images, references, face identity features or evaluation scores.

For each radius, erase both its support and the same probe patches before invoking the frozen generator. Score only the held-out probe RGB against the supplied visible context. Choose the lowest probe error, breaking ties toward the smaller support. Restore the probe context and perform the final generation at that radius. Compose the prediction only into the original requested mask, preserving every other pixel exactly. Insufficient context or failed probe generation triggers a recorded radius-zero fallback.

The hypothesis is that sensitivity to conditioning-mask extent on visible probes predicts which support best reconstructs missing facial features. This assumption can fail: small skin patches may be easy to interpolate while missing eyes or mouths remain wrong. The probe error is **not** a confidence probability or an unbiased estimate of hidden facial error.

## Frozen development screen

The full specification is `protocols/context_support_v1.json`. Four already observed removal cases, two seeds, ResShift at its native 256 pixels, exported with exact 512-pixel context preservation. Fixed radii, random support, four/five-call support averages and a five-seed original-mask average are retained. The proposed selection costs five calls, so both five-call averages control for extra compute. The shared experimental bank requires 104 generator calls and yields 80 comparison rows. No supplied reference photographs enter this single-image screen.

Selections are saved before generating the final candidate bank. The generation process has no clean-target/gallery interface. A separate frozen-metric evaluator measures FaceNet, withheld-gallery identity, ArcFace, masked/visible error, LPIPS, SSIM, PSNR, NIQE, BRISQUE and detection coverage. Seeds are averaged within identity before statistical tests; eight prespecified contrasts use Holm adjustment. Four identities cannot provide a significant two-sided exact sign-flip result at 0.05.

The progression gate is intentionally stronger than a favorable aggregate metric. Both seeds must improve identity and masked error over every fixed radius and both equal-budget controls without material gallery/perceptual regression. Full generation and detection coverage is required. A pass authorizes a separately frozen unfamiliar-development confirmation, not website promotion or a publication claim. A failure is retained without expanding the experiment or training an adapter on this premise. Numerical thresholds are practical development decisions, not validated clinical or perceptual thresholds.

No reserved-final pixels are opened. Images, model weights, per-person generated outputs and reference datasets remain local. Source signatures, numerical results, failures and provenance are shareable.

## Prior-art boundary, checked 25 September 2026

| Source | Existing contribution | Consequence here |
|---|---|---|
| [Noise2Self, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html) | Self-supervised calibration of parameterized denoisers using an independence framework. | Held-out-pixel calibration itself is established. Its noise assumptions do not make our geometric-hole score unbiased. |
| [Restore from Restored: Single-image Inpainting, 2021](https://arxiv.org/abs/2110.12822) | Test-image self-supervised fine-tuning of pretrained inpainting networks using internal self-exemplars. | Adapting from internal image context is established; keeping weights frozen is a distinction, not proof of novelty. |
| [Mask Consistency Regularization in Object Removal, 2025](https://arxiv.org/abs/2509.10259) | Training with dilated and reshaped masks and consistency regularization. | Mask perturbation and mask-shape robustness already have direct prior art. Our inference-time support selection must demonstrate an additional benefit. |
| [Test-Time Degradation Adaptation for Open-Set Image Restoration, ICML 2024](https://proceedings.mlr.press/v235/gou24a.html) | Test-time degradation adaptation and adapter-guided restoration. | Generic test-time calibration/degradation adaptation is not a defensible new claim. |

These primary-source checks establish overlap, not exhaustive proof of absence. The only candidate distinction tested here is an input-only, frozen-generator mask-support decision for missing facial features, with independent identity outcomes and compute-matched controls. Even if that decision helps, a fuller related-work comparison and unfamiliar-identity replication remain necessary.
