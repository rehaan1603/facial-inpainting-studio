# Reference-proxy blend calibration

Implemented 23 September 2026. This is a candidate preservation mechanism, not a proven novel restoration method.

## Why change the signal?

The reference-disagreement diagnostic had harmful-edit AUC 0.417–0.533. Agreement between restored outputs does not imply that an edit is correct; a model can make the same error under all reference subsets. The revised mechanism instead obtains known-error examples by synthetically damaging a supplied reference photograph. That reference is held out of proxy conditioning, so the restorer cannot condition directly on the proxy's undamaged original.

## What is learned?

For each person, five fixed damage types are applied to reference 1. References 2–4 condition a frozen ReF-LDM model. At each sampled masked proxy pixel, let `d = restored - damaged` and `e = original_reference - damaged`. A nine-coefficient linear rule predicts a restoration weight from an intercept and eight local statistics of the damaged/restored pair. Weighted ridge least squares minimizes the squared RGB error of `damaged + weight * d`, with coefficient penalty 0.01 and no intercept penalty. A separate scalar control estimates one global weight using the same proxy supervision.

At target inference, coefficients act on the damaged target and its existing all-reference restoration. Weights are clipped to [0,1] and spatially smoothed within the supplied mask; unmasked observed pixels are exact. Neither clean target, target degradation label nor withheld gallery is an input. Coefficients are mixing weights, not confidence probabilities. Actual target generation still uses four references; proxy restoration uses three because of the holdout.

This is a fitted postprocessing rule. The generator, VAE and reference-conditioning network remain frozen. It does not transfer reference pixels into the target or train a local fusion adapter.

## Required controls and limits

The frozen screen compares observed input, full restoration, fixed 50/50 blending, proxy-global blending and proxy-spatial blending. Primary comparisons are spatial versus fixed/global on identity and masked MAE, with four Holm-adjusted identity-unit tests. All other quality metrics and failures remain visible. Four previously observed mixed-damage cases are exploratory only.

Proxy-to-target distribution shift is unresolved. Only one calibration view per person is used; its illumination, expression and texture can differ from the target. Reusing aligned mask coordinates can miss corresponding semantic regions. The medium proxy bank matches the existing development regime, so apparent success would not establish adaptation to arbitrary damage. Missing-region inpainting remains a separate unsolved problem for this restoration backbone.

## Prior-art boundaries checked on 23 September

- [SAIR: Self-Supervised Face Image Restoration with a One-Shot Reference](https://arxiv.org/abs/2203.03005) already uses a supplied reference to constrain restoration semantics. Self-supervised reference restoration is not a new general claim.
- [Personalized Restoration via Dual-Pivot Tuning](https://personalized-restoration.github.io/) already personalizes restoration from a few reference photographs by tuning the prior and guidance components.
- [Test-Time Degradation Adaptation](https://arxiv.org/abs/2312.02197) already adapts to unknown degradation during testing. Test-time adaptation is not a new general claim.
- [WaveFace](https://arxiv.org/abs/2403.12760) and [WaveFreqAnchor](https://arxiv.org/abs/2608.06717) constrain restoration through frequency decomposition/anchoring; simple low-frequency preservation would not establish our novelty either.

The narrow candidate distinction is cross-reference proxy supervision of a spatial edit-reliability rule while the restorer remains frozen. Whether that distinction is original enough and useful beyond simple blending remains to be demonstrated. This source check is not an exhaustive proof of absence of equivalent prior work.
