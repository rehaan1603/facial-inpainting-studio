# Local feature correspondence — prototype and limits

Updated 20 September 2026. Development only; the eight reserved-final identities remain unused.

## What is implemented

Each reference and damaged target is detected with five landmarks. An orientation-preserving least-squares similarity transform normalizes translation, scale and in-plane rotation to a 224-pixel canonical face. Eight regions cover image-left/right eyes, nose, mouth, cheeks, forehead and chin. Frozen AlexNet convolutional features are pooled to 3×3 per region (1,728 values); a global ArcFace descriptor remains additional context. Region quality, in-bounds coverage, landmark residual and damaged-region fraction are recorded.

This is genuine regional feature extraction, not relabeling global descriptors. However, in-bounds coverage is only a visibility proxy. Five landmarks cannot solve 3D pose, expression, hair occlusion or detailed eye/mouth correspondence. No clean target or withheld gallery image is used for conditioning or alignment.

Extraction succeeded on 22 of 24 initial damaged cases. The fully removed cases for development identities 1306 and 787 had no unique detected face; they were recorded as failures with no clean-target fallback. The maximum normalized landmark residual among successful cases was approximately 0.1702. This exposes an applicability limitation for extensive missing regions.

## Feature transfer into generation

For the four mixed-damage cases, each reference is warped to the damaged target's geometry and encoded with the frozen native SDXL VAE posterior mean (4×64×64). Eight target-region Gaussian maps select/fuse those spatial features. Four policies are single-best regional quality, equal averaging, quality weighting and damage-conditioned weighting. The last uses local feature compatibility moderated by surviving-region reliability.

After the first unchanged denoising step, the fused native feature contributes 25% of the masked-image conditioning context inside supplied damage. It contributes zero outside damage. Original global FaceID tokens remain present. The intervention preserves the original initial RNG draws. No reference pixels are pasted into the final output; the standard final compositor preserves observed pixels outside damage exactly. The frozen UNet was not trained for this injected context, which is a major limitation.

## Test result and decision

The matched six-policy, two-seed comparison contains 48 scored rows: 32 new generations and 16 reused, hash-verified global controls. See `LOCAL_LATENT_RESULTS_V1.md` and `local_latent_results_v1.json` for all metrics and identity-level contrasts.

Damage-conditioned local fusion reached target FaceNet **0.6846**, compared with **0.6894** for matched global concatenation. Hole MAE increased from **0.09054** to **0.09486**. The paired target difference was **−0.00483**, 95% identity-bootstrap interval **[−0.01808, 0.01186]**, Holm p **1.0**. All 48 outputs preserved known pixels (visible MAE zero); metric coverage was 48/48 for each reported metric. Mechanistic local fusion was achieved, but useful improvement over global descriptors was not demonstrated.

Single-best local conditioning slightly increased target similarity to 0.6925, while worsening hole MAE to 0.09579. This mixed result is not a sufficient training gate. No compact adapter is trained at this stage. Candidate reranking is deferred until restoration is stable, as requested. No claim of improvement over the historical 0.8180 aggregate is justified because that aggregate uses a different damage mixture.

Seven unit tests cover three-state preservation, invalid maps, similarity recovery/degeneracy, spatial weighting, zero-gain passthrough and callback injection. These are not substitutes for the completed real generations. A full-GPU zero-gain replay passed on 21 September: raw/final/hard outputs and input/mask PNGs exactly matched the baseline on one development case. See local_zero_gain_verification_v1.json.

## Visual review

All eight contact sheets, both seeds for each development identity, were inspected on 20 September. Identity 1306: eye/gaze changes and loss of the original teeth-visible expression. Identity 1043: enlarged altered smile/teeth and changed eyes. Identity 2790: altered eye shape and lip expression. Identity 787: enlarged eyes, changed gaze and smoothed central-face texture. Similar failures appear across global and local arms; local transfer does not reliably restore the target's transient expression. This review is descriptive, not a blinded human study. Seed 29 additionally shows altered lip color/closure, enlarged or asymmetric eyes and changed teeth across the six policies. Both seeds show failure to retain the target expression.

## Next defensible direction

Prioritize observed-evidence preservation and a restoration-specific conditioning path over stronger identity forcing. Before training, demonstrate local conditioning benefit with robust correspondence and a matched evidence-preserving control. Add explicit correspondence failure handling, pose/expression stress tests and independent reference-restoration baselines. Keep final identities sealed until method and parameters are frozen.


## 22 September continuation

The 22 September external ReF-LDM comparison does not change the negative local-feature-injection result. True local transfer remains implemented but lacks demonstrated benefit. New work should separate surviving evidence from missing-region synthesis; see NOVELTY_NEXT_EXPERIMENT.md. No fusion adapter has been trained.
