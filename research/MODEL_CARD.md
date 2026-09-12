# Local mask-refinement controls

## Purpose and inputs

These checkpoints estimate a replacement mask for a cropped facial image with a possibly inaccurate supplied mask. They do not generate images themselves. Inputs are observed RGB and the supplied binary mask; inference uses a frozen LaMa or ResShift reconstruction model afterward. White means replace. The refiner operates at 128 × 128 and its probabilities are upsampled to the 256 × 256 inference image.

The six checkpoints share a 119,057-parameter convolutional encoder-decoder. Generic training uses region-balanced BCE with negative weight 1; the preservation-weighted control uses weight 4. Seeds are 17, 29 and 43, with matched sampled data per seed and a 1,500-update budget. Training sources are the 24,133 reviewed HQ training images. The original dataset and backbone licenses apply separately.

## Selection and reported performance

The checkpoints are selected on fixed tuning data. Threshold selection also uses tuning data only, minimizing FNR + 4 FPR over five fixed threshold candidates. Seed-17 thresholds are 0.5 for generic and 0.35 for weighted refinement; other thresholds are in the selection JSON files. These thresholds are study settings, not guarantees calibrated for arbitrary deployment images.

FINAL_RESULTS.md and final_results.json contain the complete measurements and paired intervals. Development evaluation shows that the weighted control reduces visible-pixel change while worsening reconstruction compared with generic refinement. Neither weighting nor threshold selection establishes a new method or an overall quality improvement. The final training losses are still decreasing, so the fitted checkpoints are budget-limited controls rather than converged estimates.

## Limitations

- Evaluated at 256 square on prepared face crops. LaPa test framing uses known landmark annotations. Full photographs, multiple faces, arbitrary resolution and automatic face alignment are unsupported by this CLI.
- Thin occluder structures and unfamiliar object appearance can remain unrepaired; false-positive predictions can replace valid content. The saved preselected examples include these failures.
- The model has synthetic supervision and no verified real-captured occlusion performance, independent identity-fidelity evidence, or demographic subgroup analysis.
- Dataset identity and pretraining exposure checks are incomplete. No inference should be interpreted as verified hidden facial identity or forensic recovery.
- Generative backbones remain external pretrained models. The LaMa export is not verified numerically equivalent to the original author model, and ResShift is used through a pinned local wrapper.

## Reproduction

Use the local environment and configs described in README.md. Checkpoint training signatures and file hashes are recorded. Do not overwrite the frozen experiment directory to improve its apparent results; create a new development protocol and reserve fresh evaluation data. No public model distribution has been performed.
