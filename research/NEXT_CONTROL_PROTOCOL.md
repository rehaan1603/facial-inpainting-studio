# Stronger controls before a learned method

This is a prospective development plan. These experiments have not run. The current matched comparison keeps the original v2 protocol unchanged.

## Extend the search on tuning data

Use the same 48 tuning identities for both backbones, retaining all three corruption families and five error conditions. Evaluate dilation radii 0, 2, 4, 8, 12, 16, 24. If the minimum remains on a search boundary, report the boundary rather than calling it optimal. Select one global radius per backbone using mean full-face LPIPS. Keep the 48 assessment identities out of selection.

For each radius, compare hard compositing with an inward feather of width 2, 4, or 8 pixels. Define alpha from the distance inside the expanded supplied mask, capped at one; alpha must be zero outside that mask. Blend the model completion and observed image using that alpha. Generate model output with the expanded hard mask, so the feathering control changes only compositing. Use identical random seeds for diffusion alternatives within each case. Never construct alpha from the true occlusion or clean semantic labels.

Retain all controls in a reconstruction-versus-visible-change plot. LPIPS alone may reward changes to visible pixels, so do not select or describe a preservation method solely using that metric. Choose application tolerances before final testing, with a stated rationale.

## Area-matched diagnostic

Construct new corruption cases with matched true missing area across eye, mouth, and nonsemantic locations, and matched false-positive and false-negative mask area across error strata. Define feasible area bins and rejection rules before sampling. Record actual fractions and skipped cases. Do not resize all masks blindly: clipping at image boundaries changes coverage. Retain these as a separate protocol version; do not mix their results into v2 tables.

## Decision gate

Proceed to a learned correction only after these controls are measured. Compare against a generic segmentation refiner and a learned alpha head with the same observed RGB and supplied mask, training examples, and tuning budget. Neither clean target RGB nor its parsing may enter inference. Any proposed training loss may use training targets, but an independent evaluation must measure its claimed behavior.

Do not use the final test split for debugging, choosing radii, or setting tolerances. Pretraining exposure and identity contamination remain separate unresolved checks even when our own tuning and assessment identities are disjoint.
