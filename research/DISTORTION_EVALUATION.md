# Synthetic distortion implementation

`src/degradation` constructs new evaluation cases without editing source photographs. Implemented types: removal, Gaussian blur, horizontal motion blur, Gaussian noise, JPEG compression, resolution loss, combined resolution/blur/noise/JPEG and illumination reduction. Every type has mild, medium and severe parameter presets; fixed seeds reproduce outputs.

Masks support rectangles, irregular strokes and coarse eye, nose, mouth, half-face and central-face regions. These are landmark-based geometric approximations, not semantic segmentations. Outside the supplied mask, bytes remain identical. Clean-image landmarks are used solely for synthetic case construction. Only the damaged RGB, mask and reference photos reach the inference interface.

The first comparison evaluates **medium** eye removal, mouth removal and central-face mixed degradation, with generation seeds 17 and 29. The other types/severities have deterministic implementation tests but are not yet a complete GPU robustness benchmark. Global degradation is supported by an explicit all-image mask at the corruption layer, but the reused benchmark metric contract requires known pixels; global-restoration evaluation is deferred.

Mixed degradation retains some degraded image evidence, but the current inpainting backbone primarily replaces the masked region. It is not a purpose-trained deblurring or blind restoration system. Supplied perfect synthetic masks do not establish automatic damage localization or robustness to real captured degradation.

The fixed generation strength is 0.99. This replacement-heavy setting preserves little opportunity to exploit partially damaged details inside the mask. Comparing restoration-specific strengths or residual conditioning is a future development experiment, not a result of this cycle.

Each generated case records parameters, source-role hashes, mask hash, observed-image hash and exact known-pixel preservation. Evaluation targets and galleries remain separate from conditioning references.
