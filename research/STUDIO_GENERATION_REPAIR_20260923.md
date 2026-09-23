# Local generation repair — 23 September 2026

This is an application repair and functional audit, not a novelty experiment or a quality benchmark. No clean target was used to select or generate a reconstruction. No new training was performed. Research generators, evaluation scripts, historical outputs and reserved final identities were not changed or accessed.

## Findings and changes

Recent user-upload runs showed LaMa blurring an erased eye band, ResShift synthesizing nearly closed eyes, and 1024-pixel reference inference synthesizing sunglasses. The earlier 512-pixel reference result generated eyes on the same upload. Higher resolution is not a guaranteed quality improvement.

The main studio's missing-area reference path now erases masked RGB to neutral gray and uses full denoising before calling the frozen generator. Final composition uses the original observation. Both all-reference and automatic-selection modes use this path; selection still analyzes the original observation. Partial-damage confidence-map generation and ReF-LDM are unchanged because they intentionally retain damaged evidence. The new wrapper records conditioning and original-input hashes separately.

This changes both masked input and denoising strength, so the audit does not isolate which change removes the sunglasses failure. It also does not establish a general quality improvement. Standard-resolution generated eyes still look different from the original person's possible expression; there is no verified hidden-eye ground truth in this check.

ResShift previously exported the entire downsampled 256-pixel input. It now infers at its native 256 pixels, resizes only the prediction for composition, and exports 512 pixels with visible input pixels preserved exactly. This does not create native 512-pixel generated detail. Mask expansion now means 8 pixels on the same 512-pixel canvas for all main modes (previously 8 native ResShift pixels represented 16 display pixels). Non-square single-image API inputs use an aspect-preserving canvas.

The website relabels the 1024-pixel option as experimental, and explains ResShift's native resolution. The local server was found stopped and restarted using the persistent launcher.

## Verification

- Five real HTTP/GPU runs completed on the failing uploaded eye-band case: LaMa, ResShift, all-reference at 512 and 1024, and automatic reference selection at 512.
- All five export 512-pixel PNGs, alter masked pixels, and preserve every pixel outside their effective masks exactly.
- All five results were visually inspected. Reference outputs generate eyes rather than sunglasses in this check. LaMa remains blurry and ResShift's eye/expression reconstruction remains unsuitable on this large missing region. Those are retained failures, not successful quality results.
- Fourteen HTTP boundary/dispatch tests plus six composition/geometry tests pass; browser JavaScript syntax passes.
- Three additional real browser runs on a second, existing development sample completed (reference, ResShift, LaMa), with HTTP 200 PNG downloads, no browser console errors and exact visible-pixel preservation. These are functional checks, not independent quality validation.
- Numerical receipts: `studio_generation_repair_20260923.json`. Local image comparison: `outputs/studio_quality_audit_v1/http/review.png`. Photos and references remain excluded from GitHub.

## Remaining work

The quality objective is not complete. LaMa is still a small-texture-repair baseline; ResShift and reference models can alter expression, gaze and identity. A broader, separately defined development comparison is needed before promoting preprocessing as a quality improvement. The current correction is not evidence of novelty and does not supersede the negative research results. Missing-region reconstruction still requires a stronger method and independent validation; the final held-out identities remain reserved.
