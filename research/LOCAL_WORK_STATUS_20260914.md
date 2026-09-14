# Local work checkpoint — 14 September 2026

Public hosting is paused at the owner's request. The Cloudflare tunnel and loopback sharing gateway were stopped; the GitHub homepage no longer points to the temporary address. The local GPU server remains available at http://127.0.0.1:8765/ without sign-in. Start Studio.cmd starts or reuses it after a reboot. No automatic public sharing is installed.

## Current local verification

- Four-reference standard inference completed through the local HTTP API: job 6a46bdd76cd54909b630a105bda472d8, 512 × 512 output, 45.297 seconds including startup and verification, 16.5 seconds inference including offload. All 19,238 masked pixels changed; every pixel outside the effective mask was preserved. The appended record is reference_webapp_repair_checks.json.
- Five additional local HTTP/GPU checks passed: LaMa with exact, expanded and learned masks; ResShift with exact and expanded masks. Results and effective masks were downloaded and verified. The new dated report local_inference_checks_20260914.json preserves the older check report.
- A further browser-driven LaMa run completed (ebee4d2e042c4ce2aa2ff63fcda2ec25). The browser displayed the reconstructed image and enabled comparison; both result and effective-mask download events completed. Inference took 0.69 seconds after loading. The local app was left open on that result.

These are execution and compositing checks. They do not establish that the generated hidden face is correct or that references consistently improve identity fidelity.

## Priorities after local verification

1. Add an independent identity-fidelity evaluator, distinct from the reference-conditioning encoder, to the completed twelve-person diagnostic. Record weights, preprocessing, failed detections and pretraining limitations before interpreting results.
2. Use that evidence together with hole error, perceptual error and visual failure inspection to choose a specific model change. Keep the current working baseline and frozen diagnostic intact; compare changes in a new version rather than overwriting results.
3. Broaden masks and reference variation, then reserve a genuinely unobserved final evaluation. Existing development outputs cannot become fresh confirmation.
4. Revisit external hosting after the local quality work. The in-app browser's opaque-origin form-login issue was still under investigation when sharing was paused. An unfinished local copy is retained in .local-share/share_gateway_opaque_origin_wip.py; it is not an active service or a completed fix.

The website and inference now work locally on the recorded examples. Improved likeness, a novel research contribution and publication readiness remain unestablished.
