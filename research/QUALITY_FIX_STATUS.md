# Face appearance investigation

The user reported incorrect reconstructed facial appearance on 12 September 2026. A completed inference and preserved known pixels do not establish acceptable facial identity or detail.

The SDXL inpainting model card documents reduced sharpness at denoising strength 1.0 and recommends 0.99 in its example. The website now passes 0.99 explicitly. Existing measured runs and the CLI's explicit diagnostic settings remain intact. Source: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (checked 12 September 2026).

A controlled comparison was attempted on the local four-reference example, but the reference environment now fails to import the regex native extension: Windows reports “An Application Control policy has blocked this file.” No comparison output was produced. The configuration correction is therefore not a locally verified quality improvement. No Windows protection was disabled or bypassed.

The server now reports this dependency failure clearly instead of a generic reconstruction failure. A detached launcher also addresses the separate stopped-server problem: Start Studio.cmd launches the server independently and reuses it when already running.

The repair and development diagnostic below have now been executed. Independent identity-fidelity assessment remains pending. The failed-attempt record and all previous outputs remain available. Neither the configuration change nor an attractive demonstration proves correct recovery of hidden facial features.

## Executed repair, 12–13 September 2026

A separate reference_env_v2 was installed from official PyPI releases, sharing only the project's CUDA base environment. The modern stack alone still imported blocked regex 2026.9.10. Compatible regex 2025.11.3 restored that import. An additional ml_dtypes 0.6.0 block was resolved with compatible onnx 1.20.1 and ml_dtypes 0.5.4. Diffusers 0.40.0, Transformers 5.17.0, Accelerate 1.15.0, PEFT 0.20.0 and tokenizers 0.23.2 are recorded in the environment lock. The final installer passes pip check, reference imports and CUDA availability checks. No operating-system policy, signing rule or endpoint protection was altered. Legacy experiment environments remain unchanged.

Actual four-reference GPU outputs completed at strength 0.99, at both 512 and 1024 model resolution, on the same identity_916 engineering example. The 1024 call used tiled VAE decoding, exported 512 pixels, and recorded 6,534,925,824 peak allocated GPU bytes and 33.79 seconds including offload. Its first attempt failed on an outdated pipeline convenience-method name; using the VAE's supported tiling methods resolved it. These examples show execution and plausible detail, not correct hidden eyes, makeup, gaze or identity. A strength-only comparison must use the matched modern-stack diagnostic, not compare old-stack and new-stack images as if only strength changed.

The UI could carry the previous person's references into a newly uploaded input. New input upload now clears references and asks for matching photos. Browser verification loaded four references, uploaded a new input, observed zero references and a disabled reconstruction button. Non-reference research samples also clear old selections. Framing changes of the same input retain its references.

Detailed processing is optional; the standard setting stays at 512. Both modes preserve processed known pixels by explicit composition. A real three-reference standard HTTP run and four-reference detailed browser run completed on 13 September. Result and mask browser downloads succeeded. Their measured records are in reference_webapp_repair_checks.json. The standard run took 149 seconds from submission through loading/download verification, of which 19.93 seconds were inference including offload; do not advertise inference-only timing as total waiting time.

Version 2 of the diagnostic stopped after five of 48 candidates when Windows/OneDrive briefly locked progress.json. Version 3 retries atomic record writes and treats unavailable progress updates as nonfatal; required score records still fail explicitly if they cannot be saved. All 48 candidates were regenerated under a new frozen signature. All 96 scored output hashes, twelve identity labels and frozen sources were verified. Failed and completed runs remain separate.

The complete base suite passes 36 tests with 3 OpenCV-dependent skips; all 9 blending tests pass in the reference environment. The HTTP test helper was corrected to omit unsolicited bodies on GET and bodyless requests after Windows aborted those connections. No failed checks were counted as successful.

## What the completed diagnostic says

At strength 0.99 with Poisson composition, reference-on LPIPS is 0.032674 versus 0.032460 with its contribution disabled. The paired difference is +0.000214, with an exploratory 95% identity bootstrap interval spanning zero. Hole MAE is worse with references (0.091549 versus 0.084491). Boundary harmonization has much larger measured effects than the reference adapter in this small study. The evidence does not support a consistent reference benefit or independent identity-fidelity claim. See REFERENCE_DIAGNOSTIC_RESULTS.md for every condition and contrast.

The unresolved task is improving and independently evaluating likeness on varied masks and real damage. The website repair is not equivalent to a novel, publication-ready method.
