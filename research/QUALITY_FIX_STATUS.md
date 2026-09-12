# Face appearance investigation

The user reported incorrect reconstructed facial appearance on 12 September 2026. A completed inference and preserved known pixels do not establish acceptable facial identity or detail.

The SDXL inpainting model card documents reduced sharpness at denoising strength 1.0 and recommends 0.99 in its example. The website now passes 0.99 explicitly. Existing measured runs and the CLI's explicit diagnostic settings remain intact. Source: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (checked 12 September 2026).

A controlled comparison was attempted on the local four-reference example, but the reference environment now fails to import the regex native extension: Windows reports “An Application Control policy has blocked this file.” No comparison output was produced. The configuration correction is therefore not a locally verified quality improvement. No Windows protection was disabled or bypassed.

The server now reports this dependency failure clearly instead of a generic reconstruction failure. A detached launcher also addresses the separate stopped-server problem: Start Studio.cmd launches the server independently and reuses it when already running.

Next verification: resolve the blocked dependency through an approved software/environment repair, rerun the same target/reference comparison, assess masked facial detail and identity separately, and continue the predeclared development diagnostic. Preserve the failed-attempt record and all previous outputs. Neither the configuration change nor an attractive demonstration proves correct recovery of hidden facial features.
