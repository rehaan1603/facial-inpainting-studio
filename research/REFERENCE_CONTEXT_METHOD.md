# Reference restoration with visible-context correction

## Question and mechanism

Can spatial reference restoration correct a completed face's identity while a constraint on surviving original pixels reduces unwanted structural changes? This is an implemented research hypothesis, not established novelty. The previous context-probe, reference-proxy and disagreement policies failed their controls.

ResShift supplies a radius-zero facial scaffold inside missing regions. Its invented eyes/mouth are hypotheses, not observed evidence. ReF-LDM receives that scaffold and four supplied personal photographs. A separate context corrector uses only the original **outside-mask** pixels, never the erased input pixels or clean target. It acts in the final four denoising steps.

Given predicted clean latent z, decode image D(z), substitute original reliable context to form P(D(z)), then calculate:

`z_corrected = z + 0.5 * [E(P(D(z))) - E(D(z))]`.

The difference of encodings isolates context substitution from generic autoencoder round-trip bias. The corrected latent is converted back into epsilon **before the sampler calculates the next state**. No generator weights are updated. Exact outside-mask final composition remains mandatory; internal latent correction is not guaranteed spatially local because the encoder has a broad receptive field.

The author sampler forwards a score-corrector argument to `ddim_sampling()`. The new runner explicitly builds the schedule and calls that loop with `corrector_kwargs={}`. A full GPU zero-gain equivalence check against `sample()` must pass bit-for-bit before the experiment proceeds. An image callback after the transition would not implement this intervention.

The initial v1 attempt failed before sampling: blanket parameter freezing conflicted with the author's EMA registration. Review also found omitted empty hook options and corrected an erroneous statement that the author did not forward the hook. V2 preserves the author's EMA flags, uses evaluation/no-gradient mode and passes the options explicitly. The failed v1 source, protocol and receipt remain separate; no method parameters or outcomes were tuned during this technical recovery.

## Causal controls and gates

The original pre-generation protocol is `protocols/reference_context_v1.json`; `protocols/reference_context_v2.json` records its technical recovery with the same design: four previously observed removal identities, seeds 17/29, 50 DDIM steps and guidance 1.5. Two factors are independently toggled: true/zero reference condition and visible-context correction off/on. The unchanged ResShift scaffold is the fifth arm. This is 32 new restoration calls, eight reused scaffolds and two equivalence calls. Additional VAE computation is measured, not represented as free or equal total compute.

The independent evaluator retains FaceNet target/gallery, diagnostic ArcFace, LPIPS, SSIM, PSNR, hole/visible MAE, NIQE, BRISQUE and all detector/runtime failures. Seeds are averaged within identity for six prespecified primary tests with Holm correction. Missing scores are not imputed or silently discarded. All cases require visual review for gaze, mouth, teeth and eye geometry.

The candidate must improve both identity and hole error over the scaffold and plain reference cascade on both seeds, retain gallery/perceptual fidelity, and show benefit over the reference-disabled correction arm. These are exploratory progression requirements. Passing would permit a separately frozen unfamiliar-development confirmation; it would not establish statistical superiority, a first-in-literature contribution, a training decision or website promotion. Reserved-final identities remain untouched.

## Prior-art boundaries checked 25 September 2026

| Primary source | Overlap and implication |
|---|---|
| [ReF-LDM](https://arxiv.org/abs/2412.05043) | Multi-reference spatial restoration is an existing author model. Its source/weights are credited and pinned. Our cascade does not make its reference attention a new contribution. |
| [Reference-Guided Large-Scale Face Inpainting](https://arxiv.org/abs/2303.07014) | Identity and component-level texture control for missing faces already exist; reference-guided completion is not new by itself. |
| [RePaint](https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf) | Known-context constraints during diffusion are established. |
| [ReSample](https://arxiv.org/abs/2307.08123) | Latent diffusion inverse problems already use data consistency during reverse sampling. Our encoding-difference correction is not a reproduction of its optimization/resampling method. |
| [InstantIR](https://arxiv.org/abs/2410.06551) | Dynamic generative references and changing restoration conditioning during sampling already exist. A hypothetical feedback extension cannot be called novel merely because it updates a scaffold. |
| [SSDiff](https://arxiv.org/abs/2510.12114) | Pseudo-reference structure and staged, region-specific guidance already address damaged old-photo faces. Coarse-to-fine guidance is not a new general idea. |

The candidate distinction is the measured interaction between an explicitly disposable completion scaffold, actual personal references and reliable original context in a frozen restoration pipeline. Its usefulness must survive the factorial controls and independent replication. The scaffold is still used as low-quality conditioning in this first screen; the method does not yet solve the risk of entrenching its invented expression. No exhaustive literature-priority claim is made.

All images, weights and private per-person features stay local. GitHub contains code, protocols, numerical receipts, sources and honest limitations.
