# Reference-intervention preflight

30 September 2026. Development feasibility, not a trained contribution.

## Literature gate: broad proposal overlaps

The [RefSTAR full text, sections 3.1–3.3](https://arxiv.org/html/2507.10470v1) is now accessible through arXiv. It already supervises reference-region selection using consistency masks, introduces dual-stream transfer, and uses mask-compatible cycle consistency. Its data engine grows an initially annotated subset into a larger curated collection. Thus supervised regional reliability is not an adequate distinction from RefSTAR. This supersedes the earlier PDF-access limitation in the feasibility document.

A narrower hypothesis is paired intervention supervision: hold the damaged target, diffusion noise and timestep fixed; alter a known region of one reference; penalize harmful output changes while retaining reconstruction supervision. Include wrong-person substitution separately from local blur/occlusion. Unlike a consistency-mask classification target alone, the intervention objective constrains the effect on the prediction. Whether that precise objective is already covered elsewhere remains unresolved. Do not call it novel until that search and a comparison with RefSTAR-style supervision are complete.

Never label a clean but different pose/expression as globally irrelevant. A synthetic corruption label establishes which region was altered, not which reference content is causally useful. Test over-suppression: a model that ignores every reference can trivially achieve invariance and must fail the usefulness criterion.

## Actual local hardware experiment

Implemented scripts/refldm_adapter_memory_probe.py. It loads checksum-verified author UNet weights, freezes them, computes reference key/value caches and inserts a neutral-initialized value-cache adapter with 273 trainable parameters. Synthetic 64 by 64 latents match the 512-pixel image latent grid. It performs an optimizer update through the real UNet using bfloat16 autocast and float32 parameters. It reads no dataset photographs and saves no model checkpoint.

| Probe | Peak tensor memory | Peak allocator reservation | Result |
|---|---:|---:|---|
| One reference, initial probe | 1626 MiB | 1732 MiB | Finite gradients and parameter update |
| Four references, initial probe | 2975 MiB | 3248 MiB | Finite gradients and parameter update |
| Four references, added integrity checks | 2975 MiB | 3248 MiB | Neutral output exactly matches baseline; no backbone gradients; adapter updates |

The verified four-reference backward/update section took about 0.38 seconds; this is a single smoke measurement, not throughput benchmarking. The synthetic loss is random-target MSE and has no reconstruction-quality interpretation. Initial receipts identify the initial script hash; the verified receipt identifies the revised script with extra checks. All receipts are retained.

These measurements exclude VQ image encoding/decoding, image-space losses and a paired intervention branch. They are a lower-bound feasibility result. Full training memory and numerical stability remain unverified. Multiplying cached values is a convenient gradient-path probe, not a validated attention-suppression mechanism: keys still affect normalization, so zero values do not remove reference influence. The production implementation needs an explicit attention-weight design and its own intervention-off equivalence checks.

## Next experiment specification

1. Design the actual attention intervention and check closest prior objectives before selecting it. Keep the existing production pipeline and frozen experiments untouched.
2. Construct a training-only manifest from official identity metadata with explicit disjointness checks against development and reserved final identities. Each target needs independent references; no angle synthesis from that target. Split whole identities before any augmentation. Do not load reserved image pixels.
3. Implement cached frozen image encoding with identity/split/source provenance. Profile one complete paired training step, including the selected losses, before a training run. Account for cache disk size and stale-cache detection.
4. Development arms: unmodified ReF-LDM; equal-capacity adapter with reconstruction loss only; adapter with reference-region supervision only; adapter with paired intervention supervision. Keep reference counts, noise draws, training examples and optimization budget matched. Include a reference-disabled control to detect trivial invariance. Do not pretend the mask-supervision arm reproduces the complete RefSTAR system.
5. Predefine useful-reference sensitivity and corrupted-reference stability together. Improvement on stability alone is insufficient. Evaluate on fresh identities only after development decisions freeze; use the paired identity-level metrics and guardrails in REFLDM_IMPROVEMENT_FEASIBILITY_20260930.md.

Current status: hardware gradient path passed; full training not run; proposed objective not implemented; novelty and superiority unproven. No website default changed.
