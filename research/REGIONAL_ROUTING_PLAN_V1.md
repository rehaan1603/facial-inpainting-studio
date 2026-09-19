# Regional identity routing — bounded second cycle

The first comparison did not establish superiority for single-reference selection. All-reference conditioning had the strongest target/gallery identity means. This cycle retains all valid references and tests the smallest available spatial-conditioning mechanism, with zero new trainable parameters.

Implementation uses the installed Diffusers 0.40 IPAdapterAttnProcessor2_0 masked branch. It computes attention separately for each reference, multiplies by downsampled spatial masks and sums the contributions. A pipeline proxy supplies `cross_attention_kwargs` without modifying the frozen generator, model weights, attention implementation or first-cycle modules.

The official guide documents image-specific spatial masks: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter . Fractional weighting here follows the inspected installed implementation; its numerical behavior will be tested. Mask routing itself is established functionality, not a new research contribution.

Before inspecting new generated outcomes, freeze `protocols/regional_routing_protocol_v1.json`, implementation hashes, first-cycle case and result hashes, and installed attention/projection/mask-processing source hashes. Reuse the original four development groups, twelve damaged inputs and two seeds. Never open the eight reserved final groups. Six policies yield 144 logical rows: original concatenation, equal per-reference attention, identity weighting, quality weighting, mask-aware global weighting and regional weighting. Original concatenation results can be reused only after exact request/hash verification.

Use the same global descriptors, references, sampler, strength, scale and composition. All routed policies use equal weights outside the marked region. Region maps are constructed from the damaged image's detected geometry only, never clean-target geometry. A missing unique detection falls back to coarse frame geometry with a warning. Per-reference masks form a partition of unity, controlling overall conditioning magnitude. Compare regional to equal weighting to isolate routing from the change in attention normalization; concatenate is a separate existing baseline.

Tests must cover weight conservation, finite/nonnegative weights, single-reference identity, permutation equivariance, distinct regional preference, invalid reference exclusion, actual stock-attention weighted-sum behavior, real GPU inference and unchanged known pixels. Score only after generation, retaining all rows. Report target/gallery identity metrics, distortion metrics, visible failures, and identity-level uncertainty. Keep the method experimental even if execution succeeds.

Only after evidence should a new website mode be enabled. Do not replace the existing default or advertise better likeness. Broader validation and genuine spatial feature transfer remain separate work.
