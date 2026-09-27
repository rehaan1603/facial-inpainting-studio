# Reference/context factorial screen — 25 September 2026

**Numerical progression gate: failed.**

Four already observed removal identities, seeds 17 and 29. Two factors: genuine versus zero reference conditioning; visible-context correction on versus off. ResShift scaffold controls are unchanged. This measures the contribution of each factor, rather than crediting a cascade gain to a new mechanism.

Generation: 32/32 new restorations plus two full GPU equivalence runs and eight reused scaffolds. Evaluation: 40/40. Disabled-correction latent equality: True. Every output preserves original visible pixels: True. Detector/gallery coverage complete: True.

| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ | Mean generation seconds |
|---|---:|---:|---:|---:|---:|
| scaffold | 0.741489 | 0.423009 | 0.062665 | 0.025402 | NA |
| no_reference | 0.758506 | 0.423336 | 0.066654 | 0.028055 | 3.929875 |
| reference | 0.780990 | 0.480085 | 0.068752 | 0.028146 | 3.279250 |
| no_reference_context | 0.753154 | 0.430160 | 0.064710 | 0.027008 | 4.396250 |
| reference_context | 0.781441 | 0.476390 | 0.065988 | 0.027110 | 4.398625 |

Correction adds four decoder/eight encoder passes; it is not equal total compute to an uncorrected cascade. Times exclude model loading and conditioning. The scaffold cost was measured separately in the context-support study.

| Primary contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |
|---|---:|---:|---|---:|
| reference_context_minus_scaffold/facenet_cosine | 4 | 0.039952 | [0.03201897442340851, 0.05144362896680832] | 0.750000 |
| reference_context_minus_scaffold/hole_mae | 4 | 0.003323 | [0.002057882009269218, 0.004490604274529075] | 0.750000 |
| reference_context_minus_reference/facenet_cosine | 4 | 0.000451 | [-0.0176142156124115, 0.018516942858695984] | 1.000000 |
| reference_context_minus_reference/hole_mae | 4 | -0.002764 | [-0.005107957863811238, -0.0007600591929986383] | 0.750000 |
| reference_context_minus_no_reference_context/facenet_cosine | 4 | 0.028288 | [0.00401759147644043, 0.05255764722824097] | 0.750000 |
| reference_context_minus_no_reference_context/hole_mae | 4 | 0.001279 | [0.0007206823354212674, 0.0022782790393729886] | 0.750000 |

Six primary tests were specified before generation. Both seeds are averaged within identity; n=4, not n=8 or n=40. Bootstrap intervals are unstable and the minimum two-sided exact p-value is 0.125. The full JSON retains per-seed gate checks, secondary comparisons, all quality metrics, failures and the descriptive factorial interaction.

The first v1 wrapper attempt failed before sampling because blanket parameter freezing conflicted with the author EMA scope. Review also fixed missing empty hook kwargs and corrected the false statement that the author sample function did not forward the hook. Recovery v2 retains cases, seeds, method parameters and gates. Original records remain intact.

Prior-art boundaries and equations are in `REFERENCE_CONTEXT_METHOD.md`. Neither cascaded restoration, reference conditioning nor known-context diffusion projection is new by itself. This screen does not establish novelty, reliable unknown-person recovery, a publication-ready method, or permission to open reserved-final identities.
