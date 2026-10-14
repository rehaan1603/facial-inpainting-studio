# Spatial Q/V prerequisite results

14 October 2026. Training prerequisites only. No new reconstruction-quality claim, checkpoint, training run or application change.

## What was implemented

An isolated rank-4 query/value residual at one late SDXL self-attention layer adds 10,240 parameters to the existing 385-parameter scorer (10,625 total). Native projections remain frozen. The operation changes only mask-selected attention outputs, although later network layers can propagate changes outside those locations. Final image compositing remains responsible for exact visible-pixel preservation.

V1 uses ordinary adapted-minus-native attention. V2 computes the same first-order derivative with local subtraction of native-input gradients before returning them to the backbone. Both remain active at zero B. No zero-weight bypass or relaxed tolerance was introduced. Both are experimental and untrained.

## Tests and full-model probes

| Check | V1 | V2 |
|---|---:|---:|
| Synthetic module checks | 9 passed | 9 passed |
| Additional derivative checks | Not applicable | 2 passed, including GPU and finite differences |
| Full-model enabled-zero prediction maximum difference | 0 | 0 |
| Original scorer gradient maximum absolute difference | 1.811786205507815e-7 | 1.811786205507815e-7 |
| Fixed atol1e-7 / rtol1e-4 gradient gate | Fail | Fail |
| Peak GPU allocation (bytes) | 7,338,135,040 | 7,338,135,040 |
| Supervisor wall time (seconds) | 75.375 | 79.359 |
| New optimizer steps / checkpoints | 0 / 0 | 0 / 0 |

Both full-model probes stopped at the gradient gate, before structural backward and the disposable optimizer step. Consequently the full-model SSIM gradient, optimizer behavior and new-model image quality remain unmeasured. CUDA backend flags were restored. Neither probe ran out of memory or timed out. Original checkpoint/source pins remain verified. A successful synthetic test is not a substitute for the failed full-model prerequisite.

## Follow-up diagnosis

A separately specified training-only diagnostic completed in 80.734 wall seconds on person6792, timestep300, noise20261011. It used no evaluation original and performed no optimization.

| Comparison | Maximum gradient difference | Outcome |
|---|---:|---|
| Native model, first versus repeated forward/backward | 0 | Exact |
| Enabled-zero scorer-only versus all-parameter request | 0 | Exact |
| Enabled-zero scorer-only versus native model | 1.811786205507815e-7 | Fails original tolerance |

The native repeated predictions also match exactly. These measurements rule out ordinary native-run variation and the parameter-request list as explanations in this diagnostic. The local cancellation repair did not resolve the issue. The remaining discrepancy is associated with the added full-model computation graph under the current mixed-precision/checkpointing setup. Its precise source has not been established. Do not describe the hypothesized cancellation mechanism as a confirmed root cause.

## Decision and remaining work

The original control-reuse recipe stops at its prerequisite. Neither prepared inference replay script ran, and no capacity training or new evaluation images followed. Existing website settings, all previous checkpoints, datasets and published numeric results remain unchanged.

Next: specify a fresh paired control that includes the same attention computation path at fixed zero residual, verify matched initial predictions and scorer gradients against the trainable counterpart, and retain the historical baseline as a separate comparator. Confirm the adapter's full-model structural gradients and memory/optimizer checks. Only then perform the pre-training 1024 replay and freeze a new training/evaluation protocol. This requires a new matched control run rather than reusing the old SSIM32 trace. Do not silently change the original preflight or loosen its tolerance. If the matched-path prerequisite also fails, localize the backward divergence before further training.

Any later quality candidate must still improve hole SSIM/PSNR and perceptual quality without identity degradation, preserve visible pixels, and undergo multiple-seed/corrupted-reference validation. These engineering tests establish no improvement in reconstruction quality.

[Complete numeric evidence and source hashes](spatial_qv_prerequisite_evidence_v1.json) · [Original experiment](MASKED_SPATIAL_QV_PREFLIGHT_V1.md) · [Repair preflight](MASKED_SPATIAL_QV_GRADIENT_REPAIR_PREFLIGHT_V2.md) · [Diagnostic plan](SPATIAL_QV_GRADIENT_DIAGNOSTIC_PLAN_V1.md).
