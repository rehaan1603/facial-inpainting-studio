# Repeat-gradient isolation V5 — 19 October 2026

Training-only diagnostic, not a quality experiment. Preserve V1–V4 evidence.
V4 failed matched scorer gradients (max 1.811786205507815e-7) despite equal predictions and identical autograd parameter counts. Its control gradient matched the historical native gradient exactly.

Hypothesis: the intervening decoded-SSIM backward or another sequential graph lifecycle operation changes the later backward. Remove the CPU VAE decode/structural backward between the two identical epsilon passes. Keep the model, mask, seed, dtype, parameters, loss and tolerances unchanged. No optimizer step, saved checkpoint, test original or image evaluation. This narrows the cause; it cannot authorize training by itself.

Measure both gradients and unchanged predictions with the original atol=1e-7, rtol=1e-4; retain failures. Maximum 600 seconds and 7.8GB CUDA allocation. Pin sources; do not overwrite previous attempts. If repetition fails, inspect graph/state changes before designing another intervention.
