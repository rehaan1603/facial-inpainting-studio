# Spatial Q/V gradient cancellation repair v2

14 October 2026. Written after the v1 prerequisite failed, before the v2 implementation or execution. Preserve all v1 sources, receipts and controls. No training is authorized by the v1 result.

V1 produced exactly equal full-model predictions at zero B, but the original scorer's epsilon gradient differed by 1.811786205507815e-7, exceeding the fixed atol1e-7/rtol1e-4 criterion. Peak allocation was 7,338,135,040 bytes. Nine CPU tests passed. The GPU probe stopped before structural backward or an optimizer update.

Hypothesis: separate low-precision backward branches accumulate the cancelling native attention contributions in different orders. Test a mathematically equivalent attention-difference autograd operation that subtracts native-input gradients inside the operation, in fp32 (fp64 for double tests), before returning them to the backbone. Preserve the original delegate gradient and the nonzero adapter gradient. No zero-weight bypass, detached upstream path, changed tolerance, changed loss or new parameters.

First repeat the nine module tests, then compare nonzero forward/backward against ordinary autograd in double precision and run finite-difference gradcheck. Add GPU synthetic zero/nonzero checks under the same FLASH/MATH policy and bf16 autocast. Preserve failures. Only if these pass, run the same training-only full-model probe with new filenames and fresh receipts, retaining all v1 tolerances and 600-second/7.8-billion-byte bounds. Record original-vs-zero prediction and scorer gradient, Q/V B gradient, frozen-weight and backend guards.

All scientific data, checkpoints, optimizer, update schedule, neutral inference replay requirement and evaluation gates remain those of MASKED_SPATIAL_QV_PREFLIGHT_V1.md. V2 must pass its prerequisites before an enabled-zero inference replay and any training. This is a numerical implementation repair, not evidence of better reconstruction. Freeze v2 source pins before GPU probing. No v1 pinned file may be changed.
