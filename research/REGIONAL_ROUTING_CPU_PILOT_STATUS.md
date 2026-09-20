# CPU-mask routing pilot — interrupted for device optimization

The first regional-routing pilot used stock fractional masks residing on the CPU. Inspection of Diffusers showed that each attention call resized and expanded these masks on the CPU before copying the expanded result to the GPU. Generation was materially slower than the original concatenation path; concurrent CPU map checks also affected early timings. Therefore the timings are not a clean benchmark of the device effect.

The pilot was stopped on 20 September 2026. Existing output files, logs, signatures and completed generation records remain under `outputs/regional_routing_v1`; an interrupted partial output is not treated as a completed row. Its implementation is preserved in Git commit `54a0914`. This incomplete set is excluded from the second experiment's quality statistics.

V2 keeps the spatial masks on the UNet execution device before denoising. It has a distinct protocol, signatures, output directory and evidence ledger. The reference weights, policies and generation settings remain the same. CPU/GPU interpolation can differ numerically, so results are not mixed or represented as an identical replay.
