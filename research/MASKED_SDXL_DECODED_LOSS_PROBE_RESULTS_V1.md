# Decoded missing-region loss: gradient and resource result

10 October 2026. The [frozen audit/probe](MASKED_SDXL_DECODED_LOSS_PREFLIGHT_V1.md) **passed** on the laptop. This tests one differentiable training example and one disposable update. It creates no reconstruction, quality score or checkpoint and makes no SSIM or identity-improvement claim.

| Measurement | Actual value |
|---|---:|
| Training identity / target HQ | 6792 / 28120 |
| Trainable shared parameters / processors | 385 / 70 |
| Timestep / noise seed | 300 / 20261011 |
| Full epsilon MSE | 0.021381464 |
| Hole epsilon MSE | 0.243648559 |
| Existing epsilon objective | 0.188081786 |
| Normalized decoded-hole RGB L1 | 0.053411108 |
| Total objective with fixed RGB coefficient 0.1 | 0.193422897 |
| Epsilon gradient norm | 0.000070340153 |
| RGB gradient norm, unweighted | 0.000090811220 |
| RGB contribution norm after weight 0.1 | 0.000009081122 |
| RGB-to-epsilon norm ratio, unweighted | 1.291030 |
| Gradient cosine | 0.998812487 |
| Combined gradient norm | 0.000079411722 |
| In-memory disposable parameter update | Yes |
| Frozen UNet / VAE gradients | Absent / absent |
| Worker elapsed seconds | 33.641 |
| Peak GPU allocated / reserved bytes | 7,337,905,664 / 7,767,851,008 |
| Peak process working set bytes | 7,437,676,544 |
| Peak process pagefile/commit counter bytes | 17,212,342,272 |
| Original checkpoint hashes and backend flags | Preserved |

CPU VAE forward took 7.640 seconds. RGB backward, including decoder recomputation and the second UNet backward, took 17.250 seconds. Epsilon backward took 2.375 seconds. Decoder activations stayed on CPU, avoiding concurrent GPU VAE storage. These are one measured case, not stable throughput estimates for all timesteps or photographs.

The 512 hole contains exactly 16,048 pixels and matches the pinned Boolean mask SHA256 and nearest64 cache tensor. DDPM alpha_bar at300 is 0.590500772. Oracle-noise inversion has maximum residual 0.003003716 and RMS 0.000428912 because inherited forward-noising coefficients/additions use float16. Decoded affine RGB is unclamped for training: minimum -0.019658625, maximum 0.994046211, fraction outside [0,1] 0.000081380. The added loss retains gradients for out-of-range values.

The RGB gradient reaches the same trainable parameters and is closely aligned with epsilon on this example. Its weighted norm is about 12.9% of the epsilon norm. This supports trying a controlled loss comparison, but does not establish that the new loss supplies useful anatomical information, fixes iris colour or changes generated quality. The 385-parameter scorer still has no verified eye correspondence and cannot abstain from all misleading appearance tokens.

The [next matched plan](MASKED_SDXL_DECODED_CONTINUATION_PREFLIGHT_V1.md) compares equal fresh continuations from the identical v2-last checkpoint. A fresh epsilon-only continuation controls extra optimization. All six quality/identity measures and per-person guards remain required. The earlier 64-update pilot and current CFG3/local outputs remain intact, including their negative results. Photographs, embeddings and checkpoints stay local.
