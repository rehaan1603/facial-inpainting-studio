# Masked SDXL compatibility feasibility results v1

10 October 2026. One conditional synthetic 512-pixel-equivalent direct-UNet
gradient check passed in **36.813 seconds supervisor wall time**, with
**7,289,674,752 bytes peak allocated GPU memory (7.29 decimal GB; 6.79 GiB)**.
This establishes a single gradient/update path on the recorded laptop GPU. It
does not establish reconstruction quality, repeated real-data training, or
1024-pixel backward feasibility.

| Retained attempt | Result | Evidence and limit |
| --- | --- | --- |
| [v1, 1024 equivalent](../outputs/masked_sdxl_compatibility_probe_v1/1024.json) | Failed during backward | Neutral maximum absolute error 0.0; `LSE is not correctly aligned (strideH)`; not reported as OOM. Worker elapsed 88.265 s; peak allocated 7,752,139,776 bytes, reserved 8,346,664,960 bytes. No completed gradient/update evidence. |
| [v2, 1024 equivalent](../outputs/masked_sdxl_compatibility_probe_v2/1024.json) | Interrupted; exact stuck stage unknown | Flash/Math backend attempt did not produce completed neutral/gradient/update measurements. Observed interval before stop was 286.63 min, CPU time 93.296875 s, with GPU utilization observed at zero and CPU time unchanged. Suspension/time changes are not excluded; that interval is not confirmed active computation and is not evidence of a GPU-memory limit. |
| [v3, 512 equivalent](../outputs/masked_sdxl_compatibility_probe_v3/512.json) | Passed | Neutral maximum error 0.0; finite nonzero aggregate gradient norm 0.00014083388668950647; disposable scorer update occurred; frozen weights had no gradients. Synthetic epsilon loss 2.1419291496276855. Forward/backward/update timing block 5.5 s; worker elapsed 26.828 s. Peak reserved 7,562,330,112 bytes. |

The [512 supervisor](../outputs/masked_sdxl_compatibility_probe_v3/512_supervisor.json)
finished with return code 0, without reaching its 300-second timeout. Its
36.813-second wall time includes process startup; worker elapsed and the
5.5-second forward/backward/update block measure different scopes. None measures
the cost of a complete real-data training run.

All attempts used the same 385-parameter shared `10 → 32 → 1` scorer, bias bound
0.5, 64-query chunks, seed 20261010, timestep 500, four synthetic references,
16 tokens per reference in each adapter branch, branch scales 0.8/0.2, frozen
float16 weights, bfloat16 autocast, float32 scorer, and UNet/scorer checkpointing.
The loaded frozen model had 3,366,130,756 parameters and 70 attention processors.
The successful run used Flash/Math backend selection and restored prior flags.
Its first-layer gradients were present and zero at the neutral initialization;
the final weight gradient was nonzero. The synthetic optimizer update was
discarded and no checkpoint was saved.

The probes excluded photographs, VAE encoding/decoding, image/identity losses,
CFG sampling, and dataset optimization. No quality metric, improved SSIM claim,
or hidden-eye recovery claim follows from their synthetic loss or gradient.
Existing reconstruction/evaluation settings are unchanged by these probes.

All three frozen [v1](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V1.md),
[v2](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V2.md), and
[v3](MASKED_SDXL_COMPATIBILITY_FEASIBILITY_PREFLIGHT_V3.md) protocols and the
unsuccessful receipts remain preserved. The companion
[photo-free numeric record](masked_sdxl_compatibility_feasibility_v1.json) records
receipt, protocol, core/wrapper/supervisor, implementation, and model-asset hashes.
Their recorded source/protocol hashes match the current retained files. This
summary was generated from JSON/source evidence without GPU work or new probes;
real-data cache preparation and any later fixed pilot are separate records.
