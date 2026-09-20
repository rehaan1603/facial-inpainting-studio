# Regional reference conditioning — implementation status

The first cycle implemented reference selection. The second cycle adds deterministic spatial weighting of multiple global FaceID descriptors through existing Diffusers IP-Adapter masks. It does not implement local reference-patch transfer, learned spatial correspondence, a trainable fusion adapter, or a new generator.

The fixed alternatives are original token concatenation, equal per-reference attention averaging, global identity weighting, quality weighting, mask-aware global weighting, and region-specific weighting. Every routed policy keeps equal weights outside supplied damage. The regional policy blends coarse facial-region weight maps with a fixed Gaussian basis; geometry comes only from the damaged input. Clean targets and withheld gallery photographs are evaluation-only.

Native separate-reference attention averaging and concatenated-token attention normalize differently. The original concatenation control is retained for that reason. Regional masks are an existing library capability, not a novelty claim: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter .

The incomplete CPU-mask pilot remains preserved separately. V2 puts routing masks on the denoiser execution device before attention, without changing the frozen weighting formula. Its completed 144-row comparison did not establish a benefit: regional target FaceNet mean was 0.8137 versus 0.8180 for original concatenation, and all five Holm-adjusted p-values were 1.0. See `REGIONAL_ROUTING_RESULTS_V2.md` and `REGIONAL_ROUTING_VISUAL_REVIEW_V2.md`. See `REGIONAL_ROUTING_PLAN_V1.md` and `protocols/regional_routing_protocol_v2.json` for the fixed design.

Outstanding research includes larger disjoint development groups, real visibility estimation, spatial feature correspondence, external reference-inpainting baselines, human assessment, and a frozen final evaluation. No local generator or regional adapter training has occurred. Pretrained identity exposure is unknown.
