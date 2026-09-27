# Masked scaffold conditioning — 25 September 2026

**Progression gate: failed.**

The preceding cascade improved identity but increased pixel error. This separately frozen follow-up suppresses low-quality scaffold latent cells overlapping missing pixels. Real references and the late original-context correction remain fixed. The global control matches the mean conditioning multiplier, not exact latent norm. A reference-disabled arm tests reference benefit. No clean truth enters inference.

24/24 new generations, two equivalence calls, 16 reused controls and 40/40 scored rows. Complete detector/gallery coverage: False. Exact outside-mask preservation: True.

**Coverage audit addendum, 27 September:** scoring completed despite two FaceNet and seven ArcFace detector failures across seven output rows. Both masked arms' FaceNet target/gallery means below use **three** identities with both seeds available; their ArcFace means in JSON use **two**. Other arms and pixel/perceptual metrics use four. Do not compare differently supported table means as paired deltas. The paired contrasts below already use the correct complete identity pairs. No scores or gates were changed by this explanatory addendum; original reporter/source hashes remain preserved.

| Arm | FaceNet ↑ | Gallery ↑ | Hole MAE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| scaffold | 0.741489 | 0.423009 | 0.062665 | 0.025402 |
| reference_context | 0.781441 | 0.476390 | 0.065988 | 0.027110 |
| masked_lq_context | 0.432041 | 0.165785 | 0.137404 | 0.092064 |
| masked_lq_no_reference_context | 0.445447 | 0.178519 | 0.140856 | 0.091680 |
| global_lq_context | 0.788347 | 0.493971 | 0.064979 | 0.028968 |

| Primary contrast | Identity n | Delta | 95% identity bootstrap interval | Holm p |
|---|---:|---:|---|---:|
| masked_lq_context_minus_scaffold/facenet_cosine | 3 | -0.329600 | [-0.3838459700345993, -0.26757723093032837] | 1.000000 |
| masked_lq_context_minus_scaffold/hole_mae | 4 | 0.074739 | [0.0622949931697523, 0.08718307343111427] | 1.000000 |
| masked_lq_context_minus_reference_context/facenet_cosine | 3 | -0.370777 | [-0.41436798870563507, -0.3040870726108551] | 1.000000 |
| masked_lq_context_minus_reference_context/hole_mae | 4 | 0.071416 | [0.05828728641285351, 0.0845440917979057] | 1.000000 |
| masked_lq_context_minus_global_lq_context/facenet_cosine | 3 | -0.366236 | [-0.4190567284822464, -0.3011200726032257] | 1.000000 |
| masked_lq_context_minus_global_lq_context/hole_mae | 4 | 0.072424 | [0.06009884133415976, 0.0847499737727575] | 1.000000 |
| masked_lq_context_minus_masked_lq_no_reference_context/facenet_cosine | 3 | -0.013406 | [-0.023609936237335205, 0.00011965632438659668] | 1.000000 |
| masked_lq_context_minus_masked_lq_no_reference_context/hole_mae | 4 | -0.003452 | [-0.0059726966568755535, -0.00093178682088009] | 1.000000 |

Eight tests specified before generation. Seeds are averaged within identity. Four identities cannot support a two-sided exact p below 0.125; the intervals are unstable. The same already observed cases informed the follow-up hypothesis, so this is development exploration, not independent confirmation. All scores, coverage and per-seed gates remain in JSON.

Latent masking does not guarantee semantic locality because the encoder has a broad receptive field. Missing input-latent values were set to zero without training the model on spatially dropped conditioning; out-of-distribution behavior is a plausible failure mode. No radius/strength search or clean-target tuning was performed in this protocol.

Mask-aware conditioning, personal reference restoration and reliability-aware guidance have prior art. No first-in-literature, publication-ready superiority or arbitrary-client accuracy claim follows. Final identities are untouched. See `NOVELTY_ESTABLISHMENT_20260925.md` for visual findings, related-work boundaries and the resulting decision.
