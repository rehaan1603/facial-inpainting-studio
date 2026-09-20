# Distortion-aware generation — development screening v1

144 GPU generations, 144 preservation outputs and 24 unchanged-input controls; 312 scored rows. Four previously observed development identities, one fixed seed, six damage kinds, one central-face mask per identity and medium severity. This is a controlled screening study, not final validation.

Strengths 0.50, 0.75 and 0.99; adapter scales 0.8 and 1.2; 30-step schedule, guidance 5.0, fixed references and Poisson composition. Lower strength also uses fewer active steps; this is not yet an equal-active-step causal test. Preservation mixes 50% observed evidence within degraded regions, zero within fully removed regions, and preserves all known pixels exactly. The supplied class/confidence is not estimated from clean truth.

| Damage | Arm (scale 0.8) | FaceNet target ↑ | Gallery ↑ | LPIPS ↓ | Hole MAE ↓ | NIQE ↓ | BRISQUE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| removal | observed | 0.4802 | 0.1833 | 0.0988 | 0.1970 | 4.9444 | 11.0495 |
| removal | standard_s0.5_a0.8 | 0.4684 | 0.2360 | 0.0626 | 0.1205 | 4.8449 | 9.3502 |
| removal | standard_s0.75_a0.8 | 0.6868 | 0.4954 | 0.0345 | 0.0847 | 4.6863 | 9.7831 |
| removal | standard_s0.99_a0.8 | 0.7079 | 0.5156 | 0.0318 | 0.0865 | 4.7308 | 9.9166 |
| removal | preserve_s0.99_a0.8 | 0.7079 | 0.5156 | 0.0318 | 0.0865 | 4.7308 | 9.9166 |
| gaussian_blur | observed | 0.9249 | 0.5011 | 0.0405 | 0.0315 | 4.8410 | 9.9993 |
| gaussian_blur | standard_s0.5_a0.8 | 0.7082 | 0.5083 | 0.0308 | 0.0630 | 4.7051 | 9.8389 |
| gaussian_blur | standard_s0.75_a0.8 | 0.7381 | 0.5319 | 0.0299 | 0.0775 | 4.6856 | 9.8751 |
| gaussian_blur | standard_s0.99_a0.8 | 0.7170 | 0.5118 | 0.0315 | 0.0878 | 4.7166 | 9.9579 |
| gaussian_blur | preserve_s0.99_a0.8 | 0.8043 | 0.5335 | 0.0302 | 0.0532 | 4.7922 | 9.9686 |
| noise | observed | 0.9846 | 0.5587 | 0.0480 | 0.0556 | 4.0876 | 5.4601 |
| noise | standard_s0.5_a0.8 | 0.7397 | 0.5138 | 0.0300 | 0.0803 | 4.4929 | 9.6314 |
| noise | standard_s0.75_a0.8 | 0.7273 | 0.5245 | 0.0301 | 0.0908 | 4.6404 | 10.0941 |
| noise | standard_s0.99_a0.8 | 0.7024 | 0.4960 | 0.0313 | 0.0931 | 4.7451 | 10.0071 |
| noise | preserve_s0.99_a0.8 | 0.8366 | 0.5711 | 0.0280 | 0.0561 | 4.1599 | 6.1928 |
| jpeg | observed | 0.9869 | 0.5626 | 0.0049 | 0.0162 | 4.3798 | 10.1805 |
| jpeg | standard_s0.5_a0.8 | 0.7507 | 0.5429 | 0.0243 | 0.0709 | 4.6498 | 10.1611 |
| jpeg | standard_s0.75_a0.8 | 0.7354 | 0.5347 | 0.0279 | 0.0835 | 4.6904 | 10.0295 |
| jpeg | standard_s0.99_a0.8 | 0.7183 | 0.5146 | 0.0310 | 0.0903 | 4.7343 | 9.9982 |
| jpeg | preserve_s0.99_a0.8 | 0.8447 | 0.5746 | 0.0154 | 0.0467 | 4.6681 | 9.9581 |
| downsample | observed | 0.9765 | 0.5456 | 0.0269 | 0.0205 | 4.7418 | 10.0710 |
| downsample | standard_s0.5_a0.8 | 0.7627 | 0.5231 | 0.0260 | 0.0629 | 4.7189 | 10.0070 |
| downsample | standard_s0.75_a0.8 | 0.7433 | 0.5261 | 0.0283 | 0.0775 | 4.7176 | 9.9240 |
| downsample | standard_s0.99_a0.8 | 0.7137 | 0.5046 | 0.0308 | 0.0885 | 4.7144 | 9.9487 |
| downsample | preserve_s0.99_a0.8 | 0.8338 | 0.5637 | 0.0274 | 0.0494 | 4.7830 | 10.0054 |
| mixed | observed | 0.8960 | 0.4802 | 0.0519 | 0.0446 | 4.0709 | 7.8703 |
| mixed | standard_s0.5_a0.8 | 0.7206 | 0.4761 | 0.0299 | 0.0770 | 4.6124 | 9.9251 |
| mixed | standard_s0.75_a0.8 | 0.7285 | 0.4961 | 0.0304 | 0.0876 | 4.6932 | 9.9937 |
| mixed | standard_s0.99_a0.8 | 0.7158 | 0.5075 | 0.0312 | 0.0919 | 4.7526 | 9.9788 |
| mixed | preserve_s0.99_a0.8 | 0.8174 | 0.5289 | 0.0312 | 0.0578 | 4.1981 | 8.5621 |

Full scale-1.2, preservation, ArcFace, SSIM, PSNR and coverage results are retained in `distortion_aware_results_v1.json`; no unfavorable arm is omitted from the evidence ledger.

| Primary FaceNet contrast | Identity n | Mean difference | 95% identity bootstrap interval | Exact p | Holm p |
|---|---:|---:|---|---:|---:|
| removal_strength_0.5_vs_.99 | 4 | -0.2394 | [-0.2970494031906128, -0.1818382889032364] | 0.1250 | 1.0000 |
| removal_strength_0.75_vs_.99 | 4 | -0.0211 | [-0.1277349591255188, 0.082211434841156] | 0.6250 | 1.0000 |
| gaussian_blur_strength_0.5_vs_.99 | 4 | -0.0088 | [-0.07571238279342651, 0.02974754571914673] | 1.0000 | 1.0000 |
| gaussian_blur_strength_0.75_vs_.99 | 4 | 0.0211 | [-0.0011749565601348877, 0.04333081841468811] | 0.3750 | 1.0000 |
| gaussian_blur_preserve_vs_standard | 4 | 0.0873 | [0.06768113374710083, 0.10686039924621582] | 0.1250 | 1.0000 |
| noise_strength_0.5_vs_.99 | 4 | 0.0373 | [0.006073981523513794, 0.06844475865364075] | 0.2500 | 1.0000 |
| noise_strength_0.75_vs_.99 | 4 | 0.0249 | [0.005319386720657349, 0.044469863176345825] | 0.1250 | 1.0000 |
| noise_preserve_vs_standard | 4 | 0.1342 | [0.09047409892082214, 0.2080196738243103] | 0.1250 | 1.0000 |
| jpeg_strength_0.5_vs_.99 | 4 | 0.0324 | [-0.0024781078100204468, 0.06207321584224701] | 0.2500 | 1.0000 |
| jpeg_strength_0.75_vs_.99 | 4 | 0.0171 | [-0.004602760076522827, 0.03880748152732849] | 0.3750 | 1.0000 |
| jpeg_preserve_vs_standard | 4 | 0.1264 | [0.09531760215759277, 0.172468900680542] | 0.1250 | 1.0000 |
| downsample_strength_0.5_vs_.99 | 4 | 0.0490 | [0.032215774059295654, 0.06581991910934448] | 0.1250 | 1.0000 |
| downsample_strength_0.75_vs_.99 | 4 | 0.0296 | [0.019710808992385864, 0.039389997720718384] | 0.1250 | 1.0000 |
| downsample_preserve_vs_standard | 4 | 0.1200 | [0.08181352913379669, 0.17685064673423767] | 0.1250 | 1.0000 |
| mixed_strength_0.5_vs_.99 | 4 | 0.0048 | [-0.04608546197414398, 0.056410133838653564] | 0.7500 | 1.0000 |
| mixed_strength_0.75_vs_.99 | 4 | 0.0127 | [-0.0022858083248138428, 0.027764856815338135] | 0.3750 | 1.0000 |
| mixed_preserve_vs_standard | 4 | 0.1016 | [0.0790296196937561, 0.12422117590904236] | 0.1250 | 1.0000 |

## Interpretation boundary

The 0.8180 historical all-reference mean was measured across a different three-condition/two-seed mixture. Comparing a new six-distortion average to that number does not establish an improvement. Use paired within-case settings here and a matched historical-protocol confirmation before making that claim.

Four identities provide weak uncertainty estimates; bootstrap intervals are unstable and the minimum two-sided exact p is 0.125. Statistical success cannot be claimed from this screen. Check unchanged-input controls: beating aggressive generation while underperforming the input is not successful restoration. ArcFace is the conditioning encoder and is diagnostic. No final identities, new adapter training or candidate reranking were used.

Local feature correspondence and novelty findings are reported separately. Seed confirmation, fresh development identities, severity/mask/reference variation and external baselines remain necessary.

## Fresh development extension completed

See `DISTORTION_EXTENSION_RESULTS_V1.md`: four additional identities, six types, two severity/mask strata and a compressed-reference stressor. 192/240 scheduled rows were scoreable; 48 retain generation/dependent failures for one identity. The three complete primary identity pairs favor lower denoising and preservation versus aggressive generation descriptively, but all four Holm p-values are 1.0. Unchanged inputs still expose fidelity loss. This does not establish a publishable superiority claim or improvement over the historical 0.8180 aggregate.
