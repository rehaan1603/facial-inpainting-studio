# First local research milestone

The GPU environment, image audit, cleaned manifests, and first LaMa diagnostic are implemented. This is an engineering milestone, not evidence of a novel method.

## Data findings

- All 52,168 source face images decoded without errors.
- CelebA-HQ identity labels are disjoint across official splits, but exact file duplicates still cross splits.
- 112 exact duplicate groups were found across the supplied collections; every member was excluded from working manifests.
- Source images and original official manifests were retained.

| Working dataset | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| celebahq | 24142 | 2987 | 2817 | 29946 |
| lapa | 18082 | 1964 | 1952 | 21998 |

Exclusion totals: 54 HQ files and 170 LaPa files. These cleaned counts differ from published official benchmark protocols. Exact hashes cannot exclude differently encoded copies or identity annotation errors; near-duplicate analysis remains necessary.

## GPU and baseline

- PyTorch 2.11.0+cu128, CUDA runtime 12.8: forward/backward passed.
- Baseline: third-party IOPaint/Sanster TorchScript export of LaMa, with verified published MD5 and recorded SHA256.
- 32 cleaned validation images from 32 identities; no final-test model evaluation.
- 4 mask conditions and 4 methods produce 512 recorded measurement rows.
- Original checkpoint equivalence, face-specific performance, and pretraining exposure are not established.

## Diagnostic findings

| Mask condition | Supplied-mask hole PSNR (dB) |
|---|---:|
| accurate | 17.56 |
| under_4px | 10.75 |
| over_4px | 16.61 |
| shift_8px | 14.06 |

These are means of per-image true-hole PSNR values from a small, fixed validation diagnostic. Higher is better. The benchmark uses textured rectangular occlusions. It does not establish real-world performance or statistical significance.

Four-pixel dilation exactly undoes four-pixel erosion for these rectangles, restoring the exact-mask baseline result. That is expected geometry, not a learned improvement. Dilation of already oversized masks worsens reconstruction and changes additional visible pixels. The supplied-mask baseline preserves pixels outside its mask by explicit compositing; zero visible change in some conditions is therefore by construction.

Measured warm GPU-forward median: 30.7 ms; peak allocated tensor memory: 245.9 MiB. Timing excludes loading, CPU processing and compositing, and the annotation scan was running concurrently. This is a feasibility measurement, not a controlled efficiency benchmark.

The inspected example grid contains severe facial artifacts even with exact masks. A face-specific backbone must be included before arguing for a new facial method.

## Next experiment

Use independently corrupted irregular and semantic masks, tune dilation/refinement controls on validation, add perceptual scoring, and compare a face-specific model. Keep the final test untouched. Only proceed to a new trainable policy if a useful gap survives these controls.

## Verification limits

GPU and four protocol tests passed; dependency consistency passed. All 372,767 HQ masks and 22,168 LaPa labels decoded and passed value checks, and all 22,168 LaPa landmark files passed syntax checks. These checks do not establish annotation accuracy or landmark bounds. Pilot manifest hashes, 512 measurement rows and 32 distinct validation identities were verified. Identity recognition metrics, near-duplicate checking, final held-out evaluation, a new method, and manuscript results remain future work.
