# Expanded validation benchmark

A simple dilation improves aggregate perceptual reconstruction while increasing changes to valid facial pixels. At this historical milestone only the frozen backbone was evaluated. Six trainable refiner controls were completed later; see FINAL_RESULTS.md.

## Protocol

- 96 distinct validation identities, excluding all identities from the first pilot.
- 48 tuning and 48 assessment identities with no overlap; final test remains unused for inference.
- Three mask families: brush, eye region, mouth region.
- Five conditions: accurate, undersized, oversized, translated, and spatially varying boundary error.
- Dilation radii 0, 2, 4, 8; one global radius chosen by tuning LPIPS only.
- LPIPS 0.1.4, AlexNet v0.1 weights and ImageNet backbone, RGB normalized to [-1,1].
- 7,200 metric rows including the repeated exact-mask diagnostic; correlated masks are not independent observations.

## Assessment results

| Radius | Full-face LPIPS | True-hole MAE | Visible-pixel MAE |
|---|---:|---:|---:|
| 0 | 0.08848 | 0.17037 | 0.00067 |
| 2 | 0.08633 | 0.16959 | 0.00113 |
| 4 | 0.07578 | 0.15062 | 0.00170 |
| 8 | 0.06312 | 0.12630 | 0.00330 |

The tuning-selected radius is 8. Its paired assessment LPIPS difference relative to radius 0 is -0.02536, with a 95% identity-bootstrap interval [-0.02808, -0.02273] over 2,000 resamples. Negative values favor dilation. This interval describes this fixed synthetic development sample, not unseen real-world performance.

## Interpretation and limits

The best radius lies at the largest tested value, so the sweep has not established an optimal dilation radius. Expand the tuning range before fixing a competitive final control. Do not present improvement over radius 0 alone as a learned contribution.

The semantic families are not area-matched, and the corruption textures are synthetic. Clean parsing is used only to construct corruptions, never as an inference input. Exact-mask results use privileged information and are diagnostic. The model is a generic third-party LaMa export with unverified numerical equivalence to the original release.

The bootstrap selects no parameters on assessment data, but this remains development validation that can inform later design. A future final evaluation needs locked decisions and untouched test data. No claims about identity preservation, real occluder removal, or state-of-the-art quality are supported yet.

## Next step

Validate a face-specific baseline, extend simple correction controls, and construct area-matched cases. A proposed learned correction must improve the reconstruction–preservation curve against these controls. See benchmark_v2_results.json and outputs/benchmark_v2/metrics.csv for the complete measurements.

## Face-specific baseline compatibility

The official ResShift face checkpoint and VAE loaded strictly against pinned source commit bb03b7d21614cace01787e097c8a6ab6b945227d. Four previously used validation cases completed at 256 pixels with four diffusion steps. This is a compatibility smoke test, not a matched model comparison or an official reproduction.

Peak allocated GPU memory was 1006.4 MiB. The three calls after initialization took about 0.22 seconds each. This is not a controlled end-to-end latency benchmark.

The preview contains plausible completions but visible differences from the hidden target, including eye details. Zero visible-region error follows from exact-mask compositing; it is not evidence of learned preservation. Inaccurate-mask behavior still needs a matched evaluation.

The original configuration references FFHQ and the VAE is named celeba256; pretraining exposure is unresolved. Retain the S-Lab noncommercial license and record this caveat in comparisons. Sources, checkpoint hashes, and adapter settings are in resshift_provenance.json and resshift_downloads.json.
