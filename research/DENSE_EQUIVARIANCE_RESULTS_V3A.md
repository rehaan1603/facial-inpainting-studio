# CLIP known-shift diagnostic v3a — results

Recorded 12 October 2026. This is a feature-correspondence diagnostic on eight existing training observations. No model was trained, no reconstructed image was generated, and reconstruction metrics are unchanged.

## Scope and controls

The [prospective protocol](DENSE_EQUIVARIANCE_PREFLIGHT_V3A.md) retained the frozen CLIP-H/14 encoder, 224-pixel input, penultimate 16x16 patch grid, matching thresholds and geometric gates from v3. Each descriptor grid has a 32-pixel sampling interval in the original 512x512 coordinate system. No feature interpolation was used.

Training identities were 6792, 2250, 2083, 4164, 674, 476, 1281 and 8748. Original image files and cached features were hash-checked. Missing RGB was set to 128 before creating translated observations. Translations move the already-sanitized observation by +8 or +32 pixels horizontally; newly exposed pixels are filled with 128. The shifted mask and padding are excluded from matching, with the same 32-pixel exclusion margin and central coordinate limits. Clean pixels inside the missing region are never supplied to feature extraction.

The identity control compares each stored feature bank with itself. It checks matching and coordinate handling, but is **not an independent encoder replay**. Translated observations are controlled transformations of the same image, not independent reference photographs or additional training examples. They do not contribute to real-reference coverage.

Two quantities are reported separately: held-out RMS measures agreement with descriptor correspondences that were not fitted; known-transform RMS measures the fitted transform against the supplied synthetic translation. A transform can be correct while held-out descriptor matches remain unreliable.

## Preserved first-attempt failure and retry

The first attempt completed all eight identity controls, then stopped at the processor-configuration equality assertion before any translated-image inference. The live processor dictionary contained tuple/list representations that differed from its JSON-loaded equivalent. The receipt records `AssertionError()` and elapsed time 6.734 seconds; it does not record a GPU memory peak.

The retry compares JSON-normalized processor dictionaries. This changes the provenance equality check, not image preprocessing, descriptors, transformations, matching thresholds or gate definitions. The failed attempt remains intact in `outputs/spatial_reference_correspondence_v3a/receipt.json`; the successful retry is separate in `outputs/spatial_reference_correspondence_v3a_retry/receipt.json`.

## Completed retry results

| Control | Cases | Passed unchanged geometry gate | Known-transform RMS |
|---|---:|---:|---:|
| Identity feature-bank control | 8 | 8 | 0 pixels for every case |
| +8-pixel translation | 8 | 0 | 8 pixels for every case |
| +32-pixel translation | 8 | 2 | 0 pixels for every case |

All +8-pixel fits selected the identity transform. All +32-pixel fits recovered the correct source-to-target translation of -32 pixels; six nevertheless failed the held-out correspondence check.

| Training identity | +8px held-out RMS | +8px gate | +32px held-out RMS | +32px gate |
|---|---:|---|---:|---|
| 6792 | 60.663004 | Fail | 21.989897 | Fail |
| 2250 | 117.645723 | Fail | 0.000000 | Pass |
| 2083 | 113.792114 | Fail | 60.164117 | Fail |
| 4164 | 53.770755 | Fail | 83.138439 | Fail |
| 674 | 114.515529 | Fail | 69.357181 | Fail |
| 476 | 74.312083 | Fail | 11.174885 | Pass |
| 1281 | 85.603738 | Fail | 54.494036 | Fail |
| 8748 | 120.468053 | Fail | 64.190194 | Fail |

RMS values are in original-image pixels, rounded to six decimal places. Full precision and all pair counts, transforms and fit diagnostics remain in the retry receipt.

| Resource or execution field | Measured value |
|---|---:|
| Retry elapsed time | 6.047 seconds |
| Retry peak GPU allocated memory | 1,302,118,912 bytes |
| Trainable parameters | 0 |
| Optimizer updates / reconstruction generations | 0 / 0 |

## Interpretation and next step

Correct identity and full-grid-shift fits support the implemented coordinate order and transform direction. The 8-pixel shifts expose coarse grid sampling; the large held-out errors even for exactly recovered 32-pixel transforms also expose unreliable descriptor matches. These observations do not justify relaxing the original 4-pixel RANSAC or 12-pixel held-out gates. They also do not establish that CLIP correspondence can align different photographs of a face.

The next separately specified experiment replaces the descriptor with frozen DINOv2-small at its native 518-pixel grid, retaining matching and geometry gates. Its results are recorded in [DINO geometry v4](DENSE_DINO_GEOMETRY_RESULTS_V4.md). The existing baseline, evaluator, application and checkpoints are unchanged.

## Evidence provenance

- Original v3 feature receipt SHA256: `932557c1ab85196650c9045ef4f0ee51011c3ca2724cb3cb9cf5216f31201787`.
- First-attempt receipt SHA256: `753b4617a35c951b024d3305ef2e7ed61402d261026570f8c98066f6238103fd`.
- Retry receipt SHA256: `152c2c926f19a4d725183e4604b72a178905caeda72fede8e9644cf3164b655d`.
- First-attempt script SHA256: `14fd47f7e4596f3b1d12a78b93ef6c85857139f779eddd0863e148108a328357`.
- Retry script SHA256: `5a92682f5bf4e079b02d0ab6f2796c2edaed702edcea397f64c8c9c4d9228174`.
- Execution code: [audit_dense_equivariance_v3a.py](../scripts/audit_dense_equivariance_v3a.py); unchanged matching and fit checks: [dense_feature_geometry_v3.py](../scripts/dense_feature_geometry_v3.py).

