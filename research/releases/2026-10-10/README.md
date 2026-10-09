# 10 October 2026 release evidence

This snapshot contains code-linked metrics and hashes for the completed local studies. Image pixels, model weights and reference feature tensors remain on the original laptop. The [export script](../../../scripts/export_release_evidence_20261010.py) validates source image hashes and exports saved records without generation or rescoring.

| File | Contents |
|---|---|
| [metrics.json](metrics.json) | Every attempt/status and per-output metric from nine masked SDXL experiment folders |
| [website_replays.json](website_replays.json) | Four actual HTTP/GPU reconstruction receipts with raw/final equivalence |
| [verification.json](verification.json) | Test counts, Python syntax, frozen generator hash and evidence checks |

The exact local source-summary hashes are stored alongside each experiment. Reused images remain marked; detector nulls and failed attempts remain present. In the guidance replication, `control` and `candidate` are CFG5/CFG3 labels while both reference conditions are clean. Different experiment folders retain different protocols; do not average across them as one benchmark.

## Current repeated comparison

Three observed people, four real references each, seeds17/29/43, DDIM30, FaceID0.8, strength1, 1024 processing/512 export and fixed Poisson finish. Guidance alone changes 5 → 3.

| Metric | CFG5 | CFG3 |
|---|---:|---:|
| Hole PSNR ↑ | 17.01785 | 17.82148 |
| Hole SSIM ↑ | 0.41415 | 0.43310 |
| Hole LPIPS ↓ | 0.20144 | 0.19185 |
| Eye LPIPS ↓ | 0.15958 | 0.14498 |
| FaceNet ↑ | 0.76920 | 0.79887 |
| ArcFace ↑ | 0.76072 | 0.78508 |

SSIM improves 8/9 pairs. Person10149's mean FaceNet falls 0.013618. Eye colour, gaze and eyebrows remain imperfect. Read [all per-person/seed results](../../FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md). The app option is unchecked by default; see [integration verification](../../GENTLER_REFERENCE_WEBSITE_V1.md).

## Unsuccessful mechanisms retained

- Euler: mean identity declines in the fixed seed17 test.
- Added appearance branch: mean SSIM declines at CFG5.
- Guidance + appearance: SSIM and ArcFace decline versus CFG3 alone.
- Registered reference texture: all six means worsen and visible local alignment artifacts remain.

See [research navigation](../../README.md) for every protocol/report. Original baseline checkpoints and previous outputs remain available. These are development observations; publication validation, unknown-person robustness and the remaining fidelity work stay open.
