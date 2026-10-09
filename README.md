# Facial Inpainting Studio

Local facial reconstruction and evaluation on a Windows laptop with an RTX 5070 Laptop GPU. Upload a damaged photograph, mark the affected region, supply optional matching reference photographs, reconstruct, compare and download the result.

**Updated 10 October 2026.** Start with the [project report](PROJECT_REPORT.md), [research index](research/README.md) and [local testing instructions](LOCAL_TESTING.md). Public hosting is paused. Dataset photographs and downloaded model weights stay on the configured laptop.

## Navigate the project

| Location | Contents |
|---|---|
| [Project report](PROJECT_REPORT.md) | Current progress, results, limitations and remaining work |
| [Application](webapp/README.md) | Local HTTP service and browser controls |
| [Scripts](scripts/README.md) | Inference, experiments, scoring and verification |
| [Preservation components](src/preservation) | Mask, evidence, correspondence and preservation code |
| [Research](research/README.md) | Protocols, dated results, metrics, papers and provenance |
| [10 October evidence](research/releases/2026-10-10/README.md) | Photo-free per-output metrics and verification snapshot |
| [Presentations](presentations/README.md) | Public decks and local illustrated deck locations |
| [Review notes](review/README.md) | Faculty preparation notes and code navigation |
| [Configuration](configs) | Settings and local-path example |
| [Tests](tests) | Geometry, mask, preservation, routing and runtime checks |
| [Repository scope](REPOSITORY_SCOPE.md) | Included artifacts and restoration requirements |

## Run on the configured laptop

Double-click **Start Studio.cmd**. It opens **http://127.0.0.1:8765/** and reuses a running server. Running servers keep their loaded backend until restarted.

The updated testing build is currently at **http://127.0.0.1:8766/**. The older 8765 process remains running. After a laptop restart, Start Studio starts the current code on 8765. Avoid concurrent jobs across both addresses: they use the same GPU.

To start a foreground server from PowerShell in this repository:

```powershell
.venv/Scripts/python.exe webapp/server.py --port 8766
```

Keep the laptop awake. A fresh clone requires the [local configuration](configs/local.example.json), CUDA environment, permitted data and model cache described in [repository scope](REPOSITORY_SCOPE.md). The static frontend alone cannot generate images.

## Implemented methods

| Method | Inputs and purpose | Current limitation |
|---|---|---|
| LaMa | Image + mask; small texture repairs | Large missing facial features can be distorted/blurry |
| ResShift | Image + mask; single-image facial inpainting | Identity, gaze and expression can differ |
| SDXL + FaceID Portrait | Image + mask + 1–4 matching photos | Hidden eye/brow details remain uncertain |
| Experimental reference selection | Same inputs; fixed heuristic selects one photo | Broad quality improvement remains unresolved |
| Experimental ReF-LDM | Blur/noise evidence map + 3–4 references | Whole-image restoration; erased-region diagnostic retains major failures |

Reference processing offers 256/512/1024 comparisons; exports are 512 × 512. White mask pixels request replacement. Visible pixels outside the effective mask are preserved. Read [local testing](LOCAL_TESTING.md) for the separate grayscale evidence-map convention.

Generative backbones and recognition encoders are pretrained. Earlier experiments trained small mask-refinement and compatibility modules. The [paper/implementation map](research/PAPERS_AND_IMPLEMENTATION_MAP.md) documents their roles.

## Latest focused eye study

Fixed persons **263, 6681 and 10149**, four real clean references each, seeds **17/29/43**, DDIM30, FaceID0.8, strength1, 1024 processing, 512 output and Poisson finishing. Only classifier-free guidance changes from 5 to 3. Eighteen outputs cover nine pairs; six seed17 images reuse hash-verified earlier outputs.

| Nine-pair mean | CFG5 control | CFG3 comparison |
|---|---:|---:|
| Hole PSNR ↑ | 17.01785 | 17.82148 |
| Hole SSIM ↑ | 0.41415 | 0.43310 |
| Hole contextual LPIPS ↓ | 0.20144 | 0.19185 |
| Eye contextual LPIPS ↓ | 0.15958 | 0.14498 |
| FaceNet cosine ↑ | 0.76920 | 0.79887 |
| ArcFace cosine ↑ | 0.76072 | 0.78508 |

SSIM improves in 8/9 pairs; person10149 loses **0.01362 mean FaceNet cosine**. Iris colour, gaze and eyebrow fidelity remain imperfect. These are three observed development people; similarity scores are not reconstruction-correctness percentages. See [all paired results](research/FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md).

The app exposes unchecked **Gentler eye details** only in 1024 reference mode with **Ignore opaque cover colour** enabled. Four live reconstructions match saved raw/final research outputs byte-for-byte. See [app verification](research/GENTLER_REFERENCE_WEBSITE_V1.md). Added appearance conditioning and registered texture transfer retain mixed/negative results in their reports.

Local test sets are under `Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09/`. Each includes inputs, masks, references, comparison-only originals and labelled outputs. Use the [official dataset/sample restoration guide](research/DATASET_DOWNLOADS_AND_SAMPLES.md).

## Evaluation and reproduction

Compare only identical input, mask, reference, seed, processing, finishing and metric protocols. The [research index](research/README.md) distinguishes masked SDXL inpainting from whole-image ReF-LDM restoration and published-paper tables.

PSNR, SSIM, LPIPS, FaceNet and ArcFace are recorded; earlier suites add NIQE/BRISQUE and other diagnostics. Detector failures remain explicit. ArcFace also provides conditioning. Hole SSIM uses valid masked windows; contextual LPIPS uses fixed crop context. See [metric code](scripts/extended_evaluation_metrics.py), [masked scorer](scripts/score_masked_sdxl_v1.py) and [evaluation protocol](research/EXTENDED_EVALUATION_PROTOCOL_V1.md).

```powershell
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_webapp.py
.venv/Scripts/python.exe -m unittest discover -s scripts -p test_studio_finishing.py
```

The [script index](scripts/README.md) identifies each separate runtime and reproduction stage. Signed experiments refuse changed inputs. Changed studies require new versioned protocols/output directories. Preserve baseline checkpoints and unsuccessful outcomes.

## Remaining research

Reliable local eye/brow correspondence and reference compatibility remain the next quality work, followed by fixed corrupted-reference and unseen-person validation. Reserved final evaluation, matched external comparisons, human evaluation and manuscript/venue review remain in [remaining work](PROJECT_REPORT.md#4-remaining-work).

See [data/model terms](research/DATA_AND_LICENSES.md), [earlier reproduction](research/REPRODUCE_EARLIER_STAGES.md) and the [previous README snapshot](https://github.com/rehaan1603/facial-inpainting-studio/blob/a6f3b68/README.md). Restricted photos, personal uploads, generated faces, downloaded weights and environments stay local. Existing project-trained artifacts already in history are retained.
