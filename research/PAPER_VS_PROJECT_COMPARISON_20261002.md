# Published papers and our measured results

Updated 2 October 2026. PSNR/SSIM higher is better; LPIPS lower is better. NR means not reported in the selected source table.

## Published benchmark context

**Different datasets, damage protocols and identity metrics: these rows are not a superiority leaderboard.** The links identify the exact published tables.

| Method / source | Year | References | PSNR dB | SSIM | LPIPS | Identity | Exact context of selected row |
|---|---|---:|---:|---:|---:|---|---|
| [LaMa-Fourier, Table 1](https://arxiv.org/html/2109.07161v1) | WACV 2022; preprint 2021 | 0 | NR | NR | .098 | NR | CelebA-HQ, 256, wide masks; not our Big-LaMa weights/protocol |
| [Extended ResShift, Table V](https://arxiv.org/html/2403.07319v1) | 2024 preprint | 0 | NR | NR | .1506 | NR | CelebA-Test inpainting, average of box/irregular/half/expand masks |
| [RefFaceInpainting, Table I](https://arxiv.org/html/2303.07014v1) | 2023 | 1 | NR | NR | .083 | ID retrieval 66.85% | Author curated identity dataset, 256, approximately 30% mask rate |
| [ReF-LDM, Table 7](https://arxiv.org/html/2412.05043v1) | NeurIPS 2024 | up to 5 | NR | NR | .429 | IDS .676 | FFHQ-Ref-Severe, 512, synthetic whole-image restoration |
| [GFRNet, Table 1](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xiaoming_Li_Learning_Warped_Guidance_ECCV_2018_paper.pdf) | ECCV 2018 | 1 | 28.55 | .947 | NR | NR | VGGFace2, 4x degradation; do not substitute WebFace numbers |
| [3DFaceFill, Figure 4(d)](https://arxiv.org/html/2110.10395v1) | WACV 2022; preprint 2021 | 0 | 29.9398 | .9492 | .0365 | NR | CelebA-HQ, 224 crop, random rectangles restricted to face, pooled mask ratios |
| [DMDNet, Table II](https://arxiv.org/html/2210.08160v1) | 2022 | multiple / dictionary | 28.97 | .902 | .166 | Id .793 | CelebRef-HQ, 512, 4x plus blur/noise/JPEG; full dual-dictionary model |
| [ReFine, Table 1](https://openaccess.thecvf.com/content/WACV2025/papers/Chong_Copy_or_Not_Reference-Based_Face_Image_Restoration_with_Fine_Details_WACV_2025_paper.pdf) | WACV 2025 | 1 | 23.77 | .803 | .158 | IPS .4649 | CASIA-WebFace restoration; IPS concerns reference detail preservation |
| [RefSTAR, Table 1](https://arxiv.org/html/2507.10470v1) | preprint 2025; author repository AAAI 2026 | 1 | 24.69 | NR | .335 | ID-GT 82.76; ID-Ref 64.53, author scale | Celeb-Ref-Test, 512, synthetic restoration; selection masks are transfer masks, not erased input masks |
| [ReSem-Face, Table 1](https://arxiv.org/html/2608.04820v1) | 2026 preprint | 5 | 28.31 | .904 | .116 | CosFace ID .766 | CelebAHQ-IDI-5, semantic masks, 40-step identity personalization; not locally reproduced |
| Proposed compatibility module | 2026 local experiment | 4 | 24.05182 | .63730 | .26477 | FaceNet .90670 | Four observed development identities, clean references, blur/noise restoration; not comparable to published rows above |

## Direct matched local comparison

Same four observed identities, four references, 512 output, seed 17 and 50 DDIM steps. Clean means clean references; the target remains blurred/noisy. This tests restoration, not erased-region completion.

| Method | Reference condition | PSNR (dB) | SSIM | LPIPS | FaceNet cosine |
|---|---|---:|---:|---:|---:|
| Local ReF-LDM baseline | Clean | 24.04671 | 0.63723 | 0.26544 | 0.91158 |
| Target-similarity/pose ablation | Clean | 24.03612 | 0.63665 | 0.26653 | 0.91174 |
| Our regional compatibility adapter | Clean | 24.05182 | 0.63730 | 0.26477 | 0.90670 |

## Corrupted-reference paired comparison

Seven corruption conditions averaged within each identity.

| Our adapter minus local ReF-LDM | Change |
|---|---:|
| PSNR (dB) | +0.005803 |
| SSIM | +0.000156 |
| LPIPS | -0.000128 |
| Relative LPIPS reduction | 0.047% |

FaceNet has three unavailable measurements across the full experiment; complete paired identity support is absent. Both progression gates are INCONCLUSIVE, and the LPIPS target is independently unmet. No superiority or novelty claim is supported.

## How the papers are used

LaMa, ResShift, RefFaceInpainting and ReF-LDM have local implementation experiments; only ReF-LDM is the backbone in the matched table above. GFRNet, 3DFaceFill, DMDNet, ReFine, RefSTAR and ReSem-Face are related-work/architecture references, not locally reproduced results. Local ReF-LDM uses separately encoded reference images, so it is not an exact reproduction of the author benchmark.

Source: [complete measurements](target_compatibility_results_v1.json); [protocol and literature details](MATCHED_COMPARISON_AND_LITERATURE_20261002.md).
