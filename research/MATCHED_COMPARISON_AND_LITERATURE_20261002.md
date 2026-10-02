# Comparison ledger and research scope

Checked 2 October 2026 against the primary sources linked below. New Human Faces training remains separate. **No superiority or publication readiness is established.**

Published numbers from different datasets and mask protocols are provided for context only and are not treated as direct evidence of superiority. No column below is a cross-paper leaderboard. NR means not reported in the cited table; UV means not verified, not zero. Identity retrieval, CosFace similarity, ArcFace similarity and reference-based IPS are different measures.

## A. Published numbers: contextual only

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

The quoted values are a deliberately identified row per source, not an exhaustive extraction or each paper's universally best score. ReF-LDM reports additional face-region LPIPS and warns about pretraining overlap. RefFace's paper text has inconsistent dataset identity counts; its split must be resolved from released metadata before reproducing the published score. ReSem-Face is a recent preprint, not an independently verified result here.

## Architecture, protocol and reproducibility inventory

| Method | Task, input and datasets | Masks / resolution | Main mechanism and practical limitation | Official assets / local status |
|---|---|---|---|---|
| LaMa | Masked RGB + mask; Places/CelebA-HQ | Narrow, wide, segmentation masks; train 256, higher-resolution inference | Fourier-convolution network and large-receptive-field loss; no identity reference | [Author code/checkpoints](https://github.com/advimman/lama); pretrained local inference executed; published benchmark not reproduced |
| ResShift | Damaged RGB; face inpainting branch trained on FFHQ, tested on CelebA-Test | Box/irregular/half/expand; local face checkpoint 256 | Latent residual-shifting diffusion; can invent hidden features | [Author code/checkpoints](https://github.com/zsyOAOA/ResShift); pinned checkpoint executed locally |
| RefFace | Masked RGB + mask + one reference; curated Celeb-ID-derived data | Free-form masks; 256 | Identity control plus component-wise texture injection/segmentation; sensitive to component correspondence | [Author code/checkpoints](https://github.com/WuyangLuo/RefFaceInpainting); 16 local predictions and 28 comparison rows completed; quality gate failed |
| ReF-LDM | Degraded RGB + variable reference set; FFHQ-Ref, CelebA-Test-Ref | Whole-image degradation, no required missing mask; 512 | CacheKV diffusion and timestep-scaled identity training; not inherently trained for erased regions | [Author code/checkpoints](https://github.com/ChiWeiHsiao/ref-ldm); local restoration and small adapter pilot executed; individual reference encoding differs from author image concatenation |
| GFRNet | Degraded RGB + one HQ reference; VGGFace2/WebFace | Blur/noise/compression/downsampling; 256 | Learned warping followed by guided reconstruction; inaccurate alignment harms transfer | [Author repository](https://github.com/csxmli2016/GFRNet); assets advertised, downloads not validated this turn; not locally reproduced |
| 3DFaceFill | Masked single face; CelebA/CelebA-HQ/MultiPIE | Face-only rectangles; align 256 then crop 224 | 3D factorization, UV albedo completion, rendering; face-surface restriction and 3D estimation assumptions | [Author code](https://github.com/human-analysis/3dfacefill); checkpoint links exist but complete asset availability unverified; not reproduced |
| DMDNet | Degraded face with optional same-person dictionary; FFHQ/CelebA-HQ/CelebRef-HQ | Whole-image restoration; 512 | Generic/specific component dictionaries; landmark failures documented | [Code and checkpoint link](https://github.com/csxmli2016/DMDNet); variable reference count, exact count for the selected table row UV; not reproduced |
| ReFine | Degraded face + one reference; CASIA-WebFace and detailed-face examples | Whole-image restoration; exact benchmark image size UV | Generative prior, spatial-minimum and cycle losses; copying reference details can differ from target truth | [Author-associated replacement release](https://github.com/JianWang-CMU/RefineFIR); original paper repository inaccessible; checkpoint retrieval UV; not reproduced |
| RefSTAR | Degraded face + one reference; CelebRef-HQ/RefSel-HQ, Celeb-Ref-Test, RealRef60 | Learned transferable-region mask; 512 | Supervised region selection, dual-stream attention and mask-compatible cycle consistency; broad compatibility/region selection already prior art | [Author code/model release](https://github.com/yinzhicun/RefSTAR); not downloaded or reproduced locally |
| ReSem-Face | Masked target, mask, text and five references; CelebAHQ-IDI-5/VGGFace2 | Eye/brow, lower face, whole face and random masks; exact pixel size UV | Identity-conditioned semantic pre-inpainting and diffusion; personalization and sizeable training budget | [Primary paper](https://arxiv.org/html/2608.04820v1); official downloadable code/checkpoints not verified; not reproduced |

## B. Matched local comparisons

Two panels are required; combining them would confound tasks.

1. **Restoration:** original 16-training/four-observed-development identity pilot. Same cached degraded targets, four references, 512 output, seed 17, 50 DDIM steps, guidance 1.5, bfloat16 and evaluator. New eight-condition comparison includes the frozen ReF-LDM baseline, equal-capacity no-target control, and target-conditioned regional candidate. Legacy reconstruction/intervention rows are comparable only for their recorded clean/wrong-person-patch conditions, not the six new corruptions. ReF-LDM and the current reference baseline are the same backbone here, not two independent competitors.
2. **Missing-region completion:** existing distortion/removal cases with the same observed image/mask and 512 composited output. LaMa, ResShift and the fixed-first-reference RefFace result belong here. Native 256-to-512 resampling must be disclosed. RefFace supports one reference: use the predeclared first of the same reference pool, never the best using target scores. A multi-reference extension may receive more information; it is not a reference-count-matched causal ablation.

Do not force a full-image restoration model and an inpainting model into an apparently identical task by silently erasing the entire image or changing masks. Do not duplicate rows under “ReF-LDM” and “current baseline.” Local archived outputs will be rescored with the same regional definitions. New method results and failure records are written under `outputs/target_compatibility_v1`, separately from all historical experiments.

## Decision and remaining evidence

Region selection alone is not novel: RefSTAR, RefFace, DMDNet and earlier adaptive spatial fusion already overlap. The present test is a **hypothesis** about observed-target compatibility under controlled reference corruption, not a novelty announcement. Ordinary attention already uses target queries; our proposed distinction is a regional log-prior computed from fixed observed-target features, pose/quality proxies and reference consensus rather than noisy denoising queries alone. Whether that distinction is useful and sufficiently original remains unresolved.

The first screen uses already-observed development identities and one seed. If it fails the declared joint progression gate, retain FAIL/INCONCLUSIVE and do not open reserved final identities. If it passes, freeze a follow-up protocol covering an independent cohort, multiple seeds, component ablations, human fidelity review, clean-reference non-inferiority and appropriate uncertainty/multiplicity reporting. Numerical engineering targets are not acceptance criteria for inventing results.
