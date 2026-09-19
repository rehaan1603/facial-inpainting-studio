# Mask-aware reference selection v1

Implemented on 19 September 2026. This is an untrained, explicitly heuristic selector in front of the existing pretrained SDXL/FaceID Portrait generator. It does not perform regional feature fusion or establish novelty.

## Inference contract

`scripts/mask_aware_inpaint.py` accepts a damaged image, a matching nonempty white-replace mask and one to four arbitrary reference paths. Dataset identity labels, a clean target and evaluation galleries are absent from this interface. It writes the selected output, existing generator metadata and a `_selection.json` sidecar. Existing files cannot be overwritten.

Each reference must have exactly one detected face. Missing/corrupt photos and duplicate encoded/decoded photos are rejected individually with recorded reasons. A nonempty valid subset can proceed. If none remains, the default is an actionable error. The explicitly requested CLI `--fallback lama` performs a recorded single-image fallback; the website does not silently substitute it. Missing masks require user marking; automatic damage detection is not implemented.

## Fixed formula

Mask overlap with eight coarse regions determines normalized demand weights. Regions use the damaged input's detected box/five landmarks, with a documented generic frame fallback if no unique face is detected. Region names denote image-left/right. Reference regions use each uploaded photograph's own geometry.

The fixed score is 0.45 regional quality + 0.20 regional visibility proxy + 0.10 global quality + 0.10 pose similarity + 0.10 identity compatibility mapped to [0,1] + 0.05 regional exposure. Regional terms are demand-weighted. Quality combines saturating Laplacian variance and nonclipped exposure. Visibility and pose are coarse five-landmark proxies; they do not detect true occlusion, glasses or hair. Sharpness can reward noise. Scores are not probabilities.

Identity compatibility uses the damaged input embedding only when a unique face is detected and at least 60% of the coarse facial region is outside the supplied mask. Otherwise it uses median pairwise reference cosine similarity. One reference has neutral consensus zero. Pairwise cosine below 0.2 produces a disagreement warning, not identity verification or automatic exclusion. Ties retain input order. These values were fixed before examining generated outcomes.

Policies: first valid; seeded random valid; identity-only; whole-face quality-only; all valid; mask-aware top one. Zero-scale is an additional conditioning control using all references with adapter contribution zero. Invalid-file filtering is common to every policy. The all-reference arm changes the information budget relative to top-one policies.

The selected photograph supplies a global FaceID embedding. A region-based selection score does **not** mean regional image details are transferred. The experiment tests the practical effect of selection under this limitation.

Reference quality scores also depend on pixel resolution and face size. The controlled cases standardize photographs to 512 × 512, whereas arbitrary runtime uploads can differ. The current scorer does not calibrate this effect. It warns about disagreement but does not reliably identify which photo belongs to a different person, and it does not estimate actual occlusion coverage. These are explicit limits of v1.

## Local use

Use `C:\Users\rehaa\.cache\facial-inpainting\reference_env_v2\Scripts\python.exe` to run `scripts/mask_aware_inpaint.py --image OBSERVED.png --mask MASK.png --references REF1.png REF2.png --output NEW_RESULT.png`. Paths are arbitrary, and one through four references are accepted. The local website exposes **Choose from 1–4 photos · experimental** and **Run diagnostics**.

See `GENERALIZATION_RESULTS.md` for measured results. Code functionality alone is not a facial-fidelity improvement claim.
