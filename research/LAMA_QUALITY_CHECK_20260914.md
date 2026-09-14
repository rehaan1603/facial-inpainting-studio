# LaMa quality check — 14 September 2026

## Finding

The reported blurry/distorted face behavior is reproducible. The checkpoint returns images and correctly consumes RGB floats in [0,1] and a white-is-missing mask. Its TorchScript wrapper zeros masked pixels before calling the generator. This was not a reversed-mask or missing-inference problem.

Visual inspection of saved LaMa website examples showed implausible textures, blurred facial features and incomplete removal when an experimental learned mask missed damage. On the fixed identity_620 eye-mask diagnostic, 256-pixel inference produced a malformed patterned eye region; 512-pixel inference filled the eye region with smooth skin rather than correct eyes. Increasing resolution does not solve semantic facial completion.

The installed Big-LaMa is a general image-completion baseline. The [official repository](https://github.com/advimman/lama) lists Big-LaMa trained on Places2 separately from its face-data models. It is not an identity-conditioned face reconstruction model. No new face-specific LaMa weights were trained or substituted during this check.

## Implemented repairs

- Website LaMa now processes and exports the 512-pixel editing frame directly, avoiding the previous mandatory reduction to 256. ResShift remains at its established 256-pixel setting; reference mode remains 512. The historical command-line/research inference remains unchanged.
- Uploaded masks matching the original photograph now use the identical crop/fit transform as the photo. Previously all masks were independently stretched to a square, which misaligned nonsquare photo/mask pairs. Already-framed 512-square masks remain supported; other mismatched sizes receive an explicit error.
- LaMa is labelled for small repairs, with guidance that large missing facial features can remain blurry or distorted. The learned refiner remains explicitly experimental.
- No sharpening, face replacement or hidden model substitution was used to imply recovered identity.

## Verification

- Five real HTTP/GPU runs passed: LaMa painted/expanded/learned at 512 and ResShift painted/expanded at 256. All outputs changed masked pixels and exactly preserved input pixels outside the effective mask. See `lama_webapp_fix_checks_20260914.json`.
- Eleven HTTP boundary/dispatch tests passed.
- Geometry checks passed for portrait crop, landscape fit, already-framed masks and mismatched-size rejection.
- The updated page loads the new framing helper and displays the correct LaMa label and 512-pixel output setting.

These checks establish functional behavior and corrected preprocessing, not satisfactory facial fidelity. LaMa still fails when it must invent important hidden facial features.

## Remaining project work

The expanded 144-output evaluation already includes ArcFace, independent FaceNet, NIQE, BRISQUE, SSIM, PSNR, LPIPS and reference-count/statistical ablations. Remaining work is reliable facial completion, mask-aware reference selection/fusion, larger and varied masks, multiple seeds, disjoint conditioning/evaluation galleries, fair external-method comparisons, fresh held-out testing and a manuscript with supported claims. The current app and evaluation are not a completed novel publishable method. Public hosting remains paused.
