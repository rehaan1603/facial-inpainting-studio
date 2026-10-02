# Single-image reconstruction: boundary matching update

1 October 2026. Scope: studio quality improvement with existing models; no training, dataset replacement or novelty claim.

## Change

LaMa and ResShift previously hard-pasted predictions into the displayed image. Boundary matching already existed for reference inference but was not available as a visible single-image option. Reused the unchanged `reference_blending.harmonize_reference` Poisson compositor in a separate CPU process using installed `reference_env_v2`. No packages or model weights were installed. Only damaged input, mask and prediction are passed to this process.

The main page now offers **Match colour at edges** for all models. It starts off for LaMa and on for ResShift/reference modes. Unchecking it preserves raw output byte-for-byte. Single-image runs expose **Before edge matching**, alongside finished image and metadata recording the actual blend and any fallback. A missing finishing environment or unsupported boundary mask produces a visible raw-result warning; unexpected worker failures remain errors with a saved log. Existing reference/evidence generation and frozen research generators remain unchanged.

## Verification

- Eight preliminary recompositions used saved outputs from the first two new-dataset photographs; these are postprocessing diagnostics, not new model generations. The first attempt in project Python stopped because OpenCV is isolated; recovery used the existing evaluation environment. No dependency installation was needed.
- Completed **18 real model generations**, retaining raw and matched outputs: 16 on four newly supplied photographs (two masks, two models) and two ResShift runs on existing development identities 3182 and 1037. These are development data. Source/pretraining overlap of the supplied dataset is unresolved. The first five completed runs were retained when a user run occupied the studio; verified resume completed thirteen more without interrupting that run.
- **18/18** raw and matched pairs preserve input outside the effective mask exactly. All **8/8** raw outputs with earlier counterparts match previous PNG hashes byte-for-byte, isolating finishing from generation.
- **23 Python checks pass**: three real finishing/contract checks and twenty HTTP boundary/dispatch tests. Three JavaScript polling checks and main-page JavaScript syntax pass. The actual browser displayed a completed ResShift result and raw-output download. This observed browser run is not an additional member of the evaluation cohort.
- All 18 comparison sheets were visually reviewed. The new-dataset sheets are copied to `Downloads/Human dataset/Ready to test - Human Faces v1/Edge matching comparisons`. Photos remain local.

## Results and limits

| Model | Paired cases | Lower masked MAE after matching | Mean raw MAE | Mean matched MAE |
|---|---:|---:|---:|---:|
| LaMa | 8 | 5/8 | 0.05775 | 0.05680 |
| ResShift | 10 | 8/10 | 0.09946 | 0.07392 |

MAE is on the 0–1 pixel scale inside the mask. ResShift mean error decreases 25.7% on this small heterogeneous development set; that is **not a 25.7% identity-accuracy gain**. Cases share source photos and are not statistically independent. No significance or generalization claim is made. LaMa's mixed results are why its default remains raw.

ResShift colour seams decrease in several large rectangular holes and scratch repairs become less conspicuous. Gaze and feature shape remain wrong in large missing regions. Photo 04 still has a glasses-like hallucination and MAE worsens from 0.10938 to 0.11057. Existing case 3182 worsens slightly from 0.06449 to 0.06552; case 1037 improves from 0.07877 to 0.07746. LaMa still fails to generate faithful eyes behind large rectangles. Edge matching is a finishing option, **not a solution to missing identity information**.

No new ArcFace/FaceNet/LPIPS assessment was performed for this finishing update. Blur/noise reconstruction and synthetic-angle references were not changed or tested here. New low-resolution photos and different masks do not constitute a matched comparison to the old dataset's quality. Equal performance across datasets is not established.

Numerical receipt: [single_image_edges_v1.json](single_image_edges_v1.json). Check: `scripts/verify_single_image_edges_v1.py` retains previous attempts and verifies hashes on resume. Existing datasets, clean comparison images and trained checkpoints were not altered.
