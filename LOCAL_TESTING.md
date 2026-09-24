# Test the local studio

Start from PowerShell in this project folder:

```powershell
.\.venv\Scripts\python.exe scripts/launch_studio.py
```

The launcher opens http://127.0.0.1:8765/ and leaves the local server running. The laptop must remain awake. No public hosting is enabled.

## Missing facial areas

On the main page, upload the damaged photo, paint the area to replace, and select reference-guided reconstruction with 3–4 clear photos of the same person. Compare the result with the input before downloading. Large missing regions can produce incorrect identity or expression.

## Blurry or noisy areas

Open http://127.0.0.1:8765/confidence.html (also linked from the main page). Upload the photo, mark **Damaged, with some detail left**, and add 3–4 close-up reference photos of the same person. Select **Restore blur or noise — experimental ReF-LDM**, then reconstruct. White/unmarked areas are retained; partially trusted areas blend surviving evidence with restoration. Download the image and settings after completion.

ReF-LDM does not fill erased regions. If any marked region is **Completely missing**, choose the reference-guided missing-area method instead. ReF-LDM uses fixed settings; the replacement-strength selector is disabled for it. Neither mode verifies that the reconstructed face is the true hidden appearance.

Use well-framed single-face photos. New photos are accepted without retraining or dataset membership lookup, but quality is not guaranteed. Confidence-page non-square inputs fit into a 512×512 canvas without stretching; padding is preserved. Output is 512×512, not original-resolution recovery.

The model files and restricted dataset photographs remain local. Research uncertainty diagnostics are not a website quality guarantee or an automatically calibrated confidence feature.


## Accuracy audit update (24 September 2026)

Keep "Ignore opaque cover colour (experimental)" off initially. It can help when an opaque cover becomes sunglasses, but reduced identity similarity in the development audit. ResShift exports 512 pixels with visible pixels preserved while its model still operates at 256 pixels. For partial blur/noise, use the evidence-map page and inspect results carefully. No mode guarantees exact hidden facial features. See research/STUDIO_ACCURACY_REVIEW_20260924.md.


## Compare processing sizes

Choose a reference-photo model, add matching photos, and select **Compare 256 / 512 / 1024 (three runs)** under Processing detail. The same input, mask, references and seed are used for three sequential GPU runs. Results appear below the workspace with individual image/settings downloads. Every export is 512 pixels; the labels describe actual model processing resolution, not export dimensions. 256 and 1024 processing are experimental. No automatic accuracy score is shown without a clean ground-truth image. LaMa and ResShift retain their fixed native model resolutions.

The first three-size check produced a severe eye-region artifact at 256 pixels. Treat 256 as an experimental comparison setting; 512 remains the default. The comparison is for inspection, not a promise of accuracy.
