# Test the local studio

Start from PowerShell in this project folder:

```powershell
.\.venv\Scripts\python.exe scripts/launch_studio.py
```

The launcher opens http://127.0.0.1:8765/ and leaves the local server running. The laptop must remain awake. No public hosting is enabled.

## Missing facial areas

On the main page, upload the damaged photo, paint the area to replace, and select reference-guided reconstruction with 3–4 clear photos of the same person. Start at 512 processing. Compare the result with the input before downloading. Large missing regions can produce incorrect identity or expression.

If the reference model says a marked area is too thin, use **Expand mask by 8 px**, paint a wider region, or use LaMa for a tiny scratch. Inspect the resulting effective mask. This check prevents disconnected marks from disappearing inside the reference model; it does not guarantee every fine boundary can be recovered. ResShift now retains thin damage when reducing its input mask to 256 pixels.

## Blurry or noisy areas

Open http://127.0.0.1:8765/confidence.html (also linked from the main page). Upload the photo, mark **Damaged, with some detail left**, and add 3–4 close-up reference photos of the same person. Select **Restore blur or noise — experimental ReF-LDM**, then reconstruct. White/unmarked areas are retained; partially trusted areas blend surviving evidence with restoration. Download the image and settings after completion.

ReF-LDM does not fill erased regions. If any marked region is **Completely missing**, choose the reference-guided missing-area method and **Strong reconstruction** instead. Gentle reconstruction is rejected for missing areas because it can retain the erasure. ReF-LDM uses fixed settings; the replacement-strength selector is disabled for it. Neither mode verifies that the reconstructed face is the true hidden appearance.

Uploading a different target clears the previous target's reference photos and result. Add matching references again. ReF-LDM checks for one sufficiently visible face per reference; rectangular references are framed without stretching. If a reference fails, replace it with a clearer close-up of the same person. These checks do not verify that different photos show the same identity.

For a saved evidence map, use an **opaque grayscale PNG matching the editor's displayed image dimensions**. Export the image and map together from the editor to retain alignment. White keeps pixels, gray preserves part of the surviving detail, and black marks complete absence. Suggestions only adjust regions already marked as partially damaged; they are heuristics, not accuracy predictions. Thin missing/partial marks remain represented when the server reduces the maps. If a running job loses connection, use **Reconnect to reconstruction** to retrieve that same job before editing or starting another.

Use well-framed single-face photos. New photos are accepted without retraining or dataset membership lookup, but quality is not guaranteed. Confidence-page non-square inputs fit into a 512×512 canvas without stretching; padding is preserved. Output is 512×512, not original-resolution recovery.

The model files and restricted dataset photographs remain local. Research uncertainty diagnostics are not a website quality guarantee or an automatically calibrated confidence feature.


## Accuracy audit update (24 September 2026)

Keep "Ignore opaque cover colour (experimental)" off initially. It can help when an opaque cover becomes sunglasses, but reduced identity similarity in the development audit. ResShift exports 512 pixels with visible pixels preserved while its model still operates at 256 pixels. For partial blur/noise, use the evidence-map page and inspect results carefully. No mode guarantees exact hidden facial features. See research/STUDIO_ACCURACY_REVIEW_20260924.md.


## Compare processing sizes

Choose a reference-photo model, add matching photos, and select **Compare 256 / 512 / 1024 (three runs)** under Processing detail. The same input, mask, references and seed are used for three sequential GPU runs. Results appear below the workspace with individual image/settings downloads. Every export is 512 pixels; the labels describe actual model processing resolution, not export dimensions. 256 and 1024 processing are experimental. No automatic accuracy score is shown without a clean ground-truth image. LaMa and ResShift retain their fixed native model resolutions.

The first three-size check and the later unfamiliar-image audit produced severe artifacts at 256 pixels. Treat 256 as experimental; 512 remains the default and becomes the main comparison preview/download when it succeeds. Individual size results remain available. The comparison is for inspection, not a promise of accuracy.

## Current verification and limits

The 24 September unfamiliar-image audit completed 25/26 generations and scored 29/30 rows on two additional development identities. Separate repair checks and recovery runs verify functioning masks, framing and restoration, with failed attempts retained. They do not establish reliable accuracy on arbitrary client photographs. Facial identity, gaze, expression and large missing features remain imperfect. See [client fixes and evidence](research/STUDIO_CLIENT_FIXES_20260924.md) and [the numerical audit](research/UNFAMILIAR_STUDIO_ACCURACY_V2.md).

## Mask refinement and result updates

Learned refinement now keeps all the pixels you marked and may add nearby pixels. Inspect its effective mask; added areas can alter valid details. LaMa is intended for small repairs and remains poor at replacing large eyes, noses or mouths. ResShift with an exact painted mask is a practical single-image facial baseline; matching reference photos at 512 processing provide another option, not a guarantee of identity.

Changing the target clears the previous person’s references. Changing the map or generation settings clears an old result; reconstruct again before downloading a result for the new settings. Refresh a previously open page to load these editor fixes, after saving any work you want to keep.
