# Development reference diagnostics

This folder contains 12 selected HQ validation identity-label groups, each with one target and four reference photographs. `manifest.json` is the entry point; `selection_audit.json` records eligibility and every attempted image; `signature.json` fixes preparation inputs and code.

Each case provides `observed.png`, a white-is-missing `mask.png`, and `reference_1.png` through `reference_4.png`. Only those files belong in an inference request. `target.png` is retained as a development comparison and must never be passed to the model. Masks are synthetic eye rectangles constructed from unobscured target landmarks; the model must not receive those landmarks.

The selected labels do not overlap any identity label used by earlier pilots, v2 tuning or assessment, v3, the original HQ object test, the earlier reference smoke example, or identity 916. This is a stronger exclusion than omitting assessment/test labels alone, but remains a development set, not a fresh final test.

Selected photographs have different source and decoded-pixel hashes, and no pair among the 60 selected photos has pHash Hamming distance at most six. Exact hashes were also compared against protected HQ train/test and excluded validation records. The earlier review quarantined known near-duplicate photographs, but its candidate search omitted within-dataset, within-split pairs. Therefore this preparation does not establish absence of near duplicates between these cases and previously used validation photographs. Hash screening can also miss crops, flips, and other transformations. Supplied identity labels are not independently verified person identities, and model pretraining overlap remains unresolved.

Selection used only dataset labels, hashes, and CPU face-detection evidence. No generated output or inpainting metric influenced selection. Face-detection failures and photo availability bias which groups qualify. Preserve all subsequently generated results; do not replace cases because reconstruction quality is disappointing.

Reproduce with the existing local reference environment:

```powershell
& 'C:/Users/rehaa/.cache/facial-inpainting/reference_env/Scripts/python.exe' scripts/prepare_reference_diagnostics.py
```

The script refuses to replace this version when its source, detector, input manifest, or exclusion sources change. Original datasets and previous experiment protocols remain untouched.
