# Accuracy findings and website decision — 24 September 2026

## Completed audit

The audit started on 23 September using deployed commit `7efabad`. It attempted 36 reconstructions across four already-observed development identities: six missing-region variants and three partial-damage variants per identity. Eight unchanged observations were controls. **34/36 generation attempts completed; 42/44 rows were scored.** All 34 generated outputs preserved observed pixels outside the exact supplied mask. Two failed attempts remain in the original results.

`STUDIO_ACCURACY_V1.md` and `studio_accuracy_v1.json` contain the complete measurements, denominators and per-image evidence. These are development results, not final-test results. To reproduce the frozen HTTP audit's default behavior, use the deployed code at `7efabad`; the revised website now makes its neutralization behavior explicit and optional. Do not silently rerun that frozen audit against a changed default.

## Visual inspection

All eight comparison sheets were reviewed against the clean targets, including every successful reconstruction and the failed slots.

- Missing-region LaMa results blur or erase eyes and mouths. Lower pixel error does not imply correct anatomy.
- ResShift generates recognizable structures but changes eye shape, mouth shape, teeth, and expression. It does not reliably recover the target's hidden features.
- Reference results retain more identity cues in several cases, but produce wrong gaze, lip shape, makeup/contrast and expressions. The neutralized 512-pixel path is often soft; 1024 pixels is sometimes clearer but still invents details.
- All partial-damage variants retain some blur or uneven texture. ReF-LDM with the fixed evidence blend has the strongest paired metrics in this small mixed-damage subset, but its visual detail is still not an exact match. It is not suitable for completely missing regions.

## Regression and deployment decision

The neutralized 512-pixel all-reference path reduced FaceNet similarity relative to the previous saved generator by **0.2047, 0.1716 and 0.1596** in the three completed matched cases. The fourth generation failed. Mean identity delta over those three complete pairs is approximately **−0.1786**. Therefore the workaround is **not retained as the default**.

The main website returns to its previous reference generation path by default. The new unchecked option, **Ignore opaque cover colour (experimental)**, retains the workaround for cases that otherwise copy the obstruction or generate sunglasses. Its warning explicitly states that resemblance can worsen. The option is unavailable with partial-damage evidence maps. ResShift's exact visible-pixel preservation fix remains active. Fifteen HTTP tests and six geometry/composition tests pass after the change; JavaScript syntax passes.

For mixed damage, ReF-LDM plus the fixed 128/255 evidence weight improves FaceNet, masked MAE and LPIPS versus the unchanged observation in **4/4** cases. Means are **0.9233 vs 0.8960** FaceNet, **0.0414 vs 0.0446** masked MAE, and **0.0364 vs 0.0519** LPIPS. Both SDXL evidence settings worsen identity and masked MAE in **4/4** cases despite improving LPIPS. These findings apply to this specific four-person subset and fixed blend, not all blur types or unknown photographs.

## Runtime failures

One run failed in OpenCV colour conversion before reference detection. A separate stress test of the same references completed 1,000 conversions without an error (OpenCV 5.0.0, default 24 threads). The other subprocess stopped at the start of sampling without a Python traceback. Causes remain unresolved. New failures now append the process exit code to the local inference log to improve diagnosis. Separate retries are recorded in `studio_accuracy_recovery_v1.json`; they never replace the original failed audit rows.

## What remains

Identity and expression accuracy inside a missing region are **not reliably maintained**. Better independent-identity testing, multiple masks/severities/seeds, explicit expression/gaze evaluation, and stronger reconstruction methods remain necessary. This audit does not validate the experimental mask refiner or every possible uploaded image. No new model training, final-test access, or novelty claim occurred. Dataset and user photographs remain local.


### Recovery verification

Both runtime retries completed with original settings, and their outputs were scored separately. The restored default verification reproduced the previous 1306 PNG **byte-for-byte**, confirming that the earlier reference behavior is restored. Original failed audit attempts remain counted. The two recovered outputs still show imperfect eye and mouth shape.
