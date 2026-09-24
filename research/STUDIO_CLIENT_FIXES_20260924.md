# Studio client reconstruction fixes — 24 September 2026

The application was checked for failures affecting unfamiliar uploaded photos, ordinary inpainting and evidence-map restoration. Concrete preprocessing, editor-state and runtime faults were repaired. Correct execution and preserved visible pixels do **not** prove that a generated face is accurate.

## Accuracy audit before the repairs

Two newly selected local development identities, 3182 and 1037, were excluded from earlier development and reserved-final groups before image access. They are now observed development cases; pretraining exposure remains unknown. The frozen audit attempted 26 generations and included four unchanged-input controls: **25/26 generations completed; 29/30 rows scored**. One OpenCV failure remains counted. Subsequent recovery checks do not replace that audit row.

All four comparison sheets were reviewed internally. Small 256-pixel reference outputs can leave coloured hole artifacts. Gentle 0.5 reconstruction retains substantial erased damage. LaMa, ResShift and reference methods still produce incorrect large facial features, gaze or expression. Better identity or image-quality scores do not establish correct hidden anatomy. The complete method table and detection denominators are in [UNFAMILIAR_STUDIO_ACCURACY_V2.md](UNFAMILIAR_STUDIO_ACCURACY_V2.md) and its numerical receipt.

## Implemented changes

| Failure | Current behavior | Remaining limitation |
|---|---|---|
| Thin mask components disappear during ResShift downsampling | Overlapping damage pixels are retained in the reduced binary mask; final composition uses the original display mask | A 256-pixel model cannot supply original-resolution detail |
| Missing/partial evidence disappears during map reduction | Mask maximum and evidence minimum are pooled over matching source footprints; padding remains reliable | Conservative map resampling, not calibrated confidence |
| Rectangular reference API photos are stretched | All source/map paths share aspect-preserving 512 framing | Output remains a processed 512-pixel canvas |
| Rectangular ReF-LDM references are stretched; unusable references reach inference | CPU face validation followed by a recorded square face crop; square references retain their original inputs | Crop framing is not learned alignment or same-person verification |
| Thin SDXL components lose all latent-mask support | Preflight rejects unsupported disconnected marks and suggests expansion, wider marking or LaMa | This contains the failure; it does not solve all fine-boundary reconstruction |
| Learned refinement removes user-painted damage | Studio refinement takes the union with the requested mask; zero marked pixels were dropped in both follow-up runs | Added regions can still replace valid detail; LaMa remains poor for large facial holes |
| A poor experimental 256 result becomes the main comparison | Successful 512 processing is preferred for the main preview/download; all sizes remain available | No automatic quality ranking or claimed optimal size |
| Evidence-page state mixes different targets or asynchronous uploads | New targets clear references/results; upload revisions and job locks prevent stale state; disconnected jobs can reconnect | Reloading the server or browser is not persistent job recovery |
| Suggestions overwrite explicit evidence labels; transparent maps are ambiguous | Suggestions affect only partial regions; transparent maps are rejected; missing regions require strong generation | Suggestions remain unvalidated heuristics |
| Intermittent native/OpenCV failure and a new wrapper dependency failure | Worker enforces one OpenCV thread through nested calls; ReF-LDM child omits the unnecessary OpenCV import; timeout terminates the child tree | Runtime mitigation and observed recovery, not proof of a universal root cause |

These are studio changes. Frozen research generator sources and evaluators remain unchanged; no model training or new inference-quality guarantee is implied.

## Verification and retained failures

| Check | Recorded result |
|---|---|
| Separate post-fix HTTP/GPU verification | **8/8 passed**: failed 1024 retry, default 512, expanded/learned LaMa and ResShift, thin ResShift mask, nonsquare reference upload |
| Default SDXL preservation | The checked 512 result is byte-identical to its pre-fix default output |
| Further client-input check | **2/4 initially passed**: both ResShift runs succeeded; both new ReF-LDM wrapper runs failed because its child environment did not install or need OpenCV |
| Separate ReF-LDM recovery after wrapper correction | **2/2 passed**, covering square and rectangular reference uploads; both preserve known pixels |
| Square ReF-LDM preservation | Output is byte-identical to the pre-fix result; all four square reference hashes remain unchanged |
| Automated regressions | **41 Python checks**: 19 HTTP and 22 geometry/preparation/runtime; **three Node suites passed** |
| Current browser workflows | Six real browser outputs verified, including missing-area generation, repeated partial restoration and all three selected-reference sizes; matching reference pixels, HTTP 200, output hashes and exact outside-mask preservation checked |
| Browser state | Changing person cleared references and disabled reconstruction; editing evidence cleared the prior result and explained the invalidation; successful 512 output was the main comparison preview; no console errors observed |
| Learned-mask correction | **2/2** follow-up generations and scores completed on one observed case; dropped requested pixels changed from **111 / 106** (LaMa / ResShift) to **0 / 0** |

Receipts: [eight repair checks](studio_fixes_verification_v2.json), [initial client-input attempts](studio_client_inputs_verification_v2.json), and [separate ReF-LDM recovery](studio_client_inputs_verification_v2_recovery.json). Each records its own code/output hashes; these are repeated functional checks on now-observed cases, not additional independent identity evidence. Restricted face images and comparison sheets remain local.

The final HTTP-suite attempt encountered one Windows `10053` connection abort during the cross-origin rejection test. An unchanged repeat passed **19/19**; both logs are retained locally. This transient test-transport failure was not an inference failure, and its underlying cause is not established.

## Measured effect and visual review

The corrected ResShift mask reduction was compared with the earlier outputs on the same two development faces, with other generation settings fixed:

| Metric | Earlier sampling | Coverage-preserving sampling |
|---|---:|---:|
| FaceNet cosine | 0.7481 | 0.8630 |
| Gallery FaceNet cosine | 0.5126 | 0.6761 |
| Original damage MAE | 0.08800 | 0.07163 |
| LPIPS | 0.02982 | 0.02037 |

All four quantities improved on each of the two cases. These are descriptive diagnostic results, not accuracy percentages or evidence of population generalization. Visual review showed better mouth/eye appearance but remaining differences from the true face. The learned-mask union also improved original-damage MAE, FaceNet and LPIPS on its single LaMa/ResShift check. LaMa still smears large eye/nose/mouth regions; learned refinement is not promoted as a quality improvement.

The initial post-fix scorer incorrectly supplied the original mask to a strict preservation check for four expanded/refined outputs. It rejected them because those modes intentionally edit a larger effective region. Those caller failures remain in `studio_fixes_quality_v2.json`. A separate corrected evaluation passes the saved effective mask, additionally measures MAE in the original damaged region, and scored **4/4** unchanged saved outputs. Neither the frozen evaluator nor model outputs were modified. See [quality receipt](studio_fixes_quality_v2.json), [mask-corrected receipt](studio_fixes_quality_v2_mask_corrected.json), [mask-union receipt](studio_refiner_union_verification_v2.json) and [browser receipt](studio_browser_verification_v2.json).

One earlier browser result was observed cleared while its old ready message remained; the invalidating event was not captured. Clearing a displayed result now updates its status. A repeated completed restoration and an explicit subsequent map edit both showed the intended display/invalidation behavior. Transient browser selector timeouts were checked against the actual DOM and loaded image; they were not counted as failed generations.

## Still required

Broaden client-photo framing, pose, expression, mask and reference-quality coverage with failures retained. Persistent large-hole facial errors require a justified method improvement and matched accuracy evaluation, not more success-only demonstrations. Blinded human assessment, larger identity-separated validation, a beneficial research contribution, method freeze and reserved-final evaluation remain outstanding. All eight reserved-final identities remain untouched; novelty and publication readiness are not established.
