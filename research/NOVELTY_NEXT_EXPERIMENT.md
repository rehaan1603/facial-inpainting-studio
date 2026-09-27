# Next contribution hypothesis — 22 September 2026

## Current decision — 27 September 2026

Two additional mechanisms have now been implemented and tested, not merely proposed. Reference restoration with original-context correction completed 32 new generations; a subsequent masked-conditioning screen completed 24. Each has 40 scored comparison rows including reused controls and two separate GPU equivalence calls. The first improves identity but worsens pixel/perceptual error against the original scaffold. The second visibly erases facial features and loses detector coverage. Both fail their frozen gates; no training or website promotion is justified. See `NOVELTY_ESTABLISHMENT_20260925.md` for metrics, visual review, integrity and prior art.

The next action is a **published missing-region reference baseline**, followed by a prospectively defined mechanism only if a reproducible structural limitation can be targeted. Existing restoration-plus-completion combinations are not sufficient novelty. RefFaceInpainting's official code accepts a damaged target, one reference and reference-only parsing; ground-truth target parsing is not needed in its inference path. Local reproduction still requires the exact author weights and preprocessing, strict loading, a frozen development comparison and failure-inclusive evaluation.

A larger fresh cohort is also unresolved: at most 11 unused local validation groups have eight photos, and the official FFHQ-Ref validation metadata adds only four predicted reference groups of that size before screening. No FFHQ pixels have been accessed. Do not take author training data or reserved-final images to enlarge development results. A different reference/gallery count is a new protocol, not a retrospective relaxation of completed gates.

The historical hypotheses below document the research trajectory. They do not override this decision or authorize further tuning on the same observed targets.

## Latest continuation — 25 September 2026

A different, single-image mask-support selection mechanism is now implemented and tested. It uses deliberately hidden reliable context to choose among four conditioning-mask radii, without updating the generator or reading target truth. The frozen screen completed 104 generations and 80 scored rows over four already observed identities and two seeds, including equal-compute controls. It failed: selected/original-mask FaceNet is 0.6111/0.7415 and hole MAE is 0.07665/0.06266. Five of eight probe-versus-hole-error rank correlations are negative. See `CONTEXT_SUPPORT_METHOD.md`, `CONTEXT_SUPPORT_RESULTS_V1.md` and `CONTEXT_SUPPORT_REVIEW_V1.md`.

Neither context-probe calibration, reference-proxy blending nor disagreement gating currently justifies more calibration data, adapter training, or a novelty claim. The next proposal must directly address facial structure/identity errors and state why its information is available for an unknown input. Surviving target evidence and reference identity evidence must be distinguished from unknowable hidden expression. No specific successor has yet earned a training or final-test gate. The historical proposals below remain rejected or unvalidated research history, not a promise of publication.

23 September continuation: a different reference-proxy calibration mechanism was implemented and tested; see `REFERENCE_PROXY_METHOD.md` and `PROXY_CALIBRATION_RESULTS_V1.md`. It uses measured errors on synthetically damaged supplied references, not disagreement, to fit spatial blending. Twenty proxy generations and twenty candidate evaluations completed. It also failed the progression gate: a small pixel-error improvement over fixed blending was accompanied by worse identity/gallery similarity and LPIPS. Neither diagnostic supports promoting a calibrated reliability claim or starting fusion-adapter training. The proposed designs below remain research history, not a validated roadmap to superiority.

## 23 September diagnostic outcome

The first uncalibrated reference-disagreement prototype is now implemented and tested; it is no longer only a proposal. Sixteen leave-one-out generations and twelve candidate evaluations are complete. Disagreement AUC for harmful pixel edits is 0.417–0.533 on the four observed mixed cases. Its gate improves FaceNet versus the all-reference baseline but loses to an edit-magnitude gate on identity and masked error; both gates worsen LPIPS. The incremental reference-uncertainty hypothesis is therefore unsupported by this diagnostic. Do not describe the map as calibrated confidence, train a fusion adapter on this premise, or claim novelty from it. See `REFERENCE_RISK_RESULTS_V1.md`. The design below is retained as the pre-experiment hypothesis; larger calibration is not automatically justified by these results.

## Decision, not a novelty claim

The completed regional/global and local-latent mechanisms did not establish an advantage. Do not train a fusion adapter merely to obtain a trained component. First finish the matched ReF-LDM restoration baseline. Its successful execution is an engineering milestone, not a contribution.

The candidate research question is: can reference-dependent uncertainty predict **harmful edits to surviving facial evidence**, allowing selective reconstruction that improves the identity–restoration tradeoff on people absent from calibration? A sharp image or reference agreement alone does not establish correct identity or expression.

## Proposed mechanism and falsifiable controls

1. Use a frozen restoration-specific backbone. Keep the same seed, observed image and mask when testing reference sensitivity. Generate the all-reference candidate and leave-one-reference-out candidates; do not choose candidates using clean targets or evaluation-gallery embeddings.
2. Measure local output disagreement, edit magnitude against the observed image, and input-only evidence reliability. Disagreement is a candidate risk feature, **not calibrated uncertainty**. References can agree on a wrong expression, and identical results can share systematic bias.
3. On separate calibration identities only, test whether these features predict an increase in local reconstruction error relative to the observation. Freeze the predictor and operating threshold before separate development validation. Do not reuse an identity across calibration and validation through different photographs.
4. Preserve surviving evidence where estimated edit risk is high. Missing pixels require a separate policy: retaining an erased pixel is not successful reconstruction. Flag unreliable completion rather than silently presenting it as recovered truth. Exact unmasked pixels remain unchanged.
5. Compare with unchanged input, raw restoration, hard mask composition, fixed confidence blending, edit-magnitude-only gating, disagreement-only gating and random gating at matched edited area. Report edit coverage and risk together so a method cannot win merely by doing nothing.

The incremental hypotheses are whether reference perturbation adds value beyond edit magnitude, whether calibration transfers to unfamiliar identities and damage types, and whether the mechanism improves fidelity without unacceptable remaining damage. These controls are required before a contribution claim. The implementation details and calibration sample size must be frozen in a separate run protocol before collecting outcomes; this document is not that protocol.

## Evaluation and progression gates

- Use FaceNet target and withheld-gallery similarity as independent identity outcomes; ArcFace stays diagnostic when used in conditioning. Also retain LPIPS, SSIM, PSNR, masked/visible MAE, NIQE, BRISQUE, detection coverage and visible failure review.
- Separate fully missing regions from partial degradation; independently vary mask, degradation severity, pose, expression and reference quality. Add unfamiliar development identities, keeping the eight reserved-final identities unopened.
- Evaluate identity-level uncertainty intervals, harm-versus-edit-coverage curves and human judgments of gaze, expression and facial geometry. All failures remain in the ledger. A benefit on the four observed screen identities is exploratory only.
- Proceed to an adapter only if the deterministic mechanism has a convincing, replicated benefit against simple controls. Otherwise reject or revise it. No automatic website promotion follows from a single favorable metric.

## Prior-art limits

ReF-LDM already uses multi-reference latent restoration; Differential Diffusion already supports spatial change control; ReFine and RefSTAR already transfer/select reference features. The existing novelty audit applies. A 22 September search also found the authors' [ID-PreFeR project](https://id-prefer.pages.dev/), which describes person-specific adaptation with mixed-quality references and degradation separation. That is an additional overlap to investigate, not a reproduced baseline here. Its public page lists code as forthcoming at inspection time. Our proposed risk-calibrated editing distinction is still a hypothesis and requires a fuller paper-level overlap check.

No claim of being first, publishable superiority, calibrated confidence, or reliable unknown-person recovery is currently justified.
