# Base-paper comparison and improvement decision

Checked 30 September 2026. Literature comparison and proposed experiments only: no new model was trained or evaluated for this decision. Novelty and superiority remain unestablished.

## Decision

Use [ReF-LDM, NeurIPS 2024](https://arxiv.org/abs/2412.05043) as the reproducible base for reference-guided blur/noise restoration. Its author implementation already runs locally. ReF-LDM uses cached reference attention and a timestep-dependent identity loss. These are inherited contributions, not ours. Author code: https://github.com/ChiWeiHsiao/ref-ldm.

This comparison does not establish missing-region inpainting quality. Completely erased eyes require a separate inpainting experiment and relevant inpainting comparators. Keep both tasks separate in tables and claims.

## Cross-check against existing contributions

| Proposed broad idea | Prior work that prevents claiming the broad idea as new |
|---|---|
| Multiple photographs preserve identity | ReF-LDM already uses multiple references. |
| Select useful reference information | [RefSTAR, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38194) explicitly addresses reference selection, transfer and reconstruction. |
| Separate reference identity and degraded-image structure | [IConFace, revised August 2026](https://arxiv.org/abs/2605.02814v2) has identity and degraded-structure pathways, including localized evidence. |
| Richer reference features and improved identity loss | [Reference-Guided Identity Preserving Face Restoration, 2025](https://arxiv.org/abs/2505.21905) introduces Composite Context, Hard Example Identity Loss and multi-reference inference adaptation. |
| Preserve localized reference details | [Copy or Not?, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Chong_Copy_or_Not_Reference-Based_Face_Image_Restoration_with_Fine_Details_WACV_2025_paper.html) studies fine-detail transfer with spatial minimum and cycle-consistency losses. |

This is a focused overlap check, not an exhaustive novelty clearance. Search also surfaced adaptive reference-attention gating in general restoration work; generic spatial gating is therefore not a defensible novelty claim. Full-text comparison of those methods is required before freezing a new mechanism. RefSTAR's author PDF could not be fetched through the browser tool; its publisher description supports the limited claim above.

## One candidate worth investigating

Hypothesis: controlled reference interventions during training can teach a small adapter to suppress misleading local reference evidence while retaining useful identity information. Freeze the backbone initially. Present paired reference sets differing only in a controlled corruption, occlusion or wrong-person substitution. Train a spatial reliability predictor with known intervention labels, alongside reconstruction objectives. The output should remain stable when irrelevant reference regions change and improve when valid evidence is supplied.

Possible contribution is the precise intervention supervision, its interaction with evidence availability, and demonstrated robustness—not the presence of a gate. This remains a candidate: it may overlap existing work or fail experimentally. It differs operationally from our previous post-generation blending and landmark warping, but that difference alone is insufficient for novelty.

Clean targets may supervise training and offline scoring only. Inference receives the damaged image, damage mask and independent reference photographs. It must never use clean-target pixels, evaluation galleries, or the known identity label to choose outputs. Synthetic angle variants derived from the target are demonstrations, not independent reference evidence.

## Minimum experiment before claiming improvement

1. Reproduce the unmodified author baseline on a documented identity-disjoint development cohort with fixed degradation, reference count, seeds, resolution and sampling budget. Audit pretraining overlap where metadata allows; disclose unknown overlap. Do not compare our local scores directly with a paper table using different data or metrics.
2. Profile one adapter training step on the 8 GB laptop, recording peak memory and time. Gradient checkpointing, mixed precision and gradient accumulation are possible engineering choices, not a promise that training fits. Stop before an expensive run if feasibility fails.
3. Compare original baseline, baseline with identical output compositing, equal-capacity adapter without intervention supervision, global reliability gate, and proposed spatial intervention supervision. Give every arm the same tuning budget. Include clean, degraded, absent and wrong-person references.
4. Before the new validation run, freeze one primary endpoint: paired identity-level change in independent identity similarity under corrupted-reference conditions. Require masked LPIPS and clean-reference identity non-inferiority as guardrails, with margins justified from development variability and practical importance before validation. ArcFace is secondary wherever it also supplies conditioning. Report masked MAE/PSNR, detector failures, visible-region changes, latency and peak memory. NIQE/BRISQUE cannot establish correct identity.
5. Average seeds within each identity; use paired identity-level confidence intervals and report complete failure counts. Determine cohort size from a development-based power analysis. Four repeatedly inspected identities cannot establish generalization. Correct exploratory multiple comparisons; retain unsuccessful runs.
6. Conduct blinded paired review of eyes, gaze, mouth, identity and artifacts with a predefined rubric. A sharper but different face is a failure. Fresh real client images without ground truth support qualitative usability testing, not numerical reconstruction accuracy.
7. Only after the method and analysis are frozen, evaluate an independent cohort once. Do not inspect or tune on the eight reserved identities during development. Preserve existing frozen protocols and negative results.

## What would support a paper

A defensible submission needs both an identifiable difference from the closest prior methods and a repeatable benefit under matched evaluation. Beating ReF-LDM alone would support a claim against that baseline, not state of the art: newer relevant methods must also be compared where reproducible implementations are available. If the adapter helps only bad-reference cases, state that restricted robustness contribution rather than universal restoration superiority.

Reject the candidate if the equal-capacity control matches it, clean-reference quality declines beyond the frozen margin, benefits disappear on new identities, or closest prior work already implements the same intervention. Do not repeatedly change the method against the final test.

## Current evidence and immediate priorities

- Existing ReF-LDM development summary contains 48 rows from four identity units; it is development evidence, not a published-benchmark reproduction.
- Previous reference/context changes traded better identity similarity for worse missing-region fidelity. Structural transport produced no candidate images because all 24 attempts failed the deformation-validity gate. See STRUCTURE_TRANSPORT_REVIEW_V1.md.
- Studio seam/noise finishing fixes are engineering improvements. They do not establish a novel reconstruction model.
- Priority order: finish closest-method full-text overlap audit; freeze a matched baseline protocol; pass the laptop training feasibility check; implement and train the candidate only if the first gates pass; then run ablations and independent validation.

No positive experimental result or publication readiness is claimed by this document.
