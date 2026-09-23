# ReF-LDM matched development comparison â€” 22 September 2026

External author baseline, not a novel project method. Four already-observed development identities; six damage types; 50 DDIM steps, seed 17, CFG 1.5. Native whole-image outputs and exact outside-mask compositions are evaluated separately. No clean target/gallery conditions generation. Pretraining overlap is unknown. Eight final identities remain untouched.

Primary analysis averages five partially degraded conditions within each identity. Removal is exploratory and out of domain. Eight predeclared tests use exact sign flips, identity bootstrap intervals and Holm correction. Four identities are underpowered. Native outputs may change visible pixels; their actual changes are recorded separately without altering frozen metric formulas.

| Arm (20 partial cases) | FaceNet | Hole MAE | LPIPS | NIQE |
|---|---:|---:|---:|---:|
| native | 0.8997704744338989 | 0.04602661873225027 | 0.22269090190529822 | 4.631849752997046 |
| composited | 0.921619114279747 | 0.04602661873225027 | 0.026691996352747084 | 4.150685197858535 |
| observed | 0.9537710040807724 | 0.033665074678170745 | 0.03443129344377667 | 4.4242053505833265 |
| sdxl_low | 0.7363824993371964 | 0.07081131188658965 | 0.02817770102992654 | 4.635801392076155 |
| sdxl_high | 0.7134480714797974 | 0.09031922147817631 | 0.031157161574810743 | 4.732604534582467 |
| preserve_high | 0.8273447155952454 | 0.05262465736112041 | 0.026424729777500034 | 4.5202478579189105 |

## Primary comparisons

| Contrast | Identities | Delta | CI95 | Holm p |
|---|---:|---:|---|---:|
| native_minus_sdxl_low/facenet_cosine | 4 | 0.16338797509670258 | [0.10588700771331788, 0.2393921703100204] | 1 |
| native_minus_sdxl_low/hole_mae | 4 | -0.024784693154339378 | [-0.031822740939049604, -0.016818690592147797] | 1 |
| native_minus_sdxl_high/facenet_cosine | 4 | 0.18632240295410157 | [0.1458455502986908, 0.2445225775241852] | 1 |
| native_minus_sdxl_high/hole_mae | 4 | -0.04429260274592603 | [-0.05146440504159901, -0.03736149705092426] | 1 |
| composited_minus_sdxl_low/facenet_cosine | 4 | 0.18523661494255067 | [0.12444091439247135, 0.25726655423641204] | 1 |
| composited_minus_sdxl_low/hole_mae | 4 | -0.024784693154339378 | [-0.031822740939049604, -0.016818690592147797] | 1 |
| composited_minus_sdxl_high/facenet_cosine | 4 | 0.20817104279994966 | [0.1719910562038422, 0.26240759491920473] | 1 |
| composited_minus_sdxl_high/hole_mae | 4 | -0.04429260274592603 | [-0.05146440504159901, -0.03736149705092426] | 1 |

All metrics and per-condition coverage are in `refldm_results_v1.json`. Historical 0.8180 covers a different condition mixture and is not a matched comparator. No website promotion or publication-ready superiority follows from this small screen.

## Visual review and decision

All four six-condition sheets were inspected. All four erased-region cases retain the gray missing areas: this restoration model is not a working missing-region completer. Partial-damage results generally preserve facial structure better than the displayed SDXL control, with remaining smoothing and eye/mouth discrepancies. Native restoration changes reliable content; hard composition prevents that outside the mask. This was not a blinded human study.

The unchanged-input control remains better on partial-case identity similarity and masked MAE. Thus this baseline is promising relative to SDXL, but does not yet justify a claim of better fidelity than the surviving observation. No generator is promoted to the website on this evidence.
