# Expanded distortion visual review

20 September 2026. Reviewed all 36 successful case sheets: identities 1485, 2087 and 2289 × six degradation kinds × two severity/mask strata. All standard/preservation arms and unchanged inputs were visible together. The twelve cases for identity 386 have no generated results because reference detection failed; their failure placeholders are retained. No reserved-final images were opened.

This is an unblinded engineering review by the assistant, not independent human validation. Severity and mask shape change together. Photographs and contact sheets remain local under outputs/distortion_extension_v1/contact_sheets. Reproduce the sheets with scripts/render_distortion_extension.py.

| Condition | Repeated visual finding | Decision |
|---|---|---|
| Fully removed | Strength 0.5 leaves gray/softened holes. Strength 0.99 creates facial features but changes eyes, lips and expression; severe removal can introduce cheek marks. Preservation is intentionally identical to generation for wholly missing pixels. | No universal low-strength default |
| Blur | Lower strength often retains soft eyes; high strength sharpens plausible but different features. Blending observed evidence retains some blur. | Sharpness does not establish fidelity |
| Noise | Lower strength leaves patterned skin/nose artifacts; high strength cleans texture while changing facial details. Preservation reintroduces some noise. | Need degradation-specific restoration |
| JPEG | Inputs retain considerable identity detail; generation unnecessarily changes lips, eyes or expression. Preservation attenuates those changes. | Keep unchanged-input control |
| Downsampling | Blurred eye/mouth detail can persist at low strength; high strength substitutes different details. | No verified recovery of true detail |
| Mixed | Severe low-strength outputs include conspicuous colored/cross-like cheek artifacts for 1485 and 2087 and altered makeup/lips for 2289. High strength changes expressions; preservation reduces artifact contrast but does not remove the underlying failure. | Do not promote based on mean metrics |

All three identities show subject-specific failure modes. Identity 2289's glasses and makeup are a challenging appearance factor, not evidence of broad pose/occlusion robustness. More independent identities and a blinded fidelity assessment remain necessary. The visual evidence supports the numerical decision not to promote local fusion or any tested preservation setting as a superior restoration method.
