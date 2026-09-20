# Distortion and local-feature phase: completion record

20 September 2026. Development-only checkpoint. No reserved-final identities used.

## 1. Files created

- `research/DISTORTION_AWARE_GENERATION_RESULTS.md`
- `research/DISTORTION_EXTENSION_RESULTS_V1.md`
- `research/LOCAL_FEATURE_CORRESPONDENCE.md`
- `research/LOCAL_LATENT_RESULTS_V1.md`
- `research/NOVELTY_GAP_ANALYSIS.md`
- `research/distortion_aware_evidence_v1.json`
- `research/distortion_aware_results_v1.json`
- `research/distortion_extension_evidence_v1.json`
- `research/distortion_extension_results_v1.json`
- `research/distortion_phase_verification_v1.json`
- `research/local_latent_evidence_v1.json`
- `research/local_latent_results_v1.json`
- `research/protocols/distortion_aware_v1.json`
- `research/protocols/distortion_extension_v1.json`
- `research/protocols/local_latent_v1.json`
- `scripts/check_local_correspondence.py`
- `scripts/distortion_aware_study.py`
- `scripts/distortion_extension_study.py`
- `scripts/evaluate_distortion_aware.py`
- `scripts/evaluate_distortion_extension.py`
- `scripts/evaluate_local_latents.py`
- `scripts/prepare_distortion_extension.py`
- `scripts/prepare_local_latents.py`
- `scripts/preservation_inpaint.py`
- `scripts/report_distortion_aware.py`
- `scripts/report_distortion_extension.py`
- `scripts/report_local_latents.py`
- `scripts/run_local_latent_comparison.py`
- `src/local_correspondence/__init__.py`
- `src/local_correspondence/features.py`
- `src/local_correspondence/latent_fusion.py`
- `src/preservation/__init__.py`
- `src/preservation/confidence.py`
- `tests/test_local_correspondence.py`
- `tests/test_local_latent_fusion.py`
- `tests/test_preservation.py`
- `webapp/dist/confidence.css`
- `webapp/dist/confidence.html`
- `webapp/dist/confidence.js`

## 2. Files modified

- `PROJECT_REPORT.md`: measured completion, results, failed rows, remaining work and current runtime blocker.
- `research/REGIONAL_FUSION_METHOD.md`: separate actual local-feature prototype from historical global routing.
- `webapp/server.py`: allow the three new static confidence-editor assets.
- `webapp/dist/index.html`: link to evidence-map editor.

## 3. Tests

Seven unit tests and twelve HTTP tests passed. JavaScript syntax passed. Real browser image/map imports and export verified against all three downloaded PNGs pixel-for-pixel. All 552 successful output hashes, three frozen generation-source signatures and 24 old checkpoints verified. Live CLI inference was attempted and failed before generation: Windows Application Control blocked a SciPy DLL. No policy was weakened. See distortion_phase_verification_v1.json.

## 4–5. Experiments and numerical results

Initial screen: 144 new generations, 312 scored rows. Local comparison: 32 new generations plus 16 reused controls, 48 scored rows. Extension: 96 planned generations, 72 successful; 192/240 scheduled rows scored. The 24 failed generations and 24 unavailable preservation rows for identity 386 remain in the ledger after compressed-reference face detection failed. These counts are not independent sample sizes. Statistical units are identities.

Matched local fusion: FaceNet 0.6846 versus global 0.6894; hole MAE 0.09486 versus 0.09054. Damage-minus-global difference −0.00483, Holm p 1.0. Fresh development complete-pair contrasts use three identities: lower-strength FaceNet +0.03731/hole MAE −0.02063; preservation FaceNet +0.06892/hole MAE −0.03298 relative to high strength. All four adjusted p-values 1.0. Unchanged-input safeguards remain unfavorable for identity/hole error.

## 6. Visual failures

All eight local-fusion sheets reviewed. Eye shape, gaze, mouth/teeth and expression change; texture smoothing remains. Second seed sometimes changes lip color/closure markedly. All 36 successful expanded cases are now visually reviewed; see DISTORTION_EXTENSION_VISUAL_REVIEW_V1.md. Blinded human assessment remains pending.

## 7–8. Supported and failed hypotheses

A: lower strength has descriptive benefit for some degraded inputs, but is not universally better and performs poorly on removal. No corrected significant result. B: retaining observed evidence reduces damage versus aggressive generation descriptively, but does not establish improvement over unchanged input. C/D: aligned local features and regional local fusion have not demonstrated a useful multi-metric advantage over global descriptors. Training gate is not met.

## 9–10. Mechanism and historical baseline

True local-feature extraction, explicit five-point alignment and spatial VAE feature injection were implemented and used in real generations. No learned fusion parameters. No improvement over historical 0.8180 is established; the new mixed-only comparison has a different condition mixture.

## 11. Before final evaluation

Restore an approved inference runtime and repeat real inference checks; complete expanded visual review and controlled compressed-reference failure diagnosis; establish a stronger preservation/restoration mechanism; reproduce close external controls; expand identity counts and independent pose/severity/reference factors; complete human review. Train only if justified; test candidate reranking only after stable restoration. Freeze method and parameters before opening the final set.

## 12. Novelty

Broad novelty is weakened by existing component dictionaries, warped guidance, spatial feature attention and spatial denoising control. See NOVELTY_GAP_ANALYSIS.md. This checkpoint provides reproducible diagnostics, not a completed novel-method paper.
