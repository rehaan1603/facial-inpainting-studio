# Generalization and restoration upgrade — first implementation cycle

Date: 19 September 2026. Scope: the user's first eight steps, before a regional fusion adapter.

## Inspected baseline and preservation boundary

The current checkpoint is 91752a6. The working tree also contains two local dataset-export scripts from the preceding task; preserve them. The repository has scripts/, configs/, data/manifests/, outputs/, research/, tests/ and webapp/. There is no existing src/ package. Historical outputs include mask benchmarks, trained refiners, object composites, partial OSOR, reference diagnostics, reference-count ablations and web runs. Do not modify their signed inference sources, protocols, checkpoints or scores.

- Entry points: Start Studio.cmd -> scripts/launch_studio.py -> webapp/server.py; webapp/dist/app.js handles 512-square framing/mask painting, HTTP jobs and downloads. The server binds loopback, validates origin/token, and serializes GPU jobs.
- Single-image inference: scripts/inpaint.py loads TorchScript Big-LaMa or resshift_adapter.py. Website LaMa exports 512, ResShift 256. The CLI and historical tests remain 256. Mask white means replace; composition preserves the processed input outside the effective mask.
- Reference inference: scripts/reference_inpaint.py accepts arbitrary image/mask paths and 1–4 reference photos. It extracts CPU buffalo_l embeddings; SDXL inpainting + FaceID Portrait projects each reference to 16 tokens. FP16 generation uses CPU offload, DDIM, then hard/Poisson composition. Existing website reference mode requires 3–4 photos. Reference inference has no dataset-identity lookup.
- Environments: .venv holds CUDA torch 2.11/cu128 and baseline tools. External reference_env_v2 shares base packages and pins diffusers 0.40/transformers 5.17/InsightFace 2.0. evaluation_env_v1 adds PyIQA; identity_eval_v1 caches FaceNet. Legacy OSOR/reference environments remain intact. RTX 5070 Laptop GPU has 8 GB VRAM.
- Checkpoints: six mask refiners (119,057 parameters each) at initial/extended budgets, plus downloaded LaMa, ResShift, SDXL and FaceID weights. Only mask refiners were locally trained. No regional fusion model exists.
- Dataset: reviewed CelebAMask-HQ manifest maps HQ photos to CelebA identity/split/source hashes. Reviewed LaPa has no established cross-dataset identity mapping and is not used to claim identity-disjoint generalization. Existing fingerprints and duplicate quarantine remain authoritative inputs.
- Evaluation: reuse frozen extended_evaluation_metrics.py (FaceNet, conditioning-encoder ArcFace, NIQE, BRISQUE, SSIM, PSNR, LPIPS, regional MAE). Keep failures and paired identity resampling. Existing 144 rows belong to twelve observed development identities; never repurpose them as a fresh test.

## Design before implementation

1. Build an exclusion ledger from ALL reviewed training identities (the mask-refiner training cache covers that partition), all earlier experiment cases and engineering/diagnostic selection attempts. Reserve new validation identities and a separate final test from the remaining official partitions. Select target/four conditioning/three gallery photos with source/decoded/pHash checks. Final identities remain reserved and are not scored in this cycle. “Unseen” is relative to recorded local development, not unknown pretraining membership.
2. Add deterministic, non-destructive corruption modules for regional removal/irregular occlusion, blur, motion blur, noise, JPEG, resolution loss, mixed and illumination degradation, with three severities. Clean target landmarks may construct synthetic masks only; only corrupted RGB and the mask enter inference/scoring.
3. Add runtime reference analysis using damaged input and uploaded references only. Map arbitrary masks onto coarse facial regions. Use reference regional sharpness/exposure, coverage/pose visibility proxies, detection confidence, target pose and cautiously gated identity compatibility. Save all components and failures. Five landmarks do not establish true occlusion/visibility: name these as proxies.
4. Freeze a small deterministic scorer before looking at generated outcomes. Add first/random/identity-only/quality-only/all/mask-aware policies. The proposed policy selects one reference; it does not perform regional feature fusion or train an adapter.
5. Wrap the unchanged reference generator in a new inference entry point. Accept 1–4 arbitrary photos, exclude invalid files with diagnostics, warn about reference disagreement, and provide an explicit no-valid-reference fallback. No mask must produce an actionable needs-mask result, not silently modify the whole face. Add an experimental website mode with 1–4 references and metadata download while preserving old modes.
6. Run a fixed small validation comparison: four fresh locally unexposed identity labels, three corruption conditions, two generator seeds. Main policies: random, identity-only, quality-only, all, mask-aware; additionally retain first and zero-reference-contribution controls. Cache identical selected-reference/seed/config requests to avoid duplicate GPU work, recording aliases. Keep all logical rows and failures. Use the same generator parameters for all policies. All-reference vs single-reference comparisons also change information budget.
7. Score only after generation using clean targets. Report per-identity summaries, jointly valid pairs, intervals, exact paired tests/Holm correction, failures and output hashes. Keep the final split unused. No current pretraining-independent, universal restoration, novelty or superiority claim is possible.

## Files to add

- src/degradation/{__init__,severity,face_region_masks,distortion_pipeline,metadata}.py
- src/reference_selection/{__init__,region_mapper,quality_estimator,pose_estimator,visibility_estimator,reference_analyzer,mask_aware_scorer,selection_baselines}.py
- src/__init__.py and src/research_integrity.py
- scripts/{audit_upgrade_baseline,prepare_unseen_identity_protocol,prepare_restoration_cases,mask_aware_inpaint,run_selection_comparison,evaluate_selection_comparison,report_selection_comparison}.py
- configs/mask_aware_selection_v1.json
- tests/test_restoration_upgrade.py
- research/protocols/unseen_identity_protocol_v1.{json,md}; research/upgrade_baseline_audit_v1.json; versioned new manifests/results under outputs/generalization_v1/
- research/{GENERALIZATION_RESULTS,MASK_AWARE_REFERENCE_SELECTION,REGIONAL_FUSION_METHOD,UNSEEN_IDENTITY_EVALUATION,DISTORTION_EVALUATION}.md

Files to modify: PROJECT_REPORT.md, README.md, webapp/server.py, webapp/dist/{index.html,app.js}, webapp/README.md; add targeted HTTP tests if needed. Additional evidence-only files may be added, with their purpose recorded. Frozen scripts/reference_inpaint.py, scripts/extended_evaluation_metrics.py and their dependencies are reused without edits.

## Validation and risks

Run the existing script/test suites before changes where the relevant environment is available. New tests cover split disjointness, target/reference role separation, source hashes, determinism, damage types/severity, pixel preservation, mask mapping, score direction, missing/corrupted/duplicate references, disagreement and missing-face/mask fallbacks. Run actual GPU cases, not mocks alone, and inspect the local UI. Record failures explicitly.

Biggest risks: noisy identity labels; unknown pretrained exposure; insufficient eight-photo groups; detector failure on damage; unreliable pose/visibility proxies; global embeddings cannot transfer spatial details; general inpainting can remove rather than restore degraded features; final-test contamination through later practice exports; laptop VRAM/latency. Scorer scores are not calibrated confidence probabilities. Whole-image quality metrics can be dominated by unchanged pixels. Code working alone is not an improvement claim.

Do not implement learned regional fusion, automatic damage detection, or candidate reranking before this cycle's evidence. These remain later phases with their own frozen protocols and evaluation roles. Public hosting stays paused.
