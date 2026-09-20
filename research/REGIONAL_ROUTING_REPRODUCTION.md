# Reproducing the regional-routing development experiment

Use the existing local dataset restoration and verified model cache. Dataset photographs and model files are intentionally excluded from Git. The case manifest is `outputs/generalization_v1/cases/manifest.json`; restore that exact version before attempting this comparison. Do not substitute the reserved final identities.

Run generation with the project's pinned reference environment:

```powershell
& "$env:USERPROFILE/.cache/facial-inpainting/reference_env_v2/Scripts/python.exe" -X utf8 scripts/run_regional_routing_comparison_v2.py
```

The runner checks the protocol, implementation and Diffusers source signature before continuing. It reuses completed rows only after checking output hashes. Do not edit the frozen method to resume a partially completed experiment. A different method or environment requires a new version.

Run evaluation with the metric environment, which contains PyIQA, FaceNet and InsightFace:

```powershell
& "$env:USERPROFILE/.cache/facial-inpainting/evaluation_env_v1/Scripts/python.exe" -X utf8 scripts/evaluate_regional_routing_v2.py
```

Then verify evidence, build the report and create the numerical plot:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/verify_regional_routing_v2.py
.venv/Scripts/python.exe -X utf8 scripts/report_regional_routing_v2.py
.venv/Scripts/python.exe -X utf8 scripts/plot_regional_routing.py
```

The evidence files use exclusive creation. Do not delete previous results to rerun a report; inspect any error and preserve existing artifacts. Contact sheets contain dataset photographs and remain under the ignored `outputs` directory. Only numerical evidence and photo-free figures belong in the public repository.

## Recovery performed on 20 September

The original v2 process ended without a completion manifest. Its log ended during a slow generation; the cause was not established. On recovery, no Python processes remained. All 121 completed records were retained. One attempt contained only two routing-map files and no generated image; it was moved intact to `outputs/regional_routing_v2/interrupted_attempts`, with file hashes recorded in `recovery_20260920.json`. The same frozen runner retried that row with the original seed and settings. No generated candidate was discarded based on quality.

The initial evaluator restart used the base environment and failed at import because PyIQA was unavailable. It produced no scores. Evaluation was restarted in the pinned metric environment, preserving the failed startup log. This is an environment-launch failure, not a model-generation or quality-metric result.

Generation timing includes interrupted-run and resumed-run conditions. It is not a controlled speed comparison. The four identities were already observed in the first cycle; this experiment cannot be described as a new unseen-identity final test.
