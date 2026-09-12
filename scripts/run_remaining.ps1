# Run from any directory. Existing signed outputs resume; failures stop the pipeline.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $projectRoot
try {
    $stages = @('prepare_refiner_data.py', 'train_refiner.py', 'evaluate_refiners.py', 'download_object_annotations.py', 'prepare_object_assets.py', 'object_test.py', 'verify_object_protocol.py', 'measure_inference.py', 'report_final.py', 'write_manuscript.py', 'verify_delivery.py', 'package_project.py')
    foreach ($stage in $stages) {
        & $projectPython (Join-Path 'scripts' $stage)
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage" }
    }
    # verify_delivery.py runs the analytic suite and writes its captured output.
} finally { Pop-Location }
