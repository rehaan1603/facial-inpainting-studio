$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'scripts/launch_studio.py')
