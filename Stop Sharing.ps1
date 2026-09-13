$projectRoot = $PSScriptRoot
$stateFile = Join-Path $projectRoot '.local-share/state.json'
if (-not (Test-Path -LiteralPath $stateFile)) { Write-Output 'No saved sharing session.'; exit }
$shareState = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
foreach ($entry in @(
    @{ Id=$shareState.gateway_pid; Expected=(Join-Path $projectRoot 'webapp/share_gateway.py') },
    @{ Id=$shareState.tunnel_pid; Expected=(Join-Path $projectRoot '.local-share/cloudflared.exe') }
)) {
    $processId = [int]$entry.Id
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId"
    if ($processInfo) {
        if (-not $processInfo.CommandLine -or -not $processInfo.CommandLine.Contains($entry.Expected)) {
            throw "Process $processId does not match this project's sharing process. It was not stopped."
        }
        Stop-Process -Id $processId -ErrorAction Stop
    }
}
Remove-Item -LiteralPath $stateFile
Write-Output 'Sharing stopped. The local GPU studio remains available on port 8765.'
