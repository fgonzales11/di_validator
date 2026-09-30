# Restart only the API after local code changes, preserving the worker and its running job.
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $projectRoot 'runtime\processes.json'
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$apiRecord = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.api)" -ErrorAction SilentlyContinue
if ($apiRecord -and $apiRecord.CommandLine -match 'di_validator.api:app') {
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $($record.api)" | Where-Object { $_.CommandLine -match 'di_validator.api:app' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
    Stop-Process -Id $record.api -ErrorAction SilentlyContinue
}
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','di_validator.api:app','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'runtime\api.out.log') -RedirectStandardError (Join-Path $projectRoot 'runtime\api.err.log')
$record.api = $process.Id
$record | ConvertTo-Json | Set-Content -LiteralPath $recordPath
$ready = $false
for ($attempt=0; $attempt -lt 120; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 2
        if ($response.status -eq 'ok') { $ready=$true; break }
    } catch { Start-Sleep -Milliseconds 500 }
    $process.Refresh()
    if ($process.HasExited) { break }
}
if (-not $ready) { throw 'API did not become ready. Inspect runtime\api.err.log.' }
Write-Host "API is ready on port $Port"
