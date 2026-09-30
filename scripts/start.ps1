param([int]$Port = 8765, [switch]$NoBrowser, [switch]$NoBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 24 LTS first: https://nodejs.org/' }
& uv sync --frozen --inexact
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& .\.venv\Scripts\python.exe -m scripts.build_notebooks
if ($LASTEXITCODE -ne 0) { throw 'Notebook workspace build failed.' }
if (-not $NoBuild) {
    Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
$runtimePath = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$baseUrl = "http://127.0.0.1:$Port"
$labHeartbeat = Join-Path $runtimePath 'meter-lab\worker.json'
$labWorkerId = $null
if (Test-Path -LiteralPath $labHeartbeat) {
    $labState = Get-Content -LiteralPath $labHeartbeat -Raw | ConvertFrom-Json
    $labProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($labState.pid)" -ErrorAction SilentlyContinue
    if ($labProcess -and $labProcess.CommandLine -match 'di_validator\.meter_lab\.worker') { $labWorkerId = $labState.pid }
}
if (-not $labWorkerId) {
    $labProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','di_validator.meter_lab.worker') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath 'meter-lab-worker.out.log') -RedirectStandardError (Join-Path $runtimePath 'meter-lab-worker.err.log')
    $labWorkerId = $labProcess.Id
}
try {
    $existing = Invoke-RestMethod -Uri "$baseUrl/api/v1/health" -TimeoutSec 2
    if ($existing.status -eq 'ok') {
        $existingRecordPath = Join-Path $runtimePath 'processes.json'
        if (Test-Path -LiteralPath $existingRecordPath) {
            $existingRecord = Get-Content -LiteralPath $existingRecordPath -Raw | ConvertFrom-Json
            $existingRecord | Add-Member -NotePropertyName meter_lab_worker -NotePropertyValue $labWorkerId -Force
            $existingRecord | ConvertTo-Json | Set-Content -LiteralPath $existingRecordPath
        }
        Write-Host "DI Validator is already running at $baseUrl"
        if (-not $NoBrowser) { Start-Process $baseUrl }
        exit 0
    }
} catch { }
$workerProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','di_validator.worker') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath 'worker.out.log') -RedirectStandardError (Join-Path $runtimePath 'worker.err.log')
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','di_validator.api:app','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath 'api.out.log') -RedirectStandardError (Join-Path $runtimePath 'api.err.log')
@{api=$apiProcess.Id;worker=$workerProcess.Id;meter_lab_worker=$labWorkerId;port=$Port;root=$projectRoot} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimePath 'processes.json')
$ready = $false
for ($attempt=0; $attempt -lt 120; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/api/v1/health" -TimeoutSec 2
        if ($response.status -eq 'ok') { $ready=$true; break }
    } catch { Start-Sleep -Milliseconds 500 }
    if ($apiProcess.HasExited) { break }
}
if (-not $ready) { throw "Startup failed. Inspect $runtimePath\api.err.log" }
Write-Host "DI Validator is ready at $baseUrl"
Write-Host 'Use scripts\stop.ps1 to stop the local services.'
if (-not $NoBrowser) { Start-Process $baseUrl }
