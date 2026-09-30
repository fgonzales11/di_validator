$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot 'runtime\processes.json'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host 'No launcher process record found.'; exit 0 }
$record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m di_validator.meter_lab.worker --stop
if ($LASTEXITCODE -ne 0) { throw 'Meter Lab shutdown did not complete. Inspect its run logs; the run remains recoverable.' }
function Stop-OwnedTree([int]$ownedProcessId) {
    # Roots are checked below; terminate their owned trainers and helpers too.
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $ownedProcessId" | ForEach-Object { Stop-OwnedTree $_.ProcessId }
    Stop-Process -Id $ownedProcessId -ErrorAction SilentlyContinue
}
foreach ($processId in @($record.worker,$record.api,$record.meter_lab_worker)) {
    if (-not $processId) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -match 'di_validator\.(worker|api|meter_lab\.worker)' -and $process.ExecutablePath -eq (Join-Path $projectRoot '.venv\Scripts\python.exe')) {
        Stop-OwnedTree $processId
    }
}
Remove-Item -LiteralPath $pidFile
Write-Host 'DI Validator services stopped. Unfinished jobs will be marked interrupted at next startup.'
