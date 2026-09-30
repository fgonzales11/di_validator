# Restart the Windows worker lane while its detached WSL supervisor keeps the run.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $projectRoot 'runtime/processes.json'
$record = [System.IO.File]::ReadAllText($recordPath) | ConvertFrom-Json
$workerRecord = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.meter_lab_worker)" -ErrorAction SilentlyContinue
if ($workerRecord -and $workerRecord.CommandLine -match 'di_validator\.meter_lab\.worker') {
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $($record.meter_lab_worker)" | Where-Object { $_.CommandLine -match 'di_validator\.meter_lab\.worker' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
    Stop-Process -Id $record.meter_lab_worker -ErrorAction SilentlyContinue
}
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-m','di_validator.meter_lab.worker') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'runtime/meter-lab-worker.out.log') -RedirectStandardError (Join-Path $projectRoot 'runtime/meter-lab-worker.err.log')
$record.meter_lab_worker = $process.Id
$record | ConvertTo-Json | Set-Content -LiteralPath $recordPath
Write-Host "Meter Lab worker restarted: $($process.Id). Active WSL runs remain owned by their detached supervisors."
