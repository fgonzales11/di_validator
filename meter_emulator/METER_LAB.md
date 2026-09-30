# Meter Lab

Meter Lab is a local DI Validator workspace for running `PVDetectionAgent` and
`FaultLocationAgent` as ARM executables under the HW 4.2 Gen5 Riva emulator. Open
<http://127.0.0.1:8765/meter-lab>. PV supports GENX_SP and GENX_PP; Fault supports
GENX_PP. The SDK baseline is firmware 50.10.312.2 / AppServ 2.0.581.0.

## Install and start

From the repository in Windows PowerShell, with the existing SDK, numerical
fixtures, and HW 4.2 image installed in Ubuntu-22.04:

```powershell
wsl.exe -d Ubuntu-22.04 -u root -- python3 /mnt/c/Users/fgonzales/git/DI_Validator/meter_emulator/lab_setup.py
./scripts/start.ps1 -NoBrowser
```

The installer creates `/root/di-sdk-setup/meter-lab`, including the separate
`di-meter-lab` LXC container, and builds emulator-only executables. Stop any active
lab run before rebuilding. It does not rebuild the SDK projects or production
packages and does not stop `di-hw42`. `scripts/stop.ps1` cancels the lab run and
waits for its shutdown before stopping the local application.

The local launcher starts a dedicated `di_validator.meter_lab.worker` lane.
Hosted deployments report Meter Lab unavailable. Set `DI_HOSTED=1` on hosted
backends and `VITE_HOSTED=true` on hosted frontend builds. Browser requests carry
registered IDs and validated parameters; only the worker calls the fixed WSL
helper. Telemetry polling never launches WSL commands.

## Run a scenario

1. Select an agent, supported meter form, preset/dataset/upload, and playback speed.
2. Adjust parameters and decimal-string agent configuration, then preview inputs.
3. Start with clean state, a compatible frozen PV warm state, or a checkpoint from
   a completed PV run. Warm continuations must follow the seed's recording time
   and use matching configuration. `warm_tail` matches the 32-day fixture.
4. Observe transmitted, received, and processed counts separately. Pause/Resume
   gates scenario delivery; Stop cancels processing and drains bounded writes.
   There is no rewind or seeking. Refresh/navigation does not stop a run.
5. Inspect ARM diagnostics, stored DataServer outcomes, checks, resources, and logs.
   Compare two saved runs or download their evidence ZIPs.

Fault results arrive after completed waveform segments are analyzed. PV analysis
uses completed UTC 15-minute intervals; normal output batches four intervals.
Generation may be unavailable: chart gaps are not zero PV. Measured net export is
shown separately. PV generation remains a conditional model estimate with an
uncalibrated daytime uncertainty band.

Synthetic constant blocks declare an exact one-second value for every second in
the block. The transport compresses only those declared repetitions; ARM still
constructs/decodes SDK LID values and processes every sample. Hourly AMI cannot be
used as a one-second input. Fault uploads require synchronized instantaneous
IA, IB, IC, UA, UB, UC, with explicit A/kA and V/kV mappings. PV uploads require
one-second aggregate meter P/Q, explicit W/kW and var/kvar units, UTC time, and
optional validity/voltage columns. CSV/Parquet and supported ASCII COMTRADE pairs
are limited to 64 MiB. Fault records are bounded to 200,001 samples; PV to 60 days.

## Clocks and input paths

Scenario replay uses recording time in the numerical cores and actual elapsed
time for SDK budgets, retries, and resource measurements. No system clock changes
are made. Fastest is bounded by acknowledgements/backpressure; a requested speed
is not guaranteed. SDK event timestamps remain actual UTC.

The separate **PV SDK metrology check** sends a CSV through the private host
DataServer's real one-second subscription path and the existing compatibility
bridge into the ARM SDK decoder. Its chart timestamps are actual UTC. It does
not accelerate time. Scenario replay does not subscribe to live metrology.

## Evidence and protocol

The existing catalog stores metadata. `runtime/meter-lab` stores immutable inputs,
scenario/configuration manifests, source hashes, warm-state identities, build
manifests, telemetry, ARM acknowledgement history, numerical diagnostics,
submission/completion logs, private SQLite snapshots, checks, and exports. Each
WSL run owns an independent database, flash tree, configuration, and logs.
`runtime-helpers/` preserves the exact installed helper files against the run's
build-manifest hashes. Historical helper versions are also included in the frozen
source archive, so later runtime changes do not erase their provenance.

ML1 frames include run ID, sequence, kind, row count, timestamps and field
validity. One frame is outstanding at a time, with at most 1,024 samples. ARM
acknowledges received and processed counts; END has an explicit completion bit.
A lab-only observer records the actual SDK API completion result without
substituting results or policy decisions. Completion waits for bounded output
attempts and checkpoints. Production code excludes `DI_EMULATOR_REPLAY`, including
the ingress adapters, diagnostics, and observer. NODAEMON is independently scoped
to the emulator's foreground daemon entry point.

Worker/API restarts reconcile persisted WSL PID/start identities. A surviving
supervisor reconnects; lost ownership is marked interrupted. A browser is never
the process owner. Failed runs and adverse results remain in history.
`scripts/restart-meter-lab-worker.ps1` restarts only the Windows lab lane and
preserves a surviving WSL run, which the replacement worker reconciles.

Checks distinguish numerical parity, scenario truth, and delivery evidence.
Unverified uploads receive structural/parity checks and no invented field-accuracy
verdict. Midday load reduction is a known PV false-positive scenario and remains
an adverse expectation result even when ARM/Python numerical parity passes.

## Reproduce validation

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_meter_lab.py -q
./.venv/Scripts/python.exe -m meter_emulator.verify_lab
./.venv/Scripts/python.exe -m meter_emulator.verify_lab --full --case pv_cold
./.venv/Scripts/python.exe -m meter_emulator.verify_lab --full --case pv_adverse
./.venv/Scripts/python.exe -m meter_emulator.verify_lab --extended --start-at fault_loop_AG
./.venv/Scripts/python.exe -m meter_emulator.summarize_lab
```

Integration reports are written under `runtime/meter-lab/verification`; each
report identifies its saved run and frozen evidence. The full cold/adverse cases
replay 35 days. Numerical tolerances remain 1e-6 absolute/relative; fault tensor
tolerances remain 1e-5 absolute / 2e-5 relative. SDK quota rejections and bounded
application suppression remain visible and never trigger quota increases.

To resume evidence collection after a test-client restart, select the matching
case with `--case CASE --resume-run RUN_ID`. The detached ARM run keeps running.
The per-case verification deadline defaults to two hours and can be set with
`--timeout-seconds`. To verify ownership during an active replay, run
`python -m meter_emulator.verify_lab_restart RUN_ID` from Windows; it pauses,
restarts the lab worker, checks the unchanged owner/clock/count, then resumes.
Browser checks use `node node_modules/@playwright/test/cli.js test
e2e/meter-lab.spec.ts --project=desktop --project=tablet --timeout=600000
--trace=off` from `frontend/`. The longer timeout accommodates first-load Plotly
and browser startup on this Windows/WSL host; screenshots and evidence downloads
are retained independently of tracing.
Set `METER_LAB_BROWSER_CHANNEL=msedge` to use installed Microsoft Edge with an
isolated Playwright profile if the downloaded Chromium cannot start locally.
Set `METER_LAB_CONTROLS=1` for the additional Start/Pause/Resume/Stop check when
the lab is idle. Browser navigation and refresh are included in that check.

Run `python3 meter_emulator/freeze_lab.py` in Ubuntu-22.04 to preserve the installed
ARM binaries, build manifest, exact installed helper sources, portable sources,
and checksums under `runtime/meter-lab/builds/<build identity>`. These are emulator
replay artifacts; they are not production DI installation packages.

Resource reports compare observed QEMU CPU/RSS with the inherited 2% CPU and
2,048 KB agent RAM/flash policies. The 65,000 KB shared container ceiling is a
separate platform limit. Emulator measurements include QEMU and diagnostic
overhead and do not qualify resource use on a physical meter. Physical acquisition,
field accuracy, production resource qualification, signing, and deployment remain
separate gates.
