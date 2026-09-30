# FaultLocationAgent

Native C++11 port of DI Validator's `di_validator/faults` numerical pipeline, using DI SDK 3.0.0 in Ubuntu-22.04 WSL. The working agent is:

```text
/root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/FaultLocationAgent
```

The maintained source kit is `C:\Users\fgonzales\git\DI_Validator\di_agent`. The existing `ThirdPartyReferenceAgent` is preserved. This port uses development agent/overlay/feature IDs `23020000`, `23050000`, and `23030000`; it is not a signed production agent.

The meter target is **HW 4.2 Gen5 Riva polyphase**. [Hardware requirements](HARDWARE_TARGET.md) distinguish the documented platform limits from the agent's resource allocation and describe the future waveform interface. Package hardware selectors are restricted to `GENX_PP`, with SDK 3.0.0 release-table firmware/AppServ minima. [Verification results and deliverables](../runtime/di-agent/VERIFICATION.md) are retained outside SDK build directories.

## Run the sample

Open the agent directory from PowerShell:

```powershell
wsl -d Ubuntu-22.04 -u root --cd /root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/FaultLocationAgent
```

Then run:

```bash
./host_run.sh
```

The included `1A_val1` fixture produces onset sample 199 (0.0995 seconds), `1ph-G`, loop `AG`, and approximately **0.890890356846 km** using the unverified example line settings. Each run creates its own `validation/replay-*` folder and prints the observed database rows. It stops the agent after processing and verifies that shutdown retained its outputs.

For another exported recording:

```bash
./host_replay.sh /path/to/recording.wave --output /path/to/new-results-folder
```

The launcher needs root because vendor host packages install under `/usr` and `/etc`. Source editing is available to `fgonzales`. The launcher uses a private D-Bus, refuses to disrupt an already running SDK agent/DataServer, and writes only this agent's feature configurations into `/usr/local/bin/database/flash/muse01.db`. It does not raise policy limits or remove old outcomes. Host configuration replays use the real SDK XML parser and registered callbacks; remote HES configuration delivery is not tested.

## Rebuild and reproduce verification

Export the Python reference fixtures from the DI Validator directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B di_agent\tools\export_fixtures.py
```

In Ubuntu-22.04, set these paths and build the SDK-independent core:

```bash
KIT=/mnt/c/Users/fgonzales/git/DI_Validator/di_agent
RESULTS=/mnt/c/Users/fgonzales/git/DI_Validator/runtime/di-agent
cmake -S "$KIT" -B /tmp/fault-core-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fault-core-build -j2
ctest --test-dir /tmp/fault-core-build --output-on-failure
python3 "$KIT/tools/verify_parity.py" --binary /tmp/fault-core-build/fault_replay --fixtures "$RESULTS/fixtures" --report "$RESULTS/parity-report.json"
```

The exporter freezes Python source hashes, calibrated input hashes, configuration, expected results, intermediate arrays, and twelve-channel tensors. It does not modify existing workbench datasets. The comparator tests all 99 supplied COMTRADE recordings plus synthetic/error cases and checks chunk invariance. The core has no Python, NumPy, pandas, BLAS, or ML runtime dependency.

To reapply maintained sources to the clone and rebuild it, run as root:

```bash
python3 "$KIT/tools/install_agent.py"
cd /root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/FaultLocationAgent
./host_build.sh
python3 "$KIT/tools/verify_integration.py" --fixtures "$RESULTS/fixtures" --output "$RESULTS/integration"
```

`install_agent.py` regenerates the clone's integration files from the reference template and maintained source kit. Make future port changes in the kit; save independent edits to the clone before reapplying. The installer never changes the reference project. The original SDK clone command, from the SDK `Agents` directory, is:

```bash
./Clone_ThirdPartyReferenceAgent.sh FaultLocationAgent "Fault Location Agent" 23020000 23050000 23030000 SDK1
```

Do not rerun the clone command over an existing project.

Build the unsigned ARM package from the clone:

```bash
./build.sh TargetRelease
./pack.sh
```

Before switching targets, verify the resulting package (replace the ZIP filename with the generated revision):

```bash
python3 "$KIT/tools/verify_package.py" \
  --package /root/DI-SDK-3.0.0/DistributedIntelligenceSDK/build/FaultLocationAgent/Release/FaultLocationAgent_0.1.0.REVISION.zip \
  --binary /root/DI-SDK-3.0.0/DistributedIntelligenceSDK/build/FaultLocationAgent/FaultLocationAgent_Daemon \
  --report "$RESULTS/package-report.json"
```

This checks ZIP integrity, nested payload hashes, XML schemas, HW 4.2 polyphase selectors, policy limits, ARM ELF/dependencies, and exclusion of host replay/Python files. It does not install or execute the package.

The vendor build scripts clear both host and ARM working output directories. Preserve needed artifacts before switching targets, then run `./host_build.sh` to restore host replay. Verified artifacts from this implementation are retained under `/root/di-sdk-setup/fault-location/artifacts/`; the final Windows report and copies of the deliverables are under `runtime/di-agent`.

## Input, processing, and outputs

`FLA1` replay files contain sampling/grid frequencies, a recording identifier, decimal configuration values, and CSV columns `offset_seconds,IA_A,IB_A,IC_A,UA_V,UB_V,UC_V`. Inputs are synchronized instantaneous waveforms in A/V. Unknown units or headers, malformed records, invalid sampling grids, and invalid configuration are rejected. NaN/missing rows and timestamp gaps split contiguous segments; missing waveforms are never replaced with fabricated samples. CRLF and LF file endings are supported.

The pipeline uses completed segments and selects the first qualifying fault in each segment. It preserves centered filtering and endpoint behavior, quarter-cycle RMS detection, interpolated DFT/Fortescue magnitudes, heuristic classification, all seven fault loops, least-squares RMS phasors, optional zero-sequence compensation, and median apparent reactance distance. No continuous or immediate-detection latency is claimed. The replay carries at most 200,001 samples with the Python analysis/window constraints. Allocation is bounded by those limits, not by the meter's 2 MB quota.

Configuration is in the clone's `config/etc/agents.d/23020000.23030000.xml` and matching descriptor. Decimal parameters use SDK strings parsed as doubles, preserving precision. `-1` represents absent optional values (`end`, `manual_onset`, `r0_ohm_km`, `x0_ohm_km`, `known_distance_km`). Both zero-sequence parameters must be supplied together. Invalid updates preserve the previous valid configuration. Feature disabling pauses pending requests and cancels active calculations; updates take effect at the next segment boundary. Selection bounds are frozen when each recording is split.

Detected faults write a version-1 JSON summary to `AgentData` (maximum 1,024 bytes) and a retained `98#FAULT#1#...` history event with alarm delivery requested (maximum 256 bytes). Both contain the same correlation ID and configuration fingerprint. Summaries include the source, onset and availability offsets, fault type/loop, distance status, quality flags, and cycle statistics. SDK envelope timestamps are processing time; recording offsets are not fabricated UTC timestamps.

Manual-onset calculations write a summary without a detection alarm. A detected fault with unavailable distance still alarms. No-fault segments produce host evidence without fault outcomes. SDK output calls are synchronous and their return codes are checked. Rejected writes fail the request without automatic retries; a summary can exist if its subsequent alarm write fails. Alarm delivery is not guaranteed, and consumers should correlate/deduplicate by ID.

## Qualification limits

Host replay, numerical parity, ARM cross-compilation, XML/package validation, and retained local outcomes are verified separately. The ARM build excludes host replay/export code and does not subscribe to live metrology; starting it reports that no live waveform adapter is configured.

The sample resource policy remains **2% CPU, 2,048 KB RAM, and 2,048 KB flash**. The HW 4.2 document lists **30% CPU, 65,000 KB RAM, 40,000 KB flash, 10 MB maximum package, 50 processes, and 1,250 open files** for the platform/container. Those are not a dedicated per-agent allocation. Baseline native host RSS is approximately **6 MB**, with **24,800 KB** observed during the integration suite including a 100,000-sample cancellation case. Both exceed the current agent RAM policy. Host glibc measurements do not determine ARM/uClibc usage. Measure and optimize the actual ARM runtime and confirm its resource allocation before meter deployment; an unsigned successful build does not establish resource qualification. No meter emulator, physical meter, cloud delivery, signing, or production deployment is claimed.

Default line impedance and nominal values are unverified examples. The 99 recordings are RTDS simulations, not independently established field fault locations. This port does not include the separate CNN/ResNet training or inference code.
