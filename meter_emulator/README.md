# HW 4.2 meter emulator for both DI agents

Configured in **Ubuntu-22.04 WSL**, using the supplied
`/root/DI-SDK-3.0.0/DistributedIntelligenceSDK-ARM-EMU-1.8.5 1.tgz`.
The archive contains `emu-tool` **1.8.0**. Its HW 4.2 ARM root filesystem runs
through QEMU user emulation in the LXC container **di-hw42**. The existing
SDK **3.0.0**, reference project, agent sources, and unsigned packages are preserved.

## Use

From PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 -u root -- di-meter start
wsl.exe -d Ubuntu-22.04 -u root -- di-meter status
wsl.exe -d Ubuntu-22.04 -u root -- di-meter disable pv
wsl.exe -d Ubuntu-22.04 -u root -- di-meter enable pv
wsl.exe -d Ubuntu-22.04 -u root -- di-meter restart
wsl.exe -d Ubuntu-22.04 -u root -- di-meter stop
```

In Ubuntu, use `sudo di-meter <command>`; the agent names for enable/disable are
`pv` and `fault`. Start launches both agents and enables their main features.
Calling start again leaves existing processes and feature selections in place.
Restart enables both main features. These are development controls, not an
automatic WSL boot service.

Use these controls for this container: the vendor `emu-tool agent-start` uses
global host services and can replace the global metrology fixture directory.
The dedicated manager instead uses a private D-Bus, database, and simulator input.

## What works and what remains outside this setup

| Agent | Installed identity | Emulator behavior |
| --- | --- | --- |
| FaultLocationAgent 0.1.0 | Agent `23020000`, feature `23030000` | ARM execution, SDK registration, configuration, feature control, stored framework output, shutdown and restart. The existing ARM build has **no waveform acquisition adapter**; it explicitly reports this and does not calculate synthetic fault results. Waveform replay remains in the existing host harness. |
| PVDetectionAgent 0.1.0 | Agent `23020001`, feature `23030001` | ARM execution and the same lifecycle, plus real DataServer subscription callbacks, aggregate real/reactive power ingestion, samplewise import/export integration, and persistent interval state. |

PV's copied configuration retains unset coordinates (`999.0`). Measured energy
works, while solar analytics are suppressed until valid coordinates are configured.
Its 30-valid-day learning requirement remains in effect. This short emulator
verification does not establish a PV classification, generation accuracy, or
physical meter polarity/scaling compatibility.

The supplied emulator image is a development runtime, **not proof of firmware
50.10.312.2 / AppServ 2.0.581.0 compatibility**. Those remain the SDK3 package
baselines. Fault's original selector is GENX_PP; PV's are GENX_SP and GENX_PP.
The metrology fixture supplies aggregate P/Q and diagnostic phase voltages; it
does not emulate both physical meter forms or sub-second waveforms.

## Input and configuration

Host input: `/root/di-sdk-setup/meter-emulator/metsimfiles/PV.csv`.
Each explicit row represents one simulated second. The supplied two-hour fixture
alternates ten seconds of +1000 W import and ten seconds of -500 W export, with
400 var aggregate reactive power and 230 V phase diagnostics. Headers are
`18153531,18153532,18153520,18153521,18153522`. This is synthetic test data.
The DataServer stops replay at EOF; restarting begins the file again.

Stop the emulator before replacing its input or editing its copied configuration:

```text
/home/fgonzales/.local/share/emu-tool/lxc/di-hw42/rootfs/etc/agents.d/
  23020000.23030000.xml   Fault settings
  23020001.23030001.xml   PV settings
```

At start, the manager loads these feature XML files into the isolated DataServer.
Use decimal strings for PV coordinates/scaling. The agent performs validation
and applies accepted updates at interval boundaries. Configuration invalidation
and warm-up rules remain those implemented by the agent.

PV receipt times are actual UTC; replay runs at wall-clock speed. Use the
portable PV harness for accelerated multi-day numerical validation. Do not
change the WSL/system clock to accelerate this runtime.

## Locations and evidence

- Runtime/controller: `/usr/local/lib/di-meter-emulator`, command `/usr/local/bin/di-meter`.
- Setup, services, evidence and build artifacts: `/root/di-sdk-setup/meter-emulator`.
- Container root: `/home/fgonzales/.local/share/emu-tool/lxc/di-hw42/rootfs`.
- Private database: `database/flash/muse01.db` below the setup directory.
  It was initialized with a SQLite backup of the host database; pre-existing
  rows are retained and must not be counted as new emulator outcomes.
- Agent SDK logs: `tmp/agent/23020000/23020000_log` and
  `tmp/agent/23020001/23020001_log` below the container root.
- Service/agent stdout logs: `logs/` below the setup directory.
- PV state: `usr/share/flash/587530241/587333633/pv/` below the container root.
- `environment.json`: archive, package, runtime hashes and resource policy.
- `foreground-build/manifest.json`: source hashes and exact ARM compiler commands.
- `validation-verified/report.json`: final installed-control verification.
- Windows evidence copy: `runtime/meter-emulator` in this workspace.

The verification starts/stops the agents, measures synthetic import/export,
checks feature disabling and checkpoint checksums, verifies registration/config
logs and retained database rows, then restarts both agents. It leaves both running.
Re-run into a **new** output directory:

```bash
sudo python3 /usr/local/lib/di-meter-emulator/verify.py \
  --seconds 35 --output /root/di-sdk-setup/meter-emulator/validation-my-run
```

## Compatibility adaptations

1. **LXC/cgroup v2:** the vendor HW4.2 template used old cgroup keys. The dedicated
   container uses `lxc.cgroup2.*` for the shared hardware limits.
2. **Configuration D-Bus signature:** the SDK3 ARM library sends GetFeatureConfig
   as `(uint32 agent, uint64 feature)`; this host DataServer expects `(uint64 feature)`.
   `config_compat.c` adapts that request for these two agent IDs and forwards the
   actual response. It does not fabricate registration or policy results.
3. **Subscription transport:** `metrology_bridge.py` forwards genuine DataServer
   broadcasts `aua(uv)` as directed ARM SubscriptionUpdate calls `a(uv)` on the
   recipient's interface. LIDs, typed values, partial updates and timing are
   preserved. Sender ownership is checked; only the configured PV recipient is routed.
4. **Foreground ARM processes:** the packaged daemon forks after logging starts;
   PV was observed blocked in the logging circular-buffer mutex after that fork.
   Emulator builds define the existing `NODAEMON` flag **only for agent_daemon.cpp**.
   All algorithm/feature files retain ARM behavior; host waveform adapters remain
   excluded. The production ZIPs and source code are unchanged. Installed
   executables are stripped; debug copies are retained outside the container.
5. **Shutdown:** SIGTERM first reaches both agents and the manager waits for them
   to exit normally. It then force-stops the idle container because the supplied
   BusyBox init restarts on the default halt signal in this environment. Neither
   agent is deregistered during normal shutdown.

These are local emulator adaptations, not components to ship to meters. Original
container config and target library copies are retained in the setup directory.
The old SDK bundled inside the emulator archive was not installed over SDK3.

## Resource interpretation

Agent policies remain **2% CPU / 2048 KB RAM / 2048 KB flash each**. The shared
HW4.2 container is limited to **65,000 KB RAM, 30% of one CPU core, 50 tasks,
and 1,250 open files per process**. The documented 40,000 KB flash ceiling is not
enforced by a disk quota in this WSL root filesystem.

The 35-second foreground test measured about **17 MB RSS per emulated process**,
including QEMU overhead: this exceeds the 2 MB agent RAM policy. The final test
measured 16,968 KB / 0.40% host CPU for Fault and 17,948 KB / 1.23% for PV;
these are not meter resource measurements or a
worst-case analysis workload. No container OOM occurred. Full history, algorithm
load, actual meter resource enforcement, signing and deployment are separate gates.
No agent quota was increased to obtain these results.

## Rebuild the emulator runtime

The source kit is `meter_emulator/` in the Windows workspace. To install revised
controls from that kit and rebuild its local ARM executables:

```bash
sudo di-meter stop
sudo python3 /mnt/c/Users/fgonzales/git/DI_Validator/meter_emulator/install_controls.py
sudo python3 /usr/local/lib/di-meter-emulator/build_runtime.py
sudo di-meter start
```

The build requires the existing SDK3 TargetRelease toolchain, headers and generated
build definitions. It compiles into the separate setup directory, preserves the
original packaged executables, and installs only to this stopped container.
It also rebuilds the small configuration compatibility library. It does not
invoke SDK clean/package scripts or create a replacement production package.
