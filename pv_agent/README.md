# PVDetectionAgent 0.1.0

Native C++11 prototype for **HW 4.2 Gen5 Riva singlephase and polyphase** meters, SDK 3.0.0. Screens for likely behind-the-meter PV and conditionally estimates total site generation from signed aggregate real/reactive power, local history and configured coordinates. There is no Python runtime, weather dependency, inverter telemetry or legacy pickle model on the meter.

The maintained kit is `pv_agent`; the separate SDK project is `/root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/PVDetectionAgent`. Development IDs are agent `23020001`, overlay `23050001`, feature `23030001` (hexadecimal). The original reference, FaultLocationAgent and existing Python workbench are preserved.

**This is experimental screening, not a production meter qualification or field-accuracy claim.** Synthetic repeated midday load reductions produce false-positive PV estimates even with an excellent nighttime fit. The tests retain this adversarial result. `no_evidence` is not confirmed absence. The uncertainty band is not calibrated for daytime PV accuracy. Every generation estimate assumes no material battery/generator contribution and negligible inverter reactive-power contribution.

## Implemented behavior

- `core/pv_core.hpp`: SDK-independent `Engine::ingest`, `finalize_interval`, `evaluate_history`, `serialize_state` and `restore_state`. Calculations use doubles; robust load regression uses centered weighted QR and ten Huber weight updates. An independent NumPy least-squares oracle is in `tools/reference.py`.
- Aggregate real power LID **18153531**, reactive power **18153532**, decoded as SDK Double W/var. Positive import, negative export; explicit polarity and transformer scaling. Normalization to kW/kvar happens once. Optional phase voltage/real-power LIDs feed hourly diagnostic logs; they never replace aggregate power or produce per-phase generation estimates.
- `ApiSubscribeToPeriodicData`, `eOnPeriodOnly`, one second. Empty callbacks preserve known unchanged values; an unknown field, invalid type/value or callback gap exceeding two seconds invalidates the corresponding cache. Values are copied before callback return into a bounded 4,096-sample queue. Lost samples produce a diagnostic and coverage loss. Host tests construct the SDK's actual typed LidList and exercise the same decoder; physical meter acquisition has not been tested.
- Samples represent the start of one measured second. The live adapter timestamps callback receipt with UTC system time; meter clock synchronization, callback latency, polarity and primary/secondary scaling still require validation on both physical meter forms. Supported clock range is 2000 through 2037, compatible with the ARM SDK's 32-bit `time_t`.
- UTC-aligned 900-second intervals. Measured import/export integrate each valid signed sample before averaging. Missing samples are never filled. Ninety percent coverage is required for analysis. A partial interval retains its actual measured energy and coverage. Estimated energy also covers only observed valid seconds; the measured-export generation floor uses that observed duration, so missing energy is not extrapolated. Invalid-time energy is retained in separate unassigned counters in local state/host diagnostics; it cannot be assigned to UTC interval or daily deliveries.
- At most 60 calendar days of compact intervals, selecting the latest 30 valid days. A valid day has at least 90% real-power coverage and four eligible intervals each above 10 degrees and below -6 degrees solar elevation. Missing coordinates suppress analytics while measured energy continues. Solar position follows the published [NOAA equations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF), evaluated at interval midpoints in UTC; daylight/profile features use local solar time.
- Daily screening uses repeated daylight export or midday depression. Export defaults: less than -0.05 kW, at least 5% of valid daylight intervals and five distinct days. Profile windows are solar hours [6,10), [10,16), [17,21), requiring respectively 12, 20, 12 quarter-hour observations. The smaller shoulder mean must exceed 10% of median absolute net load (and 1e-6 kW). At least 15 days must be eligible; valley ratio <0.55 on >=35% and median <0.80. Power/profile and model-quality/range threshold comparisons treat differences within `1e-12 * max(1, abs(value), abs(threshold))` as equality, excluding machine roundoff at exact boundaries. Three consecutive daily evaluations change the stable state. With complete data, day 30 is the first evaluation, day 32 can establish `likely_pv`, and subsequent intervals can be estimated.
- The experimental load model is `max(0, b0 + b1 * kvar)`. Earliest 25 selected days train; latest five validate. Full rank, >=20 usable observations per fit/test, R² >=0.5 and RMSE/nighttime-load-RMS <=0.25 are required. Accepted models refit all 30 days. Daytime kvar must stay within training range plus 10% of its width. Generation is clipped to zero and the measured export lower bound. The band uses the held-out 95th percentile absolute nighttime residual. Eligible night estimates are zero; twilight is unavailable. Coverage and reactive-power quality gates still apply.
- Known storage/other generation and export on at least three selected nights imply ambiguity and suppress generation. A rejected load model withholds generation independently of PV screening. Meter-only signals cannot identify all storage, generator, inverter-Q or load-profile confounders.

## Configuration and lifecycle

The feature XML/descriptor exposes `latitude`, `longitude`, `polarity`, `power_scale`, `reactive_scale`, `min_coverage`, `export_threshold_kw`, `export_fraction`, `valley_threshold`, `valley_fraction`, `valley_median`, `min_r2`, `max_nrmse`, `q_margin`, `storage`, and `other_generation`. Defaults are in `defaults.json`; `latitude=longitude=999` deliberately represents absent coordinates. General `FeatureState` controls enablement. Precision-sensitive values are decimal strings parsed as doubles, not SDK FLOAT.

Invalid updates retain the previous valid configuration. Accepted updates commit at the next interval boundary. Coordinates, polarity, scaling or algorithm identity changes clear learned history and restart warm-up. Threshold changes invalidate the accepted model until reevaluation. Aggregation/model work is on a worker; host configuration exercises the real SDK XML parser and registered callbacks. Disable and shutdown stop subscriptions, finish bounded delivery attempts, checkpoint and join workers. Normal shutdown never calls deregistration.

History/model state (`PVS1`) and delivery state (`PVDL1`) live beneath `/usr/share/flash/587530241/587333633/pv`. Each file has a version/checksum, bounded parsing and atomic temporary-file/fsync/rename replacement. Checkpoints contain completed intervals and a compact current accumulator, not raw one-second history. Corruption causes warm-up with diagnostics. Delivery is checkpointed before analytics; an interrupted pair can leave analytics one checkpoint behind while suppressing repeated output IDs. These two files are not a transactional database. Successful remote storage followed by power failure before the next checkpoint can cause a repeated stable ID; consumers should deduplicate it. Checksums detect corruption, not malicious tampering.

## Deliveries and decoding

`tools/decode.py` decodes all three versioned forms:

| Payload | Purpose | Bound |
|---|---|---|
| `PVH1:` + Base64 | Four 15-minute records per hour; shorter batches on provenance changes/shutdown | 1,024 bytes; normally 249 |
| `PVD1:` + JSON | UTC daily measured/estimated energy, coverage, provenance and delivery diagnostics | 1,024 bytes |
| `98#PV#1#...` | Stable screening-state transition, retained history event | 256 bytes |

Only transition into `likely_pv` requests alarm delivery. Routine estimates do not trigger recurring alarms. IDs include version, UTC interval/event boundary and configuration identity; batches also carry model identity. The decoder attaches the generation assumptions and uncalibrated-uncertainty label. Float32 quantization happens only on the wire; numerical parity is evaluated before quantization. Missing generation is null, not zero. Daily `mixed` identifies a day containing multiple configurations/models.

The wire header is little-endian `<qQQB` (first UTC start, config hash, model hash, row count); rows are `<B7fHHIBB` (quarter-hour offset; net kW, import kWh, export kWh, generation kW, low/high kW, generation kWh; observed/estimated seconds; quality flags; state/reason). The `PVH1` decoder is the executable format reference. Hashes are FNV-1a provenance identifiers; fixture/source manifests use SHA-256.

Quality bits: 1 gap, 2 duplicate/backward time, 4 partial real-power coverage, 8 missing reactive coverage, 16 generation assumptions, 32 uncalibrated band, 64 ambiguity, 128 missing coordinates, 256 learning, 512 unavailable, 1024 queue loss, 2048 configuration reset, 4096 reset from an established screening state.

Routine stored bytes are capped at **8,192 per actual delivery UTC day**, including attempted retries, independently of historical replay timestamps. A normal full synthetic day produced 25 stored records totaling 6,219 bytes. The outbox holds 32 messages, retries at most once after 60 seconds, and logs/counts policy/transport rejections and dropped messages. Final shutdown completes remaining bounded attempts. A bulk replay of many historical days can intentionally exceed today's allowance and drop outputs; use the portable harness for long-history validation and seeded state for representative SDK replay.

## Reproduce in Ubuntu-22.04 WSL

If creating a fresh clone, recheck these IDs before running the SDK's clone script with `PVDetectionAgent "PV Detection Agent" 23020001 23050001 23030001 SDK1`. The existing project is already cloned and owned by `fgonzales`.

```bash
KIT=/mnt/c/Users/fgonzales/git/DI_Validator/pv_agent
RESULTS=/mnt/c/Users/fgonzales/git/DI_Validator/runtime/pv-agent
BUILD=/root/di-sdk-setup/pv-detection/core-build
PROJECT=/root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/PVDetectionAgent
cmake -S "$KIT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j4
ctest --test-dir "$BUILD" --output-on-failure
python3 "$KIT/tools/verify_parity.py" --binary "$BUILD/pv_replay" \
  --fixtures "$RESULTS/fixtures" --report "$RESULTS/parity-report.json"
"$BUILD/pv_replay" "$RESULTS/fixtures/warm_state.pvr" \
  --state-out "$RESULTS/fixtures/warm_state.pvs" > /tmp/pv-warm-state.json
python3 "$KIT/tools/install_agent.py"
cd "$PROJECT"
./host_build.sh
./host_replay.sh --output /root/di-sdk-setup/pv-detection/replay-new
python3 "$KIT/tools/verify_integration.py" --binary "$BUILD/pv_replay" \
  --fixtures "$RESULTS/fixtures" --output /root/di-sdk-setup/pv-detection/integration-new
python3 "$KIT/tools/measure.py" --binary "$BUILD/pv_replay" \
  --fixtures "$RESULTS/fixtures" --output /root/di-sdk-setup/pv-detection/measurements-new
```

Use root for the vendor host package installation (`wsl.exe -d Ubuntu-22.04 -u root`). The host launcher uses a private D-Bus, refuses to disrupt another running SDK session, and uses actual `/usr/local/bin/DataServer` and its SQLite database. It writes only PV feature configuration/test outcomes; it does not remove retained outcomes. Remote HES configuration delivery is not tested. Each integration output directory should be new. `host_replay.sh --cold` omits learned seed state.

To regenerate the frozen paired synthetic fixtures using the existing Windows environment:

```powershell
.\.venv\Scripts\python.exe pv_agent/tools/export_fixtures.py
```

PVR1 files contain explicit constant one-second runs, their configuration and validity fields. They are synthetic measured-sample fixtures; they are never generated by upsampling the hourly AMI files. The fixture manifest records oracle/source/configuration/input/result provenance. `data-audit.json` explains why the six supplied hourly AMI files and structure-level registry are not individual-meter estimator ground truth.

Build and validate the unsigned ARM package:

```bash
cd "$PROJECT"
./build.sh TargetRelease
./pack.sh
python3 "$KIT/tools/verify_package.py" \
  --package /path/to/Release/PVDetectionAgent_0.1.0.REVISION.zip \
  --binary /path/to/PVDetectionAgent_Daemon --report "$RESULTS/package-report.json"
```

The SDK clears its working build output when switching targets. Copy the complete `build/PVDetectionAgent` tree outside SDK build directories before restoring the host with `./host_build.sh`. Retained artifacts for this implementation are under `/root/di-sdk-setup/pv-detection/artifacts`; Windows copies, frozen fixtures, reports and logs are in `runtime/pv-agent`. The package verifier checks ARM ELF/dependencies, nested hashes, ZIP/XML integrity, hardware selectors, firmware/AppServ constraints and host/Python exclusion.

## Acceptance boundaries

The package targets only `GENX_SP` and `GENX_PP`, firmware **50.10.312.2** / AppServ **2.0.581.0**, with the inherited **2% CPU, 2,048 KB RAM, 2,048 KB flash** policy unchanged. Package version has the SDK-generated fourth revision component after `0.1.0`.

Numerical parity requires exact states/reasons/flags and `atol=rtol=1e-6` for analytical values. Tests include positive estimates, withheld outputs, warm-up/debounce, missing fields, sign-changing export integration, cloud/load/EV confounders, storage/generators, polarity, seasonal daylight, persistence, configuration boundaries, cancellation, bounded output and chunk independence. Actual SDK tests inspect stored rows, retained alarm state, decoding and restart. SDK payload-length rejection is verified; the host accepts an unregistered feature probe, so **feature policy enforcement on a meter remains unverified**.

Host RAM exceeds 2,048 KB; this is an explicit resource overrun. The HW4.2 **65,000 KB container limit is not an agent allocation**. CPU/RAM host measurements, bounded history size, payload volume and package size are reported separately. ARM cross-compilation proves build compatibility only. Physical one-second acquisition/scaling/time conventions, ARM resource use/flash wear, on-meter policy enforcement, signing and deployment remain external acceptance stages. Field accuracy requires paired measured PV and net-meter data on held-out premises and seasons; synthetic parity cannot establish it.
