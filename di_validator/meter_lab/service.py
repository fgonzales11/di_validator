"""Catalog/run contracts. HTTP reads cached telemetry; only the worker invokes WSL."""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from .. import store
from . import scenarios
from ..faults.network_model import KM_PER_MILE

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def snapshot_bytes(path):
    """Read and close a snapshot before parsing; request cooperative sharing."""
    if os.name != "nt":
        return Path(path).read_bytes()
    import _winapi
    import msvcrt

    # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE. The open handle
    # permits replacement on supporting filesystems. Windows providers that
    # still reject replacement are handled by the producer's bounded retry.
    handle = _winapi.CreateFile(str(path), 0x80000000, 7, 0, 3, 0x80, 0)
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        _winapi.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


def local():
    if os.environ.get("DI_HOSTED") == "1" or os.name != "nt":
        raise HTTPException(
            503,
            "Meter Lab requires local Windows and Ubuntu-22.04 WSL. Hosted deployments cannot control the emulator.",
        )


def read(path, default=None):
    try:
        return json.loads(snapshot_bytes(path))
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + store.uid() + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, default=str))
    for attempt in range(200):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 199:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def capabilities():
    if os.environ.get("DI_HOSTED") == "1" or os.name != "nt":
        return {
            "available": False,
            "reason": "Meter Lab is available only on local Windows / Ubuntu-22.04 WSL.",
            "agents": [],
        }
    cached = read(scenarios.root() / "capabilities.json", {})
    heartbeat = read(scenarios.root() / "worker.json", {})
    alive = time.time() - heartbeat.get("updated_utc", 0) < 45
    descriptors = []
    for key, name, forms in [
        ("pv", "PVDetectionAgent", ["GENX_SP", "GENX_PP"]),
        ("fault", "FaultLocationAgent", ["GENX_PP"]),
    ]:
        defaults = scenarios.config(key, {})
        descriptors.append(
            {
                "id": key,
                "name": name,
                "forms": forms,
                "configuration": defaults,
                "policy": {"cpu_percent": 2, "ram_kb": 2048, "flash_kb": 2048},
                "input": "One-second signed aggregate P/Q"
                if key == "pv"
                else "IA, IB, IC, UA, UB, UC instantaneous waveforms",
                "timing": "15-minute analysis; hourly stored batches"
                if key == "pv"
                else "Analysis after segment completion",
            }
        )
    return {
        **cached,
        "available": bool(cached.get("available") and alive),
        "worker_alive": alive,
        "reason": None
        if cached.get("available") and alive
        else cached.get(
            "reason",
            "Start the local Meter Lab worker; run the installation command if the ARM runtime is not installed.",
        ),
        "agents": descriptors,
        "container": "di-meter-lab",
        "platform": "HW 4.2 Gen5 Riva",
        "firmware": "50.10.312.2",
        "appserv": "2.0.581.0",
        "speeds": ["0.1", "1", "10", "100", "1000", "fastest"],
        "metrology_available": cached.get("metrology_available", False),
        "seed_fixtures": [
            {
                "id": "warm_state",
                "name": "Frozen synthetic PV warm state · 32 days",
                "sha256": store.checksum(scenarios.fixture_root("pv") / "warm_state.pvs"),
            }
        ]
        if (scenarios.fixture_root("pv") / "warm_state.pvs").exists()
        else [],
    }


def detail(identifier):
    run = store.get("meter_lab_run", identifier)
    path = scenarios.folder("runs", identifier)
    run["telemetry"] = read(
        path / "status.json",
        {"run_id": identifier, "state": run["state"], "total_samples": run["manifest"]["total_samples"]},
    )
    run["state"] = run["telemetry"].get("state", run["state"])
    run["checks"] = read(path / "checks.json", {"verdict": "pending", "checks": []})
    run["checkpoint_available"] = run["state"] == "completed" and (path / "checkpoint.pvs").exists()
    return run


def listing():
    return [detail(r["id"]) for r in store.listing("meter_lab_run")][:200]


def create(request):
    local()
    scenario = store.get("meter_lab_scenario", request.scenario_id)
    if scenario["agent"] == "fault" and request.meter_form != "GENX_PP":
        raise ValueError("Fault Location requires a polyphase meter")
    if request.mode == "metrology" and (scenario["agent"] != "pv" or request.speed != "1"):
        raise ValueError("The PV SDK metrology check requires PV and real-time (1×) playback")
    if request.mode == "metrology" and not capabilities().get("metrology_available"):
        raise ValueError("SDK metrology check runtime is unavailable")
    if not capabilities()["available"]:
        raise ValueError(capabilities()["reason"])
    identifier = store.uid()
    manifest = {
        k: v
        for k, v in scenario.items()
        if k
        in {
            "agent",
            "configuration",
            "encoding",
            "fs",
            "f0",
            "start_time",
            "duration_seconds",
            "total_samples",
            "input_sha256",
            "feeder",
        }
    }
    manifest.update(
        version=1,
        run_id=identifier,
        scenario_id=scenario["id"],
        **request.model_dump(exclude={"scenario_id"}),
        clocks={
            "scenario": "recording UTC" if scenario["agent"] == "pv" else "recording-relative seconds",
            "sdk": "actual UTC",
            "budgets": "actual elapsed time",
        },
        source_hash=scenario.get("source_sha256"),
        parameters=scenario["request"]["parameters"],
        created_at=store.now(),
    )
    seed = None
    if request.seed_fixture:
        if scenario["agent"] != "pv" or request.mode != "replay":
            raise ValueError("The warm-state fixture requires PV scenario replay")
        seed = scenarios.fixture_root("pv") / "warm_state.pvs"
        header, values, source = scenarios.fixture("pv", "warm_state")
        if not seed.exists():
            raise ValueError("The frozen warm-state fixture has not been generated")
        if scenarios.config("pv", header) != manifest["configuration"]:
            raise ValueError("Warm-state fixture configuration is incompatible")
        if manifest["start_time"] < values[-1, 0] + values[-1, 1]:
            raise ValueError("Use warm_tail or observations after the 32-day seed history")
        manifest["seed_sha256"] = store.checksum(seed)
        manifest["seed_source_sha256"] = store.checksum(source)
    if request.seed_run_id:
        if request.mode != "replay":
            raise ValueError("Saved checkpoints are available for scenario replay only")
        previous = detail(request.seed_run_id)
        if scenario["agent"] != "pv" or previous["agent"] != "pv" or not previous["checkpoint_available"]:
            raise ValueError("Select a completed PV run with a saved checkpoint")
        if previous["manifest"]["configuration"] != manifest["configuration"]:
            raise ValueError("Checkpoint configuration is incompatible; choose clean state")
        if (
            manifest["start_time"]
            < previous["manifest"]["start_time"] + previous["manifest"]["duration_seconds"]
        ):
            raise ValueError("Checkpoint continuation must begin after the seed recording ends")
        seed = scenarios.folder("runs", previous["id"]) / "checkpoint.pvs"
        manifest["seed_sha256"] = store.checksum(seed)
    record = {
        "id": identifier,
        "agent": scenario["agent"],
        "name": scenario["name"],
        "state": "preparing",
        "created_at": store.now(),
        "manifest": manifest,
    }
    # Atomic reservation across all API processes; a separate worker owns this lane.
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for row in db.execute("select body from objects where kind='meter_lab_run'"):
            current = json.loads(row[0])
            status = read(scenarios.folder("runs", current["id"]) / "status.json", current)
            if status.get("state") not in TERMINAL:
                raise HTTPException(409, "One Meter Lab run is already active")
        path = scenarios.folder("runs", identifier)
        path.mkdir(parents=True)
        shutil.copyfile(scenarios.folder("scenarios", scenario["id"]) / "input.csv", path / "input.csv")
        atomic(path / "scenario.json", scenario)
        fixture_manifest = scenarios.folder("scenarios", scenario["id"]) / "fixture-manifest.json"
        if fixture_manifest.exists():
            shutil.copyfile(fixture_manifest, path / "fixture-manifest.json")
        truth = scenarios.folder("scenarios", scenario["id"]) / "truth.json"
        if truth.exists():
            shutil.copyfile(truth, path / "source-truth.json")
        if seed:
            shutil.copyfile(seed, path / "seed.pvs")
        atomic(path / "manifest.json", manifest)
        db.execute(
            "insert into objects values (?,?,?,?)",
            ("meter_lab_run", identifier, store.encode(record), record["created_at"]),
        )
    return detail(identifier)


def control(identifier, action):
    local()
    run = detail(identifier)
    if run["state"] in TERMINAL:
        raise HTTPException(409, "This run has finished; create a new run to change its setup")
    if run["manifest"]["mode"] == "metrology" and action != "stop":
        raise HTTPException(
            409,
            "The SDK metrology check uses actual UTC and supports Stop; pause is available in scenario replay",
        )
    if action == "pause" and run["state"] not in {"running", "paused"}:
        raise HTTPException(409, "Pause is available while streaming inputs")
    if action == "resume" and run["state"] != "paused":
        raise HTTPException(409, "Only a paused run can resume")
    atomic(
        scenarios.folder("runs", identifier) / "control.json",
        {"action": action, "requested_utc": time.time()},
    )
    return {"accepted": True, "action": action}


def diagnostic_rows(identifier):
    path = scenarios.folder("runs", identifier) / "diagnostics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in snapshot_bytes(path).splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            break  # Last line may still be in flight.
    return rows


def outcomes(identifier):
    run = detail(identifier)
    path = scenarios.folder("runs", identifier)
    result = []
    for row in read(path / "outcomes.json", []):
        if row["FeatureId"] != int("23030001" if run["agent"] == "pv" else "23030000", 16):
            continue
        payload = row["payload"]
        decoded = None
        error = None
        try:
            if run["agent"] == "pv":
                from pv_agent.tools.decode import decode

                decoded = decode(payload)
            elif payload.startswith("98#FAULT#"):
                fields = payload.split("#")
                decoded = dict(
                    version=int(fields[2]),
                    id=fields[3],
                    config=fields[4],
                    onset_s=float(fields[5]),
                    type=fields[6],
                    loop=fields[7],
                    # Withheld distances carry their status (unavailable, out_of_range, ...).
                    distance_status="estimated" if fields[8][:1].isdigit() else fields[8],
                    distance_km=float(fields[8]) if fields[8][:1].isdigit() else None,
                    line_parameters=fields[9],
                    ground=fields[10],
                )
            else:
                decoded = json.loads(payload)
        except (ValueError, IndexError, KeyError) as exc:
            error = str(exc)
        result.append(
            {
                **row,
                "source": "DataServer stored outcome",
                "bytes": len(payload.encode()),
                "decoded": decoded,
                "decode_error": error,
            }
        )
    return result


def series(identifier, start=None, end=None, max_points=1500):
    run = detail(identifier)
    scenario = store.get("meter_lab_scenario", run["manifest"]["scenario_id"])
    saved = scenarios.folder("scenarios", scenario["id"]) / "input.npy"
    data = (
        np.load(saved, mmap_mode="r", allow_pickle=False)
        if saved.exists()
        else np.loadtxt(scenarios.folder("runs", identifier) / "input.csv", delimiter=",", ndmin=2)
    )
    metrology = run["manifest"]["mode"] == "metrology"
    if metrology:
        observations = scenarios.folder("runs", identifier) / "observations.csv"
        data = (
            np.loadtxt(observations, delimiter=",", ndmin=2)
            if observations.exists() and observations.stat().st_size
            else np.empty((0, 6))
        )
    # Constant blocks can begin before a live scope window and still contain
    # its samples. Clip overlapping blocks before producing plot endpoints.
    if (
        scenario["encoding"] == "pv-constant-seconds"
        and not metrology
        and (start is not None or end is not None)
    ):
        block_end = data[:, 0] + data[:, 1] - 1
        first = np.maximum(data[:, 0], start) if start is not None else data[:, 0]
        last = np.minimum(block_end, end) if end is not None else block_end
        keep = first <= last
        data = data[keep].copy()
        data[:, 0] = first[keep]
        data[:, 1] = last[keep] - first[keep] + 1
    if start is not None:
        data = data[data[:, 0] >= start]
    if end is not None:
        data = data[data[:, 0] <= end]
    if run["agent"] == "pv":
        lengths = None
        if scenario["encoding"] == "pv-constant-seconds" and not metrology:
            values = np.column_stack([data[:, 0], data[:, 2:4] / 1000, data[:, 4:6]])
            lengths = data[:, 1]
        else:
            values = np.column_stack([data[:, 0], data[:, 1:3] / 1000, data[:, 3:5]])
        for i in [1, 2]:
            values[values[:, i + 2] == 0, i] = np.nan
        plotted = scenarios.gap_markers(values[:, :3], 1, lengths)
        if lengths is not None and (start is not None or end is not None):
            endpoints = values[:, :3].copy()
            endpoints[:, 0] += lengths - 1
            plotted = np.concatenate([plotted, endpoints])
            plotted = plotted[np.argsort(plotted[:, 0], kind="stable")]
        inputs = {
            "channels": ["Signed aggregate kW", "Aggregate kvar"],
            "rows": scenarios.bounded(plotted, max_points),
        }
        voltage = None
        if not metrology and scenario["encoding"] == "pv-samples" and np.isfinite(data[:, 6:9]).any():
            voltage = {
                "channels": ["VA", "VB", "VC"],
                "rows": scenarios.bounded(
                    scenarios.gap_markers(np.column_stack([data[:, 0], data[:, 6:9]]), 1), max_points
                ),
            }
    else:
        voltage = None
        inputs = {
            "channels": list(scenarios.PHASES),
            "rows": scenarios.bounded(scenarios.gap_markers(data, 1 / scenario["fs"]), max_points),
        }
    placements = {p["segment"]: p for p in network_placements(scenario)}
    output = []
    for r in diagnostic_rows(identifier):
        t = r["start"]
        if (start is None or t >= start) and (end is None or t <= end):
            row = {k: v for k, v in r.items() if k != "diagnostics"}
            if r.get("segment") in placements:
                row["network_location"] = placements[r["segment"]]
            output.append(row)
    total = len(output)
    if len(output) > max_points:
        matrix = np.array(
            [
                [
                    i,
                    *[
                        r.get(k) if r.get(k) is not None else np.nan
                        for k in ["generation_kw", "import_kwh", "export_kwh", "estimated_seconds", "flags"]
                    ],
                ]
                for i, r in enumerate(output)
            ]
        )
        output = [output[int(row[0])] for row in scenarios.bounded(matrix, max_points)]
    return {
        "inputs": inputs,
        "voltage": voltage,
        "diagnostics": output,
        "source": "ARM analysis diagnostics",
        "progress_time": run["telemetry"].get("scenario_time"),
        "output_available": total,
    }


def network_placements(scenario):
    """Host-side feeder-model candidates for a Wavewin scenario (not ARM output)."""
    if scenario.get("agent") != "fault" or not scenario.get("feeder"):
        return []
    scenarios.reference(scenario)
    return read(scenarios.folder("scenarios", scenario["id"]) / "network.json", [])


def telemetry(identifier, cursor=0, limit=100):
    path = scenarios.folder("runs", identifier) / "telemetry.jsonl"
    rows = []
    next_cursor = cursor
    if path.exists():
        with path.open("rb") as source:
            source.seek(min(cursor, path.stat().st_size))
            for _ in range(limit):
                line = source.readline()
                if not line.endswith(b"\n"):
                    break
                rows.append(json.loads(line))
                next_cursor = source.tell()
    return {"cursor": next_cursor, "events": rows, "status": detail(identifier)["telemetry"]}


def relay_agreement(identifier, scenario):
    """Agreement between the agent's distance and the relay's own LOCATION estimate.

    Both are single-ended estimates on the same waveform; neither is ground truth,
    so this reports how closely they agree rather than a pass/fail accuracy.
    """
    try:
        relay_mi = float(scenario["feeder"]["wavewin"]["location"])
    except (TypeError, ValueError, KeyError):
        return "unavailable", "The relay report carries no LOCATION"
    distances = [
        r["distance"]["estimated_distance_km"]
        for r in diagnostic_rows(identifier)
        if r.get("distance", {}).get("status") == "estimated"
    ]
    if not distances:
        return "unavailable", f"Agent withheld the distance; relay LOCATION {relay_mi:.2f} mi"
    agent_mi = distances[0] / KM_PER_MILE
    return (
        "unavailable",
        f"Agent {agent_mi:.2f} mi vs relay {relay_mi:.2f} mi (gap {abs(agent_mi - relay_mi):.2f} mi); "
        "agreement between two estimates, not accuracy",
    )


def evaluate(identifier):
    run = detail(identifier)
    path = scenarios.folder("runs", identifier)
    checks = []

    def check(category, name, verdict, detail=""):
        checks.append(dict(category=category, name=name, verdict=verdict, detail=detail))

    if run["state"] != "completed":
        check("structural", "Run completed", "unavailable", run["state"])
    else:
        scenario = store.get("meter_lab_scenario", run["manifest"]["scenario_id"])
        try:
            scenarios.reference(scenario)
            reference = read(scenarios.folder("scenarios", scenario["id"]) / "reference.json")
            rows = diagnostic_rows(identifier)
            if run["manifest"]["mode"] == "metrology":
                observed = np.loadtxt(path / "observations.csv", delimiter=",", ndmin=2)
                reference = scenarios.pv_reference(
                    [tuple([r[0], 1, *r[1:6]]) for r in observed],
                    scenario["configuration"],
                )
            if run["manifest"].get("seed_run_id") or run["manifest"].get("seed_fixture"):
                chain = [scenario]
                parent = run
                while parent["manifest"].get("seed_run_id"):
                    parent = detail(parent["manifest"]["seed_run_id"])
                    chain.insert(0, store.get("meter_lab_scenario", parent["manifest"]["scenario_id"]))
                inputs = []
                if parent["manifest"].get("seed_fixture"):
                    _, seed_values, _ = scenarios.fixture("pv", "warm_state")
                    inputs.extend(tuple(row) for row in seed_values)
                for s in chain:
                    v = np.loadtxt(
                        scenarios.folder("scenarios", s["id"]) / "input.csv", delimiter=",", ndmin=2
                    )
                    if s["encoding"] == "pv-samples":
                        v = np.column_stack([v[:, 0], np.ones(len(v)), v[:, 1:6]])
                    inputs.extend(tuple(row) for row in v)
                reference = scenarios.pv_reference(inputs, scenario["configuration"])
                reference["intervals"] = [
                    r for r in reference["intervals"] if r["start"] >= scenario["start_time"]
                ]
            atomic(path / "reference.json", reference)
            if run["agent"] == "pv":
                from pv_agent.tools.verify_parity import compare

                compare(reference["intervals"], rows)
                compare(reference["detection"], run["telemetry"]["detection"])
            else:
                from di_agent.tools.verify_parity import compare

                compare(reference, {"segments": rows})
            check(
                "numerical",
                "ARM / Python parity",
                "pass",
                "Unquantized output; atol=1e-6, rtol=1e-6; fault tensors atol=1e-5, rtol=2e-5",
            )
        except Exception as error:
            check("numerical", "ARM / Python parity", "fail", str(error))
        if scenario["request"]["source"] == "preset":
            name = scenario["request"]["source_id"]
            params = scenario["request"]["parameters"]
            native_values = all(
                float(params.get(key, default)) == default
                for key, default in {
                    "magnitude": 1,
                    "load_scale": 1,
                    "pv_scale": 1,
                    "cloud_variation": 0,
                    "noise": 0,
                    "polarity": 1,
                    "outage_seconds": 0,
                }.items()
            )
            if name == "midday_load_drop":
                check(
                    "expectation",
                    "Synthetic truth: no PV",
                    "fail" if run["telemetry"].get("detection", {}).get("state") == "likely_pv" else "pass",
                    "Known adverse result retained: midday load reduction can create a false positive",
                )
            elif run["agent"] == "pv" and name in ["pv_export", "pv_self_consumed"]:
                available = any(
                    r.get("available") and (r.get("generation_kw") or 0) > 0
                    for r in diagnostic_rows(identifier)
                )
                learned = run["telemetry"].get("detection", {}).get("state") == "likely_pv" and run[
                    "telemetry"
                ].get("detection", {}).get("model", {}).get("valid")
                check(
                    "expectation",
                    "Generation conditional on learning/model quality",
                    "pass" if available or not learned else "fail",
                )
            elif name == "1A_val1":
                header, _, _ = scenarios.fixture("fault", name)
                for key in ["fs", "f0", "source_id"]:
                    header.pop(key, None)
                baseline = (
                    native_values
                    and params.get("start_offset_seconds", 0) == 0
                    and params.get("duration_seconds") is None
                    and scenario["configuration"] == scenarios.config("fault", header)
                )
                r = diagnostic_rows(identifier)[0]
                d = r.get("distance", {})
                okay = (
                    abs(r.get("onset", 0) - 0.0995) < 1e-9
                    and r.get("fault_type_heuristic") == "1ph-G"
                    and d.get("fault_loop") == "AG"
                    and abs(d.get("estimated_distance_km", 0) - 0.8908903568462709) < 1e-6
                )
                check(
                    "expectation",
                    "1A_val1 regression",
                    "pass" if baseline and okay else "fail" if baseline else "unavailable",
                    "Baseline example parameters"
                    if baseline
                    else "Modified inputs/configuration; numerical parity is checked separately",
                )
            elif (
                name == "sign_changes"
                and native_values
                and run["manifest"]["mode"] == "replay"
                and scenario["total_samples"] == 900
            ):
                rows = diagnostic_rows(identifier)
                energy = 0.125 * float(scenario["configuration"]["power_scale"])
                okay = (
                    len(rows) == 1
                    and abs(rows[0]["import_kwh"] - energy) < 1e-6
                    and abs(rows[0]["export_kwh"] - energy) < 1e-6
                )
                check(
                    "expectation",
                    "Samplewise import/export integration",
                    "pass" if okay else "fail",
                    f"Zero average net power still measures {energy:g} kWh import and {energy:g} kWh export",
                )
            elif name == "comtrade_99":
                count = len(diagnostic_rows(identifier))
                check(
                    "expectation",
                    "All 99 COMTRADE recordings analyzed separately",
                    "pass" if count == 99 else "fail",
                    f"{count} contiguous recording segments; original fixture hashes and batch offsets preserved",
                )
            else:
                check(
                    "expectation",
                    "Scenario truth",
                    "unavailable",
                    "Behavior reference is available; independent physical truth has not been declared for this case",
                )
            truth_file = path / "source-truth.json"
            if (
                run["agent"] == "pv"
                and native_values
                and truth_file.exists()
                and run["manifest"]["mode"] == "replay"
                and all(
                    float(scenario["configuration"][k]) == 1
                    for k in ["power_scale", "reactive_scale", "polarity"]
                )
            ):
                truth = {r["start"]: r["pv_kw"] for r in read(truth_file)}
                errors = [
                    abs(r["generation_kw"] - truth[r["start"]])
                    for r in diagnostic_rows(identifier)
                    if r.get("available") and r["start"] in truth
                ]
                check(
                    "expectation",
                    "Paired synthetic PV generation",
                    "pass" if errors and max(errors) <= 1e-6 else "fail" if errors else "unavailable",
                    f"{len(errors)} available intervals; maximum absolute error {max(errors):.9g} kW; synthetic correctness only"
                    if errors
                    else "Generation was withheld; no accuracy claim",
                )
        else:
            check(
                "expectation",
                "Field accuracy",
                "unavailable",
                "Uploaded/recorded input has no verified measured fault distance or PV production truth",
            )
            if run["agent"] == "fault" and scenario.get("feeder"):
                check("expectation", "Relay LOCATION agreement", *relay_agreement(identifier, scenario))
        actual = outcomes(identifier)
        check(
            "delivery",
            "Payload versions and limits",
            "pass"
            if all(
                not r["decode_error"] and r["bytes"] <= (256 if r["table"] == "AgentEvents" else 1024)
                for r in actual
            )
            else "fail",
            f"{len(actual)} actual DataServer rows",
        )
        if run["agent"] == "fault":
            analyzed = [r for r in diagnostic_rows(identifier) if r["status"] == "analyzed"]
            events = [r for r in actual if r["table"] == "AgentEvents"]
            expected_ids = {identifier + "-" + str(r["segment"]) for r in analyzed}
            okay = all(r["decoded"] and r["decoded"].get("id") in expected_ids for r in actual)
            check(
                "delivery",
                "Outcome correlation IDs",
                "pass" if okay else "fail",
                f"{len(events)} retained events; {sum(bool(r.get('isUnsentAlarm')) for r in events)} alarm flags",
            )
            expected = len(analyzed) + sum(r.get("inception_source") == "detected" for r in analyzed)
            rejected = run["telemetry"].get("sdk_rejected", 0)
            check(
                "delivery",
                "Required stored outcomes",
                "pass" if len(actual) == expected else "unavailable" if rejected else "fail",
                f"Expected {expected}, stored {len(actual)}; SDK rejections: {run['telemetry'].get('rejected', 0)}",
            )
        else:
            expected = bool(diagnostic_rows(identifier))
            check(
                "delivery",
                "Stored PV output",
                "pass" if actual or not expected else "fail",
                f"SDK rejected {run['telemetry'].get('rejected', 0)} attempts; bounded drops {run['telemetry'].get('dropped', 0)}",
            )
        if (path / "submissions.jsonl").exists():
            submitted = [json.loads(line) for line in (path / "submissions.jsonl").read_text().splitlines()]
            completed = (
                [json.loads(line) for line in (path / "sdk-completions.jsonl").read_text().splitlines()]
                if (path / "sdk-completions.jsonl").exists()
                else []
            )
            pending = {r["key"] for r in submitted if r["return_code"] == 0} - {r["key"] for r in completed}

            def payload_key(row):
                h = 14695981039346656037
                for b in row["payload"].encode():
                    h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
                return format(h, "x") + ("e" if row["table"] == "AgentEvents" else "d")

            stored = {payload_key(row) for row in actual}
            successful = {r["key"] for r in completed if r["return_code"] == 0}
            check(
                "delivery",
                "SDK completions reconcile with database",
                "pass" if not pending and stored == successful else "fail",
                f"{len(submitted)} submissions, {len(completed)} SDK attempts, {len(pending)} pending; successful payload identities checked against SQLite",
            )
        check(
            "structural",
            "Processed sample count",
            "pass" if run["telemetry"].get("processed") == run["manifest"]["total_samples"] else "fail",
        )
        acknowledgements = path / "acknowledgements.jsonl"
        if acknowledgements.exists():
            acks = [json.loads(line) for line in acknowledgements.read_text().splitlines()]
            okay = (
                bool(acks)
                and all(
                    a["run_id"] == identifier and a["sequence"] == i and a["processed"] <= a["received"]
                    for i, a in enumerate(acks)
                )
                and acks[-1]["complete"]
                and acks[-1]["processed"] == run["manifest"]["total_samples"]
            )
            check(
                "structural",
                "Recorded ARM acknowledgements",
                "pass" if okay else "fail",
                f"{len(acks)} ordered acknowledgements; counters checked independently",
            )
    verdict = (
        "fail"
        if any(c["verdict"] == "fail" for c in checks)
        else "pass"
        if run["state"] == "completed"
        else "unavailable"
    )
    report = {
        "verdict": verdict,
        "checks": checks,
        "generated_at": store.now(),
        "reference_sha256": store.checksum(path / "reference.json")
        if (path / "reference.json").exists()
        else None,
    }
    atomic(path / "checks.json", report)
    return report


def export(identifier):
    run = detail(identifier)
    path = scenarios.folder("runs", identifier)
    if run["state"] not in TERMINAL:
        raise HTTPException(409, "Stop or complete this run before exporting frozen evidence")
    destination = scenarios.root() / "exports"
    destination.mkdir(exist_ok=True)
    # Downloads can overlap. Each response owns an immutable archive so a
    # second request cannot truncate a file already being streamed.
    archive = destination / (identifier + "." + store.uid() + ".zip")
    atomic(path / "run.json", run)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for file in sorted(path.rglob("*")):
            if file.is_file() and not file.name.endswith(".tmp"):
                output.write(file, file.relative_to(path).as_posix())
    return archive
