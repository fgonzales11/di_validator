"""Fixed local WSL supervisor for Meter Lab. No browser-supplied commands or paths.

The detached supervisor owns the container. Windows workers/browser clients may
restart without changing its lifetime. Only validated run IDs cross this CLI.
"""

from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

BASE = Path("/root/di-sdk-setup/meter-lab")
ROOT = BASE / "lxc/di-meter-lab/rootfs"
AGENTS = {"pv": ("23020001", "23030001"), "fault": ("23020000", "23030000")}
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(value, bytes):
        tmp.write_bytes(value)
    else:
        tmp.write_text(json.dumps(value, allow_nan=False) if not isinstance(value, str) else value)
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            break
    return result


def artifact(run_id):
    if not re.fullmatch("[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run identity")
    return Path(read(BASE / "workspace.json")["artifacts"]) / "runs" / run_id


def identity(pid):
    try:
        f = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if f[0] == "Z" else f[19]
    except (OSError, ValueError):
        return None


def live(owner):
    return bool(owner and owner.get("start") and identity(owner["pid"]) == owner["start"])


def manager(run_id, manifest):
    run = BASE / "runs" / run_id
    os.environ.update(
        DI_METER_BASE=str(run),
        DI_METER_LXC=str(BASE / "lxc"),
        DI_METER_NAME="di-meter-lab",
        DI_METER_AGENT=manifest["agent"],
        DI_METER_TMP=str(run / "tmp"),
        DI_METER_LAB_REPLAY="1" if manifest["mode"] == "replay" else "0",
        DI_METER_LAB_METROLOGY="1" if manifest["mode"] == "metrology" else "0",
        DI_METER_METROLOGY_LIMIT=str(manifest["total_samples"]) if manifest["mode"] == "metrology" else "0",
    )
    spec = importlib.util.spec_from_file_location("lab_di_meter", BASE / "tools/di_meter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(run_id, manifest, output):
    if manifest["agent"] not in AGENTS or manifest["mode"] not in {"replay", "metrology"}:
        raise ValueError("Unregistered agent/input operation")
    if manifest["mode"] == "metrology" and (manifest["agent"] != "pv" or manifest["speed"] != "1"):
        raise ValueError("The SDK metrology check requires PV at actual one-second speed")
    run = BASE / "runs" / run_id
    if run.exists():
        raise ValueError("A run directory may never be reused")
    for sub in [
        "run",
        "logs",
        "database/flash",
        "database/ram",
        "metsimfiles",
        "stream",
        "tmp",
        "flash/587530241/587333633/pv",
    ]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    # Copy the image database schema and static metrology dictionaries through
    # SQLite backups. Existing development outcomes/registrations are discarded.
    original = Path("/root/di-sdk-setup/meter-emulator/database")
    for source in original.rglob("*.db"):
        target = run / "database" / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
            for table in ["AgentData", "AgentEvents", "AgentMailbox", "AgentRegistration"]:
                if dst.execute(
                    "select 1 from sqlite_master where type='table' and name=?", (table,)
                ).fetchone():
                    dst.execute("delete from " + table)
    socket = run / "run/bus.sock"
    (run / "run/bus.conf").write_text(
        "<busconfig><type>system</type><listen>unix:path="
        + str(socket)
        + '</listen><policy context="default"><allow user="*"/><allow own="*"/>'
        '<allow send_destination="*"/><allow receive_sender="*"/></policy></busconfig>'
    )
    config = (BASE / "container-template.conf").read_text()
    config = "\n".join(line for line in config.splitlines() if "system_bus_socket" not in line)
    config += (
        "\nlxc.mount.entry = "
        + str(socket)
        + " var/run/dbus/system_bus_socket none ro,bind,create=file 0 0\n"
    )
    for source, target in [("stream", "meter-lab"), ("flash", "usr/share/flash"), ("tmp", "tmp")]:
        config += f"lxc.mount.entry = {run / source} {target} none rw,bind,create=dir 0 0\n"
    (ROOT.parent / "config").write_text(config)
    aid, fid = AGENTS[manifest["agent"]]
    # SDK configuration persistence expects the installed overlay/agent directory.
    (run / "flash" / str(int(aid, 16) + 0x30000) / str(int(aid, 16))).mkdir(parents=True, exist_ok=True)
    xml = ROOT / f"etc/agents.d/{aid}.{fid}.xml"
    tree = ET.parse(xml)
    values = {**manifest["configuration"], "FeatureState": "1"}
    found = set()
    for parameter in tree.iter("parameter"):
        name = parameter.get("name")
        if name in values:
            parameter.set("value", str(values[name]))
            found.add(name)
    if set(values) - found:
        raise ValueError("Unknown feature parameters: " + str(set(values) - found))
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    metrology = ROOT / f"etc/agents.d/{aid}.{int(aid, 16) + 0x13000:08x}.xml"
    tree = ET.parse(metrology)
    for p in tree.iter("parameter"):
        if p.get("name") == "NumPhases":
            p.set("value", "1" if manifest["meter_form"] == "GENX_SP" else "3")
        if p.get("name") == "MeterForm":
            p.set("value", "2" if manifest["meter_form"] == "GENX_SP" else "9")
    tree.write(metrology, encoding="utf-8", xml_declaration=True)
    frozen_config = output / "configuration"
    frozen_config.mkdir(exist_ok=True)
    for config_file in (ROOT / "etc/agents.d").glob(aid + "*.xml"):
        shutil.copyfile(config_file, frozen_config / config_file.name)
    stream = run / "stream"
    (stream / "identity").write_text(run_id)
    (stream / "waveform.meta").write_text(f"{manifest.get('fs', 0)} {manifest.get('f0', 0)}")
    if manifest.get("seed_sha256"):
        seed = output / "seed.pvs"
        if hashlib.sha256(seed.read_bytes()).hexdigest() != manifest["seed_sha256"]:
            raise ValueError("Seed state checksum changed")
        shutil.copyfile(seed, stream / "seed.pvs")
    source = output / "input.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("Frozen input checksum changed")
    shutil.copyfile(source, run / "input.csv")
    if manifest["mode"] == "metrology":
        with (run / "metsimfiles/PV.csv").open("w") as csvfile:
            csvfile.write("18153531,18153532,18153520,18153521,18153522\n")
            for row in samples(run, manifest):
                csvfile.write(
                    ",".join(
                        [row[1] if float(row[3]) else "nan", row[2] if float(row[4]) else "nan", *row[6:9]]
                    )
                    + "\n"
                )
    shutil.copyfile(BASE / "build-manifest.json", output / "build-manifest.json")
    helpers = output / "runtime-helpers"
    helpers.mkdir(exist_ok=True)
    for name, expected in read(output / "build-manifest.json")["helpers"].items():
        raw = (BASE / "tools" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("Installed helper changed during run preparation: " + name)
        (helpers / name).write_bytes(raw)
    return run


def samples(run, manifest):
    with (run / "input.csv").open() as source:
        for row in csv.reader(source):
            if manifest["encoding"] == "pv-constant-seconds":
                start, seconds = int(row[0]), int(row[1])
                for offset in range(seconds):
                    yield [str(start + offset), *row[2:7], *(["nan"] * 3)]
            else:
                yield row


class Cancelled(Exception):
    pass


def frames(run, manifest, chunk):
    if manifest["encoding"] != "pv-constant-seconds":
        rows = iter(samples(run, manifest))
        while batch := list(itertools.islice(rows, chunk)):
            yield "DATA", batch, len(batch), float(batch[-1][0])
        return
    batch = []
    size = 0
    last = 0
    with (run / "input.csv").open() as source:
        for row in csv.reader(source):
            start, remaining = int(row[0]), int(row[1])
            while remaining:
                count = min(remaining, chunk - size)
                batch.append([str(start), str(count), *row[2:7], "nan", "nan", "nan"])
                start += count
                remaining -= count
                size += count
                last = start - 1
                if size == chunk:
                    yield "CONST", batch, size, last
                    batch = []
                    size = 0
    if batch:
        yield "CONST", batch, size, last


def supervise(run_id):
    output = artifact(run_id)
    manifest = read(output / "manifest.json")
    owner = {"pid": os.getpid(), "start": identity(os.getpid()), "run_id": run_id}
    state = {
        "version": 1,
        "run_id": run_id,
        "state": "preparing",
        "transmitted": 0,
        "received": 0,
        "processed": 0,
        "queue": 0,
        "pending": 0,
        "rejected": 0,
        "stored_data": 0,
        "stored_events": 0,
        "diagnostics": 0,
        "scenario_time": manifest["start_time"],
        "total_samples": manifest["total_samples"],
        "started_utc": time.time(),
        "owner": owner,
        "actual_speed": 0,
        "last_activity": time.time(),
    }
    atomic(output / "owner.json", owner)
    start = time.monotonic()
    last_emit = 0.0
    paused = 0.0
    paused_since = None
    run = None
    di = None
    peak = 0
    container_peak = 0
    cpu = 0.0
    prior_cpu = None
    last_rows = None
    telemetry = (output / "telemetry.jsonl").open("a", buffering=1)

    snapshot_lock = threading.Lock()
    collector_stop = threading.Event()
    collector_errors = []
    copied = {}
    control_stop = threading.Event()
    requested_action = "resume"

    def poll_controls():
        nonlocal requested_action
        # DrvFS reads can take hundreds of milliseconds. Keep that latency out
        # of every frame/acknowledgement, without relaxing stream backpressure.
        while not control_stop.is_set():
            command = read(output / "control.json")
            if command and command.get("action") in {"pause", "resume", "stop"}:
                requested_action = command["action"]
                if requested_action == "stop":
                    return
            control_stop.wait(0.05)

    controller = threading.Thread(target=poll_controls, daemon=True)
    controller.start()

    def collect(force=False):
        nonlocal last_emit, peak, cpu, prior_cpu, last_rows, container_peak
        now = time.monotonic()
        if not force and now - last_emit < 0.5:
            return
        last_emit = now
        state["elapsed_seconds"] = now - start
        state["active_seconds"] = max(
            0.001, now - start - paused - (now - paused_since if paused_since is not None else 0)
        )
        state["actual_speed"] = (
            state["processed"]
            if manifest["mode"] == "metrology"
            else max(0, state["scenario_time"] - manifest["start_time"])
        ) / state["active_seconds"]
        state["updated_utc"] = time.time()
        if run:
            for name in [
                "diagnostics.jsonl",
                "delivery.log",
                "observations.csv",
                "submissions.jsonl",
                "sdk-completions.jsonl",
                "acknowledgements.jsonl",
                "worker-finished.json",
                "error.json",
            ]:
                source = run / "stream" / name
                if source.exists():
                    info = source.stat()
                    signature = (info.st_size, info.st_mtime_ns)
                    if not force and copied.get(name) == signature:
                        continue
                    raw = source.read_bytes()
                    try:
                        atomic(output / name, raw)
                        copied[name] = signature
                    except PermissionError:
                        if state["state"] in TERMINAL:
                            deadline = time.monotonic() + 20
                            while True:
                                try:
                                    atomic(output / name, raw)
                                    break
                                except PermissionError:
                                    if time.monotonic() >= deadline:
                                        raise RuntimeError(
                                            "Final evidence is locked by a Windows reader: " + name
                                        )
                        # Windows readers may briefly deny replacement. Keep the
                        # last complete snapshot and retry next collection cycle.
                        state["artifact_diagnostic"] = "Waiting for Windows evidence reader"
                    if name.startswith("diagnostics"):
                        state["diagnostics"] = raw.count(b"\n")
            submissions = lines(run / "stream/submissions.jsonl")
            completions = lines(run / "stream/sdk-completions.jsonl")
            outstanding = {s["key"] for s in submissions if s["return_code"] == 0} - {
                s["key"] for s in completions
            }
            state["sdk_pending"] = len(outstanding)
            state["sdk_attempts"] = len(completions)
            state["sdk_rejected"] = sum(s["return_code"] != 0 for s in completions)
            state["pending"] = state.get("application_pending", 0) + state["sdk_pending"]
            state["rejected"] = state.get("application_rejected", 0) + state["sdk_rejected"]
            database = run / "database/flash/muse01.db"
            if database.exists():
                try:
                    rows = []
                    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=0.2) as db:
                        db.row_factory = sqlite3.Row
                        for table, key in [("AgentData", "stored_data"), ("AgentEvents", "stored_events")]:
                            selected = db.execute(
                                "select * from " + table + " where AgentId=? and FeatureId=? order by Id",
                                tuple(int(v, 16) for v in AGENTS[manifest["agent"]]),
                            ).fetchall()
                            state[key] = len(selected)
                            for row in selected:
                                item = dict(row)
                                data = item.pop("Data")
                                if isinstance(data, bytes):
                                    data = data.decode("utf-8", errors="replace")
                                rows.append({"table": table, **item, "payload": data})
                    state["stored_bytes"] = sum(len(r["payload"].encode()) for r in rows)
                    state["routine_data_bytes"] = sum(
                        len(r["payload"].encode()) for r in rows if r["table"] == "AgentData"
                    )
                    if rows != last_rows:
                        atomic(output / "outcomes.json", rows)
                        last_rows = rows
                except sqlite3.Error as error:
                    state["database_diagnostic"] = str(error)
            if di:
                for pids in di.agents().values():
                    for pid in pids:
                        try:
                            status = Path(f"/proc/{pid}/status").read_text()
                            peak = max(peak, int(re.search(r"VmHWM:\s+(\d+)", status).group(1)))
                            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                            ticks = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                            cpu = max(cpu, 100 * ticks / max(0.001, state["elapsed_seconds"]))
                        except (OSError, AttributeError):
                            pass
            cgroup = Path("/sys/fs/cgroup/lxc.payload.di-meter-lab")
            for name in ["memory.peak", "memory.current"]:
                try:
                    container_peak = max(container_peak, int((cgroup / name).read_text()) // 1024)
                except OSError:
                    pass
            state["resources"] = {
                "peak_rss_kb": peak,
                "max_observed_cpu_percent": cpu,
                "agent_ram_limit_kb": 2048,
                "agent_cpu_limit_percent": 2,
                "agent_flash_limit_kb": 2048,
                "container_ram_limit_kb": 65000,
                "container_peak_kb": container_peak,
                "container_cpu_limit_percent": 30,
                "container_ram_overrun": container_peak > 65000,
                "ram_overrun": peak > 2048,
                "cpu_overrun": cpu > 2,
                "execution": "ARM under QEMU; not meter qualification",
            }
        snapshot = dict(state)
        snapshot["pending"] = snapshot.get("application_pending", 0) + snapshot.get("sdk_pending", 0)
        snapshot["rejected"] = snapshot.get("application_rejected", 0) + snapshot.get("sdk_rejected", 0)
        if snapshot["state"] in TERMINAL and not force:
            return
        atomic(output / "status.json", snapshot)
        telemetry.write(json.dumps(snapshot) + "\n")

    def emit(force=False):
        if collector_errors:
            raise RuntimeError("Telemetry collection failed: " + str(collector_errors[0]))
        if force:
            with snapshot_lock:
                collect(True)

    def collect_background():
        while not collector_stop.wait(0.75):
            try:
                with snapshot_lock:
                    # A terminal state becomes visible only after shutdown and
                    # the final database/evidence snapshot, below.
                    if state["state"] not in TERMINAL:
                        collect()
            except PermissionError:
                state["artifact_diagnostic"] = "Waiting for Windows evidence reader"
            except Exception as error:
                collector_errors.append(error)
                return

    collector = threading.Thread(target=collect_background, daemon=True)
    collector.start()

    def drain():
        state["state"] = "draining"
        deadline = time.monotonic() + 20
        while True:
            emit(True)
            if not state.get("sdk_pending"):
                return
            if time.monotonic() > deadline:
                raise RuntimeError("SDK output completion timed out; pending requests retained in evidence")
            time.sleep(0.1)

    def finish_worker():
        # A wrapper return/feature signal is not a worker acknowledgement.
        # Keep the agent and SDK output thread alive until its final flush and
        # checkpoint have finished, then reconcile actual SDK completions.
        if manifest["mode"] == "metrology":
            atomic(run / "stream/finish.request", "finish")
        deadline = time.monotonic() + 30
        while not read(run / "stream/worker-finished.json", {}).get("finished"):
            emit()
            if time.monotonic() > deadline:
                raise RuntimeError("Agent worker did not acknowledge orderly completion")
            time.sleep(0.05)
        error = read(run / "stream/error.json")
        if error and not state.get("cancel_requested"):
            raise RuntimeError(error["error"])
        drain()

    def gate():
        nonlocal paused, paused_since
        requested = requested_action
        if requested == "stop":
            raise Cancelled()
        if requested == "pause":
            began = time.monotonic()
            paused_since = began
            state["state"] = "paused"
            emit(True)
            while requested_action == "pause":
                emit()
                time.sleep(0.05)
            paused += time.monotonic() - began
            paused_since = None
            if requested_action == "stop":
                raise Cancelled()
            state["state"] = "running"
            emit(True)

    try:
        with (BASE / "control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic(BASE / "owner.json", owner)
            emit(True)
            run = prepare(run_id, manifest, output)
            di = manager(run_id, manifest)
            di.start()
            state["state"] = "running"
            emit(True)
            if manifest["mode"] == "metrology":
                until = time.monotonic() + max(90, manifest["total_samples"] * 2)
                while state["processed"] < manifest["total_samples"]:
                    action = requested_action
                    if action == "pause":
                        di.feature("pv", False)
                        gate()
                        di.feature("pv", True)
                    elif action == "stop":
                        raise Cancelled()
                    progress = read(run / "stream/metrology-progress.json", {})
                    bridge = read(run / "run/metrology.json", {})
                    state.update(progress)
                    state["application_pending"] = progress.get("pending", 0)
                    state["application_rejected"] = progress.get("rejected", 0)
                    state["transmitted"] = state["received"] = bridge.get("forwarded", 0)
                    state["metrology"] = bridge
                    state["clock"] = "actual UTC"
                    state["queue"] = max(0, state["received"] - state["processed"])
                    if progress:
                        state["last_activity"] = time.time()
                    emit()
                    if time.monotonic() > until + paused:
                        raise RuntimeError("SDK callback processing timed out")
                    time.sleep(0.1)
                finish_worker()
                di.stop()
                state["state"] = "completed"
                return
            sequence = 0
            speed = manifest["speed"]
            chunk = manifest.get("chunk_size", 1024)
            if speed != "fastest":
                chunk = min(chunk, max(1, int(float(speed) * 0.1 * (manifest.get("fs") or 1))))
            batches = iter(frames(run, manifest, chunk))
            stream_start = time.monotonic()
            pause_at_start = paused
            while True:
                gate()
                kind, batch, count, last = next(batches, ("END", [], 0, 0))
                if batch and speed != "fastest":
                    target = (last - manifest["start_time"]) / float(speed)
                    while time.monotonic() - stream_start - (paused - pause_at_start) < target:
                        gate()
                        emit()
                        time.sleep(0.01)
                frame = f"ML1 {run_id} {sequence} {kind} {len(batch)}\n"
                frame += "".join(" ".join(row) + "\n" for row in batch)
                atomic(run / "stream/frame.ready", frame)
                state["transmitted"] += count
                state["queue"] = count
                if kind == "END":
                    state["state"] = "draining"
                deadline = time.monotonic() + 180
                while True:
                    if requested_action == "stop":
                        raise Cancelled()
                    error = read(run / "stream/error.json")
                    if error:
                        raise RuntimeError(error["error"])
                    ack = read(run / "stream/ack.json", {})
                    if kind == "END" and manifest["agent"] == "fault":
                        progress = read(run / "stream/analysis-progress.json", {})
                        if progress.get("processed", 0) > state["processed"]:
                            deadline = time.monotonic() + 180
                            state["last_activity"] = time.time()
                        state["processed"] = progress.get("processed", state["processed"])
                    if ack.get("sequence") == sequence:
                        if ack.get("run_id") != run_id:
                            raise RuntimeError("Acknowledgement belongs to another run")
                        state.update(
                            {key: value for key, value in ack.items() if key not in {"version", "run_id"}}
                        )
                        state["application_pending"] = ack.get("pending", 0)
                        state["application_rejected"] = ack.get("rejected", 0)
                        state["last_activity"] = time.time()
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("ARM acknowledgement timed out")
                    emit()
                    time.sleep(0.002)
                emit()
                sequence += 1
                if kind == "END":
                    if not ack.get("complete") or ack["processed"] != manifest["total_samples"]:
                        raise RuntimeError("ARM completion did not process every input sample")
                    break
            finish_worker()
            di.stop()
            emit(True)
            state["state"] = "completed"
    except Cancelled:
        state["state"] = "draining"
        state["cancel_requested"] = True
        if di:
            try:
                di.feature(manifest["agent"], False)
                finish_worker()
            except Exception as error:
                state["drain_diagnostic"] = str(error)
                state["drain_error"] = True
        emit(True)
        state["cancel_requested"] = True
    except Exception as error:
        state["state"] = "failed"
        state["error"] = str(error)
    finally:
        if di:
            try:
                di.stop()
            except Exception as error:
                state["shutdown_error"] = str(error)
                state["state"] = "failed"
        if state.get("cancel_requested") and not state.get("shutdown_error"):
            state["state"] = "failed" if state.get("drain_error") else "cancelled"
        if run:
            for name in ["logs", "tmp/agent"]:
                source = run / name
                if source.exists():
                    shutil.copytree(
                        source, output / ("logs" if name == "logs" else "agent-logs"), dirs_exist_ok=True
                    )
            seed = run / "flash/587530241/587333633/pv/analytics.pvs"
            if seed.exists():
                shutil.copyfile(seed, output / "checkpoint.pvs")
            db = run / "database/flash/muse01.db"
            if db.exists():
                with (
                    sqlite3.connect(f"file:{db}?mode=ro", uri=True) as src,
                    sqlite3.connect(output / "dataserver.sqlite") as dst,
                ):
                    src.backup(dst)
            state["persistent_bytes"] = sum(
                p.stat().st_size for p in (run / "flash").rglob("*") if p.is_file()
            )
        state["finished_utc"] = time.time()
        control_stop.set()
        controller.join(timeout=2)
        collector_stop.set()
        collector.join()
        with snapshot_lock:
            collect(True)
        telemetry.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "supervise", "reconcile", "capabilities"])
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    if args.action == "capabilities":
        print(
            json.dumps(
                {
                    "available": (BASE / "build-manifest.json").exists(),
                    "container": "di-meter-lab",
                    "build": read(BASE / "build-manifest.json", {}),
                    "owner": read(BASE / "owner.json", {}),
                    "metrology_available": True,
                }
            )
        )
        return
    output = artifact(args.run_id)
    if args.action == "supervise":
        supervise(args.run_id)
        return
    if args.action == "reconcile":
        owner = read(output / "owner.json")
        state = read(output / "status.json", {})
        if state.get("state") not in TERMINAL and owner and not live(owner):
            di = manager(args.run_id, read(output / "manifest.json"))
            # Stop orphaned lab processes before permitting a subsequent run.
            try:
                di.stop()
            except Exception as error:
                state["shutdown_error"] = str(error)
            run = BASE / "runs" / args.run_id
            for name in [
                "diagnostics.jsonl",
                "acknowledgements.jsonl",
                "submissions.jsonl",
                "sdk-completions.jsonl",
                "observations.csv",
            ]:
                source = run / "stream" / name
                if source.exists():
                    atomic(output / name, source.read_bytes())
            for name in ["logs", "tmp/agent"]:
                source = run / name
                if source.exists():
                    shutil.copytree(
                        source, output / ("logs" if name == "logs" else "agent-logs"), dirs_exist_ok=True
                    )
            database = run / "database/flash/muse01.db"
            if database.exists():
                with (
                    sqlite3.connect(f"file:{database}?mode=ro", uri=True) as src,
                    sqlite3.connect(output / "dataserver.sqlite") as dst,
                ):
                    src.backup(dst)
            state.update(state="interrupted", error="Run supervisor exited; isolated runtime stopped")
            atomic(output / "status.json", state)
        print(json.dumps({"live": live(owner), "status": state}))
        return
    with (BASE / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read(BASE / "owner.json")
        if live(old):
            raise ValueError("A Meter Lab run already owns the emulator")
        if (output / "owner.json").exists():
            raise ValueError("Run already launched; reconcile its ownership")
        with (output / "supervisor.log").open("a") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "supervise", args.run_id],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        owner = {"pid": child.pid, "start": identity(child.pid), "run_id": args.run_id}
        atomic(BASE / "owner.json", owner)
        atomic(output / "owner.json", owner)
        print(json.dumps(owner))


if __name__ == "__main__":
    main()
