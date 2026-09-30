"""Exercise both installed ARM agents against the isolated emulator services."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import di_meter as control
from pv_checkpoint import inspect

STATE = control.ROOT / "usr/share/flash/587530241/587333633/pv/analytics.pvs"


def records():
    result = {}
    with sqlite3.connect(control.DATABASE) as db:
        for table in ["AgentData", "AgentEvents"]:
            result[table] = {str(row[0]): hashlib.sha256(repr(row).encode()).hexdigest()
                for row in db.execute(f"select * from {table} where AgentId in (?,?)",
                                      (0x23020000, 0x23020001))}
    return result


def process_metrics(pid):
    path = Path(f"/proc/{pid}")
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    values = {}
    for line in (path / "status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0])
    values["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 25:
        parser.error("At least 25 seconds are required to include import and export")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"checks": {}, "hardware_profile": "HW4.2",
              "sdk": "3.0.0", "emu_tool": "1.8.0", "archive": "ARM-EMU-1.8.5"}

    def check(name, passed):
        report["checks"][name] = bool(passed)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            raise AssertionError(name)

    control.stop()
    original_rows = records()
    baseline = inspect(STATE) if STATE.exists() else {"totals": dict.fromkeys(
        ["p_count", "q_count", "import_kwh", "export_kwh"], 0)}
    offsets = {}
    for key, (aid, _) in control.AGENTS.items():
        path = control.ROOT / f"tmp/agent/{aid}/{aid}_log"
        offsets[key] = {p.stat().st_ino: p.stat().st_size
                        for p in path.parent.glob(aid + "_log*") if p.is_file()}
    control.start()
    pids = control.agents()
    check("both_arm_processes_running", all(len(values) == 1 for values in pids.values()))
    initial = {key: process_metrics(values[0]) for key, values in pids.items()}
    began = time.monotonic()
    peaks = {key: values["VmHWM"] for key, values in initial.items()}
    while time.monotonic() - began < args.seconds:
        for key, values in pids.items():
            peaks[key] = max(peaks[key], process_metrics(values[0])["VmHWM"])
        time.sleep(.5)
    elapsed = time.monotonic() - began
    report["resources"] = {}
    for key, values in pids.items():
        end = process_metrics(values[0])
        cpu = 100 * (end["cpu_seconds"] - initial[key]["cpu_seconds"]) / elapsed
        report["resources"][key] = {"emulated_process_peak_rss_kb": peaks[key],
            "host_cpu_percent_one_core": cpu, "sample_seconds": elapsed,
            "agent_ram_policy_kb": 2048, "agent_cpu_policy_percent": 2,
            "emulated_rss_over_policy": peaks[key] > 2048,
            "includes_qemu_overhead": True}
    for key in control.AGENTS:
        control.feature(key, False)
    time.sleep(2)
    state = inspect(STATE)
    report["pv_checkpoint"] = state
    differences = {key: state["totals"][key] - baseline["totals"][key]
                   for key in state["totals"]}
    report["measured_delta"] = differences
    check("pv_real_and_reactive_samples_received", differences["p_count"] >= 20 and differences["q_count"] >= 20)
    check("pv_import_and_export_measured", differences["import_kwh"] > 0 and differences["export_kwh"] > 0)
    # Fixture samples are exactly +1000 W and -500 W; missing seconds are not filled.
    seconds = differences["import_kwh"] * 3600 + differences["export_kwh"] * 7200
    check("samplewise_energy_integration", abs(seconds - differences["p_count"]) < 1e-6)
    disabled_bytes = STATE.read_bytes()
    time.sleep(3)
    check("pv_disable_stops_checkpoint_changes", STATE.read_bytes() == disabled_bytes)
    for key, (aid, _) in control.AGENTS.items():
        path = control.ROOT / f"tmp/agent/{aid}/{aid}_log"
        chunks = []
        for part in sorted(path.parent.glob(aid + "_log*"), key=lambda p: p.stat().st_mtime):
            stat = part.stat()
            previous = offsets[key].get(stat.st_ino, 0)
            with part.open() as log:
                log.seek(previous if stat.st_size >= previous else 0)
                chunks.append(log.read())
        text = "".join(chunks)
        (args.output / f"{key}-sdk.log").write_text(text)
        check(key + "_registered", "Registration apiRegisterAgent returned=0" in text)
        check(key + "_configuration", ("PV_CONFIG_APPLIED" if key == "pv" else "FAULT_CONFIG_APPLIED") in text)
        check(key + "_feature_disabled", "AGENT=0" in text)
        if key == "fault":
            check("fault_reports_missing_waveform_adapter", "live waveform adapter is not configured" in text)
    before_stop = records()
    control.stop()
    check("orderly_shutdown", not any(control.agents().values()) and control.container_state() == "STOPPED")
    after_stop = records()
    check("retained_records_after_shutdown", all(
        after_stop[table].get(key) == value for table, rows in before_stop.items() for key, value in rows.items()))
    check("prior_records_preserved", all(
        after_stop[table].get(key) == value for table, rows in original_rows.items() for key, value in rows.items()))
    control.start()
    time.sleep(4)
    check("both_agents_restart", all(len(values) == 1 for values in control.agents().values()))
    check("checkpoint_retained_on_restart", inspect(STATE)["totals"] == state["totals"])
    after_restart = records()
    check("retained_records_after_restart", all(
        after_restart[table].get(key) == value for table, rows in before_stop.items() for key, value in rows.items()))
    report["status"] = control.status()
    check("hw42_container_limits", report["status"]["cgroup"]["memory.max"] == "66560000" and
          report["status"]["cgroup"]["cpu.max"] == "300000 1000000" and
          report["status"]["cgroup"]["pids.max"] == "50")
    check("no_container_oom", "oom_kill 0" in report["status"]["cgroup"]["memory.events"])
    report["limits"] = [
        "Fault ARM build has no waveform acquisition; only its lifecycle/integration is exercised.",
        "PV uses synthetic one-second DataServer input; no physical meter or field accuracy claim.",
        "The emulator rootfs does not establish the SDK3 firmware/AppServ baseline.",
        "QEMU process memory exceeds the 2048 KB agent policy; meter resource qualification remains open."
    ]
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": len(report["checks"]), "passed": True,
                      "measured_delta": differences, "resources": report["resources"]}, indent=2))


if __name__ == "__main__":
    with (control.BASE / "run/control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
