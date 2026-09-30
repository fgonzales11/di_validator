"""SDK/DataServer acceptance using typed fixtures, real callbacks/config and retained rows."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from decode import decode
from host_session import Session, metadata
from verify_parity import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lines = (args.fixtures / "pv_export.pvr").read_text().splitlines()
    split = lines.index("DATA") + 2
    header, runs = lines[:split], lines[split:]
    checks, evidence = [], []

    def fixture(name, rows):
        path = args.output / (name + ".pvr")
        path.write_text("\n".join(header + rows) + "\n")
        return path

    def replay(session, path):
        identifier = session.submit(path)
        result = session.result(identifier, timeout=90)
        assert result["status"] == "complete", result
        assert result["pending"] == result["rejected"] == 0, result
        rows = [
            json.loads(x)
            for x in (session.spool / (identifier + ".intervals.jsonl")).read_text().splitlines()
        ]
        return result, rows

    seed_wave = fixture("seed31", runs[: 31 * 96])
    seed = args.output / "seed31.pvs"
    subprocess.run(
        [str(args.binary), str(seed_wave), "--state-out", str(seed)], check=True, stdout=subprocess.DEVNULL
    )
    day32 = fixture("day32", runs[31 * 96 : 32 * 96])
    hour33 = fixture("hour33", runs[32 * 96 : 32 * 96 + 4])
    session = Session(args.output / "lifecycle")
    state_dir = session.output / "state"
    state_dir.mkdir(exist_ok=True)
    shutil.copy2(seed, state_dir / "analytics.pvs")
    try:
        session.start(metadata(day32))
        result, intervals = replay(session, day32)
        assert result["detection"]["state"] == "likely_pv"
        assert all(not r["available"] for r in intervals)
        rows = session.rows()
        assert len(rows["events"]) == 1 and rows["events"][0][4] == 1, rows["events"]
        event = decode(rows["events"][0][3])
        assert event["state"] == "likely_pv" and event["utc"] == intervals[-1]["start"] + 900
        assert len(rows["data"]) == 25
        checks += [
            "30_day_learning_and_three_evaluation_debounce",
            "likely_pv_transition_retained_alarm",
            "24_hourly_batches_and_daily_summary",
        ]
        result, subsequent = replay(session, hour33)
        assert any(r["available"] and r["generation_kw"] > 0 for r in subsequent)
        baseline = json.loads((args.fixtures / "pv_export.json").read_text())["intervals"][
            32 * 96 : 32 * 96 + 4
        ]
        compare(baseline, subsequent)
        assert len(session.rows()["events"]) == 1
        checks += ["sdk_intervals_match_python", "generation_without_recurring_alarm"]
        typed = json.loads((state_dir / "metrology-checks.json").read_text())
        policy = json.loads((state_dir / "policy-check.json").read_text())
        assert typed["status"] == "PASS" and policy["oversize_status"] != 0
        policy["host_enforces_policy"] = policy["unregistered_feature_status"] != 0
        checks += [
            "singlephase_polyphase_aggregate_lids",
            "unchanged_missing_invalid_and_gap_callbacks",
            "sdk_oversize_rejection_and_feature_policy_probe",
        ]
        before = session.rows()
        rejected = session.update({"power_scale": "nan"})
        assert not rejected["valid"] and rejected["enabled"]
        session.update({"power_scale": "1"}, enabled=False)
        pending = session.submit(hour33)
        time.sleep(0.25)
        assert not (session.spool / (pending + ".done.json")).exists()
        assert session.rows() == before
        session.update(enabled=True)
        assert session.result(pending)["status"] == "complete"
        assert session.rows() == before
        checks += [
            "invalid_xml_values_preserve_config",
            "feature_disable_and_reenable",
            "duplicate_interval_suppression",
        ]
        # A polarity change resets the learning state at the next boundary.
        assert session.update({"polarity": "-1"})["valid"]
        changed_wave = fixture("changed", runs[32 * 96 + 4 : 32 * 96 + 8])
        changed, changed_rows = replay(session, changed_wave)
        assert changed["detection"]["state"] == "learning" and changed["detection"]["valid_days"] == 0
        assert all(not r["available"] for r in changed_rows[1:])
        assert abs(changed_rows[0]["net_kw"] - float(runs[32 * 96 + 4].split(",")[2]) / 1000) < 1e-6
        assert abs(changed_rows[-1]["net_kw"] + float(runs[32 * 96 + 7].split(",")[2]) / 1000) < 1e-6
        reset_events = session.rows()["events"]
        assert len(reset_events) == 2 and reset_events[-1][4] == 0
        assert decode(reset_events[-1][3])["state"] == "learning"
        checks.append("configuration_frozen_until_boundary_and_identity_warmup_reset")
        checks.append("learning_reset_retained_event_without_alarm")
        # Command lifecycle stops and recreates the worker without deregistration.
        session.command(False)
        session.command(True)
        checks.append("feature_command_stop_restart")
        rows = session.rows()
        decoded = [decode(r[2]) for r in rows["data"]]
        assert all(len(r[2]) <= 1024 for r in rows["data"])
        assert sum(len(r[2]) for r in rows["data"]) <= 8192
        assert all(len(r[3]) <= 256 for r in rows["events"])
        wire = [r for batch in decoded if batch["kind"] == "hourly" for r in batch["records"]]
        assert any(r["export_kwh"] > 0 and r["generation_kw"] is None for r in wire)
        checks += ["wire_decoder_and_payload_daily_limits", "measured_export_during_learning"]
        session.stop()
        assert session.rows() == rows
        evidence.append(
            dict(
                name="lifecycle",
                rows=rows,
                decoded=decoded,
                typed=typed,
                policy=policy,
                peak_rss_kb=session.peak_rss,
                cpu_seconds=session.cpu_seconds,
                state_bytes=sum(p.stat().st_size for p in state_dir.glob("*.pvs")),
            )
        )
        checks.append("orderly_shutdown_retains_records")
    finally:
        session.close()
    # Restart reads the same state and leaves previously stored outcomes untouched.
    restart = Session(args.output / "lifecycle")
    try:
        restart.start({**metadata(hour33), "polarity": "-1"})
        restart.wait(lambda: (restart.spool / "lifecycle.json").exists())
        restart.stop()
        assert restart.rows() == {"events": [], "data": []}
        assert session.rows() == rows
        checks.append("process_restart_retains_records_and_checkpoint")
    finally:
        restart.close()
    # A running long replay must respond to disable before finishing.
    cancel = Session(args.output / "cancellation")
    try:
        cancel.start(metadata(seed_wave))
        request = cancel.submit(seed_wave)
        cancel.wait(lambda: (cancel.spool / (request + ".running")).exists())
        began = time.monotonic()
        assert not cancel.update(enabled=False)["enabled"]
        cancelled = cancel.result(request)
        assert cancelled["status"] == "error" and "cancel" in cancelled["reason"]
        elapsed = time.monotonic() - began
        assert elapsed < 5, elapsed
        checks.append("responsive_worker_cancellation")
        evidence.append(dict(name="cancellation", response_seconds=elapsed, peak_rss_kb=cancel.peak_rss))
    finally:
        cancel.close()
    report = dict(
        status="PASS",
        checks=checks,
        evidence=evidence,
        physical_meter_tested=False,
        remote_hes_config_tested=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SDK integration PASS:", len(checks), "checks")


if __name__ == "__main__":
    main()
