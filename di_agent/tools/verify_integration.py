"""Exercise replay, hot XML configuration, feature state, and retained DI outcomes."""

import argparse
import json
import math
from pathlib import Path
import time
from host_session import Session, metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    wave = args.fixtures / "1A_val1.wave"
    tests = []
    reports = []
    session = Session(args.output / "lifecycle")
    try:
        session.start(metadata(wave))

        def replay(path=wave):
            result = session.result(session.submit(path))
            assert result["status"] == "complete", result
            return result

        baseline = replay()
        distance = baseline["segments"][0]["distance"]["estimated_distance_km"]
        assert math.isclose(distance, 0.8908903568462703, abs_tol=1e-6)
        tests.append("baseline_python_parity")
        applied = session.update({"x1_ohm_km": ".8"})
        assert applied["valid"] and applied["enabled"], applied
        changed = replay()
        assert math.isclose(
            changed["segments"][0]["distance"]["estimated_distance_km"], distance / 2, abs_tol=1e-6
        )
        tests.append("hot_xml_configuration")
        applied = session.update(enabled=False)
        assert applied["valid"] and not applied["enabled"], applied
        before = session.rows()
        pending = session.submit(wave)
        time.sleep(0.3)
        assert not (session.spool / (pending + ".done.json")).exists()
        assert session.rows() == before
        session.update(enabled=True)
        assert session.result(pending)["status"] == "complete"
        tests.append("disable_queue_and_reenable")
        rejected = session.update({"x1_ohm_km": "0"})
        assert not rejected["valid"], rejected
        unchanged = replay()
        assert math.isclose(
            unchanged["segments"][0]["distance"]["estimated_distance_km"], distance / 2, abs_tol=1e-6
        )
        tests.append("invalid_configuration_preserves_previous")
        session.update({"x1_ohm_km": ".4", "manual_onset": ".0995"})
        before = session.rows()
        manual = replay()
        after = session.rows()
        assert manual["segments"][0]["inception_source"] == "manual"
        assert len(after["events"]) == len(before["events"]) and len(after["data"]) == len(before["data"]) + 1
        tests.append("manual_summary_without_alarm")
        session.update({"manual_onset": "-1", "settle_cycles": "20"})
        before = session.rows()
        unavailable = replay()
        after = session.rows()
        assert unavailable["segments"][0]["distance"]["status"] == "unavailable"
        assert (
            len(after["events"]) == len(before["events"]) + 1
            and len(after["data"]) == len(before["data"]) + 1
        )
        tests.append("unavailable_distance_still_alarms")
        session.update({"settle_cycles": "1"})
        before = session.rows()
        no_fault = replay(args.fixtures / "no_fault_2000.wave")
        assert no_fault["segments"][0]["status"] == "no_fault"
        assert session.rows() == before
        tests.append("no_fault_no_output")
        session.command(enabled=False)
        before = session.rows()
        pending = session.submit(wave)
        time.sleep(0.2)
        assert not (session.spool / (pending + ".done.json")).exists()
        assert session.rows() == before
        session.command(enabled=True)
        assert session.result(pending)["status"] == "complete"
        tests.append("sdk_feature_command_stop_restart")
        # A long recording gives the configuration thread work to interrupt.
        lines = wave.read_text().splitlines()
        header = lines.index("DATA") + 2
        source_rows = [line.split(",") for line in lines[header:]]
        long_wave = args.output / "cancellation.wave"
        with long_wave.open("w") as output:
            output.write("\n".join(lines[:header]) + "\n")
            for i in range(100000):
                output.write(",".join([format(i / 2000, ".17g"), *source_rows[i % 400][1:]]) + "\n")
        before = session.rows()
        pending = session.submit(long_wave)
        session.wait(lambda: (session.spool / (pending + ".running")).exists())
        session.update(enabled=False)
        cancelled = session.result(pending)
        assert cancelled["status"] == "error" and cancelled["reason"] == "cancelled", cancelled
        assert session.rows() == before
        session.update(enabled=True)
        tests.append("disable_cancels_active_request_without_output")
        rows = session.rows()
        summaries = [json.loads(row[2]) for row in rows["data"] if row[2].startswith("{")]
        for event in rows["events"]:
            assert len(event[3].encode()) <= 256 and event[4] == 1
            correlation = event[3].split("#")[3]
            matching = [s for s in summaries if s["id"] == correlation]
            assert len(matching) == 1 and matching[0]["inception"] == "detected"
        assert all(len(row[2].encode()) <= 1024 for row in rows["data"])
        assert len({s["id"] for s in summaries}) == len(summaries)
        assert len({s["config"] for s in summaries}) >= 4
        tests.append("correlation_uniqueness_payload_limits")
        session.stop()
        assert session.rows() == rows
        tests.append("retained_after_orderly_shutdown")
        saved_rows = rows
        reports.append({"peak_rss_kb": session.peak_rss, "cpu_seconds": session.cpu_seconds, "rows": rows})
    finally:
        session.close()
    restarted = Session(args.output / "restart")
    try:
        restarted.start(metadata(wave))
        import sqlite3
        from host_session import DATABASE, AGENT, FEATURE

        with sqlite3.connect(DATABASE) as c:
            for row in saved_rows["events"]:
                assert c.execute(
                    "select Data from AgentEvents where Id=? and AgentId=? and FeatureId=?",
                    (row[0], AGENT, FEATURE),
                ).fetchone()
            for row in saved_rows["data"]:
                assert c.execute(
                    "select Data from AgentData where Id=? and AgentId=? and FeatureId=?",
                    (row[0], AGENT, FEATURE),
                ).fetchone()
        assert restarted.result(restarted.submit(wave))["status"] == "complete"
        tests.append("retained_after_dataserver_and_agent_restart")
        restarted.stop()
        reports.append(
            {
                "peak_rss_kb": restarted.peak_rss,
                "cpu_seconds": restarted.cpu_seconds,
                "rows": restarted.rows(),
            }
        )
    finally:
        restarted.close()
    report = {
        "status": "PASS",
        "tests": tests,
        "runs": reports,
        "policy_ram_kb": 2048,
        "live_meter_qualified": False,
        "configuration_transport": "host-only ingress through SDK CConfigParser and its registered callbacks; remote HES delivery not tested",
    }
    (args.output / "integration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(tests)} DI integration scenarios; report: {args.output}/integration-report.json")


if __name__ == "__main__":
    main()
