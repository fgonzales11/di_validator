"""Reproducible local ARM integration matrix; run from the repository root.

python -m meter_emulator.verify_lab [--full] [--case LABEL]
Requires the installed lab runtime and dedicated local worker. Real SDK policy
rejections remain visible; this tool never edits quotas or expected outputs.
"""

import argparse
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765/api/v1/meter-lab"
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def api(route, body=None):
    request = Request(
        BASE + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--extended", action="store_true", help="Additional waveform loops and acquisition edge cases"
    )
    parser.add_argument("--case")
    parser.add_argument("--resume-run", help="Collect the first selected case from an existing saved run")
    parser.add_argument("--start-at", help="Continue the matrix from a named case")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Wall-clock deadline per case, including slow local evidence writes",
    )
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    cases = [
        ("fault_regression", "fault", "1A_val1", {}, {"chunk_size": 17, "speed": "10"}),
        ("fault_99", "fault", "comtrade_99", {}, {}),
        ("fault_gap", "fault", "gap", {}, {"chunk_size": 31}),
        ("fault_manual", "fault", "manual", {}, {}),
        ("pv_sign_sp", "pv", "sign_changes", {}, {"meter_form": "GENX_SP", "chunk_size": 17}),
        ("pv_sign_pp", "pv", "sign_changes", {}, {"meter_form": "GENX_PP", "chunk_size": 1024}),
        ("pv_warm", "pv", "warm_tail", {}, {"seed_fixture": "warm_state"}),
        ("pv_withheld", "pv", "missing_reactive", {"duration_seconds": 86400}, {}),
        ("pv_metrology", "pv", "sign_changes", {"duration_seconds": 30}, {"mode": "metrology", "speed": "1"}),
    ]
    if args.full:
        cases.extend(
            [
                ("pv_cold", "pv", "pv_export", {}, {}),
                ("pv_adverse", "pv", "midday_load_drop", {}, {}),
            ]
        )
    if args.extended:
        cases.extend(
            ("fault_loop_" + loop, "fault", "physics_" + loop, {}, {})
            for loop in ["AG", "BG", "CG", "AB", "BC", "CA", "POS"]
        )
        cases.extend(
            [
                ("fault_60hz", "fault", "synthetic_fault", {"frequency": 60}, {}),
                ("fault_low_current", "fault", "low_current", {}, {}),
                ("fault_missing", "fault", "missing", {}, {}),
                ("fault_no_fault", "fault", "no_fault_2500", {}, {}),
            ]
        )
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            parser.error("Unknown case")
    if args.start_at:
        labels = [case[0] for case in cases]
        if args.start_at not in labels:
            parser.error("Unknown starting case")
        cases = cases[labels.index(args.start_at) :]
    reports = Path(__file__).resolve().parent.parent / "runtime/meter-lab/verification"
    reports.mkdir(exist_ok=True)
    for index, (label, agent, preset, parameters, options) in enumerate(cases):
        print("Starting " + label, flush=True)
        if index == 0 and args.resume_run:
            run = api("/runs/" + args.resume_run)
            if run["agent"] != agent or run["name"] != "Verification: " + label:
                raise ValueError("Saved run does not match the selected matrix case")
            scenario = {"id": run["manifest"]["scenario_id"]}
        else:
            scenario = api(
                "/scenarios",
                dict(agent=agent, source_id=preset, name="Verification: " + label, parameters=parameters),
            )
            run = api("/runs", dict(scenario_id=scenario["id"], **options))
        print(json.dumps({"case": label, "run_id": run["id"]}), flush=True)
        deadline = time.monotonic() + args.timeout_seconds
        while run["state"] not in TERMINAL or run["checks"]["verdict"] == "pending":
            if time.monotonic() > deadline:
                if run["state"] not in TERMINAL:
                    api("/runs/" + run["id"] + "/control", {"action": "stop"})
                raise TimeoutError(label + ": inspect run " + run["id"])
            time.sleep(1)
            run = api("/runs/" + run["id"])
        report = run["checks"]
        (reports / (label + ".json")).write_text(
            json.dumps(
                {
                    "case": label,
                    "run_id": run["id"],
                    "scenario_id": scenario["id"],
                    "state": run["state"],
                    "telemetry": run["telemetry"],
                    "checks": report,
                },
                indent=2,
            )
        )
        with urlopen(BASE + "/runs/" + run["id"] + "/export", timeout=180) as response:
            with (reports / (label + ".zip")).open("wb") as archive:
                while chunk := response.read(1024 * 1024):
                    archive.write(chunk)
        with zipfile.ZipFile(reports / (label + ".zip")) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Evidence archive failed its CRC check")
        print(
            json.dumps(
                {"case": label, "run": run["id"], "state": run["state"], "verdict": report["verdict"]}
            ),
            flush=True,
        )
        if run["state"] != "completed":
            raise RuntimeError(label + ": " + str(run["telemetry"].get("error")))


if __name__ == "__main__":
    main()
