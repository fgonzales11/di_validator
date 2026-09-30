"""Verify paused run ownership across a Windows lab worker restart."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .verify_lab import api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Run this verification from the Windows workspace")
    root = Path(__file__).resolve().parent.parent
    endpoint = "/runs/" + args.run_id
    run = api(endpoint)
    if run["state"] != "running":
        raise RuntimeError("Select an active scenario replay")
    api(endpoint + "/control", {"action": "pause"})
    deadline = time.monotonic() + 240
    while (run := api(endpoint))["state"] != "paused":
        if time.monotonic() > deadline:
            raise TimeoutError("Pause was not acknowledged")
        time.sleep(1)
    before = run["telemetry"]
    folder = root / "runtime/meter-lab/runs" / args.run_id
    previous = json.loads((folder / "worker-reconciliation.json").read_text())
    try:
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(root / "scripts/restart-meter-lab-worker.ps1"),
            ],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.monotonic() + 300
        while True:
            observed = json.loads((folder / "worker-reconciliation.json").read_text())
            if observed["worker_pid"] != previous["worker_pid"] and observed.get("live"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Replacement worker did not reconcile the surviving owner")
            time.sleep(1)
        after = api(endpoint)["telemetry"]
        for key in ["owner", "processed", "scenario_time"]:
            if after[key] != before[key]:
                raise AssertionError("Paused " + key + " changed during worker restart")
        if after["state"] != "paused":
            raise AssertionError("Run did not remain paused")
    finally:
        api(endpoint + "/control", {"action": "resume"})
    deadline = time.monotonic() + 180
    while (resumed := api(endpoint))["telemetry"]["processed"] <= before["processed"]:
        if time.monotonic() > deadline:
            raise TimeoutError("Replay did not resume processing")
        time.sleep(1)
    evidence = {
        "run_id": args.run_id,
        "verdict": "pass",
        "before": before,
        "after": after,
        "reconciliation": observed,
        "resumed": resumed["telemetry"],
    }
    (root / "runtime/meter-lab/verification/worker-restart.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"run_id": args.run_id, "verdict": "pass", "owner": after["owner"]}))


if __name__ == "__main__":
    main()
