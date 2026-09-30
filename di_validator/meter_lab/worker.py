"""Dedicated local lane; WSL supervisor survives this process and browser restarts."""

import json
import os
import subprocess
import sys
import time
import threading
from .. import store
from . import scenarios, service

HELPER = "/root/di-sdk-setup/meter-lab/tools/lab_runtime.py"


def wsl(action, identifier=None):
    command = ["wsl.exe", "-d", "Ubuntu-22.04", "-u", "root", "--", "python3", HELPER, action]
    if identifier:
        command.append(identifier)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=50,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def main():
    service.local()
    store.initialize()
    path = scenarios.root()
    if "--stop" in sys.argv:
        for run in service.listing():
            if run["state"] not in service.TERMINAL:
                service.control(run["id"], "stop")
        deadline = time.monotonic() + 55
        while any(r["state"] not in service.TERMINAL for r in service.listing()):
            if time.monotonic() > deadline:
                raise RuntimeError("Meter Lab is still draining; inspect the saved run logs")
            time.sleep(0.2)
        return
    # Windows file lock prevents duplicate launcher invocations taking this lane.
    import msvcrt

    with (path / "worker.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"1")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return
        checked = set()

        def heartbeat():
            while True:
                try:
                    service.atomic(path / "worker.json", {"pid": os.getpid(), "updated_utc": time.time()})
                except OSError:
                    # A Windows reader can deny replacement temporarily. It must
                    # not terminate liveness reporting for an otherwise healthy lane.
                    pass
                time.sleep(1)

        threading.Thread(target=heartbeat, daemon=True).start()
        last_cap = 0
        last_reconcile = 0
        while True:
            if time.time() - last_cap > 30:
                try:
                    cap = wsl("capabilities")
                    cap.pop("build", None)
                    service.atomic(path / "capabilities.json", cap)
                except Exception as error:
                    service.atomic(path / "capabilities.json", {"available": False, "reason": str(error)})
                last_cap = time.time()
            for run in service.listing():
                directory = scenarios.folder("runs", run["id"])
                if run["state"] in service.TERMINAL:
                    if run["id"] not in checked and not (directory / "checks.json").exists():
                        try:
                            service.evaluate(run["id"])
                        except Exception as error:
                            service.atomic(
                                directory / "checks.json",
                                {
                                    "verdict": "fail",
                                    "checks": [
                                        {
                                            "category": "structural",
                                            "name": "Evidence evaluation",
                                            "verdict": "fail",
                                            "detail": str(error),
                                        }
                                    ],
                                },
                            )
                    checked.add(run["id"])
                    continue
                try:
                    if not (directory / "owner.json").exists():
                        if service.read(directory / "control.json", {}).get("action") == "stop":
                            service.atomic(
                                directory / "status.json", {"state": "cancelled", "run_id": run["id"]}
                            )
                            continue
                        wsl("launch", run["id"])
                    elif time.time() - last_reconcile > 10:
                        reconciliation = wsl("reconcile", run["id"])
                        service.atomic(
                            directory / "worker-reconciliation.json",
                            {"worker_pid": os.getpid(), "utc": time.time(), **reconciliation},
                        )
                        last_reconcile = time.time()
                except Exception as error:
                    # A failed Windows/WSL invocation is not evidence that the
                    # detached run stopped. Preserve its status and retry the
                    # identity reconciliation; the helper alone declares loss.
                    service.atomic(
                        directory / "worker-diagnostic.json",
                        {
                            "worker_pid": os.getpid(),
                            "utc": time.time(),
                            "error": str(error),
                            "action": "retry reconciliation"
                            if (directory / "owner.json").exists()
                            else "retry launch",
                        },
                    )
            time.sleep(0.5)


if __name__ == "__main__":
    main()
