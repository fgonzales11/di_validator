"""Supervise the API and one leased worker in a Cloud Run container."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from . import store
from .cloud import worker_lease
from .worker import terminate_tree


def main():
    if not os.environ.get("DI_DATABASE_URL") or not os.environ.get("DI_GATEWAY_TOKEN"):
        raise RuntimeError("Hosted mode requires a durable database and gateway authentication")
    store.initialize()
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "di_validator.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PORT", "8080"),
        ]
    )
    children.append(api)
    try:
        while not stopping and api.poll() is None:
            with worker_lease() as lease:
                if lease is None:
                    time.sleep(1)
                    continue
                worker = subprocess.Popen([sys.executable, "-m", "di_validator.worker"])
                children.append(worker)
                try:
                    while not stopping and api.poll() is None and worker.poll() is None:
                        # A lost lease stops this worker before a replacement can do more work.
                        lease.execute("SELECT 1")
                        time.sleep(0.5)
                finally:
                    if worker.poll() is None:
                        terminate_tree(worker)
                    children.remove(worker)
                if not stopping and api.poll() is None:
                    raise RuntimeError("Worker stopped unexpectedly")
    finally:
        for child in reversed(children):
            if child.poll() is None:
                terminate_tree(child)


if __name__ == "__main__":
    main()
