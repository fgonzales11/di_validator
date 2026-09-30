from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import jobs, store
from .jobs import claim as claim


def execute(job_id):
    job = store.job(job_id)
    progress = store.Progress(job_id)
    try:
        progress(0.001, "Starting job")
        if job["kind"] == "import":
            from .ingest import import_dataset

            result = import_dataset(job["config"], progress)
        elif job["kind"] == "comtrade_import":
            from .faults.comtrade_io import import_sources

            result = import_sources(job["config"], progress)
        elif job["kind"] == "wavewin_import":
            from .faults.wavewin_io import import_sources as import_wavewin_sources

            result = import_wavewin_sources(job["config"], progress)
        elif job["kind"] == "registry":
            from .labels import import_registry

            result = import_registry(job["config"]["path"])
        elif job["kind"] == "classification":
            from .classification import run

            result = run(job["config"], progress)
        elif job["kind"] == "events":
            from .events import run

            result = run(job["config"], progress)
        elif job["kind"] == "synthetic":
            from .events import synthetic

            result = synthetic(job["config"], progress)
        elif job["kind"] == "aggregate":
            from .aggregation import run

            result = run(job["config"], progress)
        elif job["kind"] == "forecast":
            from .forecasting.engine import run

            result = run(job["config"], progress)
        elif job["kind"] == "forecast_setup":
            from .forecasting.runtime import setup

            result = setup(job["config"], progress)
        else:
            raise ValueError("Unknown job type")
        jobs.complete(job_id, result, job["kind"])
    except BaseException as error:
        jobs.fail(job_id, error, traceback.format_exc())


def terminate_tree(child):
    """Windows virtualenv launchers have a child interpreter: terminate the whole owned tree."""
    import psutil

    try:
        processes = psutil.Process(child.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        processes = []
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=2)
    for process in alive:
        process.kill()
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=5)


def serve():
    import psutil

    store.initialize()
    # Hosted ownership is enforced by the supervisor's PostgreSQL advisory lock.
    # A Cloud Storage mount cannot provide SQLite/file locking semantics.
    lock = (store.workspace() if not os.environ.get("DI_DATABASE_URL") else Path("/tmp")) / "worker.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        try:
            owner = json.loads(lock.read_text())
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner["created"]) < 1:
                raise RuntimeError("A worker already owns this workspace")
        except (psutil.NoSuchProcess, ValueError, KeyError, json.JSONDecodeError):
            pass
        lock.unlink()
        handle = lock.open("x")
    handle.write(json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()}))
    handle.close()
    jobs.recover_interrupted()
    child = None
    try:
        while True:
            job_id = claim()
            if not job_id:
                time.sleep(0.4)
                continue
            child = subprocess.Popen(
                [sys.executable, "-m", "di_validator.worker", "--job", job_id],
                cwd=store.ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            while child.poll() is None:
                current = store.job(job_id)
                if current["cancel_requested"]:
                    terminate_tree(child)
                    jobs.finish_cancelled(job_id)
                    break
                time.sleep(0.3)
            jobs.fail_abandoned(job_id)
            child = None
    finally:
        if child and child.poll() is None:
            terminate_tree(child)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job")
    args = parser.parse_args()
    execute(args.job) if args.job else serve()
