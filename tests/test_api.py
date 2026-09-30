from fastapi.testclient import TestClient

from di_validator.api import app
from di_validator.worker import claim, execute


def test_job_lifecycle_failure_cancel_and_rerun():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        queued = client.post("/api/v1/datasets", json={"path": "missing.csv"}).json()
        assert queued["status"] == "queued"
        assert claim() == queued["id"]
        execute(queued["id"])
        assert client.get("/api/v1/jobs/" + queued["id"]).json()["status"] == "failed"
        rerun = client.post("/api/v1/jobs/" + queued["id"] + "/rerun").json()
        assert rerun["config"] == queued["config"] and rerun["id"] != queued["id"]
        cancelled = client.post("/api/v1/jobs/" + rerun["id"] + "/cancel").json()
        assert cancelled["status"] == "cancelled" and claim() is None
        assert client.get("/api/v1/jobs/" + queued["id"]).json()["logs"]
        assert client.post("/api/v1/bootstrap", headers={"Origin": "https://example.com"}).status_code == 403


def test_annotation_partition_and_export():
    from di_validator.events import synthetic, run

    dataset = synthetic(dict(duration_seconds=10))
    with TestClient(app) as client:
        bad = client.post(
            "/api/v1/annotations",
            json=dict(
                dataset_id=dataset["id"],
                asset_id=dataset["asset_id"],
                kind="review",
                start=1,
                end=2,
                partition="development",
            ),
        )
        assert bad.status_code == 422
        result = run(dict(dataset_id=dataset["id"]))
        export = client.get(f"/api/v1/experiments/{result['id']}/export")
        assert export.status_code == 200 and export.content[:2] == b"PK"


def test_cancellation_stops_windows_interpreter_tree():
    import subprocess
    import sys
    import time
    import psutil
    from di_validator.worker import terminate_tree

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.3)
    descendants = psutil.Process(child.pid).children(recursive=True)
    terminate_tree(child)
    assert child.poll() is not None
    assert all(not process.is_running() for process in descendants)


def test_annotation_withdrawal_retains_frozen_evidence():
    from di_validator import store
    from di_validator.events import synthetic

    dataset = synthetic(dict(duration_seconds=10))
    with TestClient(app) as client:
        original = client.get("/api/v1/annotations", params={"dataset_id": dataset["id"]}).json()
        queued = client.post("/api/v1/events/run", json={"dataset_id": dataset["id"]}).json()
        client.post("/api/v1/annotations/" + original[0]["id"] + "/withdraw")
        current = client.get("/api/v1/annotations", params={"dataset_id": dataset["id"]}).json()
        assert len(current) == len(original) - 1
        assert len(store.job(queued["id"])["config"]["annotation_snapshot"]) == len(original)


def test_worker_restart_recovers_interrupted_job():
    import os
    import subprocess
    import sys
    import time
    from di_validator import store
    from di_validator.worker import terminate_tree

    queued = store.enqueue("synthetic", {"duration_seconds": 10})
    assert claim() == queued["id"]
    child = subprocess.Popen(
        [sys.executable, "-m", "di_validator.worker"],
        cwd=store.ROOT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 15
        while store.job(queued["id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
        recovered = store.job(queued["id"])
        assert recovered["status"] == "interrupted"
        assert recovered["config"] == queued["config"]
    finally:
        terminate_tree(child)
