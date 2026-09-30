"""Queue lifecycle contracts shared by the API and worker."""

import pytest

from di_validator import events, jobs, store
from di_validator.worker import execute


def test_claim_skips_cancelled_work_and_rerun_preserves_snapshot():
    config = {"annotation_snapshot": [{"id": "frozen-evidence"}], "parameters": {"threshold": 2}}
    cancelled = store.enqueue("events", config)
    assert jobs.cancel(cancelled["id"])["status"] == "cancelled"
    retry = jobs.rerun(cancelled["id"])
    assert retry["id"] != cancelled["id"]
    assert retry["config"] == config
    assert jobs.claim() == retry["id"]
    assert jobs.claim() is None
    assert store.job(cancelled["id"])["status"] == "cancelled"


def test_running_cancellation_is_cooperative_and_logged():
    queued = store.enqueue("synthetic", {})
    assert jobs.claim() == queued["id"]
    requested = jobs.cancel(queued["id"])
    assert requested["status"] == "running"
    assert requested["cancel_requested"] == 1
    execute(queued["id"])
    result = jobs.detail(queued["id"])
    assert result["status"] == "cancelled"
    assert result["result"] is None
    assert "Cancelled by user" in result["error"]
    assert "Cancelled by user" in result["logs"][-1]["message"]
    assert not store.listing("dataset")


def test_worker_completion_survives_supervisor_cleanup(monkeypatch):
    def generate(config, progress):
        progress(0.5, "Halfway")
        return {"id": "generated-dataset", "name": "Synthetic fixture"}

    monkeypatch.setattr(events, "synthetic", generate)
    queued = store.enqueue("synthetic", {})
    assert jobs.claim() == queued["id"]
    execute(queued["id"])
    jobs.finish_cancelled(queued["id"])
    jobs.fail_abandoned(queued["id"])
    result = jobs.detail(queued["id"])
    assert result["status"] == "completed"
    assert result["progress"] == 1
    assert result["result"] == {"id": "generated-dataset", "name": "Synthetic fixture"}
    assert any(log["message"] == "Halfway" for log in result["logs"])


def test_unknown_job_failure_retains_diagnostic_log():
    queued = store.enqueue("unsupported", {})
    assert jobs.claim() == queued["id"]
    execute(queued["id"])
    result = jobs.detail(queued["id"])
    assert result["status"] == "failed"
    assert result["message"] == "Unknown job type"
    assert "ValueError: Unknown job type" in result["error"]
    assert result["logs"][-1]["message"] == result["error"]


def test_recovery_only_changes_running_jobs():
    interrupted = store.enqueue("synthetic", {"seed": 17})
    assert jobs.claim() == interrupted["id"]
    queued = store.enqueue("synthetic", {})
    cancelled = store.enqueue("synthetic", {})
    jobs.cancel(cancelled["id"])
    jobs.recover_interrupted()
    assert store.job(interrupted["id"])["status"] == "interrupted"
    assert store.job(interrupted["id"])["config"] == {"seed": 17}
    assert store.job(queued["id"])["status"] == "queued"
    assert store.job(cancelled["id"])["status"] == "cancelled"
    assert jobs.claim() == queued["id"]
    jobs.fail_abandoned(queued["id"])
    assert store.job(queued["id"])["status"] == "failed"


@pytest.mark.parametrize("operation", [jobs.detail, jobs.cancel, jobs.rerun])
def test_missing_jobs_are_rejected(operation):
    with pytest.raises(ValueError, match="Job not found"):
        operation("missing")
