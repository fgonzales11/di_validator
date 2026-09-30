import json
import os

import pytest
from fastapi.testclient import TestClient

from di_validator import store
from di_validator.api import app
from di_validator.cloud import worker_lease
from di_validator.worker import claim
from scripts.cloud_snapshot import replace_paths


def test_hosted_gateway_authentication_and_origin(monkeypatch, tmp_path):
    monkeypatch.delenv("DI_DATABASE_URL", raising=False)
    monkeypatch.setenv("DI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DI_HOSTED", "1")
    monkeypatch.setenv("DI_GATEWAY_TOKEN", "test-only-token")
    monkeypatch.setenv("DI_ALLOWED_ORIGINS", "di-validator.n2wm787gnp.chatgpt.site")
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["mode"] == "hosted"
        for path in ["/api/v1/datasets", "/api/v1/overview", "/notebooks/lab/index.html", "/docs"]:
            assert client.get(path).status_code == 401
        headers = {
            "x-di-gateway-token": "test-only-token",
            "origin": "https://di-validator.n2wm787gnp.chatgpt.site",
        }
        assert client.get("/api/v1/datasets", headers=headers).status_code == 200
        assert (
            client.get(
                "/api/v1/datasets", headers={**headers, "origin": "https://untrusted.example"}
            ).status_code
            == 403
        )
        monkeypatch.delenv("DI_GATEWAY_TOKEN")
        assert client.get("/api/v1/datasets", headers=headers).status_code == 401


def test_migration_only_rewrites_workspace_paths(tmp_path):
    original = {
        "source": str(tmp_path / "data" / "one.csv"),
        "folder": str(tmp_path / "runtime" / "datasets" / "id"),
        "checksum": "unchanged",
        "nested": [str(tmp_path / "runtime" / "experiments" / "id")],
    }
    result = replace_paths(original, tmp_path)
    assert result["source"] == "/data/sources/one.csv"
    assert result["folder"] == "/data/runtime/datasets/id"
    assert result["nested"] == ["/data/runtime/experiments/id"]
    assert result["checksum"] == "unchanged" and original["folder"] != result["folder"]


@pytest.mark.skipif(
    not os.environ.get("DI_TEST_DATABASE_URL"), reason="Requires isolated PostgreSQL test database"
)
def test_postgres_persistence_job_claim_rollback_and_exclusive_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("DI_DATABASE_URL", os.environ["DI_TEST_DATABASE_URL"])
    monkeypatch.setenv("DI_WORKSPACE", str(tmp_path))
    store.initialize()
    with store.connection() as db:
        db.execute("DELETE FROM logs")
        db.execute("DELETE FROM jobs")
        db.execute("DELETE FROM objects")
    item = store.put("dataset", {"name": "Cloud persistence"})
    assert store.get("dataset", item["id"])["name"] == "Cloud persistence"
    store.put("dataset", {**item, "name": "Updated"})
    assert len(store.listing("dataset")) == 1
    with pytest.raises(RuntimeError):
        with store.connection() as db:
            db.execute("DELETE FROM objects")
            raise RuntimeError("Rollback")
    assert store.get("dataset", item["id"])["name"] == "Updated"
    job = store.enqueue("synthetic", {"seed": 42})
    assert claim() == job["id"] and claim() is None
    store.Progress(job["id"])(0.5, "Halfway")
    assert store.job(job["id"])["progress"] == 0.5
    with store.connection() as db:
        assert dict(db.execute("SELECT time,message FROM logs").fetchone())["message"] == "Halfway"
        assert json.loads(db.execute("SELECT config FROM jobs").fetchone()[0]) == {"seed": 42}
    with worker_lease() as first:
        assert first is not None
        with worker_lease() as second:
            assert second is None
    with worker_lease() as third:
        assert third is not None
