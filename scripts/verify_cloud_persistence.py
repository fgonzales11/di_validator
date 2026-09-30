"""Check that a new Cloud Run revision can read completed analysis and accept a job."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import httpx

gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")


def value(*args):
    return subprocess.check_output([gcloud, *args, "--project=thtank"], text=True).strip()


report_path = Path("runtime/cloud-deploy/backend-verification.json")
report = json.loads(report_path.read_text())
client = httpx.Client(base_url=report["url"] + "/api/v1", timeout=60, headers={
    "Authorization": "Bearer " + value("auth", "print-identity-token"),
    "x-di-gateway-token": value("secrets", "versions", "access", "latest", "--secret=di-validator-gateway-token")})


def get(path):
    response = client.get(path)
    response.raise_for_status()
    return response.json()


overview = get("/overview")
assert all(overview[key] == report[key] for key in ["datasets", "assets", "readings"])
fault = get("/experiments/" + report["fault_run"])
assert fault["fault_analyses"][0]["distance"]["estimated_distance_km"] == report["distance_km"]
forecast = get("/forecasting/runs/" + report["forecast_run"])
assert forecast["models"] == report["forecast_models"]
assert get("/forecasting/runs/" + report["inference_run"])["mode"] == "inference"
assert get("/forecasting/runs/" + report["inference_run"] + "/predictions")
response = client.post("/events/run", json={"dataset_id": fault["dataset_ids"][0], "algorithm": "fault-distance"})
response.raise_for_status()
job_id = response.json()["id"]
for _ in range(120):
    job = get("/jobs/" + job_id)
    if job["status"] == "completed":
        break
    assert job["status"] in {"queued", "running"}, job["message"]
    time.sleep(2)
else:
    raise AssertionError("Worker did not finish the post-deployment job")
revision = value("run", "services", "describe", "di-validator-api", "--region=us-west2", "--format=value(status.latestReadyRevisionName)")
report.update(persistence_verified_revision=revision, post_restart_job=job_id)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"persistence_verified": True, "revision": revision, "worker_job": job_id}))
