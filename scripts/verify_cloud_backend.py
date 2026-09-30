"""Exercise the deployed backend using in-memory developer credentials."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import httpx

gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")


def cloud(*args):
    return subprocess.check_output([gcloud, *args, "--project=thtank"], text=True).strip()


url = cloud("run", "services", "describe", "di-validator-api", "--region=us-west2", "--format=value(status.url)")
identity = cloud("auth", "print-identity-token")
gateway = cloud("secrets", "versions", "access", "latest", "--secret=di-validator-gateway-token")
assert httpx.get(url + "/api/v1/datasets", timeout=60).status_code in {401, 403}
assert httpx.get(url + "/api/v1/datasets", headers={"Authorization": "Bearer " + identity}, timeout=60).status_code == 401
client = httpx.Client(base_url=url, timeout=120,
                     headers={"Authorization": "Bearer " + identity, "x-di-gateway-token": gateway})


def api(path, body=None):
    response = client.get("/api/v1" + path) if body is None else client.post("/api/v1" + path, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"{path}: {response.status_code}: {response.text[:1500]}")
    return response.json()


def finish(job):
    start = time.monotonic()
    previous = None
    while True:
        row = api("/jobs/" + job["id"])
        state = (row["status"], row["message"])
        if state != previous:
            print(state, flush=True)
            previous = state
        if row["status"] == "completed":
            return row["result"]
        if row["status"] in {"failed", "cancelled", "interrupted"}:
            raise RuntimeError(row.get("error") or row["message"])
        if time.monotonic() - start > 900:
            raise TimeoutError("Hosted verification job did not finish within 15 minutes")
        time.sleep(2)


summary = {"url": url, "authentication": "private IAM plus application gateway token"}
overview = api("/overview")
summary["datasets"] = overview["datasets"]
summary["assets"] = overview["assets"]
summary["readings"] = overview["readings"]
sources = api("/faults/sources")
assert len(sources) == 99 and all(source["dataset_id"] for source in sources)
example = next(source for source in sources if source["name"] == "1A_val1")
native = api(f"/notebooks/data/{example['dataset_id']}?start=0&end=0.1995")
assert len(native["rows"]) == 400
print(json.dumps({**summary, "comtrade_sources": len(sources), "native_samples": len(native["rows"])}), flush=True)
event_job = finish(api("/events/run", {"dataset_id": example["dataset_id"], "algorithm": "fault-distance"}))
fault = api("/experiments/" + event_job["id"])
distance = fault["fault_analyses"][0]["distance"]["estimated_distance_km"]
assert abs(distance - .89089036) < 1e-6
summary.update(fault_run=fault["id"], distance_km=distance)
exported = client.get("/api/v1/experiments/" + fault["id"] + "/export")
exported.raise_for_status()
with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
    assert "manifest.json" in archive.namelist() and "report.html" in archive.namelist()
models = api("/forecasting/models")
assert all(model["ready"] for model in models)
assert not {"chronos", "timesfm", "tabpfn"}.intersection(model["id"] for model in models)
config = dict(name="Cloud deployment verification - all supported models", source_file="power_anomaly_dataset_2022.csv",
              channel="Load_kW", unit="kW", horizon=12, validation_windows=2, context_length=128, max_history=300,
              models=[model["id"] for model in models], parameters={"random_forest": {"n_estimators": 20},
                "xgboost": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20},
                "gluonts": {"epochs": 1, "num_batches_per_epoch": 2, "hidden_size": 10},
                "autogluon": {"time_limit": 20}, "neuralprophet": {"epochs": 1}}, timeout_seconds=600)
if "--resume" in sys.argv:
    forecast = next(run for run in api("/forecasting/runs") if run["name"] == config["name"])
else:
    forecast_job = finish(api("/forecasting/runs", config))
    forecast = api("/forecasting/runs/" + forecast_job["id"])
summary.update(forecast_run=forecast["id"], forecast_models=forecast["models"])
failed = [model for model in forecast["models"] if model["status"] != "completed"]
assert not failed, failed
reused_job = finish(api("/forecasting/runs", {**config, "models": ["random_forest"],
                       "parameters": {"random_forest": config["parameters"]["random_forest"]},
                       "reuse_run_id": forecast["id"], "reuse_model": "random_forest"}))
reused = api("/forecasting/runs/" + reused_job["id"])
assert reused["mode"] == "inference"
summary["inference_run"] = reused["id"]
result = client.get("/api/v1/forecasting/runs/" + forecast["id"] + "/export")
result.raise_for_status()
with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
    assert "report.html" in archive.namelist()
path = Path("runtime/cloud-deploy/backend-verification.json")
path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"verified": True, "report": str(path), "distance_km": distance, "models": len(models)}), flush=True)
