"""Build the backend on Cloud Build using a deliberately bounded source archive."""
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
archive = root / "runtime/cloud-deploy/backend-source.tar.gz"
include = ["pyproject.toml", "uv.lock", ".dockerignore", "deploy/Dockerfile", "di_validator", "scripts",
           "reference", "notebooks", "fault-distance/data/data_test", "fault-distance/fault_distance.ipynb",
           "runtime/jupyterlite"]
with tarfile.open(archive, "w:gz", compresslevel=2) as bundle:
    for name in include:
        source = root / name
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            if path.is_file() and not {"__pycache__", ".pytest_cache", ".git"}.intersection(path.parts):
                bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size}), flush=True)
gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")
if not gcloud:
    raise RuntimeError("Google Cloud CLI is required")
subprocess.run([gcloud, "artifacts", "repositories", "add-iam-policy-binding", "di-validator",
                "--project=thtank", "--location=us-west2",
                "--member=serviceAccount:di-validator@thtank.iam.gserviceaccount.com",
                "--role=roles/artifactregistry.writer", "--quiet"], check=True)
source_url = "gs://thtank-di-validator/build-source/backend-source.tar.gz"
subprocess.run([gcloud, "storage", "cp", str(archive), source_url, "--quiet"], check=True)
subprocess.run([gcloud, "builds", "submit", source_url, "--project=thtank", "--region=us-west2",
                "--gcs-source-staging-dir=gs://thtank-di-validator/build-staging",
                "--config=" + str(root / "deploy/cloud-build.yaml"), "--async", "--format=json", "--quiet"], check=True)
