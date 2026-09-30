"""Provision DI Validator's dedicated credentials without writing plaintext files."""
import secrets
import shutil
import subprocess
from urllib.parse import quote

GCLOUD = shutil.which("gcloud.cmd") or shutil.which("gcloud")
PROJECT = "thtank"
ACCOUNT = "di-validator@thtank.iam.gserviceaccount.com"


def call(*args, data=None):
    process = subprocess.run([GCLOUD, *args, "--project="+PROJECT, "--quiet"], input=data,
                             capture_output=True, text=True, check=False)
    if process.returncode:
        # Commands may carry credentials. Never include command arguments in errors.
        raise RuntimeError(process.stderr[-1500:])
    return process.stdout


def exists(name):
    return subprocess.run([GCLOUD, "secrets", "describe", name, "--project="+PROJECT],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


if __name__ == "__main__":
    if not exists("di-validator-database-url"):
        password = secrets.token_urlsafe(40)
        call("sql", "users", "create", "di_validator", "--instance=scar-postgres", "--password="+password)
        value = "postgresql://di_validator:"+quote(password, safe="")+"@10.233.0.3:5432/di_validator?sslmode=require"
        call("secrets", "create", "di-validator-database-url", "--replication-policy=automatic", "--data-file=-", data=value)
    if not exists("di-validator-gateway-token"):
        call("secrets", "create", "di-validator-gateway-token", "--replication-policy=automatic", "--data-file=-",
             data=secrets.token_urlsafe(48))
    for name in ["di-validator-database-url", "di-validator-gateway-token"]:
        call("secrets", "add-iam-policy-binding", name, "--member=serviceAccount:"+ACCOUNT,
             "--role=roles/secretmanager.secretAccessor")
    call("storage", "buckets", "add-iam-policy-binding", "gs://thtank-di-validator",
         "--member=serviceAccount:"+ACCOUNT, "--role=roles/storage.objectAdmin")
    call("storage", "buckets", "add-iam-policy-binding", "gs://thtank-di-validator",
         "--member=serviceAccount:"+ACCOUNT, "--role=roles/storage.legacyBucketReader")
    print("Dedicated database and gateway secrets configured; service account access granted.")
