"""Connect Sites to private Cloud Run through API Gateway without service-account keys."""
import json
import shutil
import subprocess

gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")


def call(*arguments, data=None):
    result = subprocess.run([gcloud, *arguments, "--project=thtank", "--quiet"],
                            input=data, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout.strip()


def exists(*arguments):
    return subprocess.run([gcloud, *arguments, "--project=thtank", "--quiet"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


managed = call("api-gateway", "apis", "describe", "di-validator", "--format=value(managedService)")
call("services", "enable", managed)
secret = "di-validator-api-key"
if not exists("secrets", "describe", secret):
    key_id = "di-validator-sites"
    if not exists("services", "api-keys", "describe", key_id):
        call("services", "api-keys", "create", "--key-id=" + key_id, "--display-name=DI Validator Sites bridge",
             "--api-target=service=" + managed, "--format=value(name)")
    value = call("services", "api-keys", "get-key-string", key_id, "--format=value(keyString)")
    if not value:
        raise RuntimeError("The API key service returned an empty key")
    call("secrets", "create", secret, "--replication-policy=automatic", "--data-file=-", data=value)
print("Restricted API key stored in Secret Manager; no credential file written.", flush=True)
if not exists("api-gateway", "gateways", "describe", "di-validator", "--location=us-west2"):
    print("Creating managed API gateway.", flush=True)
    call("api-gateway", "gateways", "create", "di-validator", "--api=di-validator",
         "--api-config=di-validator-v1", "--location=us-west2")
gateway = json.loads(call("api-gateway", "gateways", "describe", "di-validator", "--location=us-west2", "--format=json"))
print(json.dumps({"state": gateway["state"], "gateway_url": "https://" + gateway["defaultHostname"],
                  "managed_service": managed}), flush=True)
