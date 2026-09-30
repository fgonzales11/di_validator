"""Copy the cloud snapshot and data files to the dedicated private bucket."""
import shutil
import subprocess

from di_validator import store


def main():
    gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")
    root = store.ROOT
    log = root/"runtime/cloud-deploy/upload.log"
    pairs = [(root/"data", "sources"), (root/"runtime/cloud-deploy/snapshot", "migration")]
    pairs += [(root/"runtime"/name, "runtime/"+name) for name in ["datasets", "experiments", "forecasts", "uploads"]]
    with log.open("w", encoding="utf-8") as stream:
        for source, target in pairs:
            print("Uploading "+target, flush=True)
            subprocess.run([gcloud, "storage", "rsync", str(source), "gs://thtank-di-validator/"+target,
                            "--recursive", "--quiet"], stdout=stream, stderr=subprocess.STDOUT, check=True)
            print("Completed "+target, flush=True)


if __name__ == "__main__":
    main()
