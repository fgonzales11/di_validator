"""Package only the completed Sites build, following the official dist layout."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from di_validator import store

root = store.ROOT/"runtime/cloud-deploy/site-source"
dist = root/"dist"
head = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=root, text=True).strip()
if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
    raise RuntimeError("Commit the exact source before packaging")
for name in [".openai/hosting.json", "server/index.js", "client/index.html"]:
    if not (dist/name).is_file():
        raise RuntimeError("Build output is incomplete: "+name)
hosting = json.loads((dist/".openai/hosting.json").read_text())
if hosting["project_id"] != json.loads((root/".openai/hosting.json").read_text())["project_id"]:
    raise RuntimeError("Hosting metadata differs from the source")
archive = root.parent/("site-"+head[:12]+".tar.gz")
with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(dist.rglob("*")):
        if path.is_file():
            relative = path.relative_to(dist).as_posix()
            archive_name = relative if relative.startswith(".openai/") else "dist/" + relative
            bundle.add(path, arcname=archive_name, recursive=False)
with tarfile.open(archive) as bundle:
    names = bundle.getnames()
    assert {".openai/hosting.json", "dist/server/index.js", "dist/client/index.html"} <= set(names)
    assert not any("node_modules" in n or n.startswith("/") or ".." in Path(n).parts for n in names)
print(json.dumps({"archive": str(archive), "commit_sha": head, "files": len(names),
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
