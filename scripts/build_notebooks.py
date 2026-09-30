"""Build a locally hosted JupyterLite + matching full Pyodide distribution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from jupyterlite_pyodide_kernel.constants import PYODIDE_VERSION

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks"
OUTPUT = ROOT / "runtime" / "jupyterlite"


def main(force=False):
    versions = {name: importlib.metadata.version(name) for name in
                ["jupyterlite-core", "jupyterlite-pyodide-kernel"]}
    fingerprint = hashlib.sha256(json.dumps(versions, sort_keys=True).encode())
    for path in [Path(__file__), *sorted(SOURCE.rglob("*"))]:
        if path.is_file() and "__pycache__" not in path.parts:
            fingerprint.update(path.read_bytes())
    version = fingerprint.hexdigest()
    marker = OUTPUT / "di-build.json"
    if not force and marker.exists() and (OUTPUT / "lab" / "index.html").exists():
        if json.loads(marker.read_text())["fingerprint"] == version:
            print("Notebook workspace is already built.", flush=True)
            return
    marker.unlink(missing_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "runtime" / "jupyterlite-cache"
    print("Building local notebooks and Pyodide. The first build downloads about 350 MB.", flush=True)
    subprocess.run([sys.executable, "-m", "jupyterlite_core", "build",
                    "--lite-dir", str(SOURCE), "--output-dir", str(OUTPUT),
                    "--pyodide", f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/pyodide-{PYODIDE_VERSION}.tar.bz2"],
                   cwd=ROOT, env={**os.environ, "JUPYTERLITE_CACHE_DIR": str(cache)}, check=True)
    marker.write_text(json.dumps(dict(fingerprint=version, pyodide=PYODIDE_VERSION, **versions)), encoding="utf-8")
    print("Notebook workspace built. Runtime and analysis packages are served locally.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args().force)
