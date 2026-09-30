from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from .. import store
from .catalog import MODELS, RUNTIMES, canonical, interpreter, runtime_root


def process(command, log, progress, timeout=3600, env=None):
    """Own the entire process tree; cancellation cannot leave an ML trainer running."""
    from ..worker import terminate_tree

    started = time.monotonic()
    with Path(log).open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command,
            cwd=store.ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while child.poll() is None:
                progress.check()
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Forecast process exceeded {timeout} seconds; see {log}")
                time.sleep(0.2)
            if child.returncode:
                raise ValueError(
                    f"Forecast process failed: {Path(log).read_text(encoding='utf-8', errors='replace')[-5000:]}"
                )
        except BaseException:
            if child.poll() is None:
                terminate_tree(child)
            raise


def request(model, body, folder, progress, timeout=600):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    body = {**body, "model": model, "response": str(folder / "response.json")}
    (folder / "request.json").write_text(store.encode(body), encoding="utf-8")
    process(
        [
            str(interpreter(model)),
            str(store.ROOT / "scripts/forecast_runner.py"),
            str(folder / "request.json"),
        ],
        folder / "process.log",
        progress,
        timeout,
    )
    result = json.loads((folder / "response.json").read_text())
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["result"]


def setup(config, progress=None):
    progress = progress or store.Progress()
    model = canonical(config["model"])
    if model not in MODELS:
        raise ValueError("Unknown forecasting model")
    group = MODELS[model]["group"]
    folder = store.workspace() / "forecast-setup" / store.uid()
    folder.mkdir(parents=True)
    if group != "base":
        plan = RUNTIMES[group]
        root = runtime_root(group)
        root.parent.mkdir(parents=True, exist_ok=True)
        progress(0.05, f"Preparing Python {plan['python']} runtime for {group}")
        if not interpreter(model).exists():
            process(["uv", "venv", "--python", plan["python"], str(root)], folder / "venv.log", progress)
        progress(0.15, f"Installing pinned {group} forecasting packages")
        process(
            ["uv", "pip", "install", "--python", str(interpreter(model)), *plan["requirements"]],
            folder / "install.log",
            progress,
        )
        progress(0.7, "Checking provider imports")
        providers = request(
            model,
            {"mode": "probe", "models": [k for k, v in MODELS.items() if v["group"] == group]},
            folder / "probe",
            progress,
            600,
        )
        state = dict(providers=providers, requirements=plan, created_at=store.now())
        (root / "di-runtime.json").write_text(store.encode(state), encoding="utf-8")
        process(
            ["uv", "pip", "freeze", "--python", str(interpreter(model))],
            root / "requirements.lock.txt",
            progress,
        )
        if not providers[model]["import_ok"]:
            raise ValueError(providers[model]["error"])
    progress(1, "Forecast runtime ready")
    return dict(id=store.uid(), name=f"{MODELS[model]['name']} runtime ready", folder=str(folder))
