"""Archive the installed emulator replay build and its exact source inputs (WSL)."""

import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path("/root/di-sdk-setup/meter-lab")
WORK = Path(__file__).resolve().parent.parent


def main():
    manifest = json.loads((BASE / "build-manifest.json").read_text())
    for name, expected in manifest["helpers"].items():
        if hashlib.sha256((BASE / "tools" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Installed helper changed before freezing: " + name)
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    destination = WORK / "runtime/meter-lab/builds" / identity
    destination.mkdir(parents=True, exist_ok=True)
    sources = set()
    sources.update((WORK / "runtime/meter-lab/helper-sources").glob("*.py"))
    for agent in manifest["agents"].values():
        for name, expected in agent["sources"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Build source changed before freezing: " + name)
        sources.update(Path(p) for p in agent["sources"])
    for folder in ["meter_emulator", "di_agent", "pv_agent", "di_validator/meter_lab"]:
        sources.update(
            p
            for p in (WORK / folder).rglob("*")
            if p.is_file()
            and p.suffix
            in {".py", ".c", ".cpp", ".h", ".hpp", ".md", ".json", ".xml", ".txt", ".cmake", ".sh", ".toml"}
            and "__pycache__" not in p.parts
        )
    sources.update(
        WORK / name
        for name in [
            "frontend/src/pages/MeterLab.tsx",
            "frontend/src/pages/meter-lab.css",
            "frontend/src/components/Chart.tsx",
            "frontend/src/main.tsx",
            "frontend/e2e/meter-lab.spec.ts",
            "frontend/package.json",
            "frontend/package-lock.json",
            "frontend/playwright.config.ts",
            "pyproject.toml",
            "uv.lock",
            "tests/test_meter_lab.py",
            "di_validator/api.py",
            "di_validator/labels.py",
            "di_validator/routes/experiments.py",
            "scripts/start.ps1",
            "scripts/stop.ps1",
            "scripts/restart-meter-lab-worker.ps1",
            "README.md",
        ]
    )
    source_hashes = {}
    with zipfile.ZipFile(destination / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            if not path.is_file():
                continue
            name = (
                str(path.relative_to(WORK))
                if path.is_relative_to(WORK)
                else "sdk-entrypoints/" + str(path).split("/Agents/", 1)[-1]
            )
            installed = BASE / "tools" / path.name
            raw = (
                installed.read_bytes()
                if path.parent == WORK / "meter_emulator" and path.name in manifest["helpers"]
                else path.read_bytes()
            )
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(name, raw)
    with zipfile.ZipFile(destination / "arm-emulator-replay.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for agent in manifest["agents"].values():
            for suffix in ["", ".debug"]:
                binary = BASE / "build" / agent["name"] / (agent["name"] + "_Daemon" + suffix)
                if not suffix and hashlib.sha256(binary.read_bytes()).hexdigest() != agent["sha256"]:
                    raise ValueError("ARM binary changed before freezing")
                archive.write(binary, binary.name)
        archive.write(BASE / "build/libdi_lab_delivery.so", "libdi_lab_delivery.so")
        archive.writestr("build-manifest.json", json.dumps(manifest, indent=2))
    (destination / "source-checksums.json").write_text(json.dumps(source_hashes, indent=2))
    (destination / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "build_id": identity,
                "path": str(destination),
                "purpose": "Emulator replay only; not a production DI install package",
            }
        )
    )


if __name__ == "__main__":
    main()
