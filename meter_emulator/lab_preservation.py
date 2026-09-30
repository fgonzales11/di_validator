"""Read-only preservation and ELF audit, executed in Ubuntu-22.04 WSL."""

import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path("/root/di-sdk-setup")
checks = [
    (
        BASE / "fault-location/artifacts/hw42-arm/Release/FaultLocationAgent_0.1.0.115475065.zip",
        "1aa64b3025456cc6813a0f2d79ef6fee611c0707036d42686235d0f6909dedea",
    ),
    (
        BASE / "pv-detection/artifacts/arm-current/Release/PVDetectionAgent_0.1.0.752799214.zip",
        "9d6f0052de409a48e32982f70984ebe3b191dd77bf0829a0f11a4d16334463d1",
    ),
    (
        Path(
            "/home/fgonzales/.local/share/emu-tool/lxc/di-hw42/rootfs/usr/bin/23020000/FaultLocationAgent_Daemon"
        ),
        "3b3b12d747f657f6142072be37665f2c70d346d4984274329c8d3a6dfc07b241",
    ),
    (
        Path(
            "/home/fgonzales/.local/share/emu-tool/lxc/di-hw42/rootfs/usr/bin/23020001/PVDetectionAgent_Daemon"
        ),
        "31aa7dbc2eca53a7133f118804b0e4fcd0b4596e6957191f549d6aa4ba5a4657",
    ),
]
result = {"preserved": [], "arm": {}}
for path, expected in checks:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    result["preserved"].append({"path": str(path), "sha256": actual, "matches_baseline": actual == expected})
for name in ["FaultLocationAgent", "PVDetectionAgent"]:
    binary = BASE / "meter-lab/build" / name / (name + "_Daemon")
    text = subprocess.check_output(["readelf", "-h", "-d", str(binary)], text=True)
    result["arm"][name] = {
        "elf": text,
        "bytes": binary.stat().st_size,
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "arm_eabi5": "ARM" in text and "Version5 EABI" in text,
    }
result["original_container"] = subprocess.check_output(
    ["lxc-info", "-P", "/home/fgonzales/.local/share/emu-tool/lxc", "-n", "di-hw42"], text=True
)
destination = Path(__file__).resolve().parent.parent / "runtime/meter-lab/verification/preservation.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {"unchanged": all(r["matches_baseline"] for r in result["preserved"]), "report": str(destination)}
    )
)
