"""Install the local di-meter command into Ubuntu-22.04 (run as root)."""
import os
from pathlib import Path
import shutil

FILES = ["di_meter.py", "metrology_bridge.py", "pv_checkpoint.py", "verify.py",
         "build_runtime.py", "config_compat.c", "README.md"]


def main():
    if os.geteuid() != 0:
        raise SystemExit("Run with sudo or wsl.exe -d Ubuntu-22.04 -u root")
    source = Path(__file__).resolve().parent
    target = Path("/usr/local/lib/di-meter-emulator")
    wrapper = Path("/usr/local/bin/di-meter")
    if wrapper.exists() and "di-meter-emulator/di_meter.py" not in wrapper.read_text():
        raise RuntimeError("An unrelated di-meter command already exists")
    target.mkdir(exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, target / name)
        (target / name).chmod(0o644)
    wrapper.write_text('#!/bin/sh\nexec /usr/bin/python3 /usr/local/lib/di-meter-emulator/di_meter.py "$@"\n')
    wrapper.chmod(0o755)
    print("Installed /usr/local/bin/di-meter")


if __name__ == "__main__":
    main()
