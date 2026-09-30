"""Inspect an unsigned SDK package without installing it (run in Ubuntu WSL)."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_elf(data):
    return data[:6] == b"\x7fELF\x01\x01" and int.from_bytes(data[18:20], "little") == 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, default=Path("/root/DI-SDK-3.0.0/DistributedIntelligenceSDK"))
    args = parser.parse_args()
    profile = json.loads((Path(__file__).resolve().parents[1] / "hardware_profile.json").read_text())
    binary = args.binary.read_bytes()
    require(arm_elf(binary), "Expected a little-endian 32-bit ARM ELF")
    require(b"PV_REPLAY_DIR" not in binary, "Host replay ingress leaked into ARM binary")
    binary_sha = hashlib.sha256(binary).hexdigest()
    with zipfile.ZipFile(args.package) as archive:
        require(archive.testzip() is None, "ZIP CRC failed")
        names = archive.namelist()
        require(not any("applicationmanifest" in n.lower() for n in names), "App manifest in OTA ZIP")
        control = json.loads(archive.read("signer-control.json"))
        unsigned_tar = control["ota"][0]["input"]
        require(unsigned_tar.startswith("tmp_"), "Expected the SDK unsigned signing input")
        with tarfile.open(fileobj=io.BytesIO(archive.read(unsigned_tar))) as outer:
            manifest_bytes = outer.extractfile("ReleaseManifest.xml").read()
            manifest = ET.fromstring(manifest_bytes)
            policy = ET.fromstring(outer.extractfile("PolicyFile.xml").read())
            payload_info = manifest.find(".//device/installPath/install/file")
            payload = outer.extractfile(payload_info.get("path")).read()
            # Vendor ReleaseManifest uses an unpadded hexadecimal MD5 integer.
            require(
                int(hashlib.md5(payload).hexdigest(), 16) == int(payload_info.get("hash"), 16),
                "Payload does not match vendor manifest hash",
            )
            with tarfile.open(fileobj=io.BytesIO(payload)) as inner:
                payload_binary = inner.extractfile("usr/bin/23020001/PVDetectionAgent_Daemon").read()
                require(payload_binary == binary, "Packaged executable differs from retained ELF")
                require(
                    not any(n.endswith((".py", ".pvr", ".pvs")) for n in inner.getnames()),
                    "Host-only files included in ARM payload",
                )
                packaged_libraries = [m.name for m in inner.getmembers() if "/lib" in m.name and m.isfile()]
                require(
                    all(arm_elf(inner.extractfile(n).read()) for n in packaged_libraries), "Non-ARM library"
                )
                unpacked_file_bytes = sum(m.size for m in inner.getmembers() if m.isfile())
        validation = subprocess.run(
            [
                "xmllint",
                "--noout",
                "--schema",
                str(args.sdk / "target_scripts/ReleaseManifestFileSchema.xsd"),
                "-",
            ],
            input=manifest_bytes,
            capture_output=True,
            check=True,
        )
        with tempfile.TemporaryDirectory(prefix="pv-package-schema-") as folder:
            temp = Path(folder)
            for name in names:
                if name.startswith("config/") and not name.endswith("/"):
                    (temp / Path(name).name).write_bytes(archive.read(name))
            for path in sorted(temp.glob("*.xml")):
                schema = "BasicDesc.xsd" if path.name.endswith(".desc.xml") else "BasicConfig.xsd"
                subprocess.run(
                    ["xmllint", "--noout", "--schema", str(temp / schema), str(path)],
                    capture_output=True,
                    check=True,
                )
    hardware_ids = sorted(int(e.get("hwid")) for e in manifest.findall("./package/hardwares/hardware"))
    require(
        hardware_ids == sorted(profile["hardware_ids_from_sdk"]),
        "Package is not restricted to HW 4.2 singlephase/polyphase",
    )
    constraints = manifest.findall("./package/firmwareConstraints//item")
    expected_versions = {"system": "50.10.312.2", "appserve": "2.0.581.0"}
    require(
        {i.get("target") for i in constraints} == set(expected_versions),
        "Missing firmware/AppServ constraint",
    )
    require(
        all(i.get("version") == expected_versions[i.get("target")] for i in constraints),
        "Unexpected firmware constraints",
    )
    agent = policy.find("Agent")
    require(agent.get("Hash") == binary_sha, "Policy executable hash differs")
    require(
        agent.find("Resources").attrib == {"Cpu": "2", "Flash": "2048", "Ram": "2048"},
        "Agent resource policy changed",
    )
    require(args.package.stat().st_size < 10_000_000, "Package exceeds HW 4.2 10 MB ceiling")
    dynamic = subprocess.run(
        ["readelf", "-d", str(args.binary)], capture_output=True, text=True, check=True
    ).stdout
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    require(not any("python" in n.lower() for n in needed), "Unexpected Python dependency")
    target = args.sdk / "RIVA_AGENTSUPPORT/Workspace/TargetRelease"
    dependency_evidence = {}
    for name in needed:
        if name.startswith("libDIAgent.so"):
            dependency_evidence[name] = {"source": "package", "files": packaged_libraries}
            continue
        candidates = list((target / "toolchain").rglob(name)) + list((target / "Libraries").rglob(name))
        candidates = [p for p in candidates if p.is_file() and arm_elf(p.read_bytes())]
        require(candidates, "Missing ARM dependency: " + name)
        dependency_evidence[name] = {"source": "SDK target sysroot", "path": str(candidates[0])}
    report = {
        "status": "PASS",
        "package": args.package.name,
        "version": agent.get("Version"),
        "signed": False,
        "package_bytes": args.package.stat().st_size,
        "package_sha256": hashlib.sha256(args.package.read_bytes()).hexdigest(),
        "binary_bytes": len(binary),
        "binary_sha256": binary_sha,
        "payload_matches_retained_binary": True,
        "payload_regular_file_bytes": unpacked_file_bytes,
        "zip_crc": "PASS",
        "manifest_xml_schema": validation.stderr.decode().strip(),
        "feature_xml_schemas": "PASS",
        "hardware_ids": hardware_ids,
        "firmware_constraints": [i.attrib for i in constraints],
        "agent_resources": agent.find("Resources").attrib,
        "elf": subprocess.run(
            ["file", "-b", str(args.binary)], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "needed": needed,
        "dependency_evidence": dependency_evidence,
        "host_replay_and_python_excluded": True,
        "meter_execution_verified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Package PASS:", args.package.name, "report:", args.report)


if __name__ == "__main__":
    main()
