"""Install/build the isolated Meter Lab. Does not stop or change di-hw42."""

from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess

KIT = Path(__file__).resolve().parent
WORK = KIT.parent
BASE = Path("/root/di-sdk-setup/meter-lab")
SDK = Path("/root/DI-SDK-3.0.0/DistributedIntelligenceSDK")
ORIGINAL = Path("/home/fgonzales/.local/share/emu-tool/lxc/di-hw42/rootfs")
ROOT = BASE / "lxc/di-meter-lab/rootfs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (
        subprocess.run(
            ["lxc-info", "-P", str(BASE / "lxc"), "-n", "di-meter-lab", "-sH"], capture_output=True, text=True
        ).stdout.strip()
        == "RUNNING"
    ):
        raise RuntimeError("Stop the Meter Lab run before rebuilding its runtime")
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "workspace.json").write_text(json.dumps({"artifacts": str(WORK / "runtime/meter-lab")}))
    if not ROOT.exists():

        def skip(path, names):
            return [
                n
                for n in names
                if n in {"proc", "sys", "dev", "tmp", "flash", "samples", "system_bus_socket"}
            ]

        shutil.copytree(ORIGINAL, ROOT, symlinks=True, ignore=skip)
        for n in ["proc", "sys", "dev", "tmp", "usr/share/flash", "meter-lab", "var/run/dbus"]:
            (ROOT / n).mkdir(parents=True, exist_ok=True)
        for n in ["null", "log", "urandom"]:
            (ROOT / "dev" / n).touch()
        (ROOT / "var/run/dbus/system_bus_socket").touch()
    installed = BASE / "tools"
    installed.mkdir(exist_ok=True)
    for name in ["di_meter.py", "metrology_bridge.py", "pv_checkpoint.py", "lab_runtime.py"]:
        shutil.copyfile(KIT / name, installed / name)
    workspace = SDK / "RIVA_AGENTSUPPORT/Workspace/TargetRelease"
    manifest = {"version": 1, "flag": "DI_EMULATOR_REPLAY", "agents": {}}
    manifest["helpers"] = {
        name: sha(KIT / name)
        for name in ["di_meter.py", "metrology_bridge.py", "pv_checkpoint.py", "lab_runtime.py"]
    }
    manifest["libraries"] = {
        str(path.relative_to(ROOT)): sha(path)
        for path in [
            ROOT / "usr/lib/libAgentAPI.so.0.0.0",
            ROOT / "usr/lib/DIAgent-4.1.3/libDIAgent.so.4",
            ROOT / "usr/lib/libdbus-1.so.3",
            ROOT / "usr/lib/libstdc++.so.6",
        ]
    }
    for key, name, aid, kit in [
        ("fault", "FaultLocationAgent", "23020000", "di_agent"),
        ("pv", "PVDetectionAgent", "23020001", "pv_agent"),
    ]:
        print("Building " + name, flush=True)
        out = BASE / "build" / name
        out.mkdir(parents=True, exist_ok=True)
        inc = out / "include"
        inc.mkdir(exist_ok=True)
        shutil.copyfile(WORK / kit / "integration/agent_feature.h", inc / "agent_feature.h")
        generated = workspace / "Projects" / name / "CMakeFiles" / (name + "_Daemon.dir")
        definitions = dict(
            line.split(" = ", 1)
            for line in (generated / "flags.make").read_text().splitlines()
            if " = " in line
        )
        link = shlex.split((generated / "link.txt").read_text())
        compiler = link[0]
        objects = []
        sources = {}
        commands = []
        files = [SDK / "Agents" / name / "src" / Path(arg).name[:-2] for arg in link if arg.endswith(".o")]
        files = [
            WORK / kit / "integration" / p.name
            if p.name in {"agent_feature.cpp", "agent_metrology.cpp"}
            else p
            for p in files
        ]
        files.append(KIT / (key + "_lab.cpp"))
        # Objects depend on the agent's headers too: a changed struct layout (e.g. fault::Config)
        # must rebuild every translation unit that includes it, not only the edited source.
        headers = sorted(
            path
            for folder in [SDK / "Agents" / name / "include", WORK / kit / "core", WORK / kit / "host", KIT]
            if folder.exists()
            for path in folder.glob("*.h*")
        )
        header_digest = "".join(sha(path) for path in headers)
        if key == "fault":
            files.append(WORK / kit / "host/replay_io.cpp")
        for source in files:
            obj = out / (source.name + ".o")
            objects.append(str(obj))
            sources[str(source)] = sha(source)
            command = [
                compiler,
                "--sysroot=" + str(workspace / "toolchain"),
                "-I" + str(inc),
                "-I" + str(KIT),
                "-I" + str(WORK / kit / "host"),
                *shlex.split(definitions["CXX_DEFINES"]),
                *shlex.split(definitions["CXX_INCLUDES"]),
                *shlex.split(definitions["CXX_FLAGS"]),
                "-DDI_EMULATOR_REPLAY",
                *(["-DNODAEMON"] if source.name == "agent_daemon.cpp" else []),
                "-c",
                str(source),
                "-o",
                str(obj),
            ]
            commands.append(command)
            stamp = out / (source.name + ".sha")
            signature = hashlib.sha256(
                (
                    json.dumps(command)
                    + sha(source)
                    + sha(inc / "agent_feature.h")
                    + sha(KIT / "lab_stream.hpp")
                    + header_digest
                ).encode()
            ).hexdigest()
            if not obj.exists() or not stamp.exists() or stamp.read_text() != signature:
                subprocess.run(command, check=True)
                stamp.write_text(signature)
        binary = out / (name + "_Daemon")
        command = [compiler, *objects]
        project = workspace / "Projects" / name
        for arg in link[1:]:
            if arg.endswith(".o"):
                continue
            if "DIAgent_Download/lib/libDIAgent.so" in arg and arg.startswith("../../"):
                arg = str(ROOT / "usr/lib/DIAgent-4.1.3/libDIAgent.so.4")
            elif arg.startswith("../../"):
                arg = str((project / arg).resolve())
            command.append(arg)
        command[command.index("-o") + 1] = str(binary)
        commands.append(command)
        subprocess.run(command, check=True)
        shutil.copyfile(binary, out / (binary.name + ".debug"))
        subprocess.run(
            [str(workspace / "toolchain/bin/armv7l-timesys-linux-uclibcgnueabi-strip"), str(binary)],
            check=True,
        )
        shutil.copyfile(binary, ROOT / "usr/bin" / aid / binary.name)
        (ROOT / "usr/bin" / aid / binary.name).chmod(0o755)
        manifest["agents"][key] = {
            "name": name,
            "id": aid,
            "sha256": sha(binary),
            "sources": sources,
            "commands": commands,
        }
    observer = BASE / "build/libdi_lab_delivery.so"
    dbus = workspace / "Libraries/dbus/dbus_Download-prefix/src/dbus_Download"
    compat_object = BASE / "build/config_compat.o"
    subprocess.run(
        [
            compiler.replace("g++", "gcc"),
            "-fPIC",
            "-O2",
            "-I" + str(dbus / "usr/include/dbus-1.0"),
            "-I" + str(dbus / "usr/lib/dbus-1.0/include"),
            "-c",
            str(KIT / "config_compat.c"),
            "-o",
            str(compat_object),
        ],
        check=True,
    )
    command = [
        compiler,
        "--sysroot=" + str(workspace / "toolchain"),
        *shlex.split(definitions["CXX_INCLUDES"]),
        "-std=c++11",
        "-Os",
        "-fPIC",
        "-fno-exceptions",
        "-fno-rtti",
        "-nodefaultlibs",
        "-shared",
        str(KIT / "lab_delivery.cpp"),
        str(compat_object),
        "-ldl",
        "-lc",
        "-o",
        str(observer),
    ]
    subprocess.run(command, check=True)
    shutil.copyfile(observer, ROOT / "usr/lib" / observer.name)
    manifest["delivery_observer"] = {
        "config_compat_source_sha256": sha(KIT / "config_compat.c"),
        "source_sha256": sha(KIT / "lab_delivery.cpp"),
        "binary_sha256": sha(observer),
        "command": command,
    }
    (BASE / "build-manifest.json").write_text(json.dumps(manifest, indent=2))
    (BASE / "container-template.conf").write_text(
        (ORIGINAL.parent / "config")
        .read_text()
        .replace(str(ORIGINAL), str(ROOT))
        .replace("lxc.uts.name = di-hw42", "lxc.uts.name = di-meter-lab")
    )
    print(json.dumps({"installed": True, "manifest": str(BASE / "build-manifest.json")}))


if __name__ == "__main__":
    main()
