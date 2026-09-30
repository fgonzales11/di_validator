"""Build emulator-only foreground ARM executables with the SDK's NODAEMON flag.

NODAEMON applies only to agent_daemon.cpp so the host-only replay adapters remain
excluded. Uses existing SDK3 TargetRelease build definitions; never modifies an SDK build
directory, original source, or unsigned package. Run with the emulator stopped.
"""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import di_meter as control


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if control.container_state() != "STOPPED":
        raise RuntimeError("Stop di-hw42 before building/installing its runtime")
    workspace = control.SDK / "RIVA_AGENTSUPPORT/Workspace/TargetRelease"
    output = control.BASE / "foreground-build"
    output.mkdir(exist_ok=True)
    manifest = {"mode": "ARM SDK3 NODAEMON foreground, emulator only", "agents": {}}
    dbus = workspace / "Libraries/dbus/dbus_Download-prefix/src/dbus_Download"
    shim_source = Path(__file__).resolve().parent / "config_compat.c"
    shim = output / "libdi_emulator_config_compat.so"
    subprocess.run([str(workspace / "toolchain/bin/armv7l-timesys-linux-uclibcgnueabi-gcc"),
        "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-O2",
        "-I" + str(dbus / "usr/include/dbus-1.0"),
        "-I" + str(dbus / "usr/lib/dbus-1.0/include"),
        str(shim_source), "-ldl", "-o", str(shim)], check=True)
    shutil.copy2(shim, control.ROOT / "usr/lib" / shim.name)
    manifest["config_compat"] = {"source_sha256": digest(shim_source), "binary_sha256": digest(shim)}
    for key, (aid, name) in control.AGENTS.items():
        project = workspace / "Projects" / name
        generated = project / "CMakeFiles" / (name + "_Daemon.dir")
        destination = output / name
        destination.mkdir(exist_ok=True)
        definitions = {}
        for line in (generated / "flags.make").read_text().splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                definitions[k] = shlex.split(v)
        link = shlex.split((generated / "link.txt").read_text())
        compiler = link[0]
        inputs = {}
        commands = []
        updated = []
        for argument in link:
            if argument.endswith(".o"):
                source = control.SDK / "Agents" / name / "src" / Path(argument).name[:-2]
                target = destination / Path(argument).name
                command = [compiler, "--sysroot=" + str(workspace / "toolchain"),
                    *definitions["CXX_DEFINES"], *definitions["CXX_INCLUDES"],
                    *definitions["CXX_FLAGS"],
                    *(["-DNODAEMON"] if source.name == "agent_daemon.cpp" else []),
                    "-c", str(source), "-o", str(target)]
                commands.append(command)
                inputs[str(source)] = digest(source)
                subprocess.run(command, check=True)
                updated.append(str(target))
            elif "DIAgent_Download/lib/libDIAgent.so" in argument and argument.startswith("../../"):
                updated.append(str(control.ROOT / "usr/lib/DIAgent-4.1.3/libDIAgent.so.4"))
            elif argument.startswith("../../"):
                updated.append(str((project / argument).resolve()))
            else:
                updated.append(argument)
        executable = destination / (name + "_Daemon")
        updated[updated.index("-o") + 1] = str(executable)
        commands.append(updated)
        subprocess.run(updated, check=True)
        shutil.copy2(executable, destination / (executable.name + ".debug"))
        subprocess.run([str(workspace / "toolchain/bin/armv7l-timesys-linux-uclibcgnueabi-strip"),
                        str(executable)], check=True)
        installed = control.ROOT / "usr/bin" / aid / executable.name
        original = destination / "packaged-daemon.original"
        if not original.exists():
            shutil.copy2(installed, original)
        shutil.copy2(executable, installed)
        manifest["agents"][key] = {"source_sha256": inputs, "commands": commands,
            "executable_sha256": digest(executable), "installed": str(installed),
            "packaged_original_sha256": digest(original)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value["executable_sha256"] for key, value in manifest["agents"].items()}, indent=2))


if __name__ == "__main__":
    main()
