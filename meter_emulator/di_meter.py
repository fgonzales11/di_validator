#!/usr/bin/env python3
"""Manage the dedicated HW4.2 ARM emulator and isolated SDK host services."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

BASE = Path(os.environ.get("DI_METER_BASE", "/root/di-sdk-setup/meter-emulator"))
SDK = Path("/root/DI-SDK-3.0.0/DistributedIntelligenceSDK")
LXC = Path(os.environ.get("DI_METER_LXC", "/home/fgonzales/.local/share/emu-tool/lxc"))
NAME = os.environ.get("DI_METER_NAME", "di-hw42")
ROOT = LXC / NAME / "rootfs"
TMP = Path(os.environ.get("DI_METER_TMP", str(ROOT / "tmp")))
HERE = Path(__file__).resolve().parent
AGENTS = {
    "fault": ("23020000", "FaultLocationAgent"),
    "pv": ("23020001", "PVDetectionAgent"),
}
if os.environ.get("DI_METER_AGENT") in AGENTS:
    AGENTS = {os.environ["DI_METER_AGENT"]: AGENTS[os.environ["DI_METER_AGENT"]]}
DATABASE = BASE / "database/flash/muse01.db"
SERVICES = BASE / "run/services.json"


def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, check=True, **kwargs)


def wait_for(predicate, seconds=20):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(.1)
    raise RuntimeError("Timed out; inspect " + str(BASE / "logs"))


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except OSError:
        return None


def agents():
    found = {key: [] for key in AGENTS}
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if "lxc.payload." + NAME not in (path / "cgroup").read_text():
                continue
            command = (path / "cmdline").read_bytes().split(b"\0")
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            for key, (aid, name) in AGENTS.items():
                if (f"/usr/bin/{aid}/{name}_Daemon").encode() in command:
                    found[key].append(int(path.name))
                    parents[int(path.name)] = int(stat[1])
        except OSError:
            pass
    # The SDK resource logger forks helpers. Before exec they briefly inherit
    # the daemon argv; count only independent daemons, not those child helpers.
    return {key: [pid for pid in pids if parents.get(pid) not in pids] for key, pids in found.items()}


def container_state():
    p = subprocess.run(["lxc-info", "-P", str(LXC), "-n", NAME, "-sH"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "MISSING"


def services():
    if not SERVICES.exists():
        return {}
    result = json.loads(SERVICES.read_text())
    return {name: item for name, item in result.items()
            if isinstance(item, dict) and identity(item["pid"]) == item["start"]}


def spawn(name, command, env=None):
    with (BASE / "logs" / (name + ".log")).open("a") as log:
        child = subprocess.Popen(command, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    item = {"pid": child.pid, "start": identity(child.pid)}
    return item


def save_services(items):
    temporary = SERVICES.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, indent=2) + "\n")
    temporary.replace(SERVICES)


def import_configuration():
    with sqlite3.connect(DATABASE) as db:
        for aid, _ in AGENTS.values():
            for path in (ROOT / "etc/agents.d").glob(aid + ".*.xml"):
                parts = path.name.split(".")
                if len(parts) != 3:
                    continue
                db.execute("insert or replace into FeatureConfiguration "
                           "(FeatureId,ConfigurationId,Level,Data,LastUpdateTime) "
                           "values (?,?,?,?,?)",
                           (int(parts[1], 16), 0, 5, gzip.compress(path.read_bytes()),
                            int(time.time())))


def namespace_server():
    # Mounts affect only this DataServer and its descendants.
    for source, target in [(BASE / "database", "/usr/local/bin/database"),
                           (BASE / "metsimfiles", "/usr/share/itron/metsimfiles")]:
        run(["mount", "--bind", str(source), target])
    os.chdir("/usr/local/bin")
    os.execv("/usr/local/bin/DataServer", ["/usr/local/bin/DataServer"])


def dbus_ready():
    env = {**os.environ, "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=" + str(BASE / "run/bus.sock")}
    p = subprocess.run(["dbus-send", "--system", "--print-reply", "--reply-timeout=1000",
                        "--dest=org.freedesktop.DBus", "/", "org.freedesktop.DBus.ListNames"],
                       env=env, capture_output=True, text=True)
    return "com.itron.owi.platform.dataserver" in p.stdout


def start_services():
    current = services()
    if current:
        if set(current) == {"bus", "dataserver", "metrology"} and dbus_ready():
            return
        raise RuntimeError("Partial emulator services are running; use stop, then start")
    socket = BASE / "run/bus.sock"
    if socket.exists():
        socket.unlink()
    import_configuration()
    items = {"bus": spawn("dbus", ["dbus-daemon", "--nofork",
                                   "--config-file=" + str(BASE / "run/bus.conf")])}
    save_services(items)
    wait_for(socket.exists)
    env = {**os.environ, "LD_LIBRARY_PATH": str(SDK / "Packages/libs"),
           "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=" + str(socket),
           "DI_METER_EMULATOR": NAME}
    with (BASE / "logs/dataserver.log").open("a") as log:
        subprocess.run(["unshare", "--mount", "--propagation", "private", sys.executable,
                        str(Path(__file__).resolve()), "_dataserver"], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=15)
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if ((path / "exe").readlink().name == "DataServer" and
                    ("DI_METER_EMULATOR=" + NAME).encode() in (path / "environ").read_bytes() and
                    (NAME != "di-meter-lab" or ("DI_METER_BASE=" + str(BASE)).encode() in (path / "environ").read_bytes())):
                items["dataserver"] = {"pid": int(path.name), "start": identity(path.name)}
        except OSError:
            pass
    save_services(items)
    if "dataserver" not in items:
        raise RuntimeError("DataServer did not start")
    wait_for(dbus_ready)
    items["metrology"] = spawn("metrology-bridge", [sys.executable,
        str(HERE / "metrology_bridge.py"), "--socket", str(socket),
        "--metrics", str(BASE / "run/metrology.json")])
    save_services(items)
    time.sleep(.3)
    if identity(items["metrology"]["pid"]) != items["metrology"]["start"]:
        raise RuntimeError("Metrology bridge failed")


def feature(key, enabled):
    aid, name = AGENTS[key]
    pids = agents()[key]
    if len(pids) != 1:
        raise RuntimeError(f"Expected one {name} process; found {pids}")
    offsets = [0x11000, 0x12000, 0x14000]
    if enabled:
        offsets.insert(0, 0x10000)
    values = [f"{int(aid, 16) + offset:08x}" for offset in offsets]
    path = TMP / f"agent/{aid}/{aid}.cmd"
    path.write_text("start\n" + "\n".join(values) + "\n")
    os.kill(pids[0], signal.SIGUSR1)


def start():
    start_services()
    if container_state() != "RUNNING":
        run(["lxc-start", "-P", str(LXC), "-n", NAME, "-o", str(BASE / "logs/lxc.log")])
    for key, (aid, name) in AGENTS.items():
        if agents()[key]:
            continue
        logfile = TMP / f"agent/{aid}/{aid}_log"
        before = logfile.stat().st_size if logfile.exists() else 0
        inode = logfile.stat().st_ino if logfile.exists() else None
        lab_environment = (["--set-var", "METER_LAB_DIR=/meter-lab"]
                           if os.environ.get("DI_METER_LAB_REPLAY") == "1" else [])
        if os.environ.get("DI_METER_LAB_METROLOGY") == "1":
            lab_environment = ["--set-var", "METER_LAB_METROLOGY_DIR=/meter-lab"]
        preload = "/usr/lib/libdi_emulator_config_compat.so"
        if lab_environment:
            preload = "/usr/lib/libdi_lab_delivery.so"
        spawn(name, ["lxc-attach", "-P", str(LXC), "-n", NAME,
                     "--set-var", "LD_LIBRARY_PATH=/usr/lib/DIAgent-4.1.3",
                     "--set-var", "LD_PRELOAD=" + preload,
                     *lab_environment,
                     "--", f"/usr/bin/{aid}/{name}_Daemon"])
        def initialized():
            if not logfile.exists():
                return False
            current = logfile.stat()
            with logfile.open() as source:
                source.seek(before if current.st_ino == inode and current.st_size >= before else 0)
                return "Initialize() returned=1" in source.read()
        wait_for(initialized, seconds=120 if lab_environment else 20)
        feature(key, True)


def stop():
    shutdown_failure = None
    processes = [pid for group in agents().values() for pid in group]
    for pid in processes:
        os.kill(pid, signal.SIGTERM)
    try:
        wait_for(lambda: not any(agents().values()), seconds=30)
    except RuntimeError as error:
        if NAME != "di-meter-lab":
            raise
        shutdown_failure = error
    if container_state() == "RUNNING":
        # The vendor BusyBox init restarts on SIGTERM in WSL. All agents have
        # already finished their normal shutdown and checkpointed above.
        run(["lxc-stop", "-P", str(LXC), "-n", NAME, "-k"], timeout=15)
    for name in ["metrology", "dataserver", "bus"]:
        item = services().get(name)
        if item:
            os.kill(item["pid"], signal.SIGTERM)
            def gone():
                if identity(item["pid"]) != item["start"]:
                    return True
                try:
                    return Path(f'/proc/{item["pid"]}/stat').read_text().rsplit(")", 1)[1].split()[0] == "Z"
                except OSError:
                    return True
            wait_for(gone)
    save_services({})
    if shutdown_failure:
        raise RuntimeError("Lab agent did not shut down orderly; isolated container was terminated") from shutdown_failure


def status():
    limits = {}
    cg = Path("/sys/fs/cgroup/lxc.payload." + NAME)
    for key in ["memory.max", "memory.current", "memory.events", "cpu.max", "pids.max", "pids.current"]:
        if (cg / key).exists():
            limits[key] = (cg / key).read_text().strip()
    rows = {}
    if DATABASE.exists():
        with sqlite3.connect(DATABASE) as db:
            for key, (aid, _) in AGENTS.items():
                rows[key] = {table: db.execute(f"select count(*) from {table} where AgentId=?",
                    (int(aid, 16),)).fetchone()[0] for table in ["AgentData", "AgentEvents"]}
    metrology = BASE / "run/metrology.json"
    return {"container": NAME, "state": container_state(), "agents": agents(),
            "services": services(), "cgroup": limits, "stored_rows": rows,
            "metrology": json.loads(metrology.read_text()) if metrology.exists() else {},
            "execution": "ARM foreground emulator builds; production packages preserved",
            "database": str(DATABASE), "input": str(BASE / "metsimfiles/PV.csv"),
            "fault_acquisition": "ARM input adapter absent; startup/lifecycle only",
            "pv_coordinates": "Use copied feature XML to configure actual coordinates"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "enable", "disable", "_dataserver"])
    parser.add_argument("agent", nargs="?", choices=list(AGENTS))
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run with sudo in Ubuntu-22.04, or wsl.exe -d Ubuntu-22.04 -u root")
    if args.command == "_dataserver":
        namespace_server()
    with (BASE / "run/control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command in ["stop", "restart"]:
            stop()
        if args.command in ["start", "restart"]:
            start()
        if args.command in ["enable", "disable"]:
            if not args.agent:
                parser.error("Specify pv or fault")
            feature(args.agent, args.command == "enable")
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
