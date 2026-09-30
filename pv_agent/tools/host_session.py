"""Project-local DI host lifecycle. Uses a private D-Bus and existing SDK database."""

import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import tarfile
import time
import uuid
import xml.etree.ElementTree as ET

AGENT = 0x23020001
FEATURE = 0x23030001
FEATURES = [0x23030001, 0x23031001, 0x23032001, 0x23033001, 0x23034001]
DATABASE = Path("/usr/local/bin/database/flash/muse01.db")
PROJECT = Path("/root/DI-SDK-3.0.0/DistributedIntelligenceSDK/Agents/PVDetectionAgent")


def metadata(wave):
    values = {}
    for line in Path(wave).read_text().splitlines()[1:]:
        if line == "DATA":
            break
        key, value = line.split("=", 1)
        if key not in ():
            values[key] = value
    return values


class Session:
    def __init__(self, output, project=PROJECT):
        self.project = project.resolve()
        self.sdk = self.project.parents[1]
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.spool = Path("/tmp/agent/23020001/replay") / uuid.uuid4().hex
        self.spool.mkdir(parents=True)
        self.bus = None
        self.agent = None
        self.server_pids = []
        self.files = []
        self.peak_rss = 0
        self.cpu_seconds = 0
        self.lock = open("/var/lock/pv-detection-host.lock", "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = {
            **os.environ,
            "LD_LIBRARY_PATH": str(self.sdk / "Packages/libs"),
            "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=" + str(self.spool / "bus.sock"),
            "PV_REPLAY_DIR": str(self.spool),
            "PV_STATE_DIR": str(self.output / "state"),
            "PV_TEST_METROLOGY": "1",
        }

    def log(self, name):
        handle = (self.output / name).open("w")
        self.files.append(handle)
        return handle

    def sample_resources(self):
        if not self.agent:
            return
        try:
            for line in Path(f"/proc/{self.agent.pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    self.peak_rss = max(self.peak_rss, int(line.split()[1]))
            fields = Path(f"/proc/{self.agent.pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            scheduled = 0
            for thread in Path(f"/proc/{self.agent.pid}/task").iterdir():
                try:
                    scheduled += int((thread / "schedstat").read_text().split()[0]) / 1e9
                except (FileNotFoundError, ProcessLookupError):
                    pass
            self.cpu_seconds = max(self.cpu_seconds, cpu, scheduled)
        except FileNotFoundError:
            pass

    def wait(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.sample_resources()
            value = predicate()
            if value:
                return value
            if self.agent and self.agent.poll() is not None:
                raise RuntimeError("Agent exited early: " + str(self.agent.returncode))
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for DI host; inspect " + str(self.output))

    def _agent_processes(self):
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                executable = (path / "exe").readlink().name
                if executable == "DataServer" or executable.endswith("_Daemon"):
                    yield int(path.name), executable
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass

    def start(self, parameters=None, enabled=True):
        if os.geteuid() != 0:
            raise RuntimeError("SDK host installation requires root; invoke with sudo or wsl -u root")
        existing = list(self._agent_processes())
        if existing:
            raise RuntimeError("Stop the existing SDK host session before replay: " + str(existing))
        package = self.sdk / "host_build/PVDetectionAgent/23020001.tar"
        with tarfile.open(package) as archive:
            for item in archive.getmembers():
                if ".." in Path(item.name).parts:
                    raise RuntimeError("Unexpected archive path")
            archive.extractall("/")
        (self.output / "state").mkdir(exist_ok=True)
        self._configure_db(parameters or {}, enabled)
        with sqlite3.connect(DATABASE) as connection:
            self.before_event = connection.execute("select coalesce(max(Id),0) from AgentEvents").fetchone()[
                0
            ]
            self.before_data = connection.execute("select coalesce(max(Id),0) from AgentData").fetchone()[0]
        busconfig = self.spool / "bus.conf"
        busconfig.write_text(
            "<busconfig><type>system</type><listen>"
            + self.env["DBUS_SYSTEM_BUS_ADDRESS"]
            + '</listen><policy context="default"><allow user="*"/><allow own="*"/><allow send_destination="*"/><allow receive_sender="*"/></policy></busconfig>'
        )
        self.bus = subprocess.Popen(
            ["dbus-daemon", "--nofork", "--config-file=" + str(busconfig)],
            stdout=self.log("dbus.log"),
            stderr=subprocess.STDOUT,
        )
        self.wait(lambda: (self.spool / "bus.sock").exists())
        server_log = self.log("dataserver.log")
        subprocess.run(
            ["/usr/local/bin/DataServer"],
            env=self.env,
            cwd="/usr/local/bin",
            stdout=server_log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=15,
        )
        self.wait(lambda: list(self._agent_processes()))
        self.server_pids = [pid for pid, name in self._agent_processes() if name == "DataServer"]
        binary = Path("/usr/bin/23020001/PVDetectionAgent_Daemon")
        self.agent = subprocess.Popen(
            [str(binary)], env=self.env, cwd="/", stdout=self.log("agent.log"), stderr=subprocess.STDOUT
        )
        self.wait(lambda: self.configuration(), timeout=30)
        commands = Path("/tmp/agent/23020001/23020001.cmd")
        commands.parent.mkdir(parents=True, exist_ok=True)
        commands.write_text("start\n" + "\n".join(f"{x:08x}" for x in FEATURES) + "\n")
        self.agent.send_signal(signal.SIGUSR1)
        return self

    def _configure_db(self, parameters, enabled):
        self.configuration_xml = None
        with sqlite3.connect(DATABASE, timeout=10) as connection:
            for feature in FEATURES:
                path = Path(f"/etc/agents.d/23020001.{feature:08x}.xml")
                if not path.exists():
                    continue
                tree = ET.parse(path)
                if feature == FEATURE:
                    for param in tree.getroot().iter("parameter"):
                        key = param.get("name")
                        if key == "FeatureState":
                            param.set("value", "1" if enabled else "0")
                        elif key in parameters:
                            param.set("value", str(parameters[key]))
                    self.configuration_xml = ET.tostring(tree.getroot(), encoding="utf-8")
                xml = ET.tostring(tree.getroot(), encoding="utf-8")
                connection.execute(
                    "insert or replace into FeatureConfiguration (FeatureId,ConfigurationId,Level,Data,LastUpdateTime) values (?,?,?,?,?)",
                    (feature, 0, 5, gzip.compress(xml), int(time.time())),
                )

    def configuration(self):
        path = self.spool / "configuration.json"
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def update(self, parameters=None, enabled=True):
        tree = ET.fromstring(self.configuration_xml)
        for param in tree.iter("parameter"):
            key = param.get("name")
            if key == "FeatureState":
                param.set("value", "1" if enabled else "0")
            elif key in (parameters or {}):
                param.set("value", str(parameters[key]))
        self.configuration_xml = ET.tostring(tree, encoding="utf-8")
        state = self.spool / "configuration.json"
        if state.exists():
            state.unlink()
        pending = self.spool / "configuration.pending.xml"
        pending.write_bytes(self.configuration_xml)
        pending.rename(self.spool / "configuration.request.xml")
        return self.wait(lambda: self.configuration())

    def command(self, enabled=True):
        path = self.spool / "lifecycle.json"
        if path.exists():
            path.unlink()
        features = FEATURES if enabled else [f for f in FEATURES if f != FEATURE]
        Path("/tmp/agent/23020001/23020001.cmd").write_text(
            "start\n" + "\n".join(f"{f:08x}" for f in features) + "\n"
        )
        self.agent.send_signal(signal.SIGUSR1)

        def ready():
            try:
                return json.loads(path.read_text())["running"] == enabled
            except (FileNotFoundError, json.JSONDecodeError):
                return False

        self.wait(ready)

    def submit(self, wave):
        identifier = uuid.uuid4().hex
        pending = self.spool / (identifier + ".tmp")
        pending.write_text(str(Path(wave).resolve()) + "\n")
        pending.rename(self.spool / (identifier + ".request"))
        return identifier

    def result(self, identifier, timeout=30):
        path = self.spool / (identifier + ".done.json")
        return self.wait(lambda: json.loads(path.read_text()) if path.exists() else None, timeout)

    def rows(self):
        with sqlite3.connect(DATABASE, timeout=10) as connection:
            events = connection.execute(
                "select Id,FeatureId,EventId,cast(Data as text),isUnsentAlarm,TimeStamp from AgentEvents where Id>? and AgentId=? and FeatureId=? and cast(Data as text) like '98#PV#1#%' order by Id",
                (self.before_event, AGENT, FEATURE),
            ).fetchall()
            data = connection.execute(
                "select Id,FeatureId,cast(Data as text),TimeStamp from AgentData where Id>? and AgentId=? and FeatureId=? and cast(Data as text) like 'PV%1:%' order by Id",
                (self.before_data, AGENT, FEATURE),
            ).fetchall()
        return {"events": events, "data": data}

    def stop(self):
        self.sample_resources()
        log = Path("/tmp/agent/23020001/23020001_log")
        if log.exists():
            (self.output / "sdk-agent.log").write_bytes(log.read_bytes())
        if self.agent and self.agent.poll() is None:
            self.agent.send_signal(signal.SIGTERM)
            try:
                self.agent.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.agent.kill()
                self.agent.wait()
                raise RuntimeError("Agent failed orderly shutdown")
        if self.agent and self.agent.returncode != 0:
            raise RuntimeError("Agent exit code " + str(self.agent.returncode))

    def close(self):
        try:
            self.stop()
        finally:
            for pid in self.server_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _ in range(30):
                alive = []
                for pid in self.server_pids:
                    try:
                        if Path(f"/proc/{pid}/exe").readlink().name == "DataServer":
                            alive.append(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                if not alive:
                    break
                time.sleep(0.1)
            for pid in alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if self.bus and self.bus.poll() is None:
                self.bus.terminate()
                self.bus.wait(timeout=5)
            for handle in self.files:
                handle.close()
            self.lock.close()
