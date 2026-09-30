"""Measure the actual host process and bounded history; host results are not ARM qualification."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import time
from host_session import Session, metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lines = (args.fixtures / "pv_export.pvr").read_text().splitlines()
    split = lines.index("DATA") + 2
    source = lines[split:]
    rows = []
    for day in range(65):
        for line in source[(day % 35) * 96 : (day % 35 + 1) * 96]:
            fields = line.split(",")
            fields[0] = str(int(fields[0]) + (day // 35) * 35 * 86400)
            rows.append(",".join(fields))
    long_wave = args.output / "65days.pvr"
    long_wave.write_text("\n".join(lines[:split] + rows) + "\n")
    began = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    state = args.output / "60days.pvs"
    with (args.output / "65days.json").open("w") as output:
        subprocess.run(
            [
                "/usr/bin/time",
                "-f",
                '{"peak_rss_kb":%M}',
                "-o",
                str(args.output / "portable-time.json"),
                str(args.binary),
                str(long_wave),
                "--state-out",
                str(state),
            ],
            stdout=output,
            check=True,
        )
    elapsed = time.monotonic() - began
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    portable = dict(
        history_calendar_days=60,
        replay_days=65,
        wall_seconds=elapsed,
        cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        peak_rss_kb=json.loads((args.output / "portable-time.json").read_text())["peak_rss_kb"],
        checkpoint_bytes=state.stat().st_size,
    )
    # Idling with an active SDK subscription captures the framework/worker cost.
    session = Session(args.output / "idle")
    try:
        session.start(metadata(args.fixtures / "warm_tail.pvr"))
        session.wait(lambda: (session.spool / "lifecycle.json").exists())
        session.sample_resources()
        start_cpu = session.cpu_seconds
        began = time.monotonic()
        while time.monotonic() - began < 15:
            time.sleep(0.1)
            session.sample_resources()
        elapsed = time.monotonic() - began
        cpu = session.cpu_seconds - start_cpu
        idle = dict(
            wall_seconds=elapsed,
            cpu_seconds=cpu,
            one_core_cpu_percent=100 * cpu / elapsed,
            peak_rss_kb=session.peak_rss,
            threads=len(list(Path(f"/proc/{session.agent.pid}/task").iterdir())),
            open_files=len(list(Path(f"/proc/{session.agent.pid}/fd").iterdir())),
        )
    finally:
        session.close()
    report = dict(
        portable=portable,
        host_idle=idle,
        agent_policy=dict(cpu_percent=2, ram_kb=2048, flash_kb=2048),
        host_ram_policy_pass=idle["peak_rss_kb"] <= 2048,
        host_idle_cpu_policy_pass=idle["one_core_cpu_percent"] <= 2,
        meter_resource_qualified=False,
        limitations="Host glibc process and accelerated replay; no ARM CPU/RAM measurement. Flash must include package, atomic checkpoint temporary files, SDK logs and filesystem overhead.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
