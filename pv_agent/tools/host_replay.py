"""Run representative samples through SDK typed LIDs, registration and DataServer."""

import argparse
import json
from pathlib import Path
import shutil
import uuid
from host_session import Session, metadata, PROJECT
from decode import decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wave", nargs="?", type=Path, default=PROJECT / "fixtures/warm_tail.pvr")
    p.add_argument("--output", type=Path, default=PROJECT / "validation" / ("replay-" + uuid.uuid4().hex))
    p.add_argument("--seed", type=Path, default=PROJECT / "fixtures/warm_state.pvs")
    p.add_argument("--cold", action="store_true")
    args = p.parse_args()
    session = Session(args.output)
    state = args.output / "state"
    state.mkdir(exist_ok=True)
    if not args.cold and args.seed.exists():
        shutil.copy2(args.seed, state / "analytics.pvs")
    try:
        session.start(metadata(args.wave))
        identifier = session.submit(args.wave)
        result = session.result(identifier, timeout=90)
        assert result["status"] == "complete", result
        assert result["pending"] == 0, "Pending SDK writes remain; inspect rejection status"
        rows = session.rows()
        session.stop()
        assert rows == session.rows(), "Shutdown removed outcomes"
        report = dict(
            result=result,
            **rows,
            decoded=[decode(r[2]) for r in rows["data"]],
            retained_after_shutdown=True,
            peak_rss_kb=session.peak_rss,
            sampled_cpu_seconds=session.cpu_seconds,
            policy_ram_kb=2048,
            within_host_ram_policy=session.peak_rss <= 2048,
            acquisition="SDK subscription plus fixture-fed SDK LidList decoder; no physical meter tested",
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy2(session.spool / (identifier + ".intervals.jsonl"), args.output / "intervals.jsonl")
        print(
            json.dumps({k: v for k, v in report.items() if k not in ["events", "data", "decoded"]}, indent=2)
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
