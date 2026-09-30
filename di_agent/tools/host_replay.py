"""Replay one normalized waveform through the actual DI host agent and DataServer."""

import argparse
import json
from pathlib import Path
import uuid
from host_session import Session, metadata, PROJECT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wave", nargs="?", type=Path, default=PROJECT / "fixtures/1A_val1.wave")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "validation" / ("replay-" + uuid.uuid4().hex)
    )
    args = parser.parse_args()
    session = Session(args.output)
    try:
        session.start(metadata(args.wave))
        identifier = session.submit(args.wave)
        result = session.result(identifier)
        assert result["status"] == "complete", result
        rows = session.rows()
        session.stop()
        retained = session.rows()
        assert rows == retained, "Normal shutdown removed outcomes"
        report = {
            "result": result,
            **rows,
            "retained_after_shutdown": True,
            "peak_rss_kb": session.peak_rss,
            "cpu_seconds": session.cpu_seconds,
            "policy_ram_kb": 2048,
            "within_host_ram_policy": session.peak_rss <= 2048,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
