"""Compare native C++ output with frozen Python results; runs with WSL stdlib."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def compare(expected, actual, path="result"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in ("tensor_channels",):
                continue
            if key not in actual:
                raise AssertionError(f"{path}.{key}: missing")
            compare(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{index}]")
    elif (
        isinstance(expected, bool)
        or expected is None
        or isinstance(expected, str)
        or isinstance(expected, int)
    ):
        if type(expected) is not type(actual) or expected != actual:
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")
    else:
        absolute, relative = (
            (1e-5, 2e-5)
            if ".X[" in path
            else (1e-6, 1e-6)
            if "distance" in path or "cycle_min_km" in path or "cycle_max_km" in path
            else (1e-8, 1e-7)
        )
        if not isinstance(actual, (float, int)) or not math.isclose(
            expected, actual, abs_tol=absolute, rel_tol=relative
        ):
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def main():
    import resource  # CLI resource accounting is Linux-only; comparison is portable.
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    results = []
    for case in manifest["cases"]:
        start = time.monotonic()
        name = case["name"]
        wave = args.fixtures / f"{name}.wave"
        assert hashlib.sha256(wave.read_bytes()).hexdigest() == case["wave_sha256"], (
            f"{name}: fixture changed"
        )
        command = [str(args.binary), str(wave), "--distance" if case["distance"] else "--diagnostics"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        try:
            if case["expected_error"]:
                assert completed.returncode != 0, f"{name}: invalid input accepted"
            else:
                assert completed.returncode == 0, completed.stderr
                expected = json.loads((args.fixtures / f"{name}.json").read_text())
                actual = json.loads(completed.stdout)
                compare(expected, actual)
                if name in ["1A_val1", "gap", "missing", "manual_balanced_2500"]:
                    for chunk in [1, 17, 401]:
                        rerun = subprocess.run(
                            command + ["--chunk", str(chunk)], capture_output=True, text=True, check=True
                        )
                        assert json.loads(rerun.stdout) == actual, f"{name}: chunk-dependent results"
            results.append({"name": name, "status": "PASS", "seconds": time.monotonic() - start})
        except Exception as error:
            results.append({"name": name, "status": "FAIL", "error": str(error)})
            print(name, str(error), flush=True)
    report = {
        "cases": len(results),
        "passed": sum(row["status"] == "PASS" for row in results),
        "child_peak_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "results": results,
        "pipeline_version": manifest["pipeline_version"],
        "source_checksums": manifest["source_checksums"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Parity: {report['passed']}/{report['cases']}; report: {args.report}")
    raise SystemExit(0 if report["passed"] == report["cases"] else 1)


if __name__ == "__main__":
    main()
