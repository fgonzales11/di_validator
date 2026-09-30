"""Validate every result against frozen Python calculations, including missing outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def compare(expected, actual, path="result"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path + ": keys"
        for k, v in expected.items():
            compare(v, actual[k], path + "." + k)
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path + ": length"
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, (bool, str)) or expected is None or isinstance(expected, int):
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"
    else:
        assert isinstance(actual, (int, float)) and math.isclose(
            expected, actual, rel_tol=1e-6, abs_tol=1e-6
        ), f"{path}: {expected!r} != {actual!r}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    results = []
    for case in manifest["cases"]:
        start = time.monotonic()
        name = case["name"]
        try:
            wave = args.fixtures / (name + ".pvr")
            oracle = args.fixtures / (name + ".json")
            assert hashlib.sha256(wave.read_bytes()).hexdigest() == case["input_sha256"]
            assert hashlib.sha256(oracle.read_bytes()).hexdigest() == case["expected_sha256"]
            command = [str(args.binary), str(wave)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=90)
            assert run.returncode == 0, run.stderr
            actual = json.loads(run.stdout)
            compare(json.loads(oracle.read_text()), actual)
            if name in ["pv_export", "gap", "sign_changes"]:
                for chunk in [1, 17, 4001]:
                    again = subprocess.run(
                        command + ["--chunk", str(chunk)], capture_output=True, text=True, check=True
                    )
                    assert json.loads(again.stdout) == actual, "chunk-dependent output"
            if name in ["pv_export", "pv_self_consumed"]:
                available = [r for r in actual["intervals"] if r["available"] and r["generation_kw"] > 0]
                assert available, "Estimator never produced positive generation"
            results.append(dict(name=name, status="PASS", seconds=time.monotonic() - start))
        except Exception as e:
            results.append(dict(name=name, status="FAIL", error=str(e)))
            print(name, e, flush=True)
    report = dict(
        cases=len(results),
        passed=sum(r["status"] == "PASS" for r in results),
        results=results,
        atol=1e-6,
        rtol=1e-6,
        oracle_source_sha256=manifest["source_sha256"],
    )
    # Restore actual binary state and continue, comparing against uninterrupted Python results.
    seed = args.fixtures / "warm_state.pvs"
    subprocess.run(
        [str(args.binary), str(args.fixtures / "warm_state.pvr"), "--state-out", str(seed)],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    resumed = subprocess.run(
        [str(args.binary), str(args.fixtures / "warm_tail.pvr"), "--state-in", str(seed)],
        capture_output=True,
        text=True,
        check=True,
    )
    continued = json.loads(resumed.stdout)
    full = json.loads((args.fixtures / "pv_export.json").read_text())
    compare(full["intervals"][32 * 96 : 33 * 96], continued["intervals"])
    assert continued["detection"]["state"] == "likely_pv"
    report["checkpoint_continuation"] = "PASS"
    report["checkpoint_sha256"] = hashlib.sha256(seed.read_bytes()).hexdigest()
    # Source provenance is checked, not simply copied into the report.
    root = Path(__file__).resolve().parents[2]
    for path, expected_hash in manifest["source_sha256"].items():
        assert hashlib.sha256((root / path.replace("\\", "/")).read_bytes()).hexdigest() == expected_hash, path
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Parity {report['passed']}/{report['cases']}")
    raise SystemExit(report["passed"] != report["cases"])


if __name__ == "__main__":
    main()
