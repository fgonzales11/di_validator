"""Report estimator behavior against paired synthetic truth, including known confounders."""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for truth_path in sorted(args.fixtures.glob("*.truth.json")):
        name = truth_path.name.removesuffix(".truth.json")
        truth = {r["start"]: r for r in json.loads(truth_path.read_text())}
        expected = json.loads((args.fixtures / (name + ".json")).read_text())
        intervals = expected["intervals"]
        errors = [
            r["generation_kw"] - truth[r["start"]]["pv_kw"]
            for r in intervals
            if r["available"] and r["start"] in truth
        ]
        rows.append(
            dict(
                case=name,
                screening=expected["detection"]["state"],
                available=len(errors),
                positive_estimates=sum(r["available"] and r["generation_kw"] > 1e-6 for r in intervals),
                measured_export_kwh=sum(r["export_kwh"] for r in intervals),
                rmse_kw=math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
                max_absolute_error_kw=max(map(abs, errors)) if errors else None,
            )
        )
    by_name = {r["case"]: r for r in rows}
    for name in ["pv_export", "pv_self_consumed", "cloudy"]:
        assert by_name[name]["positive_estimates"] > 0
        assert by_name[name]["max_absolute_error_kw"] < 1e-6
    assert by_name["pv_self_consumed"]["measured_export_kwh"] == 0
    for name in ["no_pv", "battery", "generator", "night_export", "night_model_bad", "q_rank_deficient"]:
        assert by_name[name]["available"] == 0
    assert by_name["midday_load_drop"]["positive_estimates"] > 0
    report = dict(
        status="PASS",
        cases=rows,
        interpretation="Paired synthetic correctness only. Analytical outputs were checked against C++ at 1e-6 tolerances; these are not field accuracy metrics.",
        known_false_positive="Repeated midday load reduction without PV produces likely_pv and positive estimates despite excellent nighttime fit. The meter-only model cannot resolve this confounder.",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Paired synthetic checks PASS; retained known midday-load false positive")


if __name__ == "__main__":
    main()
