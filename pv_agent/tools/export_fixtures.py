"""Freeze paired synthetic load/PV truth and the independent Python oracle."""

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from reference import DEFAULTS, analyze, solar

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runtime/pv-agent/fixtures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    base = int(dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc).timestamp())

    def save(name, runs, config=None, truth=None):
        cfg = {**DEFAULTS, "latitude": 34.0, "longitude": -118.0, **(config or {})}
        lines = (
            ["PVR1"]
            + [f"{k}={v}" for k, v in cfg.items()]
            + ["DATA", "utc_start,seconds,watts,vars,p_valid,q_valid,time_valid"]
        )
        lines.extend(",".join(format(v, ".17g") for v in row) for row in runs)
        wave = args.out / (name + ".pvr")
        wave.write_text("\n".join(lines) + "\n", encoding="ascii")
        result = analyze(runs, cfg)
        expected = args.out / (name + ".json")
        expected.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        available = sum(r["available"] for r in result["intervals"])
        entry = dict(
            name=name,
            input_sha256=hashlib.sha256(wave.read_bytes()).hexdigest(),
            expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
            config=cfg,
            available_intervals=available,
            final_state=result["detection"]["state"],
        )
        if truth is not None:
            (args.out / (name + ".truth.json")).write_text(json.dumps(truth, separators=(",", ":")) + "\n")
        cases.append(entry)
        print(name, entry["final_state"], available, flush=True)

    baseline = None
    for name in [
        "pv_export",
        "pv_self_consumed",
        "no_pv",
        "cloudy",
        "switching_ev",
        "midday_load_drop",
        "night_model_bad",
        "q_rank_deficient",
        "q_out_of_range",
        "missing_reactive",
        "battery",
        "generator",
        "night_export",
        "reversed_polarity",
        "partial_coverage",
        "outage",
        "missing_location",
        "winter",
        "southern_hemisphere",
        "learning_29_days",
    ]:
        config = {}
        lat, lon, start = 34.0, -118.0, base
        if name == "winter":
            start = int(dt.datetime(2025, 12, 1, tzinfo=dt.timezone.utc).timestamp())
        if name == "southern_hemisphere":
            lat, lon = -33.86, 151.21
            config.update(latitude=lat, longitude=lon)
        if name == "battery":
            config["storage"] = 1
        if name == "generator":
            config["other_generation"] = 1
        if name == "reversed_polarity":
            config["polarity"] = -1
        if name == "missing_location":
            config.update(latitude=999, longitude=999)
        runs, truth = [], []
        days = 29 if name == "learning_29_days" else 35
        for i in range(days * 96):
            t = start + i * 900
            elevation, hour = solar(t + 450, lat, lon)
            q = 0.5 + 0.25 * math.sin(i * 1.73) + 0.08 * math.cos(i * 0.17)
            if name == "pv_self_consumed":
                q = max(0.3, q)
            load = 1.2 + 2 * q
            pv = (
                max(0.0, math.sin(math.pi * (hour - 6) / 12)) ** 2
                * (1.75 if name == "pv_self_consumed" else 4.8)
                if 6 < hour < 18
                else 0.0
            )
            if elevation <= 0:
                pv = 0.0
            if name == "no_pv":
                pv = 0.0
            if name == "cloudy":
                pv *= 0.45 + 0.45 * abs(math.sin(i * 0.19))
            if name == "switching_ev" and (hour >= 20 or hour < 3):
                load += 2.4 if i % 4 < 2 else 0
            if name == "midday_load_drop":
                pv = 0.0
                if 10 < hour < 15:
                    load *= 0.35
            if name == "night_model_bad" and elevation < -6:
                load = 1.5 + abs(math.sin(i * 0.31)) * 3
            if name == "q_rank_deficient":
                q = 0.5
            if name == "q_out_of_range" and elevation > 10:
                q = 3.0
            p = load - pv
            if name == "night_export" and elevation < -6:
                p = -1.0
            if name == "reversed_polarity":
                p = -p
            qvalid = int(not (name == "missing_reactive" and i >= 32 * 96))
            n = 900
            if name == "partial_coverage" and i >= 32 * 96:
                n = 810 if i % 2 else 809
            if name == "outage" and 12 * 96 <= i < 15 * 96:
                continue
            runs.append((t, n, p * 1000, q * 1000, 1, qvalid, 1))
            if n != 900:
                runs.append((t + n, 900 - n, 0.0, 0.0, 0, 0, 1))
            truth.append(dict(start=t, native_load_kw=load, pv_kw=pv, synthetic=True))
        save(name, runs, config, truth)
        if name == "pv_export":
            baseline = runs
    save("sign_changes", [(base, 450, 1000.0, 300.0, 1, 1, 1), (base + 450, 450, -1000.0, 300.0, 1, 1, 1)])
    save("timestamp_order", [(base, 450, 1000.0, 300.0, 1, 1, 1), (base + 449, 451, -1000.0, 300.0, 1, 1, 1)])
    save("invalid_time", [(0, 900, 1000.0, 300.0, 1, 1, 0)])
    save("gap", [(base, 400, 1000.0, 300.0, 1, 1, 1), (base + 500, 400, -1000.0, 300.0, 1, 1, 1)])
    save("absent_p", [(base, 900, 1000.0, 300.0, 0, 1, 1)])
    # A compact explicit 32-day history supports SDK restore/replay acceptance.
    save("warm_state", baseline[: 32 * 96])
    save("warm_tail", baseline[32 * 96 : 33 * 96])
    # Exact threshold ties must not flip classifications due to accumulation order.
    for name in ["threshold_median_equal", "threshold_profile_equal", "threshold_export_equal"]:
        runs = []
        for i in range(35 * 96):
            t = base + 900 * i
            elevation, hour = solar(t + 450, 0, 0)
            power = 1.0
            if name == "threshold_export_equal":
                power = -0.05
            elif 10 <= hour < 16:
                power = 0.55 if name == "threshold_profile_equal" else (0.54 if (i // 96) % 10 < 4 else 0.8)
            runs.append((t, 900, power * 1000, 300.0, 1, 1, 1))
        save(name, runs, {"latitude": 0.0, "longitude": 0.0})
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name("reference.py")]
    }
    manifest = dict(
        algorithm="pv-meter-only-1",
        provenance="synthetic paired load/PV; not field ground truth",
        python=sys.version,
        numpy=np.__version__,
        source_sha256=sources,
        cases=cases,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
