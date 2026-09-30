"""Freeze Python reference outputs without altering DI Validator datasets."""

from __future__ import annotations
import argparse
import hashlib
import json
import importlib.metadata
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The standalone exporter imports the unchanged workbench through its workspace root.
from di_validator.faults.comtrade_io import read_pair  # noqa: E402
from di_validator.faults.pipeline import FaultConfig, PHASES, VERSION, analyze_segment  # noqa: E402
from di_validator.faults.numerics import estimate_impedance_distance  # noqa: E402
from di_validator.faults.synthetic import fault_frame  # noqa: E402


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def reference(frame, fs, f0, cfg):
    frame = frame[
        (frame.offset >= cfg.start)
        & (frame.offset <= (cfg.end if cfg.end is not None else frame.offset.iloc[-1]))
    ]
    times = frame.offset.to_numpy()
    if np.any(np.diff(times) <= 0):
        raise ValueError("unordered timestamps")
    if np.any(np.abs(np.diff(times) * fs - np.rint(np.diff(times) * fs)) > 0.01):
        raise ValueError("sampling grid")
    finite = np.isfinite(frame[list(PHASES)].to_numpy()).all(axis=1)
    breaks = np.flatnonzero(np.diff(times) > 1.5 / fs) + 1
    cuts = sorted(
        set(
            [
                0,
                len(frame),
                *breaks.tolist(),
                *np.flatnonzero(~finite).tolist(),
                *(np.flatnonzero(~finite) + 1).tolist(),
            ]
        )
    )
    gaps = [(times[i - 1] + 1 / fs, times[i]) for i in breaks] + [
        (times[i], times[i] + 1 / fs) for i in np.flatnonzero(~finite)
    ]
    if cfg.manual_onset is not None and (
        any(a <= cfg.manual_onset < b for a, b in gaps) or not times[0] <= cfg.manual_onset <= times[-1]
    ):
        raise ValueError("manual onset in gap")
    analyses = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == b or not finite[a]:
            continue
        info, arrays = analyze_segment(
            frame.iloc[a:b], {"sample_rate": fs, "config": {"nominal_frequency": f0}}, cfg, len(analyses)
        )
        if arrays is not None:
            info["diagnostics"] = {
                k: arrays[k]
                for k in ["filtered_ka_kv", "sequence_peak_ka_kv", "current_ratio", "voltage_ratio", "X"]
            }
        analyses.append(info)
    return {"segments": analyses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runtime/di-agent/fixtures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []

    def save(name, frame, fs=2000.0, f0=50.0, parameters=None, distance=False, error=False):
        parameters = parameters or {}
        cfg = FaultConfig(channel_map={p: p for p in PHASES}, **parameters)
        config = cfg.model_dump(exclude={"channel_map", "dataset_checksum", "pipeline_checksum"})
        source_hash = hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
        lines = ["FLA1", f"fs={fs:.17g}", f"f0={f0:.17g}", f"source_id={name}"]
        for key, value in config.items():
            if value is None:
                value = -1
            if isinstance(value, bool):
                value = int(value)
            lines.append(f"{key}={value}")
        lines += ["DATA", "offset_seconds,IA_A,IB_A,IC_A,UA_V,UB_V,UC_V"]
        for row in frame[["offset", *PHASES]].itertuples(index=False, name=None):
            lines.append(",".join(format(v, ".17g") for v in row))
        wave = args.out / f"{name}.wave"
        wave.write_text("\n".join(lines) + "\n", encoding="ascii")
        try:
            if error:
                raise ValueError("invalid input fixture")
            if distance:
                values = frame[list(PHASES)].to_numpy().T / 1000
                expected = estimate_impedance_distance(
                    values[:3],
                    values[3:],
                    fs,
                    f0,
                    0,
                    len(frame),
                    cfg.line_params(),
                    cfg.fault_loop,
                    cfg.settle_cycles,
                    cfg.average_cycles,
                    cfg.min_current_ka,
                    cfg.fault_current_fraction,
                    cfg.max_cycle_spread,
                )
            else:
                expected = reference(frame, fs, f0, cfg)
        except ValueError as exc:
            expected = {"expected_error": str(exc)}
        (args.out / f"{name}.json").write_text(
            json.dumps(expected, default=encode, allow_nan=False, separators=(",", ":")) + "\n"
        )
        cases.append(
            {
                "name": name,
                "distance": distance,
                "expected_error": "expected_error" in expected,
                "source_sha256": source_hash,
                "wave_sha256": hashlib.sha256(wave.read_bytes()).hexdigest(),
                "configuration": config,
            }
        )

    for path in sorted((ROOT / "fault-distance/data/data_test").glob("*.cfg")):
        rec, _ = read_pair(path, path.with_suffix(".dat"))
        values = np.asarray(rec.analog, dtype=float).T.copy()
        for index, channel in enumerate(rec.cfg.analog_channels):
            if channel.uu.strip() not in ("A", "kA", "V", "kV") or float(channel.skew) != 0:
                raise ValueError("Unsupported units or channel skew")
            if channel.uu.strip().startswith("k"):
                values[:, index] *= 1000
        frame = pd.DataFrame(values, columns=PHASES)
        frame.insert(0, "offset", np.asarray(rec.time) - rec.time[0])
        save(path.stem, frame, float(rec.cfg.sample_rates[0][0]), float(rec.cfg.frequency))
        cases[-1]["original_cfg_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[-1]["original_dat_sha256"] = hashlib.sha256(path.with_suffix(".dat").read_bytes()).hexdigest()
        if path.stem == "1A_val1":
            example = frame.copy()
    save("manual", example, parameters={"manual_onset": 0.0995})
    save("selected_window", example, parameters={"start": 0.025, "end": 0.18})
    save("known_distance", example, parameters={"known_distance_km": 1.0})
    save("unavailable_distance", example, parameters={"settle_cycles": 20})
    save("low_current", example, parameters={"min_current_ka": 1e9})
    save("ground_compensated", example, parameters={"r0_ohm_km": 0.3, "x0_ohm_km": 1.2})
    save("short", example.iloc[:20])
    save("gap", example.drop(index=[198, 199, 200]).reset_index(drop=True))
    gap = example.copy()
    gap.loc[198:200, "IA"] = np.nan
    save("missing", gap)
    save("manual_gap", gap, parameters={"manual_onset": 0.0995})
    for fs, f0 in [(2000.0, 50.0), (2400.0, 60.0), (2500.0, 60.0)]:
        times = np.arange(400) / fs
        values = np.array(
            [
                np.sin(2 * np.pi * f0 * times - phase * 2 * np.pi / 3) * scale
                for scale in (100.0, 10000.0)
                for phase in range(3)
            ]
        )
        frame = pd.DataFrame(values.T, columns=PHASES)
        frame.insert(0, "offset", times)
        save(f"no_fault_{int(fs)}", frame, fs, f0)
        save(f"manual_balanced_{int(fs)}", frame, fs, f0, {"manual_onset": 0.06})
    for loop in ["AG", "BG", "CG", "AB", "BC", "CA", "POS"]:
        fs, f0, distance = 2400.0, 60.0, 12.3
        currents = np.array([2 * np.exp(0.2j), 1.5 * np.exp(-2.1j), 1.1 * np.exp(2j)])
        z1, z0 = 0.1 + 0.4j, 0.3 + 1.2j
        voltages = currents * z1 * distance
        if loop.endswith("G"):
            phase = "ABC".index(loop[0])
            voltages[phase] = (currents[phase] + (z0 - z1) / (3 * z1) * currents.sum()) * z1 * distance
        t = np.arange(240) / fs
        waveform = np.sqrt(2) * np.real(np.r_[currents, voltages][:, None] * np.exp(2j * np.pi * f0 * t)) + 3
        frame = pd.DataFrame(waveform.T * 1000, columns=PHASES)
        frame.insert(0, "offset", t)
        save(
            "physics_" + loop,
            frame,
            fs,
            f0,
            {"fault_loop": loop, "r0_ohm_km": 0.3, "x0_ohm_km": 1.2},
            distance=True,
        )
    # Feeder-scale faults: 4 samples/cycle relay reports use the fault-current peak;
    # 2400 Hz faults that clear exercise the decayed-cycle guard; gates withhold the rest.
    feeder = dict(line_length_km=5, nominal_voltage_kv=13.2, base_power_mva=10, r1_ohm_km=0.2, x1_ohm_km=0.3)
    grounded = dict(feeder, r0_ohm_km=0.6, x0_ohm_km=0.9)
    z1, z0 = 0.2 + 0.3j, 0.6 + 0.9j
    for loop in ["AB", "BG"]:
        save(f"relay_4spc_{loop}", fault_frame(240, 60, loop, 1.5, z1, z0, fault_seconds=2 / 60), 240.0, 60.0, grounded)
    save("relay_4spc_POS", fault_frame(240, 60, "POS", 1.5, z1, z0, fault_seconds=2 / 60), 240.0, 60.0, dict(grounded, fault_loop="POS"))
    save("relay_4spc_beyond", fault_frame(240, 60, "AB", 7.5, z1, fault_seconds=2 / 60), 240.0, 60.0, feeder)
    save("relay_4spc_behind", fault_frame(240, 60, "AB", -1.0, z1, fault_seconds=2 / 60), 240.0, 60.0, feeder)
    save("cleared_3_cycles", fault_frame(2400, 60, "AB", 1.5, z1, fault_seconds=3 / 60), 2400.0, 60.0, feeder)
    save("cleared_1_cycle", fault_frame(2400, 60, "AB", 1.5, z1, fault_seconds=1 / 60), 2400.0, 60.0, feeder)
    save(
        "decay_guard_off",
        fault_frame(2400, 60, "AB", 1.5, z1, fault_seconds=3 / 60),
        2400.0,
        60.0,
        dict(feeder, fault_current_fraction=0),
    )
    bad = example.copy()
    bad.loc[10, "offset"] = bad.offset.iloc[9]
    save("unordered", bad, error=True)
    bad = example.copy()
    bad.loc[10, "offset"] += 0.0001
    save("off_grid", bad, error=True)
    save("invalid_rate", example, fs=100, f0=50, error=True)
    # Invalid metadata must be rejected by the native boundary before processing.
    for name, old, new in [
        ("partial_zero_sequence", "r0_ohm_km=-1", "r0_ohm_km=0.3"),
        ("invalid_loop", "fault_loop=AUTO", "fault_loop=ZZ"),
        ("invalid_units", "IA_A", "IA_RMS"),
    ]:
        source = (args.out / "1A_val1.wave").read_text()
        destination = args.out / f"{name}.wave"
        destination.write_text(source.replace(old, new))
        (args.out / f"{name}.json").write_text('{"expected_error":"invalid input"}')
        cases.append(
            {
                "name": name,
                "distance": False,
                "expected_error": True,
                "wave_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "di_validator/faults").glob("*.py"))
    }
    manifest = {
        "pipeline_version": VERSION,
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "comtrade", "pydantic"]
        },
        "source_checksums": sources,
        "cases": cases,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(cases)} frozen cases, including 99 COMTRADE recordings, to {args.out}")


if __name__ == "__main__":
    main()
