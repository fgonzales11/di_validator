"""Registered inputs, explicit unit mappings, immutable scenario snapshots."""

from pathlib import Path
import io
import json
import math
import re
import numpy as np
import pandas as pd
from .. import store
from ..faults.pipeline import FaultConfig, PHASES, VERSION
from .schemas import Parameters


def root():
    path = store.workspace() / "meter-lab"
    path.mkdir(exist_ok=True)
    return path


def folder(kind, identifier):
    if kind not in {"runs", "scenarios", "uploads"} or not re.fullmatch("[0-9a-f]{32}", identifier):
        raise ValueError("Invalid Meter Lab artifact identity")
    return root() / kind / identifier


def fixture_root(agent):
    return store.workspace() / ("pv-agent" if agent == "pv" else "di-agent") / "fixtures"


def catalog():
    rows = []
    for agent in ["pv", "fault"]:
        path = fixture_root(agent) / "manifest.json"
        if not path.exists():
            continue
        for case in json.loads(path.read_text())["cases"]:
            if case.get("expected_error"):
                continue
            rows.append(
                {
                    "id": case["name"],
                    "agent": agent,
                    "name": case["name"].replace("_", " "),
                    "source": "preset",
                    "synthetic": not case["name"].startswith("1A_val"),
                    "description": "Recorded COMTRADE waveform; example line parameters are unverified"
                    if case["name"].startswith("1A_val")
                    else "Deterministic synthetic observations; model behavior can differ from physical truth",
                    "warning": "Known adverse case: midday load reduction can cause a PV false positive"
                    if case["name"] == "midday_load_drop"
                    else None,
                    "configuration": {
                        k: str(-1 if v is None else int(v) if isinstance(v, bool) else v)
                        for k, v in case.get("config", case.get("configuration", {})).items()
                    },
                    "expected_state": case.get("final_state"),
                }
            )
    rows.append(
        {
            "id": "synthetic_fault",
            "agent": "fault",
            "name": "Configurable fault waveform",
            "source": "preset",
            "synthetic": True,
        }
    )
    rows.append(
        {
            "id": "comtrade_99",
            "agent": "fault",
            "name": "All 99 COMTRADE recordings",
            "source": "preset",
            "synthetic": False,
            "description": "99 distinct recordings, separated by explicit timestamp gaps; one analysis per segment.",
        }
    )
    for d in store.listing("dataset"):
        compatible = d["format"] == "recording" and all(
            any(
                c.get("phase") == p[1] and c["kind"] == ("current" if p[0] == "I" else "voltage")
                for c in d.get("channels", [])
            )
            for p in PHASES
        )
        if compatible:
            rows.append(
                {"id": d["id"], "agent": "fault", "name": d["name"], "source": "dataset", "synthetic": False}
            )
    return rows


def fixture(agent, name):
    known = next(
        (c for c in catalog() if c["agent"] == agent and c["id"] == name and c["source"] == "preset"), None
    )
    if not known or name in {"synthetic_fault", "comtrade_99"}:
        raise ValueError("Select a registered fixture")
    path = fixture_root(agent) / (name + (".pvr" if agent == "pv" else ".wave"))
    header, raw = path.read_text().split("\nDATA\n", 1)
    settings = dict(line.split("=", 1) for line in header.splitlines()[1:])
    values = np.loadtxt(io.StringIO(raw), delimiter=",", skiprows=1, ndmin=2)
    return settings, values, path


def config(agent, values):
    if agent == "fault":
        fields = FaultConfig.model_fields
        clean = {k: v for k, v in values.items() if k in fields}
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError("Unknown fault configuration: " + ", ".join(sorted(unknown)))
        for key in ["end", "manual_onset", "r0_ohm_km", "x0_ohm_km", "known_distance_km"]:
            if key in clean and float(clean[key]) == -1:
                clean[key] = None
        cfg = FaultConfig.model_validate(clean).model_dump(
            exclude={"channel_map", "dataset_checksum", "pipeline_checksum"}
        )
        return {k: str(int(v) if isinstance(v, bool) else -1 if v is None else v) for k, v in cfg.items()}
    from pv_agent.tools.reference import DEFAULTS

    if set(values) - set(DEFAULTS):
        raise ValueError("Unknown PV configuration field")
    cfg = {**DEFAULTS, **{k: float(v) for k, v in values.items()}}
    if not all(math.isfinite(v) for v in cfg.values()):
        raise ValueError("Configuration must be finite")
    lat, lon = cfg["latitude"], cfg["longitude"]
    if not ((lat == lon == 999) or (abs(lat) <= 90 and abs(lon) <= 180)):
        raise ValueError("Provide valid latitude and longitude together")
    if cfg["polarity"] not in [-1, 1] or any(
        not 0 < cfg[k] <= 1e6 for k in ["power_scale", "reactive_scale"]
    ):
        raise ValueError("Invalid polarity/scaling")
    if not 0.9 <= cfg["min_coverage"] <= 1 or not 0 <= cfg["export_threshold_kw"] <= 1e6:
        raise ValueError("Coverage must be 0.9–1; export threshold must be nonnegative")
    for key in ["export_fraction", "valley_threshold", "valley_fraction", "valley_median", "max_nrmse"]:
        if not 0 < cfg[key] <= 1:
            raise ValueError(key + " must be greater than zero and at most one")
    for key in ["min_r2", "q_margin"]:
        if not 0 <= cfg[key] <= 1:
            raise ValueError(key + " must be between zero and one")
    if any(cfg[k] not in [0, 1] for k in ["storage", "other_generation"]):
        raise ValueError("Storage/generation settings must be 0 or 1")
    return {k: str(v) for k, v in cfg.items()}


def mapped_frame(request):
    mapping = request.mapping
    limit = 5_184_000 if request.agent == "pv" else 200_001
    if request.source == "dataset":
        d = store.get("dataset", request.source_id)
        if d["format"] != "recording":
            raise ValueError("Hourly AMI cannot supply one-second metrology or instantaneous waveforms")
        paths = list((Path(d["folder"]) / "samples").glob("*.parquet"))
        import pyarrow.parquet as pq

        if sum(pq.ParquetFile(p).metadata.num_rows for p in paths) > limit:
            raise ValueError(f"Select a shorter recording: this agent supports at most {limit:,} samples")
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        mapping = mapping.model_copy(
            update={
                "timestamp": "offset",
                "timestamp_unit": "seconds",
                "sample_rate": d["sample_rate"],
                "grid_frequency": d.get("config", {}).get("nominal_frequency", 50),
                "channels": mapping.channels
                or {
                    p: next(
                        c["name"]
                        for c in d["channels"]
                        if c.get("phase") == p[1] and c["kind"] == ("current" if p[0] == "I" else "voltage")
                    )
                    for p in PHASES
                },
                "units": mapping.units or {p: "A" if p[0] == "I" else "V" for p in PHASES},
            }
        )
    else:
        upload = store.get("meter_lab_upload", request.source_id)
        base = folder("uploads", upload["id"])
        if upload["format"] == "comtrade":
            from ..faults.comtrade_io import read_pair

            rec, _ = read_pair(base / "source.cfg", base / "source.dat")
            if request.agent != "fault":
                raise ValueError("COMTRADE requires Fault Location Agent")
            frame = pd.DataFrame({"offset": rec.time})
            inferred = {}
            units = {}
            for channel, values in zip(rec.cfg.analog_channels, rec.analog):
                if float(channel.skew) != 0:
                    raise ValueError("Nonzero channel skew is unsupported; supply synchronized waveforms")
                name = channel.name.strip()
                frame[name] = values
                phase = channel.ph.strip().upper() or name[-1].upper()
                unit = channel.uu.strip()
                key = ("I" if unit in ["A", "kA"] else "U") + phase
                inferred[key] = name
                units[key] = unit
            mapping = mapping.model_copy(
                update={
                    "timestamp": "offset",
                    "timestamp_unit": "seconds",
                    "sample_rate": float(rec.cfg.sample_rates[0][0]),
                    "grid_frequency": float(rec.cfg.frequency),
                    "channels": mapping.channels or inferred,
                    "units": mapping.units or units,
                }
            )
        else:
            path = base / ("source." + upload["format"])
            columns = sorted({mapping.timestamp, *mapping.channels.values(), *mapping.validity.values()})
            if len(columns) > 16:
                raise ValueError("Map only the supported timestamp, waveform or metrology fields")
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                if pq.ParquetFile(path).metadata.num_rows > limit:
                    raise ValueError(f"Select a shorter input: at most {limit:,} samples are supported")
                frame = pd.read_parquet(path, columns=columns)
            else:
                frame = pd.read_csv(path, usecols=columns, nrows=limit + 1)
    if len(frame) > limit:
        raise ValueError(f"Select a shorter input: at most {limit:,} samples are supported")
    if mapping.timestamp not in frame:
        raise ValueError("Select a timestamp column")
    if mapping.timestamp_unit == "iso_utc":
        times = (
            pd.to_datetime(frame[mapping.timestamp], utc=True, errors="raise").astype("int64").to_numpy()
            / 1e9
        )
    else:
        times = pd.to_numeric(frame[mapping.timestamp], errors="raise").to_numpy(dtype=float)
    channels = ["P", "Q"] if request.agent == "pv" else list(PHASES)
    allowed = set(channels) | ({"VA", "VB", "VC"} if request.agent == "pv" else set())
    if (
        len(set(mapping.channels.values())) != len(mapping.channels)
        or not set(channels) <= set(mapping.channels) <= allowed
    ):
        raise ValueError("Map distinct " + ", ".join(channels) + " channels")
    if request.agent == "pv" and (
        mapping.asset_level != "meter"
        or not mapping.aggregate_power
        or mapping.sample_rate != 1
        or mapping.timestamp_unit == "seconds"
    ):
        raise ValueError(
            "PV requires genuine one-second aggregate meter P/Q in UTC; hourly or transformer measurements are incompatible"
        )
    values = []
    for key in channels:
        unit = mapping.units.get(key)
        allowed = {"P": ["W", "kW"], "Q": ["var", "kvar"]}.get(
            key, ["A", "kA"] if key.startswith("I") else ["V", "kV"]
        )
        if unit not in allowed:
            raise ValueError(key + " requires " + " or ".join(allowed))
        if mapping.channels[key] not in frame:
            raise ValueError("Mapped column does not exist: " + mapping.channels[key])
        values.append(
            pd.to_numeric(frame[mapping.channels[key]], errors="raise").to_numpy(dtype=float)
            * (1000 if unit.startswith("k") else 1)
        )
    if request.agent == "fault":
        return (
            {},
            np.column_stack([times, *values]),
            mapping.sample_rate,
            mapping.grid_frequency,
            "fault-samples",
            None,
        )
    valid = []
    for key, val in zip(channels, values):
        valid.append(
            (frame[mapping.validity[key]].to_numpy() == 1) if key in mapping.validity else np.isfinite(val)
        )
    voltages = []
    for phase in ["VA", "VB", "VC"]:
        if phase not in mapping.channels:
            voltages.append(np.full(len(times), np.nan))
            continue
        if mapping.units.get(phase) not in ["V", "kV"]:
            raise ValueError(phase + " requires V or kV")
        voltage = pd.to_numeric(frame[mapping.channels[phase]], errors="raise").to_numpy(dtype=float)
        voltages.append(voltage * (1000 if mapping.units[phase] == "kV" else 1))
    array = np.column_stack([times, *values, *valid, np.ones(len(times)), *voltages])
    return {}, array, 1, 0, "pv-samples", None


def build(request):
    recordings = []
    params = request.parameters
    settings = {}
    source_path = None
    feeder = None
    fs, f0 = 1, 0
    if request.source == "preset" and request.source_id == "comtrade_99":
        if request.agent != "fault":
            raise ValueError("COMTRADE waveforms require Fault Location Agent")
        records = []
        offset = 0.0
        for index in range(1, 100):
            header, record, source = fixture("fault", f"1A_val{index}")
            rate, frequency = float(header.pop("fs")), float(header.pop("f0"))
            header.pop("source_id", None)
            if records and (rate != fs or frequency != f0 or header != settings):
                raise ValueError("COMTRADE batch needs matching acquisition/configuration")
            fs, f0, settings = rate, frequency, header
            recordings.append(
                {
                    "name": f"1A_val{index}",
                    "offset_seconds": offset,
                    "source_sha256": store.checksum(source),
                    "samples": len(record),
                }
            )
            record[:, 0] += offset
            records.append(record)
            # A ten-sample gap keeps recordings separate without inflating the
            # native analysis window beyond the existing 200,000-sample limit.
            offset = round(record[-1, 0] * fs + 10) / fs
        values = np.concatenate(records)
        encoding = "fault-samples"
    elif request.source == "preset" and request.source_id == "synthetic_fault":
        if request.agent != "fault":
            raise ValueError("Synthetic waveform requires Fault Location Agent")
        from ..faults.synthetic import fault_frame

        # A bolted fault params.distance_km down the configured line: the loop
        # voltage is I*Z1*d, so the reactance method has a known answer.
        fs, f0 = 2400, params.frequency
        line = FaultConfig.model_validate(
            {k: (None if v == "-1" else v) for k, v in config("fault", request.configuration).items()}
        )
        frame = fault_frame(
            fs,
            f0,
            params.fault_type,
            params.distance_km,
            complex(line.r1_ohm_km, line.x1_ohm_km),
            None if line.r0_ohm_km is None else complex(line.r0_ohm_km, line.x0_ohm_km),
            onset_seconds=params.onset_seconds,
            total_seconds=params.onset_seconds + 0.2,
            nominal_kv=line.nominal_voltage_kv,
        )
        values = frame[["offset", *PHASES]].to_numpy()
        if params.fault_type != "none" and params.distance_km <= line.line_length_km:
            settings["known_distance_km"] = params.distance_km
        encoding = "fault-samples"
    elif request.source == "preset":
        settings, values, source_path = fixture(request.agent, request.source_id)
        fs = float(settings.pop("fs", 1))
        f0 = float(settings.pop("f0", 0))
        settings.pop("source_id", None)
        encoding = "pv-constant-seconds" if request.agent == "pv" else "fault-samples"
        if request.source_id.startswith("physics_"):
            settings["manual_onset"] = "0"
    else:
        settings, values, fs, f0, encoding, source_path = mapped_frame(request)
        if request.source == "dataset" and request.agent == "fault":
            dataset = store.get("dataset", request.source_id)
            if dataset.get("wavewin"):
                from ..faults.network_model import feeder_configuration

                # Relay recordings are located on their own feeder, not the generic example line.
                settings, feeder = feeder_configuration(dataset)
                feeder.update(
                    source=dataset["source"],
                    wavewin={
                        k: dataset["wavewin"].get(k) for k in ("event_type", "location", "station", "fid")
                    },
                )
    configuration = config(request.agent, {**settings, **request.configuration})
    if not len(values):
        raise ValueError("Input contains no samples")
    if not np.isfinite(values[:, 0]).all():
        raise ValueError("Timestamps must be finite")
    origin = float(values[0, 0])
    lower = origin + params.start_offset_seconds
    upper = lower + (params.duration_seconds if params.duration_seconds is not None else float("inf"))
    if request.agent == "pv" and params.days != 35:
        upper = min(upper, lower + params.days * 86400)
    if encoding == "pv-constant-seconds":
        blocks = []
        outage = origin + params.outage_start if params.outage_start is not None else float("inf")
        for row in values:
            a = max(row[0], lower)
            b = min(row[0] + row[1], upper)
            if a >= b:
                continue
            if int(a) != a or int(b) != b:
                raise ValueError("PV window boundaries must be integer seconds")
            cuts = sorted({a, b, *[c for c in [outage, outage + params.outage_seconds] if a < c < b]})
            for x, y in zip(cuts, cuts[1:]):
                block = row.copy()
                block[0] = x
                block[1] = y - x
                if outage <= x < outage + params.outage_seconds:
                    block[4:6] = 0
                blocks.append(block)
        values = np.asarray(blocks)
    else:
        values = values[(values[:, 0] >= lower) & (values[:, 0] < upper)]
    if not len(values):
        raise ValueError("The selected time window contains no samples")
    rng = np.random.default_rng(params.seed)
    if request.agent == "fault":
        if len(values) > 200001:
            raise ValueError("Fault inputs are limited to 200,001 samples; select a shorter recording")
        analysis_start = max(values[0, 0], float(configuration["start"]))
        analysis_end = (
            min(values[-1, 0], float(configuration["end"]))
            if float(configuration["end"]) >= 0
            else values[-1, 0]
        )
        if round((analysis_end - analysis_start) * fs) > 200000:
            raise ValueError(
                "Analysis window exceeds 200,000 native sample periods including gaps; select a shorter window"
            )
        if fs / f0 < 4 or fs / f0 > 10000:
            raise ValueError("Waveform requires 4–10,000 samples per electrical cycle")
        if np.any(np.diff(values[:, 0]) <= 0) or np.any(
            np.abs(np.diff(values[:, 0]) * fs - np.rint(np.diff(values[:, 0]) * fs)) > 0.01
        ):
            raise ValueError(
                "Waveform timestamps must increase on the declared sample grid; gaps are allowed"
            )
        values[:, 1:] *= params.magnitude
        if params.noise:
            values[:, 1:] += (
                rng.normal(size=values[:, 1:].shape) * np.nanmax(abs(values[:, 1:]), axis=0) * params.noise
            )
        count = len(values)
        duration = float(values[-1, 0] - values[0, 0] + 1 / fs)
    else:
        pcol = 2 if encoding == "pv-constant-seconds" else 1
        if source_path and source_path.with_suffix(".truth.json").exists():
            truth = json.loads(source_path.with_suffix(".truth.json").read_text())
            bytime = {x["start"]: x for x in truth}
            for row in values:
                point = bytime.get(row[0]) or bytime.get(math.floor(row[0] / 900) * 900)
                if point and (params.load_scale != 1 or params.pv_scale != 1 or params.cloud_variation):
                    cloud = 1 - params.cloud_variation * rng.random()
                    row[pcol] = (
                        point["native_load_kw"] * params.load_scale - point["pv_kw"] * params.pv_scale * cloud
                    ) * 1000
        elif params.load_scale != 1 or params.pv_scale != 1 or params.cloud_variation:
            raise ValueError("Separate load/PV controls require a paired synthetic truth fixture")
        values[:, pcol] *= params.magnitude * params.polarity
        values[:, pcol + 1] *= params.magnitude
        if params.noise:
            values[:, pcol] += (
                rng.normal(size=len(values)) * max(1, np.nanmax(abs(values[:, pcol]))) * params.noise
            )
        if (
            np.any(values[:, 0] < 0)
            or np.any(values[:, 0] > 4102444800)
            or np.any(values[:, 0] != np.floor(values[:, 0]))
        ):
            raise ValueError("PV requires integer UTC seconds between 1970 and 2100")
        # Native timestamps may contain outages or adversarial jumps. They are
        # never resampled. Reject coarse regular data presented as one-second.
        if encoding == "pv-samples" and len(values) > 1 and not np.any(np.diff(values[:, 0]) == 1):
            raise ValueError(
                "No genuine one-second cadence found; hourly AMI cannot be replayed as metrology"
            )
        count = int(values[:, 1].sum()) if encoding == "pv-constant-seconds" else len(values)
        duration = float(
            values[-1, 0] - values[0, 0] + (values[-1, 1] if encoding == "pv-constant-seconds" else 1)
        )
        if count > 5184000 or duration > 60 * 86400:
            raise ValueError("PV inputs are limited to 60 calendar days / 5,184,000 samples")
    if params.outage_start is not None and params.outage_seconds and encoding != "pv-constant-seconds":
        mask = (values[:, 0] >= origin + params.outage_start) & (
            values[:, 0] < origin + params.outage_start + params.outage_seconds
        )
        if request.agent == "fault":
            values[mask, 1:] = np.nan
        if encoding == "pv-samples":
            values[mask, 3:5] = 0
    return dict(
        values=values,
        configuration=configuration,
        fs=fs,
        f0=f0,
        encoding=encoding,
        total_samples=count,
        start_time=float(values[0, 0]),
        duration_seconds=duration,
        source_path=source_path,
        recordings=recordings,
        feeder=feeder,
    )


def create(request):
    result = build(request)
    identifier = store.uid()
    path = folder("scenarios", identifier)
    path.mkdir(parents=True)
    values = result.pop("values")
    np.savetxt(path / "input.csv", values, fmt="%.17g", delimiter=",")
    np.save(path / "input.npy", values, allow_pickle=False)
    source = result.pop("source_path")
    record = {
        "id": identifier,
        "name": request.name or request.source_id.replace("_", " "),
        "agent": request.agent,
        "request": request.model_dump(),
        **result,
        "input_sha256": store.checksum(path / "input.csv"),
        "source_sha256": store.checksum(source) if source else None,
        "reference": "pending",
        "provenance": "synthetic paired/declared fixture"
        if request.source == "preset" and not request.source_id.startswith("1A_val")
        else "recorded measurements; accuracy truth unverified",
    }
    # Freeze the reference only when it matches the exact unmodified fixture.
    original = Parameters()
    if (
        source
        and request.parameters == original
        and not request.configuration
        and not request.source_id.startswith("physics_")
    ):
        expected = source.with_suffix(".json")
        (path / "reference.json").write_bytes(expected.read_bytes())
        if request.agent == "fault":
            (path / "reference-version.txt").write_text(VERSION)
        record["reference"] = "frozen Python fixture"
    if source and source.with_suffix(".truth.json").exists():
        (path / "truth.json").write_bytes(source.with_suffix(".truth.json").read_bytes())
    manifest = fixture_root(request.agent) / "manifest.json"
    if request.source == "preset" and manifest.exists():
        (path / "fixture-manifest.json").write_bytes(manifest.read_bytes())
        record["fixture_manifest_sha256"] = store.checksum(manifest)
    return store.put("meter_lab_scenario", record)


def pv_reference(values, configuration):
    """Adapt parsed numeric arrays to the reference's integer UTC/second contract."""
    from pv_agent.tools.reference import analyze

    return analyze(
        [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), int(r[4]), int(r[5]), int(r[6])) for r in values],
        {k: float(v) for k, v in configuration.items()},
    )


def reference(scenario):
    path = folder("scenarios", scenario["id"])
    version = path / "reference-version.txt"
    stale = scenario["agent"] == "fault" and (not version.exists() or version.read_text() != VERSION)
    if (path / "reference.json").exists() and not stale:
        return
    values = np.loadtxt(path / "input.csv", delimiter=",", ndmin=2)
    if scenario["agent"] == "pv":
        if scenario["encoding"] == "pv-samples":
            values = np.column_stack([values[:, 0], np.ones(len(values)), values[:, 1:6]])
        result = pv_reference(values, scenario["configuration"])
    else:
        from di_agent.tools.export_fixtures import reference as fault_reference, encode

        cfg = {k: (None if v == "-1" else v) for k, v in scenario["configuration"].items()}
        cfg = FaultConfig.model_validate({**cfg, "channel_map": {phase: phase for phase in PHASES}})
        result = fault_reference(
            pd.DataFrame(values, columns=["offset", *PHASES]), scenario["fs"], scenario["f0"], cfg
        )
        result = json.loads(json.dumps(result, default=encode, allow_nan=False))
        if scenario.get("feeder"):
            placements = network_placements(scenario, values, cfg, result["segments"])
            (path / "network.json").write_text(json.dumps(placements, default=encode, allow_nan=False))
    (path / "reference.json").write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))
    if scenario["agent"] == "fault":
        version.write_text(VERSION)


def network_placements(scenario, values, cfg, segments):
    """Host-side candidates on the feeder model for each segment, as in the offline Wavewin workflow.

    The agent reports one equivalent-line distance; branched feeders give several
    crossings of the measured reactance, located from the same waveform here.
    """
    from ..faults.network_model import network_location

    dataset = {"source": scenario["feeder"]["source"], "wavewin": scenario["feeder"]["wavewin"]}
    placements = []
    for segment in segments:
        rows = values[(values[:, 0] >= segment["start"]) & (values[:, 0] <= segment["end"])]
        if segment["status"] == "insufficient_data" or len(rows) < 2:
            continue
        loop = (segment.get("distance") or {}).get("fault_loop") or cfg.fault_loop
        placement = network_location(
            dataset,
            rows[:, 1:4].T / 1000,
            rows[:, 4:7].T / 1000,
            scenario["fs"],
            scenario["f0"],
            segment.get("center_sample") or 0,
            len(rows),
            loop,
            cfg,
        )
        placements.append({"segment": segment["segment"], **placement})
    return placements


def preview(scenario, max_points=900):
    values = np.loadtxt(folder("scenarios", scenario["id"]) / "input.csv", delimiter=",", ndmin=2)
    if scenario["agent"] == "pv":
        if scenario["encoding"] == "pv-constant-seconds":
            values = np.column_stack([values[:, 0], values[:, 2:4] / 1000, values[:, 4:6]])
        else:
            values = np.column_stack([values[:, 0], values[:, 1:3] / 1000, values[:, 3:5]])
        for index in [1, 2]:
            values[values[:, index + 2] == 0, index] = np.nan
        channels = ["Signed aggregate kW", "Aggregate kvar"]
        values = values[:, :3]
    else:
        channels = list(PHASES)
    return {
        "channels": channels,
        "rows": bounded(values, max_points),
        "start": scenario["start_time"],
        "end": scenario["start_time"] + scenario["duration_seconds"],
        "samples": scenario["total_samples"],
    }


def gap_markers(values, period, lengths=None):
    if len(values) < 2:
        return values
    ends = values[:-1, 0] + (lengths[:-1] if lengths is not None else period)
    gaps = np.flatnonzero(values[1:, 0] - ends > period * 0.5)
    if not len(gaps):
        return values
    extra = np.full((len(gaps), values.shape[1]), np.nan)
    extra[:, 0] = ends[gaps]
    combined = np.concatenate([values, extra])
    return combined[np.argsort(combined[:, 0], kind="stable")]


def bounded(values, limit):
    """Union extrema and missing-value markers in time buckets, bounded output."""
    if len(values) > limit:
        buckets = max(1, (limit - 2) // (3 * (values.shape[1] - 1) + 2))
        indices = {0, len(values) - 1}
        for block in np.array_split(np.arange(len(values)), buckets):
            indices.update([int(block[0]), int(block[-1])])
            for col in range(1, values.shape[1]):
                finite = np.isfinite(values[block, col])
                indices.update(block[~finite][:1].tolist())
                if finite.any():
                    good = block[finite]
                    indices.add(int(good[np.argmin(values[good, col])]))
                    indices.add(int(good[np.argmax(values[good, col])]))
        values = values[sorted(indices)]
    return [[float(x) if np.isfinite(x) else None for x in row] for row in values]
