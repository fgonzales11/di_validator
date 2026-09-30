from __future__ import annotations

import math
import html
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lfilter

from . import store
from .adapters import EVENT_DETECTORS, load_plugins
from .classification import environment
from .labels import active_annotations
from .ingest import summarize_recording, write_frame
from .query import files
from .schemas import EventConfig, SyntheticConfig


def synthetic(config, progress=None):
    progress = progress or store.Progress()
    cfg = SyntheticConfig.model_validate(config)
    dataset_id = store.uid()
    folder = store.workspace() / "datasets" / dataset_id
    folder.mkdir(parents=True)
    fs, duration = cfg.sample_rate, cfg.duration_seconds
    count = int(fs * duration)
    origin = pd.Timestamp("2025-01-01T00:00:00Z")
    asset_id = f"transformer:SYNTHETIC:{dataset_id[:8]}"
    # Fixed first-minute events include one on a 10-second boundary and a marked data gap.
    scale = min(1, duration / 120)
    on, off = 10 * scale, 35 * scale
    dip, rise, outage = (45 * scale, 50 * scale), (65 * scale, 69 * scale), (85 * scale, 89 * scale)
    gap = (100 * scale, 101 * scale)
    truth = [
        dict(kind="load_on", start=on, end=on),
        dict(kind="load_off", start=off, end=off),
        dict(kind="voltage_dip", start=dip[0], end=dip[1]),
        dict(kind="voltage_rise", start=rise[0], end=rise[1]),
        dict(kind="interruption", start=outage[0], end=outage[1]),
    ]
    random = np.random.default_rng(cfg.seed)
    written = 0
    for part, index in enumerate(range(0, count, 250000)):
        progress(index / count * 0.9, f"Generating synthetic samples: {index:,} / {count:,}")
        t = np.arange(index, min(index + 250000, count), dtype=np.int64) / fs
        keep = ~((t >= gap[0]) & (t < gap[1]))
        t = t[keep]
        power = 1.2 + 4.2 * ((t >= on) & (t < off)) + random.normal(0, 0.008, len(t))
        voltage = np.full(len(t), 120.0)
        voltage[(t >= dip[0]) & (t < dip[1])] *= 0.72
        voltage[(t >= rise[0]) & (t < rise[1])] *= 1.18
        voltage[(t >= outage[0]) & (t < outage[1])] *= 0.03
        phase = 2 * np.pi * 60 * t
        v = np.sqrt(2) * voltage * np.sin(phase) + random.normal(0, 0.08, len(t))
        current = np.sqrt(2) * (power * 1000 / 120) * np.sin(phase)
        frame = pd.DataFrame(
            dict(
                t_ns=origin.value + np.rint(t * 1e9).astype(np.int64),
                offset=t,
                voltage=v,
                current=current,
                power=power,
                voltage_rms=voltage + random.normal(0, 0.03, len(t)),
            )
        )
        write_frame(frame, folder / "samples" / f"part-{part:05d}.parquet")
        write_frame(summarize_recording(frame), folder / "summaries" / f"part-{part:05d}.parquet")
        written += len(frame)
    from .recording_index import build

    build(folder)
    channels = [
        dict(name="voltage", kind="voltage", unit="V", phase="A"),
        dict(name="current", kind="current", unit="A", phase="A"),
        dict(name="power", kind="power", unit="kW", phase="A"),
        dict(name="voltage_rms", kind="voltage_rms", unit="V", phase="A"),
    ]
    meta = dict(
        id=dataset_id,
        version=1,
        name=cfg.name,
        format="recording",
        asset_level="transformer",
        circuit="SYNTHETIC",
        asset_id=asset_id,
        assets=1,
        synthetic=True,
        source="deterministic synthetic generator v1",
        source_checksum=store.digest(cfg.model_dump()),
        channels=channels,
        timezone="UTC",
        sample_rate=fs,
        interval_position="start",
        interval_seconds=1 / fs,
        folder=str(folder),
        rows=count,
        valid_rows=written,
        readings=written * 4,
        start=origin.isoformat(),
        end=(origin + pd.Timedelta(seconds=(count - 1) / fs)).isoformat(),
        duration_seconds=(count - 1) / fs,
        config=dict(**cfg.model_dump(), nominal_frequency=60, nominal_voltage=120),
        quality=dict(gaps=1, missing_readings=(count - written) * 4),
    )
    (folder / "manifest.json").write_text(store.encode(meta), encoding="utf-8")
    store.put(
        "asset",
        dict(id=asset_id, source_id=dataset_id[:8], circuit="SYNTHETIC", level="transformer"),
        asset_id,
    )
    for event in [
        dict(kind="review", start=0, end=gap[0]),
        dict(kind="review", start=gap[1], end=duration),
        *truth,
    ]:
        store.put(
            "annotation",
            dict(
                dataset_id=dataset_id,
                asset_id=asset_id,
                partition="evaluation",
                synthetic=True,
                note="Synthetic ground truth",
                **event,
            ),
        )
    return store.put("dataset", meta, dataset_id)


class Detector:
    """Causal chunk processor. State survives arbitrary chunk boundaries, never data gaps.

    Events enter `events` when confirmed; ongoing voltage-event ends are updated in place.
    RMS uses the nearest whole-sample cycle window; it is an engineering baseline, not a compliance meter.
    """

    def __init__(self, dataset, config):
        self.dataset, self.config = dataset, EventConfig.model_validate(config)
        self.fs = dataset["sample_rate"]
        self.period = 1 / self.fs
        self.window = max(1, round(self.fs / dataset["config"].get("nominal_frequency", 60)))
        self.nominal = dataset["config"].get("nominal_voltage", 120)
        self.channels = {c["kind"]: c for c in dataset["channels"]}
        if len(self.channels) != len(dataset["channels"]):
            raise ValueError(
                "Baseline detectors require one channel per measurement kind; use a phase-specific recording or a custom multi-phase adapter"
            )
        self.events = []
        self.last = None
        self.reset()
        if not set(self.channels) & {"power", "voltage", "voltage_rms", "current", "current_rms"}:
            raise ValueError("No supported detector channels")

    def reset(self):
        self.filters = {}
        self.warm = 0
        self.load_tail = np.array([])
        self.states = {"voltage": None, "load": None}
        self.last_step = -math.inf

    def smooth(self, key, values):
        zi = self.filters.get(key, np.zeros(self.window - 1))
        output, zi = lfilter(np.ones(self.window) / self.window, [1.0], values, zi=zi)
        self.filters[key] = zi
        return output

    def segments(self, family, labels, times):
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        names = (
            {1: "voltage_dip", 2: "voltage_rise", 3: "interruption"}
            if family == "voltage"
            else {1: "load_on", 2: "load_off"}
        )
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            label = int(labels[a])
            state = self.states[family]
            if state is not None and label != state["label"]:
                self.states[family] = state = None
            if label == 0:
                continue
            if state is None:
                state = dict(label=label, start=float(times[a]), event=None)
                self.states[family] = state
            end = float(times[b - 1] + self.period)
            duration = end - state["start"]
            if state["event"] is None and duration + 1e-9 >= self.config.minimum_duration:
                if family == "load" and state["start"] - self.last_step < 0.25:
                    continue
                confirmed = (
                    state["start"]
                    + math.ceil(self.config.minimum_duration * self.fs - 1e-8) / self.fs
                    - self.period
                )
                event = dict(
                    id=store.uid(),
                    asset_id=self.dataset["asset_id"],
                    kind=names[label],
                    start=state["start"],
                    end=end if family == "voltage" else state["start"],
                    emitted_at=confirmed,
                )
                self.events.append(event)
                state["event"] = event
                if family == "load":
                    self.last_step = state["start"]
            if state["event"] is not None and family == "voltage":
                state["event"]["end"] = end

    def process_chunk(self, frame):
        if frame.empty:
            return self.events
        times = frame.offset.to_numpy()
        if self.last is not None and times[0] - self.last > self.period * 1.5:
            self.reset()
        # NaNs and timestamp gaps are barriers, including within a storage chunk.
        relevant = [c["name"] for c in self.channels.values()]
        valid = np.isfinite(frame[relevant].to_numpy()).all(axis=1)
        cuts = sorted(
            set(
                [
                    0,
                    len(frame),
                    *(np.flatnonzero(np.diff(times) > self.period * 1.5) + 1).tolist(),
                    *np.flatnonzero(~valid).tolist(),
                    *(np.flatnonzero(~valid) + 1).tolist(),
                ]
            )
        )
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a > 0:
                self.reset()
            if a == b or not valid[a]:
                self.reset()
                continue
            part = frame.iloc[a:b]
            t = times[a:b]

            def get(kind):
                return part[self.channels[kind]["name"]].to_numpy()

            voltage, load, derived_voltage, derived_load = None, None, False, False
            if "voltage_rms" in self.channels:
                voltage = get("voltage_rms")
            elif "voltage" in self.channels:
                voltage = np.sqrt(np.maximum(0, self.smooth("v2", get("voltage") ** 2)))
                derived_voltage = True
            if "power" in self.channels:
                load = get("power") / (1000 if self.channels["power"]["unit"] == "W" else 1)
            elif (
                "voltage" in self.channels
                and "current" in self.channels
                and self.channels["voltage"]["phase"] == self.channels["current"]["phase"]
            ):
                load = self.smooth("power", get("voltage") * get("current")) / 1000
                derived_load = True
            elif "current_rms" in self.channels:
                load = get("current_rms")
            elif "current" in self.channels:
                load = np.sqrt(np.maximum(0, self.smooth("i2", get("current") ** 2)))
                derived_load = True
            valid_after = max(0, self.window - 1 - self.warm)
            if voltage is not None:
                ratio = voltage / self.nominal
                labels = np.where(
                    ratio < self.config.interruption_ratio,
                    3,
                    np.where(
                        ratio < self.config.dip_ratio, 1, np.where(ratio > self.config.rise_ratio, 2, 0)
                    ),
                )
                if derived_voltage:
                    labels[:valid_after] = 0
                self.segments("voltage", labels, t)
            if load is not None:
                lag = max(1, round(0.1 * self.fs))
                extended = np.r_[self.load_tail, load]
                old = np.arange(len(self.load_tail), len(extended)) - lag
                delta = np.zeros(len(load))
                okay = old >= 0
                delta[okay] = load[okay] - extended[old[okay]]
                labels = np.where(
                    delta >= self.config.step_threshold,
                    1,
                    np.where(delta <= -self.config.step_threshold, 2, 0),
                )
                if derived_load:
                    labels[: max(0, self.window - 1 + lag - self.warm)] = 0
                self.segments("load", labels, t)
                self.load_tail = extended[-lag:]
            self.warm += len(part)
        self.last = float(times[-1])
        self.events.sort(key=lambda event: (event["emitted_at"], event["kind"]))
        return self.events


def reviewed_union(annotations):
    intervals = sorted((a["start"], a["end"]) for a in annotations if a["kind"] == "review")
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def evaluate(predictions, annotations, tolerance=0.1, gaps=None):
    reviewed = reviewed_union(annotations)
    # Remove unobserved periods from reviewed duration and event eligibility.
    for gap_start, gap_end in gaps or []:
        revised = []
        for a, b in reviewed:
            if b <= gap_start or a >= gap_end:
                revised.append([a, b])
            else:
                if a < gap_start:
                    revised.append([a, gap_start])
                if b > gap_end:
                    revised.append([gap_end, b])
        reviewed = revised

    def inside(event):
        return any(a <= event["start"] < b and event["end"] <= b + 1e-8 for a, b in reviewed)

    truth = [a for a in annotations if a["kind"] != "review" and inside(a)]
    pred = [a for a in predictions if inside(a)]
    candidates = sorted(
        (abs(p["start"] - t["start"]), i, j)
        for i, p in enumerate(pred)
        for j, t in enumerate(truth)
        if p["kind"] == t["kind"]
        and p["asset_id"] == t["asset_id"]
        and abs(p["start"] - t["start"]) <= tolerance
    )
    used_p, used_t, matched = set(), set(), []
    for error, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        p, t = pred[i], truth[j]
        union = max(p["end"], t["end"]) - min(p["start"], t["start"])
        overlap = max(0, min(p["end"], t["end"]) - max(p["start"], t["start"]))
        matched.append(
            dict(
                prediction_id=p["id"],
                annotation_id=t.get("id"),
                onset_error=p["start"] - t["start"],
                duration_iou=overlap / union if union > 0 and t["end"] > t["start"] else None,
                latency=p["emitted_at"] - t["start"],
            )
        )
    hours = sum(b - a for a, b in reviewed) / 3600
    tp, fp, fn = len(matched), len(pred) - len(matched), len(truth) - len(matched)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        reviewed_hours=hours,
        precision=precision if hours else None,
        recall=recall if hours else None,
        f1=2 * tp / (2 * tp + fp + fn) if hours and 2 * tp + fp + fn else None,
        false_alarms_per_hour=fp / hours if hours else None,
        onset_mae=float(np.mean([abs(m["onset_error"]) for m in matched])) if matched else None,
        mean_latency=float(np.mean([m["latency"] for m in matched])) if matched else None,
        matches=matched,
        ignored_predictions=len(predictions) - len(pred),
        tolerance_seconds=tolerance,
    )


def run(config, progress=None, batch_size=100000):
    if config.get("algorithm") == "fault-distance":
        from .faults.workflow import run as run_faults

        return run_faults(config, progress, batch_size)
    load_plugins()
    progress = progress or store.Progress()
    cfg = EventConfig.model_validate(config)
    dataset = store.get("dataset", cfg.dataset_id)
    if dataset["format"] != "recording":
        raise ValueError("Event detection requires a recording")
    annotations = cfg.annotation_snapshot if cfg.annotation_snapshot is not None else active_annotations()
    selected = [
        a for a in annotations if a["dataset_id"] == dataset["id"] and a["partition"] == cfg.partition
    ]
    if cfg.algorithm == "baseline-events":
        detector = Detector(dataset, cfg.model_dump())
    else:
        spec = EVENT_DETECTORS.get(cfg.algorithm)
        if not spec or not spec["channels"].issubset({c["kind"] for c in dataset["channels"]}):
            raise ValueError("Event adapter is unavailable or requires incompatible channels")
        detector = spec["factory"](dataset, cfg.parameters)
    done, gaps, previous = 0, [], None
    for path in files(dataset, "samples"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            progress.check()
            frame = batch.to_pandas()
            offsets = frame.offset.to_numpy()
            if previous is not None and offsets[0] - previous > 1.5 / dataset["sample_rate"]:
                gaps.append((previous + 1 / dataset["sample_rate"], float(offsets[0])))
            for i in np.flatnonzero(np.diff(offsets) > 1.5 / dataset["sample_rate"]):
                gaps.append((float(offsets[i] + 1 / dataset["sample_rate"]), float(offsets[i + 1])))
            missing = frame[[c["name"] for c in dataset["channels"]]].isna().any(axis=1).to_numpy()
            edges = np.diff(np.r_[False, missing, False].astype(int))
            gaps.extend(
                (float(offsets[a]), float(offsets[b - 1] + 1 / dataset["sample_rate"]))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            )
            previous = float(offsets[-1])
            detector.process_chunk(frame)
            done += len(frame)
            progress(
                min(0.95, done / dataset["valid_rows"] * 0.95),
                f"Processed {done:,} native samples; {len(detector.events)} events",
            )
    result_id = store.uid()
    folder = store.workspace() / "experiments" / result_id
    folder.mkdir(parents=True)
    result = dict(
        id=result_id,
        kind="events",
        name=cfg.name,
        dataset_ids=[dataset["id"]],
        config=cfg.model_dump(),
        synthetic=dataset["synthetic"],
        events=detector.events,
        metrics=evaluate(detector.events, selected, cfg.match_tolerance, gaps),
        evidence_id=store.digest(selected),
        split_id=store.digest([dataset["asset_id"], cfg.partition]),
        folder=str(folder),
        environment=environment(),
        interpretation="Synthetic fixture evaluation"
        if dataset["synthetic"]
        else "Reviewed event evaluation",
    )
    pd.DataFrame(detector.events).to_csv(folder / "events.csv", index=False)
    (folder / "manifest.json").write_text(store.encode(result), encoding="utf-8")
    (folder / "annotations.json").write_text(store.encode(selected), encoding="utf-8")
    report = "<!doctype html><html><meta charset='utf-8'><title>Event evaluation</title><style>body{font:15px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>"
    report += f"<h1>{html.escape(cfg.name)}</h1><p>{html.escape(result['interpretation'])}</p>"
    report += pd.DataFrame(detector.events).to_html(index=False)
    report += f"<h2>Evaluation and provenance</h2><pre>{html.escape(store.encode(result))}</pre></html>"
    (folder / "report.html").write_text(report, encoding="utf-8")
    return store.put("experiment", result, result_id)
