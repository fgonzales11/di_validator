"""Persisted offline Event Lab jobs, evidence and downloadable fault artifacts."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import store
from ..classification import environment
from ..events import evaluate
from ..labels import active_annotations
from ..query import files
from ..schemas import EventConfig
from .pipeline import PHASES, VERSION, analyze_segment, validate_dataset


def run(config, progress=None, batch_size=100000):
    progress = progress or store.Progress()
    event_config = EventConfig.model_validate(config)
    dataset = store.get("dataset", event_config.dataset_id)
    cfg = validate_dataset(dataset, event_config.parameters)
    frozen = event_config.model_copy(update={"parameters": cfg.model_dump()}).model_dump()
    columns = ["offset", *[cfg.channel_map[p] for p in PHASES]]
    chunks = []
    for path in files(dataset, "samples", cfg.start, cfg.end):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            progress.check()
            chunk = batch.to_pandas()
            chunks.append(chunk[(chunk.offset >= cfg.start) & (chunk.offset <= cfg.end)])
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    if frame.empty or len(frame) > 200001:
        raise ValueError("Analysis window must contain 1–200,001 native samples")
    times = frame.offset.to_numpy(dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Fault analysis requires strictly ordered sample timestamps")
    fs = dataset["sample_rate"]
    if np.any(np.abs(np.diff(times) * fs - np.rint(np.diff(times) * fs)) > 0.01):
        raise ValueError("Samples must lie on the declared sampling grid; no interpolation is performed")
    finite = np.isfinite(frame[columns[1:]].to_numpy()).all(axis=1)
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
    gaps = [(float(times[i - 1] + 1 / fs), float(times[i])) for i in breaks]
    gaps += [(float(times[i]), float(times[i] + 1 / fs)) for i in np.flatnonzero(~finite)]
    if cfg.manual_onset is not None and (
        any(a <= cfg.manual_onset < b for a, b in gaps) or not times[0] <= cfg.manual_onset <= times[-1]
    ):
        raise ValueError("Manual onset falls in a data gap or outside observed samples")
    result_id = store.uid()
    folder = store.workspace() / "experiments" / result_id
    folder.mkdir(parents=True)
    analyses, events = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        progress(a / len(frame) * 0.9, f"Analyzing native segment {len(analyses) + 1}")
        if a == b or not finite[a]:
            continue
        info, arrays = analyze_segment(frame.iloc[a:b], dataset, cfg, len(analyses))
        analyses.append(info)
        if arrays is None:
            continue
        prefix = f"segment-{info['segment']:03d}"
        info["artifact_prefix"] = prefix
        np.savez_compressed(folder / f"{prefix}-tensor.npz", **arrays, metadata=store.encode(info))
        phase_table = pd.DataFrame(
            arrays["filtered_ka_kv"].T, columns=[p + "_" + ("kA" if p[0] == "I" else "kV") for p in PHASES]
        )
        phase_table.insert(0, "offset_seconds", arrays["time_seconds"])
        phase_table.to_csv(folder / f"{prefix}-filtered.csv", index=False)
        pd.DataFrame(info["distance"].get("cycles", [])).to_csv(
            folder / f"{prefix}-distance-cycles.csv", index=False
        )
        # A user-selected inception is a calculation input, never a detector prediction.
        if info["inception_source"] == "detected":
            events.append(
                dict(
                    id=store.uid(),
                    asset_id=dataset["asset_id"],
                    kind="fault",
                    start=info["onset"],
                    end=info["onset"],
                    emitted_at=info["emitted_at"],
                    segment=info["segment"],
                )
            )
    annotations = (
        event_config.annotation_snapshot
        if event_config.annotation_snapshot is not None
        else active_annotations()
    )
    selected = [
        a
        for a in annotations
        if a["dataset_id"] == dataset["id"]
        and a["partition"] == event_config.partition
        and a["kind"] in {"review", "fault"}
    ]
    # Restrict review exposure to the data actually analyzed.
    selected_window = []
    for annotation in selected:
        row = dict(annotation)
        if row["kind"] == "review":
            row.update(
                start=max(row["start"], float(times[0])), end=min(row["end"], float(times[-1] + 1 / fs))
            )
            if row["end"] <= row["start"]:
                continue
        selected_window.append(row)
    metrics = evaluate(events, selected_window, event_config.match_tolerance, gaps)
    metrics["mean_latency"] = None  # An offline analysis does not measure a causal detector's latency.
    if cfg.manual_onset is not None:
        metrics.update(
            precision=None,
            recall=None,
            f1=None,
            false_alarms_per_hour=None,
            onset_mae=None,
            reason="Manual inception was supplied; detection metrics are disabled",
        )
    interpretation = (
        "Simulated recording. " if dataset.get("synthetic") else ""
    ) + "Offline fault inception and apparent reactance distance; fault types and AUTO loops are heuristic."
    if not cfg.line_parameters_verified:
        interpretation += " Line parameters are unverified examples."
    result = dict(
        id=result_id,
        name=event_config.name,
        kind="events",
        dataset_ids=[dataset["id"]],
        config=frozen,
        synthetic=dataset.get("synthetic", False),
        events=events,
        metrics=metrics,
        fault_analyses=analyses,
        evidence_id=store.digest(selected),
        split_id=store.digest([dataset["asset_id"], event_config.partition]),
        folder=str(folder),
        environment=environment(),
        interpretation=interpretation,
        algorithm_version=VERSION,
        source_checksum=dataset["source_checksum"],
        gaps=gaps,
        source_metadata={
            k: dataset.get(k)
            for k in ["comtrade", "clock_verified", "clock_basis", "sample_rate", "channels"]
        },
    )
    pd.DataFrame(events, columns=["id", "asset_id", "kind", "start", "end", "emitted_at", "segment"]).to_csv(
        folder / "events.csv", index=False
    )
    for name, value in [
        ("manifest", result),
        ("configuration", frozen),
        ("annotations", selected),
        ("fault_distance", analyses),
        ("dataset", dataset),
    ]:
        (folder / f"{name}.json").write_text(store.encode(value), encoding="utf-8")
    report = f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Fault analysis</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}pre{{white-space:pre-wrap}}</style><h1>{html.escape(event_config.name)}</h1><p>{html.escape(interpretation)}</p>"
    report += "<p>Distances are measured from the recording terminal. Ground estimates without r0/x0 are uncompensated. Cycle spread is variation, not a confidence interval. No trained checkpoint or Hugging Face model is used.</p>"
    report += f"<h2>Results and provenance</h2><pre>{html.escape(store.encode(result))}</pre></html>"
    (folder / "report.html").write_text(report, encoding="utf-8")
    progress(1, f"Completed {len(analyses)} segments; {len(events)} fault candidates")
    return store.put("experiment", result, result_id)
