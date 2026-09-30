from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import store
from .schemas import ImportConfig


def normalize(value):
    return str(value).strip().strip("\"'").strip()


def asset_key(level, circuit, source_id):
    return f"{level}:{circuit}:{normalize(source_id)}"


def clock(values, timezone):
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", format="mixed"))
    if parsed.tz is None:
        parsed = parsed.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.tz_convert("UTC")


def batches(path, size=1000):
    if path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=size):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, chunksize=size, encoding="utf-8-sig")


def write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def import_dataset(config: dict, progress=None):
    progress = progress or store.Progress()
    cfg = ImportConfig.model_validate(config)
    path = Path(cfg.path).resolve()
    if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Select an existing CSV or Parquet file")
    progress(0.01, "Fingerprinting source")
    fingerprint = store.checksum(path)
    spec = cfg.model_dump()
    circuit = cfg.circuit or path.stem.removeprefix("AMI_")
    version = store.digest({"sha256": fingerprint, "config": spec, "importer": 1})[:24]
    try:
        return store.get("dataset", version)
    except ValueError:
        pass
    # Each attempt writes a new staging directory; only a successful catalog commit makes it visible.
    folder = store.workspace() / "datasets" / f"{version}-{store.uid()[:8]}"
    folder.mkdir(parents=True)
    meta = dict(
        id=version,
        name=cfg.name or path.stem,
        format=cfg.format,
        asset_level=cfg.asset_level,
        circuit=circuit,
        source=str(path),
        source_checksum=fingerprint,
        config=spec,
        timezone=cfg.timezone,
        channels=[c.model_dump() for c in cfg.channels],
        interval_position=cfg.interval_position,
        interval_seconds=cfg.interval_seconds,
        folder=str(folder),
        synthetic=False,
        version=1,
    )
    if cfg.format == "wide_ami":
        meta.update(_intervals(path, folder, cfg, circuit, progress))
    else:
        meta.update(_recording(path, folder, cfg, circuit, progress))
        from .recording_index import build

        build(folder)
    (folder / "manifest.json").write_text(store.encode(meta), encoding="utf-8")
    progress(0.98, "Committing immutable dataset version")
    return store.put("dataset", meta, version)


def _intervals(path, folder, cfg, circuit, progress):
    if len(cfg.channels) != 1 or cfg.channels[0].kind != "net_energy":
        raise ValueError(
            "Wide AMI input requires one net-energy channel; use recording format for other channels"
        )
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            original = next(csv.reader(stream))
    else:
        original = pq.read_schema(path).names
    if cfg.timestamp_column not in original:
        raise ValueError(f"Missing timestamp column {cfg.timestamp_column}")
    value_cols = [c for c in original if c != cfg.timestamp_column]
    ids = [normalize(c) for c in value_cols]
    if not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Asset identifiers must be nonempty and unique after normalization")
    keys = [asset_key(cfg.asset_level, circuit, i) for i in ids]
    observed = np.zeros(len(ids), dtype=np.int64)
    negative = observed.copy()
    minimum, maximum = np.full(len(ids), np.inf), np.full(len(ids), -np.inf)
    sums = np.zeros(len(ids))
    seen, audit, timestamps = set(), [], []
    total, numeric_invalid, kept = 0, 0, 0
    channel = cfg.channels[0]
    for part, frame in enumerate(batches(path)):
        progress.check()
        time = clock(frame[cfg.timestamp_column], cfg.timezone)
        good = np.ones(len(frame), dtype=bool)
        for n, t in enumerate(time):
            reason = ""
            if pd.isna(t):
                reason = "invalid_or_unresolved_local_time"
            elif t.value in seen:
                reason = "duplicate_timestamp"
            else:
                seen.add(t.value)
            if reason:
                audit.append(
                    {
                        "source_row": total + n + 2,
                        "timestamp": str(frame[cfg.timestamp_column].iloc[n]),
                        "reason": reason,
                    }
                )
                good[n] = False
        total += len(frame)
        if not good.any():
            continue
        time = time[good]
        values = frame.loc[good, value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        numeric_invalid += int((~np.isfinite(values) & frame.loc[good, value_cols].notna().to_numpy()).sum())
        values[~np.isfinite(values)] = np.nan
        observed += np.isfinite(values).sum(axis=0)
        negative += (values < 0).sum(axis=0)
        minimum = np.minimum(minimum, np.where(np.isfinite(values), values, np.inf).min(axis=0))
        maximum = np.maximum(maximum, np.where(np.isfinite(values), values, -np.inf).max(axis=0))
        sums += np.nansum(values, axis=0)
        timestamps.extend(time.asi8.tolist())
        kept += len(time)
        wide = pd.DataFrame(values, columns=keys)
        wide.insert(0, "t_ns", time.asi8)
        months = time.tz_convert(cfg.timezone).strftime("%Y-%m")
        for month in sorted(set(months)):
            long = wide.loc[months == month].melt(id_vars="t_ns", var_name="asset_id", value_name="value")
            long["channel"] = channel.name
            # Missing rows are retained as nulls; coverage cannot be confused with actual zero usage.
            write_frame(long, folder / "intervals" / month / f"part-{part:05d}.parquet")
        if part % 3 == 0:
            progress(min(0.88, 0.05 + part * 0.035), f"Indexed {kept:,} intervals across {len(ids):,} assets")
    if not timestamps:
        raise ValueError("No valid timestamps after clock validation")
    ticks = np.sort(np.asarray(timestamps, dtype=np.int64))
    step = int(cfg.interval_seconds * 1e9)
    offgrid = int(((ticks - ticks[0]) % step != 0).sum())
    if offgrid:
        raise ValueError(f"{offgrid} timestamps fall outside the declared interval grid")
    expected = int((ticks[-1] - ticks[0]) // step + 1)
    quality = []
    for i, source_id in enumerate(ids):
        coverage = int(observed[i]) / expected
        quality.append(
            dict(
                asset_id=keys[i],
                source_id=source_id,
                observed=int(observed[i]),
                expected=expected,
                coverage=coverage,
                negative_reads=int(negative[i]),
                constant=bool(minimum[i] == maximum[i] or observed[i] == 0),
                mean=float(sums[i] / observed[i]) if observed[i] else None,
            )
        )
        store.put(
            "asset", dict(id=keys[i], source_id=source_id, circuit=circuit, level=cfg.asset_level), keys[i]
        )
    (folder / "quality.json").write_text(store.encode(quality), encoding="utf-8")
    (folder / "timestamp_audit.json").write_text(store.encode(audit), encoding="utf-8")
    return dict(
        assets=len(ids),
        rows=total,
        valid_rows=kept,
        readings=kept * len(ids),
        start=pd.Timestamp(ticks[0], tz="UTC").isoformat(),
        end=pd.Timestamp(ticks[-1], tz="UTC").isoformat(),
        quality=dict(
            excluded_timestamps=len(audit),
            missing_intervals=expected - kept,
            missing_readings=int(expected * len(ids) - observed.sum()),
            invalid_values=numeric_invalid,
            constant_profiles=sum(q["constant"] for q in quality),
            mean_coverage=float(np.mean(observed / expected)),
        ),
    )


def summarize_recording(frame):
    channel_cols = [c for c in frame if c not in {"t_ns", "offset"}]
    groups = np.floor(frame.offset).astype(np.int64)
    tables = []
    for channel in channel_cols:
        summary = (
            frame.groupby(groups)[channel].agg(["min", "max", "mean", "count"]).reset_index(names="offset")
        )
        summary["channel"] = channel
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def _recording(path, folder, cfg, circuit, progress):
    channels = [c.name for c in cfg.channels]
    if len(set(channels)) != len(channels) or set(channels) & {"t_ns", "offset"}:
        raise ValueError("Channel names must be unique and cannot be t_ns or offset")
    origin = clock([cfg.start_time], cfg.timezone)[0] if cfg.start_time else None
    if origin is not None and pd.isna(origin):
        raise ValueError("Recording start has an unresolved timestamp")
    previous, total, valid, gaps, invalid, part_count = None, 0, 0, 0, 0, 0
    audit = []
    expected_ns = 1e9 / cfg.sample_rate
    first_ns, last_ns = None, None
    for part, frame in enumerate(batches(path, 250000)):
        progress.check()
        if not set(channels).issubset(frame.columns):
            raise ValueError(f"Required recording channels: {channels}")
        if cfg.time_mode == "timestamp":
            if cfg.timestamp_column not in frame:
                raise ValueError("Timestamp column missing")
            time = clock(frame[cfg.timestamp_column], cfg.timezone)
            okay = ~time.isna()
            for n in np.flatnonzero(~okay):
                audit.append(dict(source_row=total + int(n) + 2, reason="invalid_or_unresolved_local_time"))
            t_ns = time[okay].asi8
            frame = frame.loc[okay]
        else:
            if origin is None:
                raise ValueError("Offset or sample-index recordings require start_time")
            if cfg.time_mode == "offset_seconds":
                offset = pd.to_numeric(frame[cfg.timestamp_column], errors="raise").to_numpy(dtype=float)
            else:
                sample = (
                    pd.to_numeric(frame[cfg.timestamp_column], errors="raise").to_numpy(dtype=float)
                    if cfg.timestamp_column in frame
                    else np.arange(total, total + len(frame))
                )
                offset = sample / cfg.sample_rate
            if not np.isfinite(offset).all() or (offset < 0).any():
                raise ValueError("Sample offsets must be finite and nonnegative")
            t_ns = origin.value + np.rint(offset * 1e9).astype(np.int64)
        total += len(frame) + (int((~okay).sum()) if cfg.time_mode == "timestamp" else 0)
        if not len(t_ns):
            continue
        delta = np.diff(np.r_[previous, t_ns] if previous is not None else t_ns)
        if (delta <= 0).any():
            raise ValueError(
                "Recording timestamps must increase strictly; reconcile duplicates/order at source"
            )
        gaps += int((delta > expected_ns * 1.5).sum())
        if (delta < expected_ns * 0.5).any():
            raise ValueError("Declared sampling rate disagrees with recording timestamps")
        first_ns = int(t_ns[0]) if first_ns is None else first_ns
        last_ns, previous = int(t_ns[-1]), int(t_ns[-1])
        values = frame[channels].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        invalid += int(values.isna().sum().sum())
        values.insert(0, "offset", (t_ns - first_ns) / 1e9)
        values.insert(0, "t_ns", t_ns)
        write_frame(values, folder / "samples" / f"part-{part:05d}.parquet")
        write_frame(summarize_recording(values), folder / "summaries" / f"part-{part:05d}.parquet")
        valid += len(values)
        part_count += 1
        if part % 10 == 0:
            progress(0.1, f"Indexed {valid:,} native samples")
    if first_ns is None:
        raise ValueError("Recording has no valid samples")
    key = asset_key(cfg.asset_level, circuit, cfg.asset_id)
    store.put("asset", dict(id=key, source_id=cfg.asset_id, circuit=circuit, level=cfg.asset_level), key)
    (folder / "timestamp_audit.json").write_text(store.encode(audit), encoding="utf-8")
    return dict(
        assets=1,
        asset_id=key,
        rows=total,
        valid_rows=valid,
        readings=valid * len(channels),
        start=pd.Timestamp(first_ns, tz="UTC").isoformat(),
        end=pd.Timestamp(last_ns, tz="UTC").isoformat(),
        duration_seconds=(last_ns - first_ns) / 1e9,
        sample_rate=cfg.sample_rate,
        chunks=part_count,
        quality=dict(excluded_timestamps=len(audit), gaps=gaps, missing_readings=invalid),
    )


def load_quality(dataset):
    path = Path(dataset["folder"]) / "quality.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
