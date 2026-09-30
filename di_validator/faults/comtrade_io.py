"""Immutable COMTRADE imports, with explicit clock and instrument provenance."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from comtrade import Comtrade

from .. import store
from ..ingest import summarize_recording, write_frame
from ..recording_index import build

IMPORT_VERSION = "comtrade-1"
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def sources():
    root = store.ROOT / "fault-distance" / "data" / "data_test"
    imported = {}
    imported_paths = {}
    for dataset in store.listing("dataset"):
        if dataset.get("comtrade"):
            imported.setdefault(dataset["comtrade"]["source_id"], dataset)
            imported_paths.setdefault(dataset["source"].replace("\\", "/"), dataset)
    rows = []
    for cfg in sorted(root.glob("*.cfg")):
        dat = cfg.with_suffix(".dat")
        if not dat.is_file():
            continue
        identifier = store.digest(cfg.relative_to(store.ROOT).as_posix())[:24]
        existing = imported.get(identifier) or imported_paths.get(cfg.as_posix())
        if existing:
            identifier = existing["comtrade"]["source_id"]
        rows.append(
            dict(
                id=identifier,
                name=cfg.stem,
                cfg_path=str(cfg),
                dat_path=str(dat),
                bytes=cfg.stat().st_size + dat.stat().st_size,
                dataset_id=existing["id"] if existing else None,
            )
        )
    return rows


def source(identifier):
    match = next((s for s in sources() if s["id"] == identifier), None)
    if not match:
        raise ValueError("Choose a COMTRADE recording from the source catalog")
    return match


def normalized_header(path):
    """The RTDS 1991 headers use month/day/two-digit-year; never edit the source."""
    text = Path(path).read_text(encoding="utf-8-sig")
    assumptions = []
    if len(text.splitlines()[0].split(",")) == 2:

        def expand(match):
            assumptions.append("RTDS 1991 dates use month/day/year; two-digit years interpreted as 2000 + YY")
            return f"{match[1]}/{match[2]}/20{match[3]},{match[4]}"

        text = re.sub(r"(?m)^(\d{1,2})/(\d{1,2})/(\d{2}),(.*)$", expand, text)
    return text, sorted(set(assumptions))


def read_pair(cfg_path, dat_path):
    cfg_path, dat_path = Path(cfg_path), Path(dat_path)
    if cfg_path.stat().st_size + dat_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("COMTRADE pairs are limited to 64 MiB per recording")
    header, assumptions = normalized_header(cfg_path)
    rec = Comtrade(use_numpy_arrays=True, use_double_precision=True)
    # Parsing bytes and calibration is delegated to the pinned COMTRADE reader.
    from comtrade import Cfg

    cfg = Cfg()
    cfg.read(io.StringIO(header))
    payload = dat_path.read_bytes()
    if cfg.ft != "ASCII":
        raise ValueError(
            "This import preset supports ASCII CFG/DAT recordings; convert binary COMTRADE to ASCII first"
        )
    raw_text = payload.decode("utf-8-sig")
    native_timing = np.loadtxt(io.StringIO(raw_text), delimiter=",", usecols=(0, 1), ndmin=2)
    if len(native_timing) != int(cfg.sample_rates[-1][1]) or np.any(np.diff(native_timing[:, 0]) <= 0):
        raise ValueError("DAT sample count/order does not reconcile with CFG")
    stream = io.StringIO(raw_text)
    rec.read(io.StringIO(header), stream)
    rates = rec.cfg.sample_rates
    if len(rates) != 1 or rates[0][0] <= 0:
        raise ValueError("This importer requires one declared, positive COMTRADE sample rate")
    # The library otherwise reconstructs timestamps from the sampling rate, even
    # when DAT supplies them. Preserve supplied timing (and gaps) at float64 precision.
    times = native_timing[:, 1] * float(rec.cfg.timemult) * float(rec.cfg.time_base)
    if np.any(times < 0):
        raise ValueError("This preset requires explicit nonnegative DAT timestamps")
    rec.time[:] = times
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("COMTRADE sample times must be finite and strictly increasing")
    if len(times) != int(rates[0][1]):
        raise ValueError("DAT sample count does not reconcile with the CFG endpoint")
    fs = float(rates[0][0])
    if np.any(np.abs(np.diff(times) * fs - np.rint(np.diff(times) * fs)) > 0.01):
        raise ValueError("Sample timestamps do not lie on the declared sample grid")
    return rec, assumptions


def import_pair(cfg_path, dat_path, source_id, name=None, progress=None):
    progress = progress or store.Progress()
    checksums = dict(cfg=store.checksum(cfg_path), dat=store.checksum(dat_path))
    source_checksum = store.digest(checksums)
    dataset_id = store.digest([source_id, checksums, IMPORT_VERSION])[:32]
    existing = next((d for d in store.listing("dataset") if d["id"] == dataset_id), None)
    if existing:
        return existing
    rec, assumptions = read_pair(cfg_path, dat_path)
    fs = float(rec.cfg.sample_rates[0][0])
    origin = pd.Timestamp("1970-01-01T00:00:00Z")
    times = np.asarray(rec.time, dtype=float)
    offsets = times - times[0]
    frame = pd.DataFrame(dict(t_ns=origin.value + np.rint(offsets * 1e9).astype(np.int64), offset=offsets))
    channels, original_channels = [], []
    for index, channel in enumerate(rec.cfg.analog_channels):
        unit = channel.uu.strip()
        if unit not in {"A", "kA", "V", "kV"}:
            raise ValueError(f"Channel {channel.name}: unsupported waveform unit {unit!r}")
        phase = channel.ph.strip().upper()
        if phase not in {"A", "B", "C"}:
            matched = re.search(r"[IU]([ABC])$", channel.name.upper())
            phase = matched[1] if matched else "unverified"
        kind = "current" if unit in {"A", "kA"} else "voltage"
        channel_name = channel.name.strip()
        if channel_name in frame.columns:
            raise ValueError("COMTRADE analog channel names must be unique")
        channels.append(
            dict(name=channel_name, kind=kind, phase=phase, unit="A" if kind == "current" else "V")
        )
        values = np.asarray(rec.analog[index], dtype=np.float64)
        frame[channel_name] = values * (1000 if unit.startswith("k") else 1)
        original_channels.append(dict(vars(channel)))
    if not channels:
        raise ValueError("No current/voltage waveform channels in this recording")
    # Digital states are preserved in native storage and in the immutable source pair.
    digital_channels = []
    for index, channel in enumerate(rec.cfg.status_channels):
        column = f"status_{index + 1}"
        frame[column] = np.asarray(rec.status[index], dtype=np.int8)
        digital_channels.append(dict(column=column, **vars(channel)))
    folder = store.workspace() / "datasets" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, len(frame), 100000)):
        progress.check()
        chunk = frame.iloc[start : start + 100000]
        write_frame(chunk, folder / "samples" / f"part-{part:05d}.parquet")
        write_frame(
            summarize_recording(chunk[["t_ns", "offset", *[c["name"] for c in channels]]]),
            folder / "summaries" / f"part-{part:05d}.parquet",
        )
    build(folder)
    # Freeze source bytes alongside data for reruns and notebook reads.
    (folder / "original.cfg").write_bytes(Path(cfg_path).read_bytes())
    (folder / "original.dat").write_bytes(Path(dat_path).read_bytes())
    if checksums != dict(
        cfg=store.checksum(folder / "original.cfg"), dat=store.checksum(folder / "original.dat")
    ):
        raise ValueError("COMTRADE source changed during import; rerun the import")
    asset_id = "terminal:" + source_id
    missing = int(frame[[c["name"] for c in channels]].isna().sum().sum())
    gaps = np.flatnonzero(np.diff(offsets) > 1.5 / fs)
    details = dict(
        source_id=source_id,
        revision=str(rec.cfg.rev_year),
        station=rec.station_name,
        recorder=rec.rec_dev_id,
        original_channels=original_channels,
        digital_channels=digital_channels,
        source_checksums=checksums,
        date_assumptions=assumptions,
        reported_start=rec.start_timestamp.isoformat(),
        reported_trigger=rec.trigger_timestamp.isoformat(),
        first_sample_offset_seconds=float(times[0]),
        timing_note="Reported clock has no verified timezone. UTC epoch is an elapsed-time storage reference, not the recording's absolute time.",
        instrument_note="CFG calibration applied. Primary/secondary interpretation and current reference direction require verification.",
        import_version=IMPORT_VERSION,
    )
    meta = dict(
        id=dataset_id,
        version=1,
        name=name or "COMTRADE · " + Path(cfg_path).stem,
        format="recording",
        asset_level="terminal",
        circuit="RTDS",
        asset_id=asset_id,
        assets=1,
        synthetic=True,
        source=str(Path(cfg_path).resolve()),
        source_checksum=source_checksum,
        channels=channels,
        timezone="UTC",
        clock_verified=False,
        clock_basis="elapsed time from first sample",
        sample_rate=fs,
        interval_seconds=1 / fs,
        interval_position="start",
        folder=str(folder),
        rows=len(frame),
        valid_rows=len(frame),
        readings=len(frame) * len(channels),
        start=origin.isoformat(),
        end=pd.Timestamp(int(frame.t_ns.iloc[-1]), tz="UTC").isoformat(),
        duration_seconds=float(offsets[-1]),
        config=dict(nominal_frequency=float(rec.frequency)),
        comtrade=details,
        quality=dict(
            missing_readings=missing,
            gaps=len(gaps),
            duplicate_timestamps=0,
            missing_intervals=int(np.sum(np.rint(np.diff(offsets)[gaps] * fs) - 1)),
            constant_channels=sum(frame[c["name"]].nunique() <= 1 for c in channels),
            excluded_observations=0,
        ),
    )
    (folder / "manifest.json").write_text(store.encode(meta), encoding="utf-8")
    store.put("asset", dict(id=asset_id, source_id=source_id, circuit="RTDS", level="terminal"), asset_id)
    return store.put("dataset", meta, dataset_id)


def import_sources(config, progress=None):
    progress = progress or store.Progress()
    selected = config["sources"]
    results = []
    for index, item in enumerate(selected):
        progress(index / len(selected), f"Importing COMTRADE {index + 1}/{len(selected)}: {item['name']}")
        if (
            dict(cfg=store.checksum(item["cfg_path"]), dat=store.checksum(item["dat_path"]))
            != item["checksums"]
        ):
            raise ValueError("COMTRADE source changed after queueing; choose the source again")
        results.append(import_pair(item["cfg_path"], item["dat_path"], item["id"], progress=progress)["id"])
    return store.put(
        "comtrade_import", dict(name=f"Imported {len(results)} COMTRADE recordings", dataset_ids=results)
    )
