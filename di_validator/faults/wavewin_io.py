"""Imports for SEL "Compressed ASCII Event" (.CEV) reports exported by Wavewin.

The format is undocumented outside vendor tooling and varies by relay model
(observed: SEL-351S and SEL-451 report different summary field sets around a
common MONTH/DAY/YEAR/.../FREQ/EVENT/waveform-table skeleton). Parsing is
therefore driven by field *names*, not fixed column positions, and anything
the parser cannot reconcile raises rather than guessing at relay semantics.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import store
from ..ingest import summarize_recording, write_frame
from ..recording_index import build

IMPORT_VERSION = "wavewin-cev-1"
MAX_SOURCE_BYTES = 16 * 1024 * 1024
DATE_FIELDS = ["MONTH", "DAY", "YEAR", "HOUR", "MIN", "SEC", "MSEC"]
REQUIRED_SUMMARY_FIELDS = ["FREQ", "SAM/CYC_A", "SAM/CYC_D", "NUM_OF_CYC", "EVENT"]
CURRENT_NAME = re.compile(r"^I([ABCNG])$")
VOLTAGE_NAME = re.compile(r"^V(A|B|C|S1|S2|S)$")
# Same per-step tolerance Meter Lab and the meter agent enforce; frequency-tracked
# rows that drift further cannot be replayed there.
GRID_TOLERANCE = 0.01


def on_sample_grid(frame, fs):
    steps = np.diff(frame["offset"].to_numpy()) * fs
    return bool(np.all(np.abs(steps - np.rint(steps)) <= GRID_TOLERANCE))


def _grid_status(files):
    """Parsing every report takes ~12 s, so results persist keyed by path, size and mtime."""
    path = store.workspace() / "wavewin_grid.json"
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    status, fresh, changed = {}, {}, False
    for cev in files:
        stat = cev.stat()
        key = f"{cev.relative_to(store.ROOT).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
        if key not in cache:
            changed = True
            try:
                frame, _, _, details = parse_cev(cev)
                cache[key] = on_sample_grid(frame, details["samples_per_cycle"] * details["nominal_frequency"])
            except ValueError:
                cache[key] = False
        fresh[key] = status[cev] = cache[key]
    if changed or len(fresh) != len(cache):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(fresh), encoding="utf-8")
        temporary.replace(path)
    return status


def sources():
    root = store.ROOT / "fault-distance" / "data" / "Wavewin_Logs"
    imported = {}
    imported_paths = {}
    for dataset in store.listing("dataset"):
        if dataset.get("wavewin"):
            imported.setdefault(dataset["wavewin"]["source_id"], dataset)
            imported_paths.setdefault(dataset["source"].replace("\\", "/"), dataset)
    rows = []
    if root.is_dir():
        files = [c for c in sorted(root.rglob("*.CEV")) if not c.name.lower().endswith(".cev.session")]
        replayable = _grid_status(files)
        for cev in files:
            if not replayable[cev]:
                continue
            identifier = store.digest(cev.relative_to(store.ROOT).as_posix())[:24]
            existing = imported.get(identifier) or imported_paths.get(cev.as_posix())
            if existing:
                identifier = existing["wavewin"]["source_id"]
            rows.append(
                dict(
                    id=identifier,
                    name=cev.stem,
                    station=cev.parent.name,
                    cev_path=str(cev),
                    bytes=cev.stat().st_size,
                    dataset_id=existing["id"] if existing else None,
                )
            )
    return rows


def source(identifier):
    match = next((s for s in sources() if s["id"] == identifier), None)
    if not match:
        raise ValueError("Choose a Wavewin event from the source catalog")
    return match


def _read_text(path):
    payload = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Wavewin CEV file is not readable as text")


def _rows(text):
    prefix = re.match(r"\s*CEV\s*(\d+)", text)
    if not prefix:
        raise ValueError("Not a recognized CEV file: missing CEV header")
    body = text[prefix.end() :]
    return prefix[1], [row for row in csv.reader(io.StringIO(body)) if row]


def parse_cev(path):
    """Parse one .CEV report into (frame, channels, aux_names, details)."""
    text = _read_text(path)
    version, rows = _rows(text)
    date_header_idx = next(
        (i for i, r in enumerate(rows) if r[: len(DATE_FIELDS)] == DATE_FIELDS), None
    )
    if date_header_idx is None or date_header_idx + 3 >= len(rows):
        raise ValueError("Unrecognized CEV structure: no MONTH/DAY/YEAR/.../MSEC header")
    date_data = rows[date_header_idx + 1]
    if len(date_data) < len(DATE_FIELDS):
        raise ValueError("CEV date/time row is shorter than its header")
    try:
        month, day, year, hour, minute, second, msec = (int(v) for v in date_data[:7])
        origin = pd.Timestamp(
            year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=msec * 1000
        ).tz_localize("UTC")
    except (ValueError, pd.errors.OutOfBoundsDatetime) as error:
        raise ValueError("Unrecognized CEV date/time fields") from error

    summary_header_idx = date_header_idx + 2
    if summary_header_idx + 1 >= len(rows):
        raise ValueError("CEV file ends before an event-summary block")
    summary_header, summary_data = rows[summary_header_idx], rows[summary_header_idx + 1]
    if len(summary_header) != len(summary_data):
        raise ValueError("Event-summary header/data length mismatch")
    summary = dict(zip(summary_header, summary_data))
    missing = [k for k in REQUIRED_SUMMARY_FIELDS if k not in summary]
    if missing:
        raise ValueError(f"Unsupported CEV event-summary layout: missing {', '.join(missing)}")
    try:
        samples_per_cycle_a = float(summary["SAM/CYC_A"])
        samples_per_cycle_d = float(summary["SAM/CYC_D"])
    except ValueError as error:
        raise ValueError("CEV samples-per-cycle fields are not numeric") from error
    if samples_per_cycle_a != samples_per_cycle_d or samples_per_cycle_a <= 0:
        raise ValueError("Mismatched or invalid analog/digital sample rates are not supported")
    try:
        nominal_frequency = float(summary.get("NFREQ", 60))
    except ValueError:
        nominal_frequency = 60.0

    wave_header_idx = summary_header_idx + 2
    if wave_header_idx >= len(rows):
        raise ValueError("CEV file ends before a waveform table")
    wave_header = rows[wave_header_idx]
    if len(wave_header) < 3:
        raise ValueError("Waveform header is too short to be a CEV sample table")
    data_rows = []
    for row in rows[wave_header_idx + 1 :]:
        if len(row) != len(wave_header):
            break
        try:
            float(row[0])
        except ValueError:
            break
        data_rows.append(row)
    if not data_rows:
        raise ValueError("No waveform samples found after the CEV table header")

    checksum_index = len(wave_header) - 1
    multi_token = [i for i, name in enumerate(wave_header[:-1]) if " " in name.strip()]
    if len(multi_token) > 1:
        raise ValueError("Unsupported CEV waveform header: multiple digital-label columns")
    digital_index = multi_token[0] if multi_token else None
    digital_labels = wave_header[digital_index].split() if digital_index is not None else []

    channels, aux_names = [], []
    numeric_columns, string_columns = {}, {}
    freq_column = None
    if digital_index is not None:
        string_columns["digital_word"] = [row[digital_index].strip() for row in data_rows]
    for index, raw_name in enumerate(wave_header):
        if index in {checksum_index, digital_index}:
            continue
        name = raw_name.strip()
        if not name or name in numeric_columns or name in string_columns:
            raise ValueError("Wavewin waveform columns must be named and unique")
        values = [row[index].strip() for row in data_rows]
        try:
            parsed = np.array([float(v) for v in values], dtype=np.float64)
        except ValueError:
            string_columns[name] = values
            continue
        base = re.sub(r"\(.*\)$", "", name).strip().upper()
        current = CURRENT_NAME.fullmatch(base)
        voltage = VOLTAGE_NAME.fullmatch(base)
        if current:
            channels.append(dict(name=name, kind="current", unit="A", phase=current[1]))
            numeric_columns[name] = parsed
        elif voltage:
            scale = 1000.0 if "(kv)" in name.lower() else 1.0
            channels.append(dict(name=name, kind="voltage", unit="V", phase=voltage[1]))
            numeric_columns[name] = parsed * scale
        else:
            aux_names.append(name)
            numeric_columns[name] = parsed
            if base == "FREQ":
                freq_column = name
    if not channels:
        raise ValueError("No current/voltage waveform channels recognized in this CEV file")

    freq = numeric_columns.get(freq_column) if freq_column else None
    freq = freq if freq is not None else np.full(len(data_rows), nominal_frequency)
    freq_safe = np.where(np.isfinite(freq) & (freq > 0), freq, nominal_frequency)
    dt = 1.0 / (samples_per_cycle_a * freq_safe)
    offsets = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    t_ns = origin.value + np.rint(offsets * 1e9).astype(np.int64)

    frame = pd.DataFrame({"t_ns": t_ns, "offset": offsets})
    for name, values in numeric_columns.items():
        frame[name] = values
    for name, values in string_columns.items():
        frame[name] = values

    details = dict(
        fid=next((r[0] for r in rows if r and r[0].startswith("FID=")), None),
        cev_version=version,
        station=Path(path).parent.name,
        event_type=summary.get("EVENT"),
        location=summary.get("LOCATION"),
        shot=summary.get("SHOT"),
        targets_bits=summary.get("TARGETS"),
        samples_per_cycle=samples_per_cycle_a,
        nominal_frequency=nominal_frequency,
        cycles_reported=summary.get("NUM_OF_CYC"),
        reported_samples=len(data_rows),
        digital_labels=digital_labels,
        auxiliary_channels=aux_names,
        reported_event_timestamp=origin.isoformat(),
        timing_note=(
            "Row timing is reconstructed from each row's tracked FREQUENCY at "
            f"{samples_per_cycle_a:g} samples/cycle; the relay-reported timestamp is treated as the "
            "first-sample origin and its alignment with any specific sample has not been verified."
        ),
        instrument_note=(
            "Analog values are the relay's raw device units. No CT/PT ratio is stored in this report, so "
            "values are not converted to primary or verified secondary quantities."
        ),
        import_version=IMPORT_VERSION,
    )
    return frame, channels, aux_names, details


def import_file(cev_path, source_id, name=None, progress=None):
    progress = progress or store.Progress()
    cev_path = Path(cev_path)
    if cev_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Wavewin CEV files are limited to 16 MiB each")
    checksum = store.checksum(cev_path)
    dataset_id = store.digest([source_id, checksum, IMPORT_VERSION])[:32]
    existing = next((d for d in store.listing("dataset") if d["id"] == dataset_id), None)
    if existing:
        return existing
    frame, channels, aux_names, details = parse_cev(cev_path)
    fs = details["samples_per_cycle"] * details["nominal_frequency"]
    if not on_sample_grid(frame, fs):
        raise ValueError(
            "Relay frequency tracking moves these samples off a uniform grid; Meter Lab and the meter agent cannot replay them"
        )
    folder = store.workspace() / "datasets" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c["name"] for c in channels] + aux_names
    for part, start in enumerate(range(0, len(frame), 100000)):
        progress.check()
        chunk = frame.iloc[start : start + 100000]
        write_frame(chunk, folder / "samples" / f"part-{part:05d}.parquet")
        write_frame(
            summarize_recording(chunk[["t_ns", "offset", *numeric_cols]]),
            folder / "summaries" / f"part-{part:05d}.parquet",
        )
    build(folder)
    (folder / "original.cev").write_bytes(cev_path.read_bytes())
    if checksum != store.checksum(folder / "original.cev"):
        raise ValueError("Wavewin source changed during import; rerun the import")
    station = details["station"]
    asset_id = "terminal:" + source_id
    missing = int(frame[[c["name"] for c in channels]].isna().sum().sum())
    offsets = frame["offset"].to_numpy()
    gaps = np.flatnonzero(np.diff(offsets) > 3.0 / fs) if len(offsets) > 1 else np.array([])
    meta = dict(
        id=dataset_id,
        version=1,
        name=name or f"Wavewin · {station} · {cev_path.stem}",
        format="recording",
        asset_level="terminal",
        circuit=station,
        asset_id=asset_id,
        assets=1,
        synthetic=False,
        source=str(cev_path.resolve()),
        source_checksum=checksum,
        channels=channels,
        timezone="UTC",
        clock_verified=False,
        clock_basis="relay-reported event timestamp, treated as the first-sample origin",
        sample_rate=fs,
        interval_seconds=1 / fs,
        interval_position="start",
        folder=str(folder),
        rows=len(frame),
        valid_rows=len(frame),
        readings=len(frame) * len(channels),
        start=pd.Timestamp(int(frame["t_ns"].iloc[0]), tz="UTC").isoformat(),
        end=pd.Timestamp(int(frame["t_ns"].iloc[-1]), tz="UTC").isoformat(),
        duration_seconds=float(offsets[-1]),
        config=dict(nominal_frequency=details["nominal_frequency"]),
        wavewin={**details, "source_id": source_id},
        quality=dict(
            missing_readings=missing,
            gaps=int(len(gaps)),
            duplicate_timestamps=0,
            constant_channels=sum(frame[c["name"]].nunique() <= 1 for c in channels),
            excluded_observations=0,
        ),
    )
    (folder / "manifest.json").write_text(store.encode(meta), encoding="utf-8")
    store.put("asset", dict(id=asset_id, source_id=source_id, circuit=station, level="terminal"), asset_id)
    return store.put("dataset", meta, dataset_id)


def import_sources(config, progress=None):
    progress = progress or store.Progress()
    selected = config["sources"]
    results = []
    for index, item in enumerate(selected):
        progress(index / len(selected), f"Importing Wavewin event {index + 1}/{len(selected)}: {item['name']}")
        if store.checksum(item["cev_path"]) != item["checksum"]:
            raise ValueError("Wavewin source changed after queueing; choose the source again")
        results.append(import_file(item["cev_path"], item["id"], progress=progress)["id"])
    return store.put(
        "wavewin_import", dict(name=f"Imported {len(results)} Wavewin events", dataset_ids=results)
    )
