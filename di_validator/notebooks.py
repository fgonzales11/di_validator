"""Bounded, read-only native data access for browser Python notebooks."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import store
from .query import db, files, sql_name

SITE = store.ROOT / "runtime" / "jupyterlite"
SOURCE_READ_LIMIT = 20 * 1024 * 1024
router = APIRouter(prefix="/api/v1/notebooks", tags=["Notebooks"])


class DataPage(BaseModel):
    columns: list[str]
    rows: list[dict]
    next_cursor: str | None
    metadata: dict


@router.get("/files/{name}")
def source_csv(name: str):
    """Expose small, unmodified source tables without inventing import metadata."""
    root = (store.ROOT / "data").resolve()
    path = (root / name).resolve()
    if path.parent != root or path.name != name or path.suffix.lower() != ".csv" or not path.is_file():
        raise HTTPException(404, "Choose a CSV file from the data directory")
    if path.stat().st_size > SOURCE_READ_LIMIT:
        raise HTTPException(
            413, "Source CSV reads are limited to 20 MiB; import this file and use paged readings"
        )
    return FileResponse(
        path, filename=path.name, media_type="text/csv", headers={"X-Source-SHA256": store.checksum(path)}
    )


@router.get("/status")
def status():
    ready = (SITE / "di-build.json").is_file() and (SITE / "lab" / "index.html").is_file()
    return dict(
        ready=ready,
        notebooks=[p.name for p in sorted((store.ROOT / "notebooks" / "files").glob("*.ipynb"))],
        url="/notebooks/lab/index.html?path=01%20-%20Getting%20started.ipynb",
        row_limit=100000,
        storage="browser",
    )


def clock_ns(value, timezone):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Missing timestamp")
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return int(stamp.value)
    except Exception as error:
        raise ValueError("Provide a valid, unambiguous ISO timestamp with a timezone offset") from error


def decode_cursor(cursor, identity):
    if not cursor:
        return None
    try:
        if len(cursor) > 2048:
            raise ValueError("Cursor is too large")
        item = json.loads(base64.urlsafe_b64decode(cursor))
        if item["selection"] != identity or not isinstance(item["asset"], str):
            raise ValueError("Different selection")
        return int(item["time"]), item["asset"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
        raise ValueError("Invalid cursor; start a new query when changing the selection") from error


def native_page(dataset_id, asset_ids=None, channels=None, start=None, end=None, limit=10000, cursor=None):
    if not 1 <= limit <= 100000:
        raise ValueError("Choose between 1 and 100,000 rows per page")
    dataset = store.get("dataset", dataset_id)
    is_recording = dataset["format"] == "recording"
    assets = sorted(set(filter(None, (asset_ids or "").split(","))))
    selected = sorted(set(filter(None, (channels or "").split(","))))
    if len(assets) > 64:
        raise ValueError("Select at most 64 assets per query")
    channel_map = {c["name"]: c for c in dataset["channels"]}
    if set(selected) - channel_map.keys():
        raise ValueError("Unknown channel for this dataset")
    selected = selected or list(channel_map)
    if is_recording:
        if assets and assets != [dataset["asset_id"]]:
            raise ValueError("The selected asset is not in this recording")
        lower, upper = float(start or 0), float(end) if end is not None else dataset["duration_seconds"]
        if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0 or upper < lower:
            raise ValueError("Recording bounds must be finite, ordered seconds from origin")
        origin = clock_ns(dataset["start"], dataset["timezone"])
        start_ns, end_ns = origin + round(lower * 1e9), origin + round(upper * 1e9)
        paths = files(dataset, "samples", lower, upper)
        projection = ["t_ns", "offset", *selected]
    else:
        if not assets:
            raise ValueError("Select one or more asset_ids for interval data")
        prefix = f"{dataset['asset_level']}:{dataset['circuit']}:"
        if any(not a.startswith(prefix) for a in assets):
            raise ValueError("Asset identity must match this dataset's level and circuit")
        start_ns = clock_ns(start or dataset["start"], dataset["timezone"])
        end_ns = clock_ns(end or dataset["end"], dataset["timezone"])
        if end_ns < start_ns:
            raise ValueError("End must not precede start")
        paths = files(dataset, "intervals")
        projection = ["t_ns", "asset_id", "value"]
        # Preserve aggregation coverage if these readings were derived from meters.
        if paths:
            names = pq.read_schema(paths[0]).names
            projection += [
                c for c in ["expected_meters", "contributing_meters", "meter_coverage"] if c in names
            ]
    identity = store.digest([dataset_id, assets, selected, start_ns, end_ns])
    after = decode_cursor(cursor, identity)
    where = ["t_ns BETWEEN ? AND ?"]
    params = [paths, start_ns, end_ns]
    if not is_recording:
        where.append("asset_id IN (SELECT unnest(?))")
        params.append(assets)
    if after:
        if is_recording:
            where.append("t_ns > ?")
            params.append(after[0])
        else:
            where.append("(t_ns > ? OR (t_ns = ? AND asset_id > ?))")
            params.extend([after[0], after[0], after[1]])
    order = "t_ns" if is_recording else "t_ns,asset_id"
    if paths:
        with db() as conn:
            frame = conn.execute(
                f"SELECT {','.join(map(sql_name, projection))} FROM read_parquet(?) "
                f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?",
                [*params, limit + 1],
            ).df()
    else:
        frame = pd.DataFrame(columns=projection)
    next_cursor = None
    if len(frame) > limit:
        frame = frame.iloc[:limit].copy()
        token = dict(
            selection=identity,
            time=str(int(frame["t_ns"].iloc[-1])),
            asset=dataset["asset_id"] if is_recording else frame["asset_id"].iloc[-1],
        )
        next_cursor = base64.urlsafe_b64encode(store.encode(token).encode()).decode()
    # ISO strings avoid JavaScript's 53-bit integer rounding of UTC nanoseconds.
    frame.insert(0, "timestamp", [pd.Timestamp(int(t), tz="UTC").isoformat() for t in frame.pop("t_ns")])
    if is_recording:
        frame.insert(1, "asset_id", dataset["asset_id"])
    frame = frame.astype(object).where(pd.notna(frame), None)
    metadata = {
        k: dataset.get(k)
        for k in [
            "id",
            "name",
            "source_checksum",
            "asset_level",
            "timezone",
            "interval_position",
            "interval_seconds",
            "sample_rate",
            "synthetic",
            "start",
            "end",
            "clock_verified",
            "clock_basis",
            "comtrade",
        ]
    }
    metadata.update(channels=[channel_map[c] for c in selected], resolution="native", gap_filling="none")
    return dict(
        columns=frame.columns.tolist(),
        rows=frame.to_dict("records"),
        next_cursor=next_cursor,
        metadata=metadata,
    )


@router.get("/data/{dataset_id}", response_model=DataPage)
def data(
    dataset_id: str,
    asset_ids: str | None = None,
    channels: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(10000, ge=1, le=100000),
    cursor: str | None = None,
):
    return native_page(dataset_id, asset_ids, channels, start, end, limit, cursor)


@router.get("/sources/{dataset_id}")
def original_source(dataset_id: str):
    dataset = store.get("dataset", dataset_id)
    path = Path(dataset["source"])
    if dataset.get("synthetic") or not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise HTTPException(404, "This dataset has no downloadable source file")
    return FileResponse(path, filename=path.name)


@router.get("/registries")
def registries():
    return store.listing("registry")


@router.get("/registries/{registry_id}")
def registry(registry_id: str):
    version = store.get("registry", registry_id)
    frame = pd.read_parquet(version["path"]).astype(object)
    frame = frame.where(pd.notna(frame), None)
    return dict(columns=frame.columns.tolist(), rows=frame.to_dict("records"), metadata=version)
