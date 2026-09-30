from __future__ import annotations

from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

from . import store


def sql_name(value):
    return '"' + value.replace('"', '""') + '"'


def db():
    conn = duckdb.connect()
    conn.execute("SET memory_limit='2GB'")
    conn.execute("SET threads=2")
    return conn


def files(dataset, kind, start=None, end=None):
    folder = Path(dataset["folder"])
    if kind == "summaries" and (folder / "overview.parquet").exists():
        return [str(folder / "overview.parquet")]
    if kind == "samples" and start is not None and (folder / "chunk_index.json").exists():
        chunks = json.loads((folder / "chunk_index.json").read_text(encoding="utf-8"))
        return [
            str(folder / "samples" / c["file"]) for c in chunks if c["end"] >= start and c["start"] <= end
        ]
    return [str(p) for p in sorted((folder / kind).rglob("*.parquet"))]


def interval_frame(dataset, assets=None, start=None, end=None):
    where, params = [], [files(dataset, "intervals")]
    if assets:
        where.append("asset_id IN (SELECT unnest(?))")
        params.append(assets)
    for key, op in [(start, ">="), (end, "<=")]:
        if key:
            t = pd.Timestamp(key)
            t = t.tz_localize(dataset["timezone"]) if t.tzinfo is None else t
            where.append(f"t_ns {op} ?")
            params.append(t.value)
    with db() as conn:
        return conn.execute(
            "SELECT * FROM read_parquet(?)"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY asset_id,t_ns",
            params,
        ).df()


def clean(values):
    return [None if not np.isfinite(v) else float(v) for v in values]


def series(dataset_id, asset_ids, start=None, end=None, view="series", power=False, max_points=10000):
    dataset = store.get("dataset", dataset_id)
    if dataset["format"] != "wide_ami":
        raise ValueError("Use recording window queries for native samples")
    if not asset_ids or len(asset_ids) > 8:
        raise ValueError("Select between one and eight assets")
    frame = interval_frame(dataset, asset_ids, start, end)
    unit = dataset["channels"][0]["unit"]
    if power:
        frame["value"] *= (1 if unit == "kWh" else 0.001) / (dataset["interval_seconds"] / 3600)
        unit = "kW"
    traces = []
    for asset, group in frame.groupby("asset_id", sort=False):
        time = pd.to_datetime(group.t_ns, utc=True).dt.tz_convert(dataset["timezone"])
        s = pd.Series(group.value.to_numpy(), index=pd.DatetimeIndex(time))
        # Reindex the selected window so DST exclusions and missing timestamps draw as gaps.
        if len(s):
            s = s.reindex(
                pd.date_range(
                    s.index.min(), s.index.max(), freq=pd.Timedelta(seconds=dataset["interval_seconds"])
                )
            )
        if view == "daily":
            out = s.groupby(s.index.hour).mean()
            traces.append(dict(asset_id=asset, x=out.index.tolist(), y=clean(out.values)))
        elif view == "seasonal":
            out = s.groupby(s.index.month).mean()
            traces.append(dict(asset_id=asset, x=out.index.tolist(), y=clean(out.values)))
        elif view == "distribution":
            counts, bins = np.histogram(s.dropna(), bins=40)
            traces.append(dict(asset_id=asset, x=((bins[:-1] + bins[1:]) / 2).tolist(), y=counts.tolist()))
        elif view == "heatmap":
            table = pd.DataFrame(
                {"day": s.index.strftime("%Y-%m-%d"), "hour": s.index.hour, "value": s.values}
            )
            grid = table.pivot_table(index="day", columns="hour", values="value", aggfunc="mean").reindex(
                columns=range(24)
            )
            traces.append(
                dict(
                    asset_id=asset,
                    x=list(range(24)),
                    y=grid.index.tolist(),
                    z=[clean(r) for r in grid.to_numpy()],
                )
            )
        else:
            if len(s) > max_points:
                # Temporal min/max envelope retains spikes; null bins preserve gaps.
                bins = np.arange(len(s)) // int(np.ceil(len(s) / max(1, max_points // 3)))
                out = []
                for _, bucket in s.groupby(bins):
                    if bucket.notna().any():
                        pair = [(bucket.idxmin(), bucket.min()), (bucket.idxmax(), bucket.max())]
                        out.extend(sorted(pair, key=lambda a: a[0]))
                        if bucket.isna().any():
                            out.append((bucket.index[bucket.isna()][0], np.nan))
                    else:
                        out.append((bucket.index[0], np.nan))
                out.sort(key=lambda a: a[0])
                # The null sentinel can add a point; bound output even for pathological missingness.
                out = out[:max_points]
                traces.append(
                    dict(asset_id=asset, x=[x.isoformat() for x, _ in out], y=clean([y for _, y in out]))
                )
            else:
                traces.append(dict(asset_id=asset, x=[t.isoformat() for t in s.index], y=clean(s.values)))
    return dict(
        traces=traces,
        unit=unit,
        timezone=dataset["timezone"],
        view=view,
        resolution_seconds=dataset["interval_seconds"],
        aggregation="min/max envelope when point limit is exceeded" if view == "series" else view,
        max_points=max_points,
    )


def recording_window(dataset_id, start=0.0, end=None, max_points=10000):
    dataset = store.get("dataset", dataset_id)
    if dataset["format"] != "recording":
        raise ValueError("Select a recording")
    end = min(end if end is not None else dataset["duration_seconds"], dataset["duration_seconds"])
    if start < 0 or end < start:
        raise ValueError("Invalid recording window")
    raw = (end - start) * dataset["sample_rate"] + 1 <= max_points
    traces = []
    with db() as conn:
        if raw:
            paths = files(dataset, "samples", start, end)
            if not paths:
                return dict(
                    traces=[dict(channel=c["name"], unit=c["unit"], x=[], y=[]) for c in dataset["channels"]],
                    aggregation="native samples",
                    start=start,
                    end=end,
                    origin=dataset["start"],
                    sample_rate=dataset["sample_rate"],
                    max_points=max_points,
                )
            frame = conn.execute(
                'SELECT * FROM read_parquet(?) WHERE "offset" BETWEEN ? AND ? ORDER BY "offset"',
                [paths, start, end],
            ).df()
            offsets = frame.offset.to_numpy()
            gap_indices = np.flatnonzero(np.diff(offsets) > 1.5 / dataset["sample_rate"])
            raw = len(frame) + len(gap_indices) <= max_points
            if raw:
                insert_at = gap_indices + 1
                x = np.insert(
                    offsets, insert_at, (offsets[gap_indices] + offsets[gap_indices + 1]) / 2
                ).tolist()
                for channel in dataset["channels"]:
                    y = np.insert(frame[channel["name"]].to_numpy(), insert_at, np.nan)
                    traces.append(dict(channel=channel["name"], unit=channel["unit"], x=x, y=clean(y)))
        if not raw:
            width = max(1.0, (end - start) / max(1, max_points // 3))
            rows = conn.execute(
                """SELECT channel, floor("offset"/?) * ? AS x,
                min("min") AS lo, max("max") AS hi, sum("count") AS n
                FROM read_parquet(?) WHERE "offset" BETWEEN ? AND ? GROUP BY channel,x ORDER BY channel,x""",
                [width, width, files(dataset, "summaries"), float(np.floor(start)), float(np.ceil(end))],
            ).df()
            for channel in dataset["channels"]:
                group = rows.loc[rows.channel == channel["name"]]
                x, y = [], []
                previous = None
                for row in group.itertuples():
                    if previous is not None and row.x - previous > width * 1.5:
                        x.append(previous + width)
                        y.append(None)
                    x.extend([row.x, row.x])
                    y.extend(clean([row.lo, row.hi]))
                    previous = row.x
                traces.append(
                    dict(channel=channel["name"], unit=channel["unit"], x=x[:max_points], y=y[:max_points])
                )
    return dict(
        traces=traces,
        aggregation="native samples" if raw else "1-second min/max envelope",
        start=start,
        end=end,
        origin=dataset["start"],
        sample_rate=dataset["sample_rate"],
        max_points=max_points,
    )
