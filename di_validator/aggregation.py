from __future__ import annotations

import pandas as pd

from . import store
from .ingest import write_frame
from .query import db, files
from .schemas import AggregateConfig


def run(config, progress=None):
    progress = progress or store.Progress()
    cfg = AggregateConfig.model_validate(config)
    datasets = [store.get("dataset", i) for i in cfg.dataset_ids]
    if any(d["asset_level"] != "meter" or d["format"] != "wide_ami" for d in datasets):
        raise ValueError("Aggregation requires meter-level interval datasets")
    compatible = {
        (d["interval_seconds"], d["interval_position"], d["timezone"], store.encode(d["channels"]))
        for d in datasets
    }
    if len(compatible) != 1 or datasets[0]["interval_position"] == "unknown":
        raise ValueError(
            "Confirm matching interval start/end conventions, durations, channels, units, and timezones before aggregation"
        )
    relationships = cfg.relationship_snapshot or []
    if not relationships:
        raise ValueError("Import meter-to-transformer relationships, including the expected meter inventory")
    zone = datasets[0]["timezone"]

    def ns(v):
        t = pd.Timestamp(v)
        return (t.tz_localize(zone) if t.tzinfo is None else t).value

    mapping = pd.DataFrame(
        [
            dict(
                meter_id=r["meter_id"],
                transformer_id=r["transformer_id"],
                from_ns=ns(r["valid_from"]),
                to_ns=ns(r["valid_to"]),
            )
            for r in relationships
        ]
    )
    for _, group in mapping.groupby("meter_id"):
        ordered = group.sort_values("from_ns")
        if any(b <= a for a, b in zip(ordered.to_ns.iloc[:-1], ordered.from_ns.iloc[1:])):
            raise ValueError("Meter relationships overlap in time; reconcile the inventory")
    paths = [p for d in datasets for p in files(d, "intervals")]
    result_id = store.uid()
    folder = store.workspace() / "datasets" / result_id
    folder.mkdir(parents=True)
    with db() as conn:
        conn.register("mapping", mapping)
        conn.read_parquet(paths).create_view("source")
        if conn.execute("SELECT 1 FROM source GROUP BY asset_id,t_ns HAVING count(*)>1 LIMIT 1").fetchone():
            raise ValueError("Multiple source versions contain the same meter interval")
        bounds = conn.execute("SELECT min(t_ns),max(t_ns) FROM source").fetchone()
        step = int(datasets[0]["interval_seconds"] * 1e9)
        conn.execute(
            "CREATE TEMP TABLE ticks AS SELECT range AS t_ns FROM range(?,?,?)",
            [bounds[0], bounds[1] + step, step],
        )
        progress(0.15, "Joining effective-dated meter inventory")
        conn.execute(
            """CREATE TEMP TABLE aggregate AS
            SELECT m.transformer_id AS asset_id,t.t_ns,
                   CASE WHEN count(s.value)::DOUBLE/count(*) >= ? THEN sum(s.value) ELSE NULL END AS value,
                   count(s.value) AS contributing_meters,count(*) AS expected_meters,
                   count(s.value)::DOUBLE/count(*) AS meter_coverage
            FROM ticks t JOIN mapping m ON t.t_ns BETWEEN m.from_ns AND m.to_ns
            LEFT JOIN source s ON s.asset_id=m.meter_id AND s.t_ns=t.t_ns
            GROUP BY m.transformer_id,t.t_ns""",
            [cfg.min_meter_coverage],
        )
        # Use explicit UTC month boundaries for physical partitions; timestamps retain their UTC instants.
        first = pd.Timestamp(bounds[0], tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        edges = pd.date_range(first, pd.Timestamp(bounds[1], tz="UTC") + pd.offsets.MonthBegin(1), freq="MS")
        for index in range(len(edges) - 1):
            progress(
                0.2 + 0.65 * index / max(1, len(edges) - 1), "Writing transformer totals and meter coverage"
            )
            frame = conn.execute(
                "SELECT * FROM aggregate WHERE t_ns>=? AND t_ns<?",
                [edges[index].value, edges[index + 1].value],
            ).df()
            frame["channel"] = datasets[0]["channels"][0]["name"]
            write_frame(frame, folder / "intervals" / edges[index].strftime("%Y-%m") / "part.parquet")
        stats = conn.execute(
            "SELECT asset_id,count(value) AS observed,count(*) AS expected, avg(meter_coverage) AS meter_coverage, min(value)=max(value) AS constant,avg(value) AS mean FROM aggregate GROUP BY asset_id"
        ).df()
        total = conn.execute("SELECT count(*) FROM aggregate").fetchone()[0]
    quality = []
    for row in stats.to_dict("records"):
        asset = store.get("asset", row["asset_id"])
        quality.append(
            {**row, "source_id": asset["source_id"], "coverage": row["observed"] / row["expected"]}
        )
    (folder / "quality.json").write_text(store.encode(quality), encoding="utf-8")
    (folder / "timestamp_audit.json").write_text("[]", encoding="utf-8")
    meta = {
        **datasets[0],
        "id": result_id,
        "name": cfg.name,
        "asset_level": "transformer",
        "assets": len(stats),
        "folder": str(folder),
        "source": "meter aggregation",
        "source_checksum": store.digest(config),
        "rows": int((bounds[1] - bounds[0]) / step) + 1,
        "readings": total,
        "valid_rows": int((bounds[1] - bounds[0]) / step) + 1,
        "quality": {
            "mean_coverage": float(stats.observed.sum() / stats.expected.sum()),
            "excluded_timestamps": 0,
        },
        "aggregation": cfg.model_dump(),
        "created_at": store.now(),
        "config": {**datasets[0]["config"], "asset_level": "transformer"},
    }
    (folder / "manifest.json").write_text(store.encode(meta), encoding="utf-8")
    return store.put("dataset", meta, result_id)
