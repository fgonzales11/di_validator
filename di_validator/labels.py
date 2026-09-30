from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import store

REGISTRY_COLUMNS = {
    "STRUCT_NUM",
    "PROJECT_ID",
    "TECH_TYPE",
    "INTERCONNECTION_STATUS",
    "PERMTO_OPERATE_DT",
    "CONTRACT_END_DT",
    "SERV_PNT_ID",
    "CIRCUIT_KEY",
    "EDW_MODIFIED_DATE",
}


def import_registry(path):
    path = Path(path).resolve()
    version = store.checksum(path)[:24]
    try:
        return store.get("registry", version)
    except ValueError:
        pass
    frame = pd.read_csv(path, dtype="string", usecols=lambda c: c in REGISTRY_COLUMNS)
    required = REGISTRY_COLUMNS - {"CONTRACT_END_DT", "SERV_PNT_ID"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Registry needs {sorted(required)}")
    modified = pd.to_datetime(frame.EDW_MODIFIED_DATE, errors="coerce")
    if not modified.notna().any():
        raise ValueError("Registry needs a valid evidence snapshot date")
    folder = store.workspace() / "datasets" / f"registry-{version}"
    folder.mkdir(exist_ok=True)
    frame.to_parquet(folder / "registry.parquet", index=False)
    return store.put(
        "registry",
        dict(
            id=version,
            source=str(path),
            checksum=store.checksum(path),
            path=str(folder / "registry.parquet"),
            rows=len(frame),
            cutoff=modified.max().normalize().isoformat(),
            circuits=sorted(frame.CIRCUIT_KEY.dropna().unique().tolist()),
        ),
        version,
    )


def registry_for_assets(registry, assets, start, end, provisional, folder):
    # Registry analysis loads the training stack; local UI startup does not
    # need it until a registry comparison is requested.
    from .reference_analysis import registry_pv_labels

    data = pd.read_parquet(registry["path"])
    rows, audits = [], []
    for circuit in sorted({a["circuit"] for a in assets}):
        selected = [a for a in assets if a["circuit"] == circuit and a["level"] == "transformer"]
        if not selected:
            continue
        records = data.loc[data.CIRCUIT_KEY.eq(circuit)]
        if records.empty:
            # Missing circuit coverage is not evidence of a negative, even in exploratory mode.
            rows.extend(
                dict(asset_id=a["id"], value=None, evidence="circuit_not_in_registry") for a in selected
            )
            continue
        path = Path(folder) / f"registry-{store.digest(circuit)[:10]}.csv"
        records.to_csv(path, index=False)
        _, summary, audit, _ = registry_pv_labels(
            path,
            pd.Index([a["source_id"] for a in selected]),
            start,
            end,
            no_record_as_negative=provisional,
            expected_circuit=circuit,
        )
        for asset in selected:
            row = summary.loc[asset["source_id"]]
            rows.append(
                dict(
                    asset_id=asset["id"],
                    value=None if pd.isna(row.has_pv) else int(row.has_pv),
                    evidence=row.label_basis,
                    verification="provisional" if pd.notna(row.has_pv) and row.has_pv == 0 else "registry",
                )
            )
        audits.append(audit)
    return rows, audits


def validated_period(start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(a) or pd.isna(b) or a > b:
        raise ValueError("Evidence dates must be valid and ordered")


def manual_for_window(snapshot, asset_id, target, start, end):
    start, end = pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
    matches = [
        x
        for x in snapshot
        if x["asset_id"] == asset_id
        and x["target"] == target
        and x["verification"] == "confirmed"
        and pd.Timestamp(x["valid_from"]).tz_localize(None) <= start
        and pd.Timestamp(x["valid_to"]).tz_localize(None) >= end
    ]
    return max(matches, key=lambda a: a["created_at"]) if matches else None


def active_annotations():
    history = store.listing("annotation")
    superseded = {r["supersedes"] for r in history if r.get("supersedes")}
    return [r for r in history if r["id"] not in superseded and not r.get("withdrawn")]
