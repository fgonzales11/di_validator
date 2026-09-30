"""Datasets HTTP endpoints."""

import csv
import json
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile

from .. import responses, store
from ..ingest import load_quality
from ..query import recording_window, series
from ..schemas import AggregateConfig, ImportConfig

router = APIRouter(prefix="/api/v1")


@router.get("/sources")
def sources():
    imported = {d["source"] for d in store.listing("dataset")}
    rows = []
    for p in sorted((store.ROOT / "data").glob("*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream), [])
        layout = "mapping_required"
        if p.name == "PV_LABELS.csv":
            layout = "label_registry"
        elif p.name.startswith("AMI_") and "REPORTED_DTTM" in columns:
            layout = "wide_ami"
        rows.append(
            dict(
                name=p.name,
                path=str(p.resolve()),
                bytes=p.stat().st_size,
                registered=str(p.resolve()) in imported,
                format=layout,
            )
        )
    return rows


@router.post("/bootstrap", response_model=list[responses.Job])
def bootstrap():
    results = []
    pending = {j["config"].get("path") for j in store.jobs() if j["status"] in {"queued", "running"}}
    registries = store.listing("registry")
    for source in sources():
        if source["path"] in pending:
            continue
        if source["name"] == "PV_LABELS.csv":
            if not registries:
                results.append(store.enqueue("registry", {"path": source["path"]}))
        elif source["format"] == "wide_ami" and not source["registered"]:
            results.append(store.enqueue("import", ImportConfig(path=source["path"]).model_dump()))
    return results


@router.get("/datasets", response_model=list[responses.Dataset])
def datasets():
    return store.listing("dataset")


@router.post("/datasets", response_model=responses.Job)
def import_data(config: ImportConfig):
    return store.enqueue("import", config.model_dump())


@router.post("/uploads")
async def upload(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".csv", ".parquet"}:
        raise ValueError("Upload CSV or Parquet")
    path = store.workspace() / "uploads" / (store.uid() + extension)
    with path.open("wb") as stream:
        while chunk := await file.read(1024 * 1024):
            stream.write(chunk)
    return {"path": str(path), "name": Path(file.filename).stem}


@router.get("/datasets/{dataset_id}", response_model=responses.Dataset)
def dataset(dataset_id: str):
    return store.get("dataset", dataset_id)


@router.get("/datasets/{dataset_id}/quality")
def quality(dataset_id: str):
    selected = store.get("dataset", dataset_id)
    audit = Path(selected["folder"]) / "timestamp_audit.json"
    return dict(
        summary=selected["quality"],
        assets=load_quality(selected),
        timestamp_audit=json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else [],
    )


@router.get("/assets", response_model=list[responses.Asset])
def assets(dataset_id: str | None = None, search: str = "", limit: int = Query(200, ge=1, le=10000)):
    rows = store.listing("asset")
    if dataset_id:
        selected = store.get("dataset", dataset_id)
        allowed = (
            {q["asset_id"] for q in load_quality(selected)}
            if selected["format"] == "wide_ami"
            else {selected["asset_id"]}
        )
        rows = [r for r in rows if r["id"] in allowed]
    return [r for r in rows if search.lower() in (r["source_id"] + r["circuit"]).lower()][:limit]


@router.get("/series", response_model=responses.Series)
def time_series(
    dataset_id: str,
    asset_ids: str,
    start: str | None = None,
    end: str | None = None,
    view: str = "series",
    power: bool = False,
    max_points: int = Query(10000, ge=100, le=10000),
):
    if view not in {"series", "daily", "seasonal", "distribution", "heatmap"}:
        raise ValueError("Unknown plot view")
    return series(dataset_id, asset_ids.split(","), start, end, view, power, max_points)


@router.get("/recordings/{dataset_id}/window", response_model=responses.Series)
def window(
    dataset_id: str,
    start: float = 0,
    end: float | None = None,
    max_points: int = Query(10000, ge=100, le=10000),
):
    return recording_window(dataset_id, start, end, max_points)


@router.post("/aggregate", response_model=responses.Job)
def aggregate(config: AggregateConfig):
    body = config.model_dump()
    body["relationship_snapshot"] = store.listing("relationship")
    return store.enqueue("aggregate", body)
