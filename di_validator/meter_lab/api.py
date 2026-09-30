from pathlib import Path
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from .. import store
from . import scenarios, service
from .schemas import (
    ScenarioRequest,
    RunRequest,
    Control,
    AgentDescriptor,
    Scenario,
    Run,
    Outcomes,
    TelemetryPage,
)

router = APIRouter(prefix="/api/v1/meter-lab", tags=["Meter Lab"])


@router.get("/capabilities")
def capabilities():
    return service.capabilities()


@router.get("/agents", response_model=list[AgentDescriptor])
def agents():
    return service.capabilities()["agents"]


@router.get("/scenarios", dependencies=[Depends(service.local)])
def listing():
    return {"catalog": scenarios.catalog(), "saved": store.listing("meter_lab_scenario")}


@router.post("/scenarios", dependencies=[Depends(service.local)], response_model=Scenario)
def create(body: ScenarioRequest):
    return scenarios.create(body)


@router.post("/scenarios/validate", dependencies=[Depends(service.local)])
def validate(body: ScenarioRequest):
    result = scenarios.build(body)
    result.pop("values")
    result.pop("source_path")
    return {"valid": True, **result}


@router.get("/scenarios/{identifier}/preview", dependencies=[Depends(service.local)])
def preview(identifier: str):
    return scenarios.preview(store.get("meter_lab_scenario", identifier))


@router.post("/uploads", dependencies=[Depends(service.local)])
async def upload(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 2:
        raise ValueError("Upload one CSV/Parquet file, or one CFG/DAT pair")
    extensions = [Path(f.filename or "").suffix.lower().lstrip(".") for f in files]
    if not (extensions in [["csv"], ["parquet"]] or sorted(extensions) == ["cfg", "dat"]):
        raise ValueError("Supported formats: CSV, Parquet, ASCII COMTRADE CFG/DAT pair")
    identifier = store.uid()
    path = scenarios.folder("uploads", identifier)
    path.mkdir(parents=True)
    total = 0
    for file, extension in zip(files, extensions):
        with (path / ("source." + extension)).open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise ValueError("Uploads are limited to 64 MiB; select a shorter input")
                stream.write(chunk)
    columns = []
    fmt = "comtrade" if len(files) == 2 else extensions[0]
    if fmt != "comtrade":
        if fmt == "csv":
            import pandas as pd

            columns = list(pd.read_csv(path / "source.csv", nrows=0).columns)
        else:
            import pyarrow.parquet as pq

            columns = pq.ParquetFile(path / "source.parquet").schema.names
    return store.put(
        "meter_lab_upload",
        {
            "id": identifier,
            "format": fmt,
            "bytes": total,
            "columns": columns,
            "name": Path(files[0].filename or "Upload").stem,
        },
    )


@router.post("/runs", dependencies=[Depends(service.local)], response_model=Run)
def start(body: RunRequest):
    return service.create(body)


@router.get("/runs", dependencies=[Depends(service.local)], response_model=list[Run])
def runs():
    return service.listing()


@router.get("/runs/{identifier}", dependencies=[Depends(service.local)], response_model=Run)
def run(identifier: str):
    return service.detail(identifier)


@router.post("/runs/{identifier}/control", dependencies=[Depends(service.local)])
def control(identifier: str, body: Control):
    return service.control(identifier, body.action)


@router.get(
    "/runs/{identifier}/telemetry", dependencies=[Depends(service.local)], response_model=TelemetryPage
)
def telemetry(identifier: str, cursor: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=100)):
    return service.telemetry(identifier, cursor, limit)


@router.get("/runs/{identifier}/series", dependencies=[Depends(service.local)])
def series(
    identifier: str,
    start: float | None = None,
    end: float | None = None,
    max_points: int = Query(1500, ge=100, le=10000),
):
    return service.series(identifier, start, end, max_points)


@router.get("/runs/{identifier}/outcomes", dependencies=[Depends(service.local)], response_model=Outcomes)
def outcomes(identifier: str, cursor: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
    rows = service.outcomes(identifier)
    return {
        "rows": rows[cursor : cursor + limit],
        "cursor": min(len(rows), cursor + limit),
        "total": len(rows),
    }


@router.get("/runs/{identifier}/logs", dependencies=[Depends(service.local)])
def logs(identifier: str):
    service.detail(identifier)
    path = scenarios.folder("runs", identifier)
    result = []
    files = [
        path / "supervisor.log",
        path / "delivery.log",
        *sorted((path / "logs").glob("*.log")),
        *sorted((path / "agent-logs").rglob("*_log")),
    ]
    for file in files:
        if not file.is_file():
            continue
        with file.open("rb") as source:
            source.seek(max(0, file.stat().st_size - 8000))
            result.append(
                {"name": file.relative_to(path).as_posix(), "text": source.read().decode("utf-8", "replace")}
            )
    return result


@router.get("/compare", dependencies=[Depends(service.local)])
def compare(left: str, right: str, start: float | None = None, end: float | None = None):
    a, b = service.detail(left), service.detail(right)
    if a["agent"] != b["agent"]:
        raise ValueError("Compare two runs of the same agent")
    return {
        "left": a,
        "right": b,
        "left_series": service.series(left, start, end),
        "right_series": service.series(right, start, end),
        "alignment": "scenario time; no resampling",
        "configuration_equal": a["manifest"]["configuration"] == b["manifest"]["configuration"],
    }


@router.get("/runs/{identifier}/export", dependencies=[Depends(service.local)])
def export(identifier: str):
    from starlette.background import BackgroundTask

    archive = service.export(identifier)
    return FileResponse(
        archive,
        media_type="application/zip",
        filename="meter-lab-" + identifier + ".zip",
        background=BackgroundTask(archive.unlink, missing_ok=True),
    )
