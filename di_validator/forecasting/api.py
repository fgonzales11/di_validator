from pathlib import Path
import json
import zipfile

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from .. import store, responses
from .contracts import ForecastConfig, ForecastRun, RuntimeSetup
from .catalog import MODELS, canonical, catalog
from .engine import freeze, source_columns

router = APIRouter(prefix="/api/v1/forecasting", tags=["Forecasting"])


@router.get("/models")
def models():
    return catalog()


@router.get("/source-columns")
def columns(name: str):
    return source_columns(name)


@router.post("/setup", response_model=responses.Job)
def setup(body: RuntimeSetup):
    if canonical(body.model) not in MODELS:
        raise ValueError("Unknown forecasting model")
    return store.enqueue("forecast_setup", body.model_dump())


@router.get("/runs", response_model=list[ForecastRun])
def runs():
    return store.listing("forecast")


@router.post("/runs", response_model=responses.Job)
def create(body: ForecastConfig):
    return store.enqueue("forecast", freeze(body.model_dump()))


@router.get("/runs/{run_id}", response_model=ForecastRun)
def detail(run_id: str):
    return store.get("forecast", run_id)


@router.get("/runs/{run_id}/predictions")
def predictions(run_id: str, model: str = "", phase: str = "", limit: int = Query(20000, ge=1, le=100000)):
    run = store.get("forecast", run_id)
    rows = json.loads((Path(run["folder"]) / "predictions.json").read_text())
    return [r for r in rows if (not model or r["model"] == model) and (not phase or r["phase"] == phase)][
        :limit
    ]


@router.get("/runs/{run_id}/export")
def export(run_id: str):
    run = store.get("forecast", run_id)
    folder = Path(run["folder"])
    path = store.workspace() / "exports" / ("forecast-" + run_id + ".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in folder.rglob("*"):
            relative = item.relative_to(folder)
            if item.is_file() and (len(relative.parts) == 1 or relative.parts[0] == "artifacts"):
                archive.write(item, relative.as_posix())
    return FileResponse(path, filename=f"di-forecast-{run_id[:8]}.zip")
