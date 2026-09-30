"""Workspace HTTP endpoints."""

from fastapi import APIRouter

from .. import store
from ..forecasting.contracts import ForecastConfig
from ..schemas import EventConfig, ExperimentConfig, ImportConfig

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health():
    return dict(status="ok", workspace=str(store.workspace()), version="0.1.0")


@router.get("/overview")
def overview():
    datasets = store.listing("dataset")
    assets = store.listing("asset")
    experiments = store.listing("experiment")
    return dict(
        datasets=len(datasets),
        assets=len(assets),
        readings=sum(d["readings"] for d in datasets),
        experiments=len(experiments),
        jobs=store.jobs()[:8],
        registries=store.listing("registry"),
    )


@router.get("/presets")
def presets():
    return store.listing("preset")


@router.post("/presets")
def preset(body: dict):
    if body.get("kind") not in {"import", "classification", "events", "forecast"} or not body.get("name"):
        raise ValueError("Preset needs a name and supported kind")
    schema = {
        "import": ImportConfig,
        "classification": ExperimentConfig,
        "events": EventConfig,
        "forecast": ForecastConfig,
    }[body["kind"]]
    body["config"] = schema.model_validate(body["config"]).model_dump()
    return store.put("preset", body)
