"""Experiments HTTP endpoints."""

import json
import zipfile
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import responses, store
from ..schemas import ExperimentConfig

router = APIRouter(prefix="/api/v1")


@router.get("/algorithms", response_model=list[responses.Algorithm])
def algorithms():
    from ..adapters import EVENT_DETECTORS, load_plugins
    from ..classification import ALLOWED_PARAMS, MODEL_NAMES

    load_plugins()
    current = [
        dict(
            id=code,
            name=name,
            family="Classification",
            status="ready",
            asset_levels=["meter", "transformer"],
            sampling="hourly",
            required_channels=["net_energy"],
            parameters=sorted(ALLOWED_PARAMS.get(code, [])),
        )
        for code, name in MODEL_NAMES.items()
    ]
    current += [
        dict(
            id="load-step",
            name="Causal load-step detector",
            family="Events",
            status="ready",
            sampling="native samples",
            required_channels=["power or current RMS"],
        ),
        dict(
            id="voltage-events",
            name="Voltage disturbance detector",
            family="Events",
            status="ready",
            sampling="native samples",
            required_channels=["voltage RMS or voltage waveform"],
        ),
    ]
    current.append(
        dict(
            id="fault-distance",
            name="Fault inception + distance",
            family="Events",
            status="ready",
            sampling="native waveforms; offline analysis",
            required_channels=["IA", "IB", "IC", "UA", "UB", "UC"],
            asset_levels=["terminal", "meter", "transformer"],
            method="Single-ended simple reactance; configurable line impedance",
        )
    )
    current += [
        dict(
            id=identifier,
            name=spec["name"],
            family="Events",
            status="ready",
            sampling="native samples",
            required_channels=sorted(spec["channels"]),
        )
        for identifier, spec in EVENT_DETECTORS.items()
    ]
    for p in (store.ROOT / "reference" / "tools" / "ModelManager" / "Models").glob("*"):
        if p.suffix in {".pickle", ".h5"}:
            current.append(
                dict(
                    id=p.stem,
                    name=p.name,
                    family="Legacy inventory",
                    status="unverified",
                    reason="Environment, feature order, training granularity and channel compatibility require verification; not loaded automatically.",
                )
            )
    return current


@router.get("/experiments", response_model=list[responses.Experiment])
def experiments():
    return store.listing("experiment")


@router.post("/experiments", response_model=responses.Job)
def experiment(config: ExperimentConfig):
    from ..classification import freeze

    return store.enqueue("classification", freeze(config.model_dump()))


@router.get("/experiments/{experiment_id}", response_model=responses.Experiment)
def experiment_detail(experiment_id: str):
    return store.get("experiment", experiment_id)


@router.get("/experiments/{experiment_id}/predictions")
def predictions(experiment_id: str, model: str = "", errors_only: bool = False):
    selected = store.get("experiment", experiment_id)
    path = Path(selected["folder"]) / "predictions.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return [
        r
        for r in rows
        if (not model or r["model"] == model) and (not errors_only or r["prediction"] != r["label"])
    ]


@router.get("/experiments/{experiment_id}/export")
def export(experiment_id: str):
    selected = store.get("experiment", experiment_id)
    folder = Path(selected["folder"])
    path = store.workspace() / "exports" / f"{experiment_id}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in folder.iterdir():
            if item.is_file():
                archive.write(item, item.name)
    return FileResponse(path, filename=f"di-validator-{experiment_id[:8]}.zip")
