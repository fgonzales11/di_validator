"""Labels HTTP endpoints."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from .. import responses, store
from ..labels import validated_period
from ..schemas import LabelConfig, RelationshipConfig

router = APIRouter(prefix="/api/v1")


@router.get("/labels", response_model=list[responses.Label])
def labels():
    return store.listing("label")


@router.post("/labels", response_model=responses.Label)
def label(config: LabelConfig):
    store.get("asset", config.asset_id)
    validated_period(config.valid_from, config.valid_to)
    return store.put("label", config.model_dump())


@router.post("/labels/import", response_model=list[responses.Label])
def label_import(body: responses.PathInput):
    path = Path(body.path)
    frame = pd.read_csv(path, dtype={"asset_id": "string"}).astype(object)
    rows = frame.where(pd.notna(frame), None).to_dict("records")
    checked = [LabelConfig.model_validate(r) for r in rows]
    for c in checked:
        store.get("asset", c.asset_id)
        validated_period(c.valid_from, c.valid_to)
    return [label(c) for c in checked]


@router.post("/registry", response_model=responses.Job)
def registry(body: responses.PathInput):
    return store.enqueue("registry", {"path": body.path})


@router.get("/relationships")
def relationships():
    return store.listing("relationship")


@router.post("/relationships")
def relationship(config: RelationshipConfig):
    validated_period(config.valid_from, config.valid_to)
    meter, transformer = store.get("asset", config.meter_id), store.get("asset", config.transformer_id)
    if meter["level"] != "meter" or transformer["level"] != "transformer":
        raise ValueError("Select a meter and transformer, respectively")
    return store.put("relationship", config.model_dump())


@router.post("/relationships/import")
def relationship_import(body: responses.PathInput):
    rows = pd.read_csv(body.path, dtype="string").to_dict("records")
    configs = [RelationshipConfig.model_validate(r) for r in rows]
    for c in configs:
        validated_period(c.valid_from, c.valid_to)
        for key, level in [(c.meter_id, "meter"), (c.transformer_id, "transformer")]:
            pieces = key.split(":", 2)
            if len(pieces) != 3 or pieces[0] != level:
                raise ValueError("Relationship IDs use level:circuit:source_id")
    # Inventory may include expected meters without readings and transformers without direct recordings.
    for c in configs:
        for key, level in [(c.meter_id, "meter"), (c.transformer_id, "transformer")]:
            _, circuit, source_id = key.split(":", 2)
            store.put("asset", dict(id=key, circuit=circuit, source_id=source_id, level=level), key)
    return [relationship(c) for c in configs]
