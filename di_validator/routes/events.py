"""Events HTTP endpoints."""

from fastapi import APIRouter

from .. import responses, store
from ..labels import active_annotations
from ..schemas import AnnotationConfig, EventConfig, SyntheticConfig

router = APIRouter(prefix="/api/v1")


@router.get("/annotations", response_model=list[responses.Annotation])
def annotations(dataset_id: str):
    return [r for r in active_annotations() if r["dataset_id"] == dataset_id]


@router.post("/annotations", response_model=responses.Annotation)
def annotation(config: AnnotationConfig):
    selected = store.get("dataset", config.dataset_id)
    if selected.get("asset_id") != config.asset_id or config.end > selected.get(
        "duration_seconds", 0
    ) + 1 / selected.get("sample_rate", 1):
        raise ValueError("Annotation must refer to the recording asset and lie inside its duration")
    existing = [r for r in store.listing("annotation") if r["asset_id"] == config.asset_id]
    if any(a["partition"] != config.partition for a in existing):
        raise ValueError("Keep each asset and all its recordings in one development/evaluation partition")
    return store.put("annotation", config.model_dump())


@router.post("/annotations/{annotation_id}/withdraw", response_model=responses.Annotation)
def withdraw_annotation(annotation_id: str):
    previous = store.get("annotation", annotation_id)
    revision = {k: v for k, v in previous.items() if k not in {"id", "created_at"}}
    return store.put("annotation", {**revision, "supersedes": annotation_id, "withdrawn": True})


@router.post("/events/run", response_model=responses.Job)
def detect(config: EventConfig):
    body = config.model_dump()
    if config.algorithm == "fault-distance":
        from ..faults.pipeline import validate_dataset

        body["parameters"] = validate_dataset(
            store.get("dataset", config.dataset_id), config.parameters
        ).model_dump()
    body["annotation_snapshot"] = active_annotations()
    return store.enqueue("events", body)


@router.post("/synthetic", response_model=responses.Job)
def generate(config: SyntheticConfig):
    return store.enqueue("synthetic", config.model_dump())
