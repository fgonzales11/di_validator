"""Public response contracts. Algorithm-specific evidence remains extensible."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Channel


class Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class Record(Response):
    id: str
    created_at: str


class Dataset(Record):
    name: str
    version: int
    format: Literal["wide_ami", "recording"]
    asset_level: Literal["meter", "transformer", "terminal"]
    source: str
    source_checksum: str
    channels: list[Channel]
    timezone: str
    interval_position: Literal["unknown", "start", "end"]
    quality: dict[str, Any]


class Asset(Record):
    source_id: str
    circuit: str
    level: Literal["meter", "transformer", "terminal"]


class Label(Record):
    asset_id: str
    target: Literal["PV", "EV"]
    value: Literal[0, 1] | None
    evidence: str
    valid_from: str
    valid_to: str
    verification: Literal["confirmed", "provisional"]


class Annotation(Record):
    dataset_id: str
    asset_id: str
    kind: str
    start: float
    end: float
    partition: Literal["development", "evaluation"]


class Experiment(Record):
    name: str
    kind: Literal["classification", "events"]
    config: dict[str, Any]
    interpretation: str


class Job(Response):
    id: str
    kind: str
    status: Literal["queued", "running", "completed", "cancelled", "failed", "interrupted"]
    config: dict[str, Any]
    progress: float = Field(ge=0, le=1)
    message: str
    result: dict[str, Any] | None
    error: str | None
    created_at: str
    updated_at: str
    cancel_requested: int


class Trace(Response):
    x: list[str | float | int]
    y: list[str | float | int | None]


class Series(Response):
    traces: list[Trace]
    aggregation: str
    max_points: int


class Algorithm(Response):
    id: str
    name: str
    family: str
    status: Literal["ready", "unverified"]


class PathInput(BaseModel):
    path: str
