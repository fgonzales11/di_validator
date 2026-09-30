from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Channel(Model):
    name: str = Field(min_length=1)
    kind: Literal["net_energy", "power", "voltage", "current", "voltage_rms", "current_rms"]
    unit: Literal["kWh", "Wh", "kW", "W", "V", "A"]
    phase: str = "A"

    @model_validator(mode="after")
    def compatible_unit(self):
        units = {
            "net_energy": {"kWh", "Wh"},
            "power": {"kW", "W"},
            "voltage": {"V"},
            "voltage_rms": {"V"},
            "current": {"A"},
            "current_rms": {"A"},
        }
        if self.unit not in units[self.kind]:
            raise ValueError(f"{self.kind} is incompatible with {self.unit}")
        return self


class ImportConfig(Model):
    path: str
    name: str = ""
    format: Literal["wide_ami", "recording"] = "wide_ami"
    asset_level: Literal["transformer", "meter"] = "transformer"
    circuit: str = ""
    asset_id: str = "recording-asset"
    timestamp_column: str = "REPORTED_DTTM"
    timezone: str = "America/Los_Angeles"
    interval_seconds: float = Field(default=3600, gt=0)
    interval_position: Literal["unknown", "start", "end"] = "unknown"
    sample_rate: float = Field(default=1000, gt=0, le=1000000)
    start_time: str | None = None
    time_mode: Literal["timestamp", "offset_seconds", "sample_index"] = "timestamp"
    channels: list[Channel] = Field(
        default_factory=lambda: [Channel(name="net_energy", kind="net_energy", unit="kWh")], min_length=1
    )
    nominal_frequency: float = Field(default=60, gt=0, le=100)
    nominal_voltage: float = Field(default=120, gt=0)


class ExperimentConfig(Model):
    name: str = "Classification benchmark"
    dataset_ids: list[str] = Field(min_length=1)
    asset_ids: list[str] = Field(default_factory=list)
    target: Literal["PV", "EV"] = "PV"
    label_policy: Literal["confirmed", "registry", "provisional", "heuristic"] = "registry"
    start: str | None = None
    end: str | None = None
    min_coverage: float = Field(default=0.9, gt=0, le=1)
    seed: int = 42
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    folds: int = Field(default=5, ge=2, le=10)
    models: list[str] = Field(default_factory=lambda: ["logistic", "svm", "forest", "boosting"], min_length=1)
    parameters: dict = Field(default_factory=dict)
    threshold_policy: Literal["default", "training_f1"] = "default"
    feature_groups: list[Literal["load", "calendar", "ramps", "shape"]] = Field(
        default_factory=lambda: ["load", "calendar", "ramps", "shape"], min_length=1
    )
    label_snapshot: list[dict] | None = None
    relationship_snapshot: list[dict] | None = None
    registry_version: str | None = None
    code_version: str | None = None


class LabelConfig(Model):
    asset_id: str
    target: Literal["PV", "EV"]
    value: Literal[0, 1] | None
    evidence: str = Field(min_length=1)
    valid_from: str
    valid_to: str
    verification: Literal["confirmed", "provisional"] = "confirmed"


class AnnotationConfig(Model):
    dataset_id: str
    asset_id: str
    kind: Literal["review", "load_on", "load_off", "voltage_dip", "voltage_rise", "interruption", "fault"]
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    note: str = ""
    partition: Literal["development", "evaluation"] = "evaluation"

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("End must not precede start")
        return self


class EventConfig(Model):
    dataset_id: str
    algorithm: str = "baseline-events"
    parameters: dict = Field(default_factory=dict)
    name: str = "Event detection"
    step_threshold: float = Field(default=1.0, gt=0)
    dip_ratio: float = Field(default=0.9, gt=0, lt=1)
    rise_ratio: float = Field(default=1.1, gt=1)
    interruption_ratio: float = Field(default=0.1, gt=0, lt=1)
    minimum_duration: float = Field(default=0.02, ge=0.001)
    match_tolerance: float = Field(default=0.1, gt=0)
    partition: Literal["development", "evaluation"] = "evaluation"
    annotation_snapshot: list[dict] | None = None

    @model_validator(mode="after")
    def voltage_threshold_order(self):
        if self.interruption_ratio >= self.dip_ratio:
            raise ValueError("Interruption threshold must be below the voltage dip threshold")
        return self


class SyntheticConfig(Model):
    name: str = "Switching & voltage disturbances"
    duration_seconds: float = Field(default=120, ge=10, le=86400)
    sample_rate: int = Field(default=1000, ge=100, le=10000)
    seed: int = 42


class RelationshipConfig(Model):
    meter_id: str
    transformer_id: str
    valid_from: str
    valid_to: str


class AggregateConfig(Model):
    dataset_ids: list[str] = Field(min_length=1)
    name: str = "Meter aggregation"
    min_meter_coverage: float = Field(default=1, gt=0, le=1)
    relationship_snapshot: list[dict] | None = None
