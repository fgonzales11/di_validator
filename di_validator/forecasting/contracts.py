from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from ..schemas import Model


class ForecastConfig(Model):
    name: str = "Forecast benchmark"
    dataset_id: str | None = None
    asset_id: str = ""
    channel: str = ""
    source_file: str | None = None
    timestamp_column: str = "Timestamp"
    unit: str = "unverified"
    start: str | None = None
    end: str | None = None
    horizon: int = Field(default=24, ge=1, le=512)
    validation_windows: int = Field(default=3, ge=2, le=8)
    context_length: int = Field(default=2048, ge=32, le=200000)
    max_history: int = Field(default=10000, ge=100, le=200000)
    season_length: int = Field(default=24, ge=1, le=10000)
    lags: int = Field(default=24, ge=1, le=1024)
    models: list[str] = Field(
        default_factory=lambda: ["seasonal_naive", "random_forest", "xgboost", "lightgbm", "ensemble"],
        min_length=1,
    )
    parameters: dict[str, dict] = Field(default_factory=dict)
    missing_policy: Literal["reject", "forward_fill", "interpolate"] = "reject"
    max_gap_steps: int = Field(default=3, ge=1, le=100)
    min_coverage: float = Field(default=0.9, ge=0.5, le=1)
    seed: int = 42
    timeout_seconds: int = Field(default=600, ge=10, le=86400)
    source_snapshot: dict | None = None
    code_version: str | None = None
    reuse_run_id: str | None = None
    reuse_model: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        if bool(self.dataset_id) == bool(self.source_file):
            raise ValueError("Choose exactly one indexed dataset or source CSV")
        if not self.channel:
            raise ValueError("Choose the measurement to forecast")
        if self.context_length <= self.lags + 10:
            raise ValueError("Context must contain at least lags + 11 observations")
        if self.season_length > self.context_length:
            raise ValueError("Season length must not exceed the context window")
        if self.max_history < self.context_length:
            raise ValueError("Maximum history must be at least the context window")
        if bool(self.reuse_run_id) != bool(self.reuse_model):
            raise ValueError("Saved-model inference needs a run and model ID")
        return self


class RuntimeSetup(Model):
    model: str


class ForecastRun(Model):
    id: str
    name: str
    created_at: str
    config: dict
    source: dict
    selected_model: str
    models: list[dict]
    mode: str
    folder: str
    selection_policy: str
    population_id: str
    preprocessing: list[dict] = Field(default_factory=list)
