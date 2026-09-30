from typing import Literal
from pydantic import ConfigDict, Field, model_validator
from ..schemas import Model

Agent = Literal["pv", "fault"]
RunState = Literal[
    "preparing", "running", "paused", "draining", "completed", "cancelled", "failed", "interrupted"
]


class Mapping(Model):
    timestamp: str = "timestamp"
    timestamp_unit: Literal["utc_seconds", "iso_utc", "seconds"] = "utc_seconds"
    channels: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    validity: dict[str, str] = Field(default_factory=dict)
    asset_level: Literal["meter", "terminal", "transformer"] = "meter"
    aggregate_power: bool = False
    sample_rate: float = Field(default=1, gt=0, le=100000)
    grid_frequency: float = Field(default=50, ge=45, le=65)


class Parameters(Model):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    magnitude: float = Field(default=1, gt=0, le=100)
    load_scale: float = Field(default=1, ge=0, le=20)
    pv_scale: float = Field(default=1, ge=0, le=20)
    cloud_variation: float = Field(default=0, ge=0, le=1)
    noise: float = Field(default=0, ge=0, le=1)
    polarity: Literal[-1, 1] = 1
    outage_start: float | None = Field(default=None, ge=0)
    outage_seconds: float = Field(default=0, ge=0, le=86400 * 10)
    onset_seconds: float = Field(default=0.1, ge=0.02, le=2)
    fault_type: Literal["AG", "BG", "CG", "AB", "BC", "CA", "POS", "none"] = "AG"
    distance_km: float = Field(default=10, ge=0, le=1000)
    frequency: Literal[50, 60] = 50
    days: int = Field(default=35, ge=1, le=60)
    start_offset_seconds: float = Field(default=0, ge=0, le=60 * 86400)
    duration_seconds: float | None = Field(default=None, gt=0, le=60 * 86400)


class ScenarioRequest(Model):
    agent: Agent
    name: str = Field(default="", max_length=100)
    source: Literal["preset", "dataset", "upload"] = "preset"
    source_id: str = Field(min_length=1, max_length=100)
    mapping: Mapping = Field(default_factory=Mapping)
    parameters: Parameters = Field(default_factory=Parameters)
    configuration: dict[str, str | float | int] = Field(default_factory=dict)


class RunRequest(Model):
    scenario_id: str = Field(pattern="^[0-9a-f]{32}$")
    meter_form: Literal["GENX_SP", "GENX_PP"] = "GENX_PP"
    speed: Literal["0.1", "1", "10", "100", "1000", "fastest"] = "fastest"
    mode: Literal["replay", "metrology"] = "replay"
    seed_run_id: str | None = Field(default=None, pattern="^[0-9a-f]{32}$")
    seed_fixture: Literal["warm_state"] | None = None
    chunk_size: int = Field(default=1024, ge=1, le=1024)

    @model_validator(mode="after")
    def one_seed(self):
        if self.seed_run_id and self.seed_fixture:
            raise ValueError("Select one initial state")
        return self


class Control(Model):
    action: Literal["pause", "resume", "stop"]


class Telemetry(Model):
    model_config = ConfigDict(extra="allow")
    version: int = 1
    run_id: str
    state: RunState
    transmitted: int = 0
    received: int = 0
    processed: int = 0
    total_samples: int = 0
    queue: int = 0
    pending: int = 0
    rejected: int = 0
    stored_data: int = 0
    stored_events: int = 0
    diagnostics: int = 0
    scenario_time: float = 0


class AgentDescriptor(Model):
    id: Agent
    name: str
    forms: list[Literal["GENX_SP", "GENX_PP"]]
    configuration: dict[str, str]
    policy: dict[str, int]
    input: str
    timing: str


class Scenario(Model):
    model_config = ConfigDict(extra="allow")
    id: str
    agent: Agent
    name: str
    request: ScenarioRequest
    encoding: str
    total_samples: int
    input_sha256: str


class Check(Model):
    category: Literal["numerical", "expectation", "delivery", "structural"]
    name: str
    verdict: Literal["pass", "fail", "unavailable"]
    detail: str = ""


class CheckReport(Model):
    model_config = ConfigDict(extra="allow")
    verdict: Literal["pending", "pass", "fail", "unavailable"]
    checks: list[Check]


class Run(Model):
    id: str
    agent: Agent
    name: str
    state: RunState
    created_at: str
    manifest: dict
    telemetry: Telemetry
    checks: CheckReport
    checkpoint_available: bool


class Outcome(Model):
    model_config = ConfigDict(extra="allow")
    source: Literal["DataServer stored outcome"]
    table: Literal["AgentData", "AgentEvents"]
    Id: int
    AgentId: int
    FeatureId: int
    TimeStamp: int
    bytes: int
    payload: str
    decoded: dict | None
    decode_error: str | None


class Outcomes(Model):
    rows: list[Outcome]
    cursor: int
    total: int


class TelemetryPage(Model):
    cursor: int
    events: list[Telemetry]
    status: Telemetry
