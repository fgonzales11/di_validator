"""Serializable arc-flash study inputs and public result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from multiconductor.arcflash.ieee1584 import ElectrodeConfiguration


ARC_FLASH_LOCATION_COLUMNS = [
    "name",
    "bus",
    "equipment_id",
    "access_area",
    "equipment_type",
    "electrode_configuration",
    "conductor_gap_mm",
    "working_distance_mm",
    "enclosure_height_mm",
    "enclosure_width_mm",
    "enclosure_depth_mm",
    "protection_device_ids",
    "site_ppe",
    "limited_approach_boundary_mm",
    "restricted_approach_boundary_mm",
    "glove_class",
    "label_count",
    "in_service",
]

ARC_FLASH_SCENARIO_COLUMNS = [
    "name",
    "fault_case",
    "operating_condition",
    "description",
    "actions",
    "in_service",
]

ARC_FLASH_SOURCE_COLUMNS = [
    "name",
    "element_type",
    "element",
    "source_type",
    "decrement_profile",
    "in_service",
]

RES_ARC_FLASH_COLUMNS = [
    "location_id",
    "scenario_id",
    "location_name",
    "scenario_name",
    "equipment_id",
    "access_area",
    "bus",
    "fault_case",
    "operating_condition",
    "arcing_current_case",
    "nominal_voltage_kv",
    "working_distance_mm",
    "phase_currents_ka",
    "phase_current_spread_percent",
    "bolted_fault_current_ka",
    "initial_arcing_current_ka",
    "clearing_time_s",
    "duration_capped",
    "incident_energy_cal_cm2",
    "arc_flash_boundary_mm",
    "controlling_device_ids",
    "segment_count",
    "segments",
    "status",
    "warnings",
    "provenance",
]

RES_ARC_FLASH_WORST_COLUMNS = RES_ARC_FLASH_COLUMNS + [
    "site_ppe",
    "limited_approach_boundary_mm",
    "restricted_approach_boundary_mm",
    "glove_class",
    "report_number",
    "revision",
    "issue_date",
    "label_count",
]

RES_ARC_FLASH_OPERATION_COLUMNS = [
    "location_id",
    "scenario_id",
    "arcing_current_case",
    "sequence",
    "event_time_s",
    "event_type",
    "device_id",
    "switch_id",
    "stage",
    "current_ka",
    "operating_fraction",
    "opening_time_s",
    "details",
]


def _ensure_table(net, name: str, columns: Sequence[str]) -> pd.DataFrame:
    table = net.get(name)
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in table:
            table[column] = pd.Series(dtype=object)
    net[name] = table
    return table


def ensure_arc_flash_tables(net) -> None:
    """Add all input and result tables without altering existing rows."""

    _ensure_table(net, "arc_flash_location", ARC_FLASH_LOCATION_COLUMNS)
    _ensure_table(net, "arc_flash_scenario", ARC_FLASH_SCENARIO_COLUMNS)
    _ensure_table(net, "arc_flash_source", ARC_FLASH_SOURCE_COLUMNS)
    _ensure_table(net, "res_arc_flash", RES_ARC_FLASH_COLUMNS)
    _ensure_table(net, "res_arc_flash_worst", RES_ARC_FLASH_WORST_COLUMNS)
    _ensure_table(net, "res_arc_flash_operation", RES_ARC_FLASH_OPERATION_COLUMNS)


def _next_index(table: pd.DataFrame, index: int | None) -> int:
    if index is not None:
        selected = int(index)
        if selected in table.index:
            raise ValueError(f"index {selected} already exists")
        return selected
    numeric = [int(value) for value in table.index if isinstance(value, (int, np.integer))]
    return max(numeric, default=-1) + 1


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} value")
    return value


@dataclass(frozen=True, slots=True)
class ArcFlashStudyConfig:
    """Approved project inputs controlling an arc-flash study.

    The duration limit is mandatory.  It is an engineering input rather than
    an implicit software default; cases reaching it are marked as capped.
    """

    max_arc_duration_s: float
    study_id: str
    report_number: str
    revision: str
    issue_date: str | date
    phase_imbalance_tolerance_percent: float = 5.0
    frequency_hz: float = 60.0
    standard: str = "IEEE 1584-2018 with active errata"

    def __post_init__(self) -> None:
        _positive(self.max_arc_duration_s, "max_arc_duration_s")
        _positive(self.frequency_hz, "frequency_hz")
        tolerance = _positive(
            self.phase_imbalance_tolerance_percent,
            "phase_imbalance_tolerance_percent",
            allow_zero=True,
        )
        if tolerance > 100.0:
            raise ValueError("phase_imbalance_tolerance_percent cannot exceed 100")
        for field_name in ("study_id", "report_number", "revision"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        if not str(self.issue_date).strip():
            raise ValueError("issue_date is required")

    @property
    def issue_date_text(self) -> str:
        return self.issue_date.isoformat() if isinstance(self.issue_date, date) else str(self.issue_date)


@dataclass(slots=True)
class ArcFlashStudyResult:
    """All case, worst-case, and event audit tables returned by a study."""

    all_cases: pd.DataFrame
    worst_cases: pd.DataFrame
    operations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    study_config: ArcFlashStudyConfig | None = None


def create_arc_flash_location(
    net,
    bus: int,
    equipment_id: str,
    access_area: str,
    *,
    conductor_gap_mm: float,
    working_distance_mm: float,
    electrode_configuration: str,
    enclosure_height_mm: float | None = None,
    enclosure_width_mm: float | None = None,
    enclosure_depth_mm: float | None = None,
    protection_device_ids: Iterable[int] = (),
    site_ppe: str,
    limited_approach_boundary_mm: float,
    restricted_approach_boundary_mm: float,
    glove_class: str,
    label_count: int = 1,
    equipment_type: str | None = None,
    name: str | None = None,
    in_service: bool = True,
    index: int | None = None,
) -> int:
    """Register an equipment access area for analysis and labeling."""

    ensure_arc_flash_tables(net)
    table = net.arc_flash_location
    selected = _next_index(table, index)
    if not str(equipment_id).strip() or not str(access_area).strip():
        raise ValueError("equipment_id and access_area are required")
    if not str(site_ppe).strip() or not str(glove_class).strip():
        raise ValueError("site_ppe and glove_class are required site inputs")
    if int(label_count) < 1:
        raise ValueError("label_count must be at least 1")
    ec = ElectrodeConfiguration.normalize(electrode_configuration).value
    devices = sorted({int(value) for value in protection_device_ids})
    row = {
        "name": name or f"{equipment_id} - {access_area}",
        "bus": int(bus),
        "equipment_id": str(equipment_id),
        "access_area": str(access_area),
        "equipment_type": equipment_type,
        "electrode_configuration": ec,
        "conductor_gap_mm": _positive(conductor_gap_mm, "conductor_gap_mm"),
        "working_distance_mm": _positive(working_distance_mm, "working_distance_mm"),
        "enclosure_height_mm": enclosure_height_mm,
        "enclosure_width_mm": enclosure_width_mm,
        "enclosure_depth_mm": enclosure_depth_mm,
        "protection_device_ids": devices,
        "site_ppe": str(site_ppe),
        "limited_approach_boundary_mm": _positive(
            limited_approach_boundary_mm, "limited_approach_boundary_mm"
        ),
        "restricted_approach_boundary_mm": _positive(
            restricted_approach_boundary_mm, "restricted_approach_boundary_mm"
        ),
        "glove_class": str(glove_class),
        "label_count": int(label_count),
        "in_service": bool(in_service),
    }
    for column, value in row.items():
        table.at[selected, column] = value
    return selected


def _normalize_actions(actions: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for position, action in enumerate(actions or ()):
        if not isinstance(action, Mapping):
            raise TypeError(f"scenario action {position} must be a mapping")
        missing = {"table", "index", "column", "value"}.difference(action)
        if missing:
            raise ValueError(f"scenario action {position} is missing {sorted(missing)}")
        normalized.append(
            {
                "table": str(action["table"]),
                "index": action["index"],
                "column": str(action["column"]),
                "value": action["value"],
            }
        )
    return normalized


def create_arc_flash_scenario(
    net,
    name: str,
    fault_case: str,
    *,
    operating_condition: str = "normal",
    description: str = "",
    actions: Iterable[Mapping[str, Any]] | None = None,
    in_service: bool = True,
    index: int | None = None,
) -> int:
    """Register a max/min source and operating-topology scenario."""

    ensure_arc_flash_tables(net)
    table = net.arc_flash_scenario
    selected = _next_index(table, index)
    case = str(fault_case).strip().lower()
    if case not in {"max", "min"}:
        raise ValueError('fault_case must be "max" or "min"')
    condition = str(operating_condition).strip().lower()
    if condition not in {"normal", "emergency"}:
        raise ValueError('operating_condition must be "normal" or "emergency"')
    if not str(name).strip():
        raise ValueError("scenario name is required")
    row = {
        "name": str(name),
        "fault_case": case,
        "operating_condition": condition,
        "description": str(description),
        "actions": _normalize_actions(actions),
        "in_service": bool(in_service),
    }
    for column, value in row.items():
        table.at[selected, column] = value
    return selected


def _normalize_profile(profile: Iterable[Sequence[float]] | None) -> list[list[float]]:
    values: list[list[float]] = []
    for pair in profile or ():
        if len(pair) != 2:
            raise ValueError("decrement_profile entries must be (time_s, multiplier) pairs")
        values.append([float(pair[0]), float(pair[1])])
    if values:
        times = np.asarray([pair[0] for pair in values])
        multipliers = np.asarray([pair[1] for pair in values])
        if (
            np.any(~np.isfinite(times))
            or np.any(~np.isfinite(multipliers))
            or np.any(times < 0.0)
            or np.any(multipliers < 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "decrement_profile times must increase and all values must be finite and non-negative"
            )
    return values


def register_arc_flash_source(
    net,
    element_type: str,
    element: Any,
    source_type: str,
    *,
    decrement_profile: Iterable[Sequence[float]] | None = None,
    frequency_hz: float = 60.0,
    name: str | None = None,
    in_service: bool = True,
    index: int | None = None,
) -> int:
    """Register a utility, generator, or motor source and decrement schedule."""

    ensure_arc_flash_tables(net)
    if element_type not in net or not isinstance(net[element_type], pd.DataFrame):
        raise KeyError(f"network table {element_type!r} does not exist")
    element_key = tuple(element) if isinstance(element, list) else element
    if element_key not in net[element_type].index:
        if not (
            isinstance(net[element_type].index, pd.MultiIndex)
            and not isinstance(element_key, tuple)
            and element_key in net[element_type].index.get_level_values(0)
        ):
            raise KeyError(f"{element_type} element {element!r} does not exist")
    source = str(source_type).strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"utility", "current_source_generator", "induction_motor"}
    if source not in allowed:
        raise ValueError(f"source_type must be one of {sorted(allowed)}")
    profile = _normalize_profile(decrement_profile)
    if source == "induction_motor" and not profile:
        frequency = _positive(frequency_hz, "frequency_hz")
        profile = [[0.0, 1.0], [5.0 / frequency, 0.0]]
    table = net.arc_flash_source
    selected = _next_index(table, index)
    row = {
        "name": name or f"{source}:{element_type}:{element}",
        "element_type": str(element_type),
        "element": element,
        "source_type": source,
        "decrement_profile": profile,
        "in_service": bool(in_service),
    }
    for column, value in row.items():
        table.at[selected, column] = value
    return selected
