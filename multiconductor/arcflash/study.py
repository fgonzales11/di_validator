"""Scenario, source-decrement, and stateful-protection arc-flash studies."""

from __future__ import annotations

import copy
from dataclasses import asdict
from math import isfinite
from typing import Any, Mapping

import numpy as np
import pandas as pd

from multiconductor.arcflash.ieee1584 import (
    ArcFlashInputError,
    IEEE1584Input,
    calculate_ieee1584,
    combine_ieee1584_results,
)
from multiconductor.arcflash.models import (
    ARC_FLASH_LOCATION_COLUMNS,
    RES_ARC_FLASH_COLUMNS,
    RES_ARC_FLASH_OPERATION_COLUMNS,
    RES_ARC_FLASH_WORST_COLUMNS,
    ArcFlashStudyConfig,
    ArcFlashStudyResult,
    ensure_arc_flash_tables,
)


class ArcFlashStudyError(ValueError):
    """Raised when a study definition is incomplete or inconsistent."""


def _copy_network(net):
    try:
        return copy.deepcopy(net)
    except Exception as exc:  # pragma: no cover - unusual custom network objects
        raise ArcFlashStudyError("the network could not be copied for a scenario study") from exc


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _record_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise ArcFlashStudyError(f"expected a serialized list, got {type(value).__name__}")


def _selected_rows(table: pd.DataFrame, key: Any) -> pd.Index:
    if isinstance(key, list):
        key = tuple(key)
    if key in table.index:
        return table.loc[[key]].index
    if isinstance(table.index, pd.MultiIndex) and not isinstance(key, tuple):
        mask = table.index.get_level_values(0) == key
        if mask.any():
            return table.index[mask]
    raise KeyError(f"element index {key!r} does not exist")


def _apply_scenario_actions(net, actions: Any) -> None:
    for position, action in enumerate(_record_list(actions)):
        if not isinstance(action, Mapping):
            raise ArcFlashStudyError(f"scenario action {position} must be a mapping")
        missing = {"table", "index", "column", "value"}.difference(action)
        if missing:
            raise ArcFlashStudyError(f"scenario action {position} is missing {sorted(missing)}")
        table_name = str(action["table"])
        column = str(action["column"])
        if table_name not in net or not isinstance(net[table_name], pd.DataFrame):
            raise ArcFlashStudyError(f"scenario action table {table_name!r} does not exist")
        table = net[table_name]
        if column not in table:
            raise ArcFlashStudyError(
                f"scenario action column {table_name}.{column} does not exist"
            )
        rows = _selected_rows(table, action["index"])
        value = action["value"]
        if isinstance(value, list) and len(value) == len(rows):
            table.loc[rows, column] = value
        else:
            for row in rows:
                table.at[row, column] = value


def _phase_rows(table: pd.DataFrame, element: int) -> pd.DataFrame:
    if not isinstance(table.index, pd.MultiIndex):
        return table.loc[[element]]
    try:
        return table.xs(element, level=0, drop_level=False)
    except KeyError as exc:
        raise KeyError(f"bus {element} does not exist") from exc


def _bus_voltage(net, bus: int) -> float:
    rows = _phase_rows(net.bus, bus)
    if isinstance(rows.index, pd.MultiIndex):
        phases = set(int(value) for value in rows.index.get_level_values(1))
        missing = {1, 2, 3}.difference(phases)
        if missing:
            raise ArcFlashStudyError(
                f"arc-flash location bus {bus} is not three-phase; missing phases {sorted(missing)}"
            )
        rows = rows.loc[[(bus, phase) for phase in (1, 2, 3)]]
    voltages = pd.to_numeric(rows["vn_kv"], errors="coerce").to_numpy(float)
    if len(voltages) < 3 or np.any(~np.isfinite(voltages)) or not np.allclose(voltages, voltages[0]):
        raise ArcFlashStudyError(f"bus {bus} must have three equal finite phase voltages")
    return float(voltages[0])


def _phase_fault_currents(net, bus: int) -> list[float]:
    if "res_bus_sc" not in net or not isinstance(net.res_bus_sc, pd.DataFrame):
        raise ArcFlashStudyError("short-circuit calculation did not create res_bus_sc")
    rows = _phase_rows(net.res_bus_sc, bus)
    values: list[float] = []
    if isinstance(rows.index, pd.MultiIndex):
        for phase in (1, 2, 3):
            try:
                values.append(float(rows.loc[(bus, phase), "ikss_ka"]))
            except KeyError as exc:
                raise ArcFlashStudyError(
                    f"short-circuit results are missing phase {phase} at bus {bus}"
                ) from exc
    elif all(f"ikss_{phase}_ka" in rows for phase in ("a", "b", "c")):
        record = rows.iloc[0]
        values = [float(record[f"ikss_{phase}_ka"]) for phase in ("a", "b", "c")]
    elif "ikss_ka" in rows and len(rows) >= 3:
        values = pd.to_numeric(rows["ikss_ka"], errors="coerce").iloc[:3].astype(float).tolist()
    else:
        raise ArcFlashStudyError("three phase-resolved short-circuit currents are required")
    if len(values) != 3 or any(not isfinite(value) or value <= 0.0 for value in values):
        raise ArcFlashStudyError("all three phase fault currents must be finite and positive")
    return values


def _solve_fault(net, bus: int, fault_case: str, resistance_ohm: float = 0.0) -> list[float]:
    from multiconductor.shortcircuit import FaultSpec, calc_sc

    spec = FaultSpec(
        fault="LLL",
        bus=int(bus),
        phases=(1, 2, 3),
        case=str(fault_case),
        r_fault_ohm=float(resistance_ohm),
    )
    calc_sc(net, fault_spec=spec, branch_results=True)
    return _phase_fault_currents(net, bus)


def _mean_and_spread(currents: list[float], tolerance: float) -> tuple[float, float]:
    values = np.asarray(currents, dtype=float)
    mean = float(values.mean())
    spread = float((values.max() - values.min()) / mean * 100.0)
    if spread > tolerance + 1e-12:
        raise ArcFlashStudyError(
            f"phase current spread {spread:.3f}% exceeds the approved {tolerance:.3f}% tolerance"
        )
    return mean, spread


def _ieee_input(location: pd.Series, voltage: float, ibf: float, duration: float, case: str):
    def optional(name: str):
        value = location.get(name)
        return None if value is None or pd.isna(value) else float(value)

    return IEEE1584Input(
        voltage_kv=voltage,
        bolted_fault_current_ka=ibf,
        gap_mm=float(location["conductor_gap_mm"]),
        working_distance_mm=float(location["working_distance_mm"]),
        electrode_configuration=str(location["electrode_configuration"]),
        arc_duration_s=duration,
        enclosure_height_mm=optional("enclosure_height_mm"),
        enclosure_width_mm=optional("enclosure_width_mm"),
        enclosure_depth_mm=optional("enclosure_depth_mm"),
        arcing_current_case=case,
    )


def _solve_resistive_arc(
    net,
    *,
    bus: int,
    fault_case: str,
    target_current_ka: float,
    tolerance_fraction: float = 2e-4,
) -> tuple[float, list[float]]:
    """Solve a common per-phase arc resistance matching IEEE arcing current."""

    zero_currents = _solve_fault(net, bus, fault_case, 0.0)
    zero_mean = float(np.mean(zero_currents))
    if target_current_ka >= zero_mean:
        return 0.0, zero_currents

    low, high = 0.0, 0.01
    high_currents = _solve_fault(net, bus, fault_case, high)
    while float(np.mean(high_currents)) > target_current_ka and high < 1e6:
        low = high
        high *= 2.0
        high_currents = _solve_fault(net, bus, fault_case, high)
    if float(np.mean(high_currents)) > target_current_ka:
        raise ArcFlashStudyError("could not bracket an equivalent resistive arc solution")

    final_currents = high_currents
    for _ in range(32):
        middle = (low + high) / 2.0
        final_currents = _solve_fault(net, bus, fault_case, middle)
        current = float(np.mean(final_currents))
        if abs(current - target_current_ka) <= tolerance_fraction * target_current_ka:
            return middle, final_currents
        if current > target_current_ka:
            low = middle
        else:
            high = middle
    resistance = (low + high) / 2.0
    return resistance, _solve_fault(net, bus, fault_case, resistance)


def _scale_value(value: Any, multiplier: float):
    if value is None:
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        scaled = np.asarray(value, dtype=float) * multiplier
        return scaled.tolist() if isinstance(value, list) else scaled
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric * multiplier if np.isfinite(numeric) else value


_SOURCE_CURRENT_COLUMNS = (
    "max_ik_ka",
    "sc_current_ka",
    "ikss_ka",
    "s_sc_max_mva",
    "s_sc_min_mva",
)


def _prepare_sources(net) -> dict[int, dict]:
    controllers: dict[int, dict] = {}
    table = net.get("arc_flash_source", pd.DataFrame())
    seen: set[tuple[str, str]] = set()
    for source_id, source in table.iterrows():
        if not _as_bool(source.get("in_service"), True):
            continue
        table_name = str(source["element_type"])
        element = source["element"]
        signature = (table_name, repr(element))
        if signature in seen:
            raise ArcFlashStudyError(
                f"more than one arc-flash source record controls {table_name} {element!r}"
            )
        seen.add(signature)
        element_table = net.get(table_name)
        if not isinstance(element_table, pd.DataFrame):
            raise ArcFlashStudyError(f"source table {table_name!r} does not exist")
        rows = _selected_rows(element_table, element)
        profile = _record_list(source.get("decrement_profile"))
        normalized = sorted((float(pair[0]), float(pair[1])) for pair in profile)
        if any(time < 0 or multiplier < 0 for time, multiplier in normalized):
            raise ArcFlashStudyError("source decrement values must be non-negative")
        base_values = {
            (row, column): copy.deepcopy(element_table.at[row, column])
            for row in rows
            for column in _SOURCE_CURRENT_COLUMNS
            if column in element_table
        }
        base_service = {
            row: _as_bool(element_table.at[row, "in_service"], True)
            for row in rows
            if "in_service" in element_table
        }
        controllers[int(source_id)] = {
            "table": table_name,
            "rows": rows,
            "profile": normalized,
            "base_values": base_values,
            "base_service": base_service,
            "name": source.get("name"),
        }
    return controllers


def _source_multiplier(profile: list[tuple[float, float]], time_s: float) -> float:
    value = 1.0
    for event_time, multiplier in profile:
        if event_time <= time_s + 1e-12:
            value = multiplier
        else:
            break
    return value


def _apply_sources(net, controllers: dict[int, dict], time_s: float) -> None:
    for controller in controllers.values():
        table = net[controller["table"]]
        multiplier = _source_multiplier(controller["profile"], time_s)
        for (row, column), base_value in controller["base_values"].items():
            table.at[row, column] = _scale_value(base_value, multiplier)
        for row, base_service in controller["base_service"].items():
            table.at[row, "in_service"] = bool(base_service and multiplier > 0.0)


def _future_source_events(controllers: dict[int, dict], time_s: float) -> list[tuple[float, int]]:
    return sorted(
        (event_time, source_id)
        for source_id, controller in controllers.items()
        for event_time, _ in controller["profile"]
        if event_time > time_s + 1e-12
    )


def _approved_devices(net, location: pd.Series) -> dict[int, Any]:
    from multiconductor.protection import Fuse
    from multiconductor.protection.base import protection_devices

    approved = {int(value) for value in _record_list(location.get("protection_device_ids"))}
    table = net.get("protection", pd.DataFrame())
    missing = approved.difference(int(value) for value in table.index)
    if missing:
        raise ArcFlashStudyError(f"approved protection devices do not exist: {sorted(missing)}")
    lookup = dict(protection_devices(net))
    selected: dict[int, Any] = {}
    for device_id in sorted(approved):
        row = table.loc[device_id]
        device = lookup[device_id]
        if not _as_bool(row.get("in_service"), True) or not bool(device.in_service):
            continue
        if isinstance(device, Fuse):
            curve_kind = str(getattr(device, "curve_kind", "") or "").strip().lower()
            if curve_kind not in {"manufacturer_total_clearing", "manufacturer total clearing"}:
                raise ArcFlashStudyError(
                    f"fuse protection device {device_id} requires a manufacturer total-clearing curve"
                )
        elif getattr(device, "opening_time_s", None) is None:
            raise ArcFlashStudyError(
                f"relay protection device {device_id} requires explicit opening_time_s"
            )
        selected[device_id] = device
    return selected


def _validate_location(location: pd.Series) -> None:
    required_text = ("equipment_id", "access_area", "site_ppe", "glove_class")
    missing = [name for name in required_text if not str(location.get(name, "")).strip()]
    if missing:
        raise ArcFlashStudyError(f"location is missing required site data: {missing}")
    for name in ("limited_approach_boundary_mm", "restricted_approach_boundary_mm"):
        value = float(location.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ArcFlashStudyError(f"location requires a positive {name}")


def _arc_state(
    net,
    location: pd.Series,
    scenario: pd.Series,
    config: ArcFlashStudyConfig,
    arcing_case: str,
) -> dict:
    bus = int(location["bus"])
    voltage = _bus_voltage(net, bus)
    bolted_phases = _solve_fault(net, bus, str(scenario["fault_case"]), 0.0)
    ibf, spread = _mean_and_spread(
        bolted_phases, config.phase_imbalance_tolerance_percent
    )
    ieee = calculate_ieee1584(_ieee_input(location, voltage, ibf, 1.0, arcing_case))
    resistance, arc_phases = _solve_resistive_arc(
        net,
        bus=bus,
        fault_case=str(scenario["fault_case"]),
        target_current_ka=ieee.arcing_current_ka,
    )
    return {
        "voltage": voltage,
        "bolted_phases": bolted_phases,
        "ibf": ibf,
        "spread": spread,
        "iarc": ieee.arcing_current_ka,
        "arc_phases": arc_phases,
        "arc_resistance_ohm": resistance,
    }


def _operation_row(
    *, location_id, scenario_id, arcing_case, sequence, time_s, event_type,
    device_id=None, switch_id=None, stage=None, current_ka=None,
    fraction=None, opening_time_s=None, details=None,
) -> dict:
    return {
        "location_id": location_id,
        "scenario_id": scenario_id,
        "arcing_current_case": arcing_case,
        "sequence": sequence,
        "event_time_s": time_s,
        "event_type": event_type,
        "device_id": device_id,
        "switch_id": switch_id,
        "stage": stage,
        "current_ka": current_ka,
        "operating_fraction": fraction,
        "opening_time_s": opening_time_s,
        "details": details or {},
    }


def _run_case(
    base_net,
    location_id: int,
    location: pd.Series,
    scenario_id: int,
    scenario: pd.Series,
    config: ArcFlashStudyConfig,
    arcing_case: str,
) -> tuple[dict, list[dict]]:
    from multiconductor.protection import Fuse
    from multiconductor.protection.base import switch_current_ka

    work = _copy_network(base_net)
    _apply_scenario_actions(work, scenario.get("actions"))
    _validate_location(location)
    devices = _approved_devices(work, location)
    for device in devices.values():
        device.reset_device()
    controllers = _prepare_sources(work)
    _apply_sources(work, controllers, 0.0)

    time_s = 0.0
    cap = float(config.max_arc_duration_s)
    fractions = {device_id: 0.0 for device_id in devices}
    operated: set[int] = set()
    opening_due: dict[int, float] = {}
    controlling: list[int] = []
    segments = []
    segment_results = []
    operations: list[dict] = []
    sequence = 0
    state = _arc_state(work, location, scenario, config, arcing_case)
    initial = dict(state)
    duration_capped = False
    cleared = False

    while time_s < cap - 1e-12 and not cleared:
        timing: dict[int, dict] = {}
        candidates: list[tuple[float, int, int]] = []
        for device_id, device in devices.items():
            if device_id in operated:
                continue
            try:
                current = switch_current_ka(work, device.switch_index, "sc") / device.ct_ratio
            except (KeyError, ValueError):
                continue
            trip, curve_time, stage = device.trip_time(current)
            if trip and isfinite(curve_time):
                curve_time = float(curve_time)
                timing[device_id] = {
                    "current": current,
                    "curve_time": curve_time,
                    "stage": stage,
                }
                remaining = max(0.0, 1.0 - fractions[device_id])
                due = time_s if curve_time <= 0.0 else time_s + remaining * curve_time
                candidates.append((due, 1, device_id))
        for device_id, due in opening_due.items():
            candidates.append((due, 2, device_id))
        future_sources = _future_source_events(controllers, time_s)
        if future_sources:
            candidates.append((future_sources[0][0], 0, future_sources[0][1]))
        candidates.append((cap, 3, -1))
        next_time = min(value[0] for value in candidates)
        dt = max(0.0, next_time - time_s)

        if dt > 0.0:
            segment = calculate_ieee1584(
                _ieee_input(location, state["voltage"], state["ibf"], dt, arcing_case)
            )
            segment_results.append(segment)
            segments.append(
                {
                    "start_time_s": time_s,
                    "end_time_s": next_time,
                    "duration_s": dt,
                    "bolted_fault_current_ka": state["ibf"],
                    "arcing_current_ka": state["iarc"],
                    "arc_resistance_ohm": state["arc_resistance_ohm"],
                    "incident_energy_cal_cm2": segment.incident_energy_cal_cm2,
                }
            )
            for device_id, values in timing.items():
                curve_time = values["curve_time"]
                fractions[device_id] = (
                    1.0 if curve_time <= 0.0 else min(1.0, fractions[device_id] + dt / curve_time)
                )
        time_s = next_time

        due_now = sorted(
            (priority, identifier)
            for due, priority, identifier in candidates
            if abs(due - time_s) <= 1e-9
        )
        topology_changed = False
        source_changed = False
        for priority, identifier in due_now:
            if priority == 0:
                _apply_sources(work, controllers, time_s)
                sequence += 1
                operations.append(
                    _operation_row(
                        location_id=location_id,
                        scenario_id=scenario_id,
                        arcing_case=arcing_case,
                        sequence=sequence,
                        time_s=time_s,
                        event_type="source_decrement",
                        details={"source_id": identifier},
                    )
                )
                source_changed = True
            elif priority == 1 and identifier not in operated:
                values = timing.get(identifier)
                if values is None or fractions[identifier] < 1.0 - 1e-8:
                    continue
                device = devices[identifier]
                operated.add(identifier)
                sequence += 1
                operations.append(
                    _operation_row(
                        location_id=location_id,
                        scenario_id=scenario_id,
                        arcing_case=arcing_case,
                        sequence=sequence,
                        time_s=time_s,
                        event_type="fuse_clearing" if isinstance(device, Fuse) else "relay_operation",
                        device_id=identifier,
                        switch_id=device.switch_index,
                        stage=values["stage"],
                        current_ka=values["current"],
                        fraction=fractions[identifier],
                        opening_time_s=device.opening_time_s,
                    )
                )
                if isinstance(device, Fuse) or float(device.opening_time_s) <= 0.0:
                    device.tripped = True
                    device.status_to_net(work)
                    controlling.append(identifier)
                    topology_changed = True
                    if not isinstance(device, Fuse):
                        sequence += 1
                        operations.append(
                            _operation_row(
                                location_id=location_id,
                                scenario_id=scenario_id,
                                arcing_case=arcing_case,
                                sequence=sequence,
                                time_s=time_s,
                                event_type="breaker_open",
                                device_id=identifier,
                                switch_id=device.switch_index,
                                opening_time_s=device.opening_time_s,
                            )
                        )
                else:
                    opening_due[identifier] = time_s + float(device.opening_time_s)
            elif priority == 2 and identifier in opening_due:
                device = devices[identifier]
                device.tripped = True
                device.status_to_net(work)
                del opening_due[identifier]
                controlling.append(identifier)
                topology_changed = True
                sequence += 1
                operations.append(
                    _operation_row(
                        location_id=location_id,
                        scenario_id=scenario_id,
                        arcing_case=arcing_case,
                        sequence=sequence,
                        time_s=time_s,
                        event_type="breaker_open",
                        device_id=identifier,
                        switch_id=device.switch_index,
                        opening_time_s=device.opening_time_s,
                    )
                )
            elif priority == 3:
                duration_capped = True

        if duration_capped:
            break
        if topology_changed or source_changed:
            try:
                state = _arc_state(work, location, scenario, config, arcing_case)
            except Exception as exc:
                cleared = True
                sequence += 1
                operations.append(
                    _operation_row(
                        location_id=location_id,
                        scenario_id=scenario_id,
                        arcing_case=arcing_case,
                        sequence=sequence,
                        time_s=time_s,
                        event_type="arc_cleared",
                        details={"post_event_solve": f"{type(exc).__name__}: {exc}"},
                    )
                )

    total_energy, boundary = combine_ieee1584_results(segment_results)
    warnings = []
    if duration_capped:
        warnings.append(
            f"arc duration was capped at the study-approved {config.max_arc_duration_s:g} s"
        )
    row = {
        "location_id": location_id,
        "scenario_id": scenario_id,
        "location_name": location.get("name"),
        "scenario_name": scenario.get("name"),
        "equipment_id": location.get("equipment_id"),
        "access_area": location.get("access_area"),
        "bus": int(location["bus"]),
        "fault_case": scenario.get("fault_case"),
        "operating_condition": scenario.get("operating_condition"),
        "arcing_current_case": arcing_case,
        "nominal_voltage_kv": initial["voltage"],
        "working_distance_mm": float(location["working_distance_mm"]),
        "phase_currents_ka": initial["bolted_phases"],
        "phase_current_spread_percent": initial["spread"],
        "bolted_fault_current_ka": initial["ibf"],
        "initial_arcing_current_ka": initial["iarc"],
        "clearing_time_s": time_s,
        "duration_capped": duration_capped,
        "incident_energy_cal_cm2": total_energy,
        "arc_flash_boundary_mm": boundary,
        "controlling_device_ids": sorted(set(controlling)),
        "segment_count": len(segments),
        "segments": segments,
        "status": "valid",
        "warnings": warnings,
        "provenance": {
            "study_id": config.study_id,
            "standard": config.standard,
            "scenario_actions": _record_list(scenario.get("actions")),
        },
    }
    return row, operations


def _invalid_row(
    location_id: int,
    location: pd.Series,
    scenario_id: int,
    scenario: pd.Series,
    arcing_case: str,
    error: Exception,
    config: ArcFlashStudyConfig,
) -> dict:
    return {
        "location_id": location_id,
        "scenario_id": scenario_id,
        "location_name": location.get("name"),
        "scenario_name": scenario.get("name"),
        "equipment_id": location.get("equipment_id"),
        "access_area": location.get("access_area"),
        "bus": location.get("bus"),
        "fault_case": scenario.get("fault_case"),
        "operating_condition": scenario.get("operating_condition"),
        "arcing_current_case": arcing_case,
        "status": "invalid",
        "duration_capped": False,
        "warnings": [f"{type(error).__name__}: {error}"],
        "provenance": {"study_id": config.study_id, "standard": config.standard},
    }


def _validate_study_tables(net) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_arc_flash_tables(net)
    locations = net.arc_flash_location.loc[
        net.arc_flash_location["in_service"].map(lambda value: _as_bool(value, True))
    ]
    scenarios = net.arc_flash_scenario.loc[
        net.arc_flash_scenario["in_service"].map(lambda value: _as_bool(value, True))
    ]
    if locations.empty:
        raise ArcFlashStudyError("at least one in-service arc_flash_location is required")
    if scenarios.empty:
        raise ArcFlashStudyError("at least one in-service arc_flash_scenario is required")
    cases = set(str(value).strip().lower() for value in scenarios["fault_case"])
    missing = {"max", "min"}.difference(cases)
    if missing:
        raise ArcFlashStudyError(
            f"explicit maximum and minimum fault scenarios are required; missing {sorted(missing)}"
        )
    return locations, scenarios


def run_arc_flash(net, config: ArcFlashStudyConfig) -> ArcFlashStudyResult:
    """Run every in-service location/scenario for full and reduced arcing current.

    The input network is never used as a calculation work surface.  Results are
    copied back only after all scenario-network copies have completed.
    """

    if not isinstance(config, ArcFlashStudyConfig):
        raise TypeError("config must be an ArcFlashStudyConfig")
    locations, scenarios = _validate_study_tables(net)
    rows: list[dict] = []
    operation_rows: list[dict] = []
    warning_manifest: list[str] = []
    for location_id, location in locations.iterrows():
        for scenario_id, scenario in scenarios.iterrows():
            for arcing_case in ("full", "reduced"):
                try:
                    row, operations = _run_case(
                        net,
                        int(location_id),
                        location,
                        int(scenario_id),
                        scenario,
                        config,
                        arcing_case,
                    )
                    rows.append(row)
                    operation_rows.extend(operations)
                except Exception as exc:
                    row = _invalid_row(
                        int(location_id),
                        location,
                        int(scenario_id),
                        scenario,
                        arcing_case,
                        exc,
                        config,
                    )
                    rows.append(row)
                    warning_manifest.append(
                        f"location {location_id}, scenario {scenario_id}, {arcing_case}: {exc}"
                    )

    all_cases = pd.DataFrame(rows).reindex(columns=RES_ARC_FLASH_COLUMNS)
    operations = pd.DataFrame(operation_rows).reindex(columns=RES_ARC_FLASH_OPERATION_COLUMNS)
    valid = all_cases.loc[all_cases["status"] == "valid"].copy()
    worst_rows: list[dict] = []
    if not valid.empty:
        valid["_energy"] = pd.to_numeric(valid["incident_energy_cal_cm2"], errors="coerce")
        for location_id, group in valid.groupby("location_id", sort=True):
            selected = group.sort_values(
                ["_energy", "scenario_id", "arcing_current_case"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0].drop(labels=["_energy"]).to_dict()
            location = locations.loc[location_id]
            selected.update(
                {
                    "site_ppe": location.get("site_ppe"),
                    "limited_approach_boundary_mm": location.get(
                        "limited_approach_boundary_mm"
                    ),
                    "restricted_approach_boundary_mm": location.get(
                        "restricted_approach_boundary_mm"
                    ),
                    "glove_class": location.get("glove_class"),
                    "report_number": config.report_number,
                    "revision": config.revision,
                    "issue_date": config.issue_date_text,
                    "label_count": int(location.get("label_count", 1)),
                }
            )
            worst_rows.append(selected)
    worst = pd.DataFrame(worst_rows).reindex(columns=RES_ARC_FLASH_WORST_COLUMNS)
    net.res_arc_flash = all_cases.copy(deep=True)
    net.res_arc_flash_worst = worst.copy(deep=True)
    net.res_arc_flash_operation = operations.copy(deep=True)
    return ArcFlashStudyResult(
        all_cases=all_cases,
        worst_cases=worst,
        operations=operations,
        warnings=warning_manifest,
        study_config=config,
    )
