"""Protection timing and optional trip application."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from multiconductor.protection.base import _ensure_protection_table, protection_devices


_RESULT_COLUMNS = [
    "device_index",
    "switch_id",
    "protection_type",
    "trip_melt",
    "trip",
    "stage",
    "activation_parameter",
    "activation_parameter_value",
    "operating_time_s",
    "opening_time_s",
    "trip_melt_time_s",
    "trip_time_s",
]
_COORDINATION_COLUMNS = [
    "coordination_group",
    "device_index",
    "switch_id",
    "protection_type",
    "trip_time_s",
    "asserted_order",
    "role",
    "primary_device_index",
    "primary_trip_time_s",
    "grading_margin_s",
    "minimum_margin_s",
    "margin_violation",
    "operation_order",
]


def protection_coordination_report(
    net,
    timing: pd.DataFrame | None = None,
    *,
    scenario: str = "sc",
    minimum_margin_s: float = 0.0,
) -> pd.DataFrame:
    """Build a deterministic primary/backup grading report for asserted devices."""

    scenario = str(scenario).lower()
    if scenario not in {"sc", "pp"}:
        raise ValueError('scenario must be either "sc" or "pp"')
    minimum_margin_s = float(minimum_margin_s)
    if not np.isfinite(minimum_margin_s) or minimum_margin_s < 0.0:
        raise ValueError("minimum_margin_s must be a finite non-negative value")
    if timing is None:
        timing = net.get("res_protection")
        if not isinstance(timing, pd.DataFrame):
            timing = calculate_protection_times(net, scenario=scenario)
    if not isinstance(timing, pd.DataFrame) or timing.empty:
        report = pd.DataFrame(columns=_COORDINATION_COLUMNS)
        net.res_protection_coordination = report
        return report

    trip_mask = timing["trip"].astype(bool)
    trip_times = pd.to_numeric(timing["trip_time_s"], errors="coerce")
    asserted = timing.loc[trip_mask & np.isfinite(trip_times)].copy()
    if asserted.empty:
        report = pd.DataFrame(columns=_COORDINATION_COLUMNS)
        net.res_protection_coordination = report
        return report

    if "operation_order" in asserted:
        op_values = pd.to_numeric(asserted["operation_order"], errors="coerce")
        asserted["_operation_order"] = op_values.fillna(1).astype(int)
    else:
        asserted["_operation_order"] = 1
    asserted["_trip_time"] = pd.to_numeric(asserted["trip_time_s"], errors="coerce")
    asserted = asserted.sort_values(
        by=["_operation_order", "_trip_time", "device_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict] = []
    for operation_order, group in asserted.groupby("_operation_order", sort=True):
        group = group.reset_index(drop=True).copy()
        group["asserted_order"] = np.arange(1, len(group) + 1)
        primary_trip_time = float(group.loc[0, "_trip_time"])
        primary_rows = group.loc[np.isclose(group["_trip_time"], primary_trip_time)]
        primary_ids = tuple(int(value) for value in primary_rows["device_index"])
        primary_device_index = primary_ids[0]
        coordination_group = f"operation-{int(operation_order)}"
        for _, row in group.iterrows():
            trip_time = float(row["_trip_time"])
            role = "primary" if np.isclose(trip_time, primary_trip_time) else "backup"
            grading_margin = trip_time - primary_trip_time
            rows.append(
                {
                    "coordination_group": coordination_group,
                    "device_index": int(row["device_index"]),
                    "switch_id": row["switch_id"],
                    "protection_type": row["protection_type"],
                    "trip_time_s": trip_time,
                    "asserted_order": int(row["asserted_order"]),
                    "role": role,
                    "primary_device_index": primary_device_index,
                    "primary_trip_time_s": primary_trip_time,
                    "grading_margin_s": grading_margin,
                    "minimum_margin_s": minimum_margin_s,
                    "margin_violation": bool(
                        role == "backup" and grading_margin < minimum_margin_s
                    ),
                    "operation_order": int(operation_order),
                }
            )
    report = pd.DataFrame(rows, columns=_COORDINATION_COLUMNS)
    net.res_protection_coordination = report
    return report


def calculate_protection_times(net, scenario: str = "sc") -> pd.DataFrame:
    """Evaluate every in-service protection device for current network results."""

    scenario = str(scenario).lower()
    if scenario not in {"sc", "pp"}:
        raise ValueError('scenario must be either "sc" or "pp"')
    table = _ensure_protection_table(net)
    results = []
    for index, device in protection_devices(net):
        row = table.loc[index]
        if not bool(row["in_service"]) or not bool(device.in_service):
            continue
        results.append(device.protection_function(net=net, scenario=scenario))
    result = pd.DataFrame(results, columns=_RESULT_COLUMNS)
    net.res_protection = result
    protection_coordination_report(net, result, scenario=scenario)
    return result


def reset_protection_devices(net, *, close_switches: bool = False) -> None:
    """Reset latched trip decisions, optionally closing their switches."""

    table = _ensure_protection_table(net)
    for index, device in protection_devices(net):
        device.reset_device()
        table.loc[index, "tripped"] = False
        if close_switches:
            device.status_to_net(net)


def run_protection(
    net,
    scenario: str = "sc",
    *,
    trip_switches: bool = False,
    apply_trips: bool | None = None,
    reset: bool = True,
    fault_spec=None,
    calc_sc_kwargs: Mapping | None = None,
    ordered: bool = False,
    max_operations: int = 20,
) -> pd.DataFrame:
    """Evaluate protection and optionally apply trip decisions to switches.

    With ``fault_spec`` the short-circuit solve is run automatically.  When
    ``ordered`` and ``trip_switches`` are both true, only the fastest asserted
    device operates before the fault is re-solved on the changed topology.
    """

    if apply_trips is not None:
        trip_switches = bool(apply_trips)
    if max_operations <= 0:
        raise ValueError("max_operations must be positive")
    if reset:
        reset_protection_devices(net)

    from multiconductor.shortcircuit import (
        ShortCircuitCalculationError,
        ShortCircuitSourceError,
    )

    def solve_fault() -> None:
        if fault_spec is None:
            return
        if str(scenario).lower() != "sc":
            raise ValueError("fault_spec can only be used with scenario='sc'")
        from multiconductor.shortcircuit import FaultSpec, calc_sc

        selected_fault = FaultSpec(**fault_spec) if isinstance(fault_spec, Mapping) else fault_spec
        if not isinstance(selected_fault, FaultSpec):
            raise TypeError("fault_spec must be a FaultSpec or mapping")
        solve_kwargs = dict(calc_sc_kwargs or {})
        solve_kwargs["branch_results"] = True
        calc_sc(net, fault_spec=selected_fault, **solve_kwargs)

    solve_fault()
    if not ordered or not trip_switches:
        result = calculate_protection_times(net, scenario=scenario)
        if trip_switches:
            _ensure_protection_table(net)
            for _, device in protection_devices(net):
                if device.in_service and device.has_tripped():
                    device.status_to_net(net)
        return result

    _ensure_protection_table(net)
    device_lookup = dict(protection_devices(net))
    evaluations = []
    elapsed_time = 0.0
    stop_reason = "no-trip"
    for operation in range(1, max_operations + 1):
        result = calculate_protection_times(net, scenario=scenario).copy()
        result["operation_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["operated"] = False
        result["elapsed_time_s"] = elapsed_time
        tripping = result.loc[
            result["trip"].astype(bool)
            & np.isfinite(pd.to_numeric(result["trip_time_s"], errors="coerce"))
        ]
        if tripping.empty:
            evaluations.append(result)
            break
        fastest = float(tripping["trip_time_s"].min())
        selected = tripping.loc[np.isclose(tripping["trip_time_s"], fastest)]
        selected_ids = set(selected["device_index"].astype(int))
        elapsed_time += fastest
        selected_mask = result["device_index"].isin(selected_ids)
        result.loc[selected_mask, "operation_order"] = operation
        result.loc[selected_mask, "operated"] = True
        result.loc[selected_mask, "elapsed_time_s"] = elapsed_time
        evaluations.append(result)
        for device_id in selected_ids:
            device = device_lookup[device_id]
            device.tripped = True
            device.status_to_net(net)
        if fault_spec is None:
            stop_reason = "ordered-recalculation-requires-fault-spec"
            break
        try:
            solve_fault()
        except (ShortCircuitCalculationError, ShortCircuitSourceError) as exc:
            stop_reason = f"post-trip-solve-stopped: {type(exc).__name__}: {exc}"
            break
    else:
        stop_reason = "max-operations"

    combined = pd.concat(evaluations, ignore_index=True) if evaluations else pd.DataFrame()
    combined["stop_reason"] = stop_reason
    net.res_protection = combined
    protection_coordination_report(net, combined, scenario=scenario)
    return combined
