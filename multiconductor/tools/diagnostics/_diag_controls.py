"""Category 8 — Regulator / Capacitor / Inverter Control Errors.

Checks: ctrl_01 through ctrl_05.
"""
from __future__ import annotations

import math
from typing import Any

from ._common import (
    get_table,
    unique_bus_ids,
    is_in_service,
    safe_float,
    dedup_iter,
    issue,
)


def check_controls(net: Any) -> list[dict[str, Any]]:
    """Run all control-element checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_ctrl_01_control_references_nonexistent_bus(net))
    issues.extend(_ctrl_02_regulator_setpoint(net))
    issues.extend(_ctrl_03_capacitor_thresholds_inverted(net))
    issues.extend(_ctrl_04_inverter_reactive_exceeds_capability(net))
    issues.extend(_ctrl_05_multiple_controllers_same_bus(net))
    return issues


# ---------------------------------------------------------------------------
# ctrl_01 — Control element references non-existent bus
# ---------------------------------------------------------------------------

def _ctrl_01_control_references_nonexistent_bus(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    bus_ids = unique_bus_ids(bus)

    for table_name in ("shunt", "asymmetric_shunt", "asymmetric_sgen"):
        table = get_table(net, table_name)
        if table.empty or "bus" not in table.columns:
            continue
        for eid, row in dedup_iter(table):
            if not is_in_service(row):
                continue
            bus_val = row.get("bus")
            if bus_val is not None and bus_val not in bus_ids:
                issues.append(issue(
                    "high", "control_error", table_name, eid, "bus",
                    f"{table_name} {eid} references bus {bus_val} which does not exist.",
                    "Map to a valid bus index.",
                ))
    return issues


# ---------------------------------------------------------------------------
# ctrl_02 — Regulator setpoint outside normal range
# ---------------------------------------------------------------------------

def _ctrl_02_regulator_setpoint(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    # Check ext_grid vm_pu as a proxy for regulator setpoints
    for table_name in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, table_name)
        if eg.empty or "vm_pu" not in eg.columns:
            continue
        for eid, row in dedup_iter(eg):
            if not is_in_service(row):
                continue
            vm = safe_float(row.get("vm_pu"))
            if vm is not None and (vm < 0.90 or vm > 1.10):
                issues.append(issue(
                    "high", "control_error", table_name, eid, "vm_pu",
                    f"Source {eid} voltage setpoint vm_pu={vm:.4f} is outside [0.90, 1.10] pu.",
                    "Verify voltage setpoint is within ANSI/IEEE limits.",
                ))
    return issues


# ---------------------------------------------------------------------------
# ctrl_03 — Capacitor switching thresholds inverted
# ---------------------------------------------------------------------------

def _ctrl_03_capacitor_thresholds_inverted(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in ("shunt", "asymmetric_shunt"):
        table = get_table(net, table_name)
        if table.empty:
            continue
        if "v_threshold_on" not in table.columns or "v_threshold_off" not in table.columns:
            continue
        for eid, row in dedup_iter(table):
            if not is_in_service(row):
                continue
            v_on = safe_float(row.get("v_threshold_on"))
            v_off = safe_float(row.get("v_threshold_off"))
            if v_on is not None and v_off is not None and v_on >= v_off:
                issues.append(issue(
                    "medium", "control_error", table_name, eid,
                    "v_threshold_on/v_threshold_off",
                    f"Capacitor {eid} thresholds inverted: on={v_on}, off={v_off}.",
                    "Set v_threshold_on < v_threshold_off to prevent oscillation.",
                ))
    return issues


# ---------------------------------------------------------------------------
# ctrl_04 — Inverter reactive power exceeds capability
# ---------------------------------------------------------------------------

def _ctrl_04_inverter_reactive_exceeds_capability(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sgen = get_table(net, "asymmetric_sgen")
    if sgen.empty:
        return issues
    if "p_mw" not in sgen.columns or "q_mvar" not in sgen.columns or "sn_mva" not in sgen.columns:
        return issues

    for eid, row in dedup_iter(sgen):
        if not is_in_service(row):
            continue
        p = safe_float(row.get("p_mw"))
        q = safe_float(row.get("q_mvar"))
        sn = safe_float(row.get("sn_mva"))
        if p is None or q is None or sn is None or sn <= 0:
            continue
        q_max = math.sqrt(max(sn**2 - p**2, 0))
        if abs(q) > q_max * 1.05:  # 5% tolerance
            issues.append(issue(
                "medium", "control_error", "asymmetric_sgen", eid, "q_mvar",
                f"Inverter {eid} |q_mvar|={abs(q):.6f} exceeds reactive capability "
                f"{q_max:.6f} MVAr (from sn_mva={sn}, p_mw={p}).",
                "Reduce reactive output or increase inverter rating.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ctrl_05 — Multiple controllers on same bus
# ---------------------------------------------------------------------------

def _ctrl_05_multiple_controllers_same_bus(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus_controllers: dict[Any, list[tuple[str, Any]]] = {}

    for table_name in ("shunt", "asymmetric_shunt"):
        table = get_table(net, table_name)
        if table.empty or "bus" not in table.columns:
            continue
        for eid, row in dedup_iter(table):
            if not is_in_service(row):
                continue
            bus_val = row.get("bus")
            if bus_val is not None:
                bus_controllers.setdefault(bus_val, []).append((table_name, eid))

    for bus_val, controllers in bus_controllers.items():
        if len(controllers) > 1:
            desc = ", ".join(f"{t}[{e}]" for t, e in controllers)
            issues.append(issue(
                "low", "control_error", "bus", bus_val, "controllers",
                f"Bus {bus_val} has multiple controllers: {desc}.",
                "Verify coordinated control or remove duplicate controllers.",
            ))
    return issues
