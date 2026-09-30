"""Category 6 — Open Conductor Issues.

Checks: oc_01 through oc_04.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ._common import (
    get_table,
    elem_id,
    is_in_service,
    safe_float,
    dedup_iter,
    get_element_circuits,
    issue,
    build_network_graph,
)


def check_open_conductor(net: Any) -> list[dict[str, Any]]:
    """Run all open conductor checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_oc_01_phase_mismatch_between_ends(net))
    issues.extend(_oc_02_single_phasing(net))
    issues.extend(_oc_03_series_element_with_oos_bus(net))
    issues.extend(_oc_04_open_switch_isolates_loads(net))
    return issues


# ---------------------------------------------------------------------------
# oc_01 — Phase present at from_bus but missing at to_bus
# ---------------------------------------------------------------------------

def _oc_01_phase_mismatch_between_ends(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty or "from_phase" not in line.columns or "to_phase" not in line.columns:
        return issues

    seen: set = set()
    for idx, row in line.iterrows():
        eid = elem_id(idx)
        if eid in seen:
            continue
        if not is_in_service(row):
            continue
        seen.add(eid)

        circuits = get_element_circuits(line, eid)
        if circuits is None or circuits.empty:
            continue
        from_phases = set(circuits["from_phase"].dropna().astype(int).tolist())
        to_phases = set(circuits["to_phase"].dropna().astype(int).tolist())

        if from_phases and to_phases and from_phases != to_phases:
            missing_at_to = from_phases - to_phases
            missing_at_from = to_phases - from_phases
            if missing_at_to:
                issues.append(issue(
                    "critical", "open_conductor", "line", eid, "from_phase/to_phase",
                    f"Line {eid}: phases {sorted(missing_at_to)} present at from_bus "
                    f"but missing at to_bus — possible open conductor.",
                    "Check for broken conductor or incorrect phase assignment.",
                ))
            if missing_at_from:
                issues.append(issue(
                    "critical", "open_conductor", "line", eid, "from_phase/to_phase",
                    f"Line {eid}: phases {sorted(missing_at_from)} present at to_bus "
                    f"but missing at from_bus.",
                    "Check for broken conductor or incorrect phase assignment.",
                ))
    return issues


# ---------------------------------------------------------------------------
# oc_02 — Single-phasing: 3-phase bus with only 2 phases served
# ---------------------------------------------------------------------------

def _oc_02_single_phasing(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    line = get_table(net, "line")
    if bus.empty or line.empty:
        return issues

    if not isinstance(bus.index, pd.MultiIndex):
        return issues

    # Build expected phases per bus from bus table
    bus_expected_phases: dict[Any, set[int]] = {}
    for idx in bus.index:
        bus_id, phase = idx[0], int(idx[1])
        if phase != 0:  # exclude neutral
            bus_expected_phases.setdefault(bus_id, set()).add(phase)

    # Build served phases per bus from incoming lines
    bus_served_phases: dict[Any, set[int]] = {}
    for _, row in line.iterrows():
        if not is_in_service(row):
            continue
        fb = row.get("from_bus")
        tb = row.get("to_bus")
        fp = safe_float(row.get("from_phase")) if "from_phase" in line.columns else None
        tp = safe_float(row.get("to_phase")) if "to_phase" in line.columns else None
        if fb is not None and fp is not None and int(fp) != 0:
            bus_served_phases.setdefault(fb, set()).add(int(fp))
        if tb is not None and tp is not None and int(tp) != 0:
            bus_served_phases.setdefault(tb, set()).add(int(tp))

    three_phase_buses = {b for b, phases in bus_expected_phases.items() if len(phases) == 3}
    for b in three_phase_buses:
        served = bus_served_phases.get(b, set())
        if len(served) == 2:
            missing = bus_expected_phases[b] - served
            issues.append(issue(
                "high", "open_conductor", "bus", b, "phase",
                f"3-phase bus {b} is only served by 2 phases — "
                f"missing phase(s): {sorted(missing)}.",
                "Check upstream lines for open conductor or disconnected phase.",
            ))
    return issues


# ---------------------------------------------------------------------------
# oc_03 — Series element with one end out-of-service
# ---------------------------------------------------------------------------

def _oc_03_series_element_with_oos_bus(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    line = get_table(net, "line")
    if bus.empty or line.empty:
        return issues

    # Build set of in-service buses
    in_service_buses: set = set()
    if isinstance(bus.index, pd.MultiIndex):
        for idx in bus.index:
            if "in_service" not in bus.columns or bus.at[idx, "in_service"]:
                in_service_buses.add(idx[0])
    else:
        for idx, row in bus.iterrows():
            if is_in_service(row):
                in_service_buses.add(idx)

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        fb = row.get("from_bus")
        tb = row.get("to_bus")
        fb_ok = fb in in_service_buses
        tb_ok = tb in in_service_buses
        if fb_ok != tb_ok:
            oos_bus = fb if not fb_ok else tb
            issues.append(issue(
                "high", "open_conductor", "line", eid, "bus",
                f"Line {eid} is in-service but endpoint bus {oos_bus} is out of service.",
                "Disable the line or restore the bus to service.",
            ))
    return issues


# ---------------------------------------------------------------------------
# oc_04 — Open switch isolates downstream loads
# ---------------------------------------------------------------------------

def _oc_04_open_switch_isolates_loads(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sw = get_table(net, "switch")
    if sw.empty:
        return issues

    load = get_table(net, "asymmetric_load")
    load_buses: set = set()
    if not load.empty and "bus" in load.columns:
        load_buses = set(load["bus"].dropna().unique().tolist())

    if not load_buses:
        return issues

    # Build graph WITH all switches closed to get full topology
    graph_full = build_network_graph(net)

    # For each open switch, check if removing it would strand loads
    for eid, row in dedup_iter(sw):
        if bool(row.get("closed", True)):
            continue
        # This switch is open — check if there are loads only reachable via this switch
        bus_val = row.get("bus")
        element_val = row.get("element")
        et = row.get("et")
        if et != "b" or bus_val is None or element_val is None:
            continue
        # Temporarily check: if this switch were closed, does it connect to loads?
        if element_val not in graph_full:
            continue
        component = set()
        # Do a BFS from element_val side excluding the switch endpoint
        import networkx as nx
        temp_graph = graph_full.copy()
        if temp_graph.has_edge(bus_val, element_val):
            temp_graph.remove_edge(bus_val, element_val)
        if element_val in temp_graph:
            component = nx.node_connected_component(temp_graph, element_val)
        stranded_loads = component.intersection(load_buses)
        if stranded_loads:
            issues.append(issue(
                "medium", "open_conductor", "switch", eid, "closed",
                f"Open switch {eid} isolates {len(stranded_loads)} bus(es) with loads.",
                "Verify this is intentional or close the switch to restore service.",
            ))
    return issues
