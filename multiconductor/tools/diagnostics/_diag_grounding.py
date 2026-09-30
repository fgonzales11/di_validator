"""Category 3 — Floating / Weakly Grounded Nodes.

Checks: gnd_01 through gnd_04.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from ._common import (
    get_table,
    unique_bus_ids,
    is_in_service,
    safe_float,
    issue,
    build_network_graph,
)


def check_grounding(net: Any) -> list[dict[str, Any]]:
    """Run all grounding / floating node checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_gnd_01_bus_with_no_connections(net))
    issues.extend(_gnd_02_no_ground_path_from_neutral(net))
    issues.extend(_gnd_03_substation_ground_missing(net))
    issues.extend(_gnd_04_dead_end_buses(net))
    return issues


# ---------------------------------------------------------------------------
# gnd_01 — Bus with no connected elements
# ---------------------------------------------------------------------------

_BRANCH_TABLES = ("line", "trafo", "trafo1ph")
_SHUNT_TABLES = ("asymmetric_load", "asymmetric_sgen", "asymmetric_shunt", "ext_grid", "ext_grid_sequence")


def _gnd_01_bus_with_no_connections(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty:
        return issues
    bus_ids = unique_bus_ids(bus)
    connected_buses: set = set()

    # Branch elements
    for table_name in _BRANCH_TABLES:
        table = get_table(net, table_name)
        if table.empty:
            continue
        if table_name == "trafo1ph" and isinstance(table.index, pd.MultiIndex) and "bus" in table.index.names:
            bus_lv = table.index.names.index("bus")
            connected_buses.update(table.index.get_level_values(bus_lv).unique().tolist())
            continue
        for col in ("from_bus", "to_bus", "hv_bus", "lv_bus", "bus"):
            if col in table.columns:
                connected_buses.update(table[col].dropna().unique().tolist())

    # Shunt elements
    for table_name in _SHUNT_TABLES:
        table = get_table(net, table_name)
        if table.empty:
            continue
        if "bus" in table.columns:
            connected_buses.update(table["bus"].dropna().unique().tolist())

    # Switches
    sw = get_table(net, "switch")
    if not sw.empty:
        if "bus" in sw.columns:
            connected_buses.update(sw["bus"].dropna().unique().tolist())
        if "element" in sw.columns:
            # For bus-bus switches, element is a bus ID
            bb_sw = sw[sw.get("et", pd.Series(dtype=str)) == "b"] if "et" in sw.columns else pd.DataFrame()
            if not bb_sw.empty:
                connected_buses.update(bb_sw["element"].dropna().unique().tolist())

    orphans = bus_ids - connected_buses
    for b in orphans:
        issues.append(issue(
            "critical", "grounding", "bus", b, "connectivity",
            f"Bus {b} has no connected elements (completely isolated).",
            "Connect the bus to the network or remove it.",
        ))
    return issues


# ---------------------------------------------------------------------------
# gnd_02 — No ground path from neutral
# ---------------------------------------------------------------------------

def _gnd_02_no_ground_path_from_neutral(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty or "grounded" not in bus.columns:
        return issues
    if not isinstance(bus.index, pd.MultiIndex):
        return issues

    # Find buses that have a neutral phase (phase=0)
    neutral_buses: set = set()
    grounded_buses: set = set()

    for idx in bus.index:
        bus_id, phase = idx[0], idx[1]
        if int(phase) == 0:
            neutral_buses.add(bus_id)
            grounded = bus.at[idx, "grounded"] if "grounded" in bus.columns else False
            if grounded:
                grounded_buses.add(bus_id)

    # Build per-phase graph for neutral conductor
    line = get_table(net, "line")
    neutral_graph = nx.Graph()
    neutral_graph.add_nodes_from(neutral_buses)

    for idx, row in line.iterrows():
        if not is_in_service(row):
            continue
        fp = safe_float(row.get("from_phase"))
        tp = safe_float(row.get("to_phase"))
        if fp is not None and int(fp) == 0 and tp is not None and int(tp) == 0:
            fb, tb = row.get("from_bus"), row.get("to_bus")
            if fb in neutral_buses and tb in neutral_buses:
                neutral_graph.add_edge(fb, tb)

    # Check each neutral bus can reach a grounded bus
    for nb in neutral_buses:
        if nb in grounded_buses:
            continue
        if nb not in neutral_graph:
            continue
        reachable = nx.node_connected_component(neutral_graph, nb)
        if not reachable.intersection(grounded_buses):
            issues.append(issue(
                "high", "grounding", "bus", nb, "neutral",
                f"Bus {nb} has neutral conductor but no ground path to a grounded bus.",
                "Add grounding at this bus or ensure neutral continuity to a grounded node.",
            ))
    return issues


# ---------------------------------------------------------------------------
# gnd_03 — Substation ground reference missing
# ---------------------------------------------------------------------------

def _gnd_03_substation_ground_missing(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty:
        return issues

    source_buses: set = set()
    for table_name in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, table_name)
        if not eg.empty and "bus" in eg.columns:
            source_buses.update(eg["bus"].dropna().unique().tolist())

    if not isinstance(bus.index, pd.MultiIndex) or "grounded" not in bus.columns:
        return issues

    for sb in source_buses:
        try:
            bus_sub = bus.xs(sb, level=0)
            grounded_vals = bus_sub["grounded"].tolist()
            if not any(grounded_vals):
                issues.append(issue(
                    "high", "grounding", "bus", sb, "grounded",
                    f"Source bus {sb} has no grounded phase — substation ground reference may be missing.",
                    "Add grounding to the source bus neutral or verify grounding configuration.",
                ))
        except (KeyError, TypeError):
            continue
    return issues


# ---------------------------------------------------------------------------
# gnd_04 — Degree-1 buses (dead-ends) that aren't loads/gens/sources
# ---------------------------------------------------------------------------

def _gnd_04_dead_end_buses(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    graph = build_network_graph(net)

    # Buses with loads, gens, or sources
    has_equipment: set = set()
    for table_name in _SHUNT_TABLES:
        table = get_table(net, table_name)
        if not table.empty and "bus" in table.columns:
            has_equipment.update(table["bus"].dropna().unique().tolist())

    for node in graph.nodes():
        if graph.degree(node) == 1 and node not in has_equipment:
            issues.append(issue(
                "medium", "grounding", "bus", node, "degree",
                f"Bus {node} is a dead-end (degree 1) with no load, generation, or source.",
                "Verify this bus is intentional or connect equipment / remove it.",
            ))
    return issues
