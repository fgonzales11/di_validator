"""Category 10 — Bad Topology / Connectivity.

Checks: top_01 through top_06.
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
    dedup_iter,
    issue,
    build_network_graph,
    bfs_from_sources,
)


def check_topology(net: Any) -> list[dict[str, Any]]:
    """Run all topology / connectivity checks."""
    graph = build_network_graph(net)
    issues: list[dict[str, Any]] = []
    issues.extend(_top_01_unreachable_from_source(net, graph))
    issues.extend(_top_02_parallel_sources(net, graph))
    issues.extend(_top_03_switch_element_type_mismatch(net))
    issues.extend(_top_04_radial_violation(net, graph))
    issues.extend(_top_05_long_radial_path(net, graph))
    issues.extend(_top_06_dead_end_non_leaf_load(net, graph))
    return issues


# ---------------------------------------------------------------------------
# top_01 — Bus not reachable from any source
# ---------------------------------------------------------------------------

def _top_01_unreachable_from_source(net: Any, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    reachable = bfs_from_sources(net, graph)

    # Only flag in-service buses
    in_service_buses = set()
    if isinstance(bus.index, pd.MultiIndex):
        for idx in bus.index:
            if "in_service" not in bus.columns or bus.at[idx, "in_service"]:
                in_service_buses.add(idx[0])
    else:
        for idx, row in bus.iterrows():
            if is_in_service(row):
                in_service_buses.add(idx)

    unreachable = in_service_buses - reachable
    for b in unreachable:
        issues.append(issue(
            "critical", "topology", "bus", b, "reachability",
            f"Bus {b} is in-service but not reachable from any source.",
            "Connect to the network or mark out of service.",
        ))
    return issues


# ---------------------------------------------------------------------------
# top_02 — Source-to-source path (unintended parallel sources)
# ---------------------------------------------------------------------------

def _top_02_parallel_sources(net: Any, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    source_buses: set = set()
    for tbl in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, tbl)
        if not eg.empty and "bus" in eg.columns:
            source_buses.update(eg["bus"].dropna().unique().tolist())

    if len(source_buses) < 2:
        return issues

    source_list = list(source_buses)
    for i in range(len(source_list)):
        for j in range(i + 1, len(source_list)):
            s1, s2 = source_list[i], source_list[j]
            if s1 in graph and s2 in graph and nx.has_path(graph, s1, s2):
                issues.append(issue(
                    "high", "topology", "network", f"{s1},{s2}", "parallel_sources",
                    f"Sources at bus {s1} and {s2} are connected — potential parallel source conflict.",
                    "Verify intentional parallel operation or insert an open switch.",
                ))
    return issues


# ---------------------------------------------------------------------------
# top_03 — Switch references wrong element type
# ---------------------------------------------------------------------------

def _top_03_switch_element_type_mismatch(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sw = get_table(net, "switch")
    if sw.empty or "et" not in sw.columns or "element" not in sw.columns:
        return issues

    bus_ids = unique_bus_ids(get_table(net, "bus"))
    line_ids: set = set()
    line = get_table(net, "line")
    if not line.empty:
        if isinstance(line.index, pd.MultiIndex):
            line_ids = set(line.index.get_level_values(0).unique().tolist())
        else:
            line_ids = set(line.index.tolist())

    for eid, row in dedup_iter(sw):
        et = str(row.get("et", "")).lower()
        element = row.get("element")
        if element is None:
            continue
        if et == "b" and element not in bus_ids:
            issues.append(issue(
                "high", "topology", "switch", eid, "element",
                f"Switch {eid} type='b' (bus-bus) but element {element} is not a valid bus.",
                "Correct switch element reference or change switch type.",
            ))
        elif et == "l" and element not in line_ids:
            issues.append(issue(
                "high", "topology", "switch", eid, "element",
                f"Switch {eid} type='l' (line) but element {element} is not a valid line.",
                "Correct switch element reference or change switch type.",
            ))
    return issues


# ---------------------------------------------------------------------------
# top_04 — Radial violation — mesh detected (with element suggestion)
# ---------------------------------------------------------------------------

def _top_04_radial_violation(net: Any, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    cycles = nx.cycle_basis(graph)
    for cycle in cycles:
        # Find which element creates the loop
        cycle_edges = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            edge_data = graph.get_edge_data(u, v)
            if edge_data:
                cycle_edges.append(f"{edge_data.get('element', '?')}[{edge_data.get('index', '?')}]")
        issues.append(issue(
            "medium", "topology", "network", ",".join(map(str, cycle[:5])),
            "loop",
            f"Loop detected involving buses {cycle[:5]}{'...' if len(cycle)>5 else ''} "
            f"via {', '.join(cycle_edges[:3])}.",
            "Open one switching point to restore radial topology.",
        ))
    return issues


# ---------------------------------------------------------------------------
# top_05 — Long radial path (voltage drop risk)
# ---------------------------------------------------------------------------

def _top_05_long_radial_path(
    net: Any, graph: nx.Graph, *, max_segments: int = 50, max_km: float = 30.0,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    source_buses: set = set()
    for tbl in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, tbl)
        if not eg.empty and "bus" in eg.columns:
            source_buses.update(eg["bus"].dropna().unique().tolist())

    if not source_buses:
        return issues

    line = get_table(net, "line")
    # Build edge-length map
    edge_length: dict[tuple, float] = {}
    if not line.empty and "length_km" in line.columns:
        for eid, row in dedup_iter(line):
            if not is_in_service(row):
                continue
            fb, tb = row.get("from_bus"), row.get("to_bus")
            length = safe_float(row.get("length_km")) or 0.0
            edge_length[(fb, tb)] = length
            edge_length[(tb, fb)] = length

    # Check leaf buses (degree 1)
    for node in graph.nodes():
        if graph.degree(node) != 1:
            continue
        # Find shortest path to any source
        min_path = None
        for src in source_buses:
            if src not in graph:
                continue
            try:
                path = nx.shortest_path(graph, src, node)
                if min_path is None or len(path) < len(min_path):
                    min_path = path
            except nx.NetworkXNoPath:
                continue

        if min_path is None:
            continue

        n_segments = len(min_path) - 1
        total_km = sum(
            edge_length.get((min_path[i], min_path[i+1]), 0.0)
            for i in range(n_segments)
        )

        if n_segments > max_segments:
            issues.append(issue(
                "medium", "topology", "bus", node, "path_length",
                f"Leaf bus {node} is {n_segments} segments from source (>{max_segments}).",
                "Check for voltage drop issues on long radial paths.",
            ))
        elif total_km > max_km:
            issues.append(issue(
                "medium", "topology", "bus", node, "path_km",
                f"Leaf bus {node} is {total_km:.1f} km from source (>{max_km} km).",
                "Check for voltage drop issues on long feeders.",
            ))
    return issues


# ---------------------------------------------------------------------------
# top_06 — Degree-1 buses that aren't leaf loads
# ---------------------------------------------------------------------------

def _top_06_dead_end_non_leaf_load(net: Any, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    has_equipment: set = set()
    for tbl in ("asymmetric_load", "asymmetric_sgen", "ext_grid", "ext_grid_sequence"):
        table = get_table(net, tbl)
        if not table.empty and "bus" in table.columns:
            has_equipment.update(table["bus"].dropna().unique().tolist())

    for node in graph.nodes():
        if graph.degree(node) == 1 and node not in has_equipment:
            issues.append(issue(
                "low", "topology", "bus", node, "dead_end",
                f"Bus {node} is a dead-end with no load, generation, or source.",
                "Remove unused bus or connect equipment.",
            ))
    return issues
