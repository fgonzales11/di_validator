"""Shared helpers for the diagnostics package.

Re-exports useful functions from network_validators and adds graph utilities
needed by multiple diagnostic modules.
"""
from __future__ import annotations

import inspect
import re
from typing import Any

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Issue builder (same contract as network_validators._issue)
# ---------------------------------------------------------------------------

def issue(
    severity: str,
    check: str,
    element_type: str,
    element_index: Any,
    field: str,
    message: str,
    suggestion: str,
    code: str | None = None,
    *,
    phase: int | None = None,
    evidence: Any | None = None,
) -> dict[str, Any]:
    if code is None:
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        code = "unknown"
        while caller_frame is not None:
            match = re.match(r"^_([a-z]+_\d{2})_", caller_frame.f_code.co_name)
            if match:
                code = match.group(1)
                break
            caller_frame = caller_frame.f_back
    return {
        "code": code,
        "category": check,
        "severity": severity,
        "check": check,
        "element_type": element_type,
        "element_index": element_index,
        "phase": phase,
        "field": field,
        "evidence": {"field": field} if evidence is None else evidence,
        "message": message,
        "recommendation": suggestion,
        "suggestion": suggestion,
    }


# ---------------------------------------------------------------------------
# Table / index helpers
# ---------------------------------------------------------------------------

def get_table(net: Any, name: str) -> pd.DataFrame:
    table = getattr(net, name, None)
    if isinstance(table, pd.DataFrame):
        return table
    return pd.DataFrame()


def unique_bus_ids(bus: pd.DataFrame) -> set:
    if bus.empty:
        return set()
    if isinstance(bus.index, pd.MultiIndex):
        return set(bus.index.get_level_values(0).unique().tolist())
    return set(bus.index.tolist())


def elem_id(idx: Any) -> Any:
    """Extract the scalar element id from a possibly-tuple MultiIndex key."""
    return idx[0] if isinstance(idx, tuple) else idx


def get_bus_vn_kv(bus: pd.DataFrame, bus_id: Any) -> float | None:
    if bus.empty or "vn_kv" not in bus.columns:
        return None
    if isinstance(bus.index, pd.MultiIndex):
        try:
            sub = bus.xs(bus_id, level=0)
            vn = sub["vn_kv"]
            return safe_float(vn.iloc[0] if hasattr(vn, "iloc") else vn)
        except (KeyError, TypeError, IndexError):
            return None
    try:
        return safe_float(bus.at[bus_id, "vn_kv"])
    except (KeyError, TypeError):
        return None


def is_in_service(row: pd.Series) -> bool:
    if "in_service" not in row.index:
        return True
    return bool(row.get("in_service", True))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def dedup_iter(table: pd.DataFrame):
    """Yield (element_id, first_row) for each unique element, skipping MultiIndex duplicates."""
    seen: set = set()
    for idx, row in table.iterrows():
        eid = elem_id(idx)
        if eid in seen:
            continue
        seen.add(eid)
        yield eid, row


def get_element_circuits(table: pd.DataFrame, eid: Any) -> pd.DataFrame | None:
    """Get all circuit/phase rows for an element from a possibly-MultiIndex table."""
    if not isinstance(table.index, pd.MultiIndex):
        try:
            return table.loc[[eid]]
        except KeyError:
            return None
    try:
        return table.xs(eid, level=0)
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Graph builders (used by topology, grounding, phase, open-conductor checks)
# ---------------------------------------------------------------------------

def build_network_graph(net: Any, per_phase: bool = False) -> nx.Graph:
    """Build a networkx graph from the network.

    If *per_phase* is False, nodes are bus IDs and edges are branch elements.
    If *per_phase* is True, nodes are (bus_id, phase) tuples.
    """
    bus = get_table(net, "bus")
    bus_ids = unique_bus_ids(bus)
    graph = nx.Graph()

    if per_phase:
        # Nodes are (bus, phase) tuples
        if isinstance(bus.index, pd.MultiIndex):
            for idx in bus.index:
                graph.add_node((idx[0], idx[1]))
        # Lines: per-circuit edges
        line = get_table(net, "line")
        for idx, row in line.iterrows():
            if not is_in_service(row):
                continue
            fb, tb = row.get("from_bus"), row.get("to_bus")
            fp, tp = safe_float(row.get("from_phase")), safe_float(row.get("to_phase"))
            if fp is not None and tp is not None and fb in bus_ids and tb in bus_ids:
                graph.add_edge((fb, int(fp)), (tb, int(tp)), element="line", index=elem_id(idx))
        # trafo1ph edges
        _add_trafo1ph_edges(net, graph, bus_ids, per_phase=True)
    else:
        graph.add_nodes_from(bus_ids)
        # Lines
        line = get_table(net, "line")
        seen_lines: set = set()
        for idx, row in line.iterrows():
            eid = elem_id(idx)
            if eid in seen_lines:
                continue
            seen_lines.add(eid)
            if not is_in_service(row):
                continue
            fb, tb = row.get("from_bus"), row.get("to_bus")
            if fb in bus_ids and tb in bus_ids:
                graph.add_edge(fb, tb, element="line", index=eid)
        # trafo1ph
        _add_trafo1ph_edges(net, graph, bus_ids, per_phase=False)
        # switches
        sw = get_table(net, "switch")
        seen_sw: set = set()
        for idx, row in sw.iterrows():
            eid = elem_id(idx)
            if eid in seen_sw:
                continue
            seen_sw.add(eid)
            if bool(row.get("closed", True)) and row.get("et") == "b":
                u, v = row.get("bus"), row.get("element")
                if u in bus_ids and v in bus_ids:
                    graph.add_edge(u, v, element="switch", index=eid)

    return graph


def _add_trafo1ph_edges(net: Any, graph: nx.Graph, bus_ids: set, *, per_phase: bool) -> None:
    trafo1ph = get_table(net, "trafo1ph")
    if trafo1ph.empty or not isinstance(trafo1ph.index, pd.MultiIndex):
        return
    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
    if bus_level_name is None:
        return
    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        try:
            sub = trafo1ph.xs(tid, level=idx_level)
            bus_lv = sub.index.names.index(bus_level_name) if bus_level_name in sub.index.names else 0
            trafo_buses = list(dict.fromkeys(sub.index.get_level_values(bus_lv).tolist()))
        except Exception:
            continue
        if len(trafo_buses) < 2:
            continue
        if per_phase:
            # Connect each circuit's phases through the transformer
            for _, row in sub.iterrows():
                fp = safe_float(row.get("from_phase"))
                tp = safe_float(row.get("to_phase"))
                if fp is not None and tp is not None:
                    graph.add_edge(
                        (trafo_buses[0], int(fp)), (trafo_buses[1], int(tp)),
                        element="trafo1ph", index=tid,
                    )
        else:
            if trafo_buses[0] in bus_ids and trafo_buses[1] in bus_ids:
                graph.add_edge(trafo_buses[0], trafo_buses[1], element="trafo1ph", index=tid)


def trafo1ph_bus_pair(trafo1ph: pd.DataFrame, trafo_id: Any) -> tuple[Any, Any] | None:
    """Extract (hv_bus, lv_bus) for a trafo1ph element from its MultiIndex."""
    if trafo1ph.empty or not isinstance(trafo1ph.index, pd.MultiIndex):
        return None
    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
    if bus_level_name is None:
        return None
    try:
        sub = trafo1ph.xs(trafo_id, level=idx_level)
        bus_lv = sub.index.names.index(bus_level_name) if bus_level_name in sub.index.names else 0
        buses = list(dict.fromkeys(sub.index.get_level_values(bus_lv).tolist()))
    except Exception:
        return None
    if len(buses) < 2:
        return None
    # Convention: first bus is HV, second is LV (by index order in the MultiIndex)
    return (buses[0], buses[1])


def bfs_from_sources(net: Any, graph: nx.Graph | None = None) -> set:
    """Return set of bus IDs reachable from any ext_grid or ext_grid_sequence bus."""
    if graph is None:
        graph = build_network_graph(net)

    source_buses: set = set()
    eg = get_table(net, "ext_grid")
    if not eg.empty and "bus" in eg.columns:
        source_buses.update(eg["bus"].dropna().unique().tolist())
    egs = get_table(net, "ext_grid_sequence")
    if not egs.empty and "bus" in egs.columns:
        source_buses.update(egs["bus"].dropna().unique().tolist())

    reachable: set = set()
    for src in source_buses:
        if src in graph:
            reachable.update(nx.node_connected_component(graph, src))
    return reachable
