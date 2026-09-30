"""Category 4 — Bad Phase Connectivity.

Checks: ph_01 through ph_05.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd

from ._common import (
    get_table,
    elem_id,
    is_in_service,
    safe_float,
    dedup_iter,
    issue,
)


def check_phase_connectivity(net: Any) -> list[dict[str, Any]]:
    """Run all phase connectivity checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_ph_01_load_on_unreachable_phase(net))
    issues.extend(_ph_02_phase_renumbering_across_transformer(net))
    issues.extend(_ph_03_neutral_without_ground_return(net))
    issues.extend(_ph_04_load_phase_not_on_bus(net))
    issues.extend(_ph_05_phase_ordering_anomalies(net))
    return issues


# ---------------------------------------------------------------------------
# ph_01 — Load on phase not served by upstream line (per-phase BFS)
# ---------------------------------------------------------------------------

def _ph_01_load_on_unreachable_phase(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    line = get_table(net, "line")
    load = get_table(net, "asymmetric_load")
    if bus.empty or line.empty or load.empty:
        return issues
    if "from_phase" not in load.columns or "bus" not in load.columns:
        return issues
    if "from_phase" not in line.columns:
        return issues

    # Find source buses
    source_buses: set = set()
    for tbl in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, tbl)
        if not eg.empty and "bus" in eg.columns:
            source_buses.update(eg["bus"].dropna().unique().tolist())

    # Build per-phase adjacency: {(bus, phase): set of (bus, phase)}
    phase_adj: dict[tuple, set[tuple]] = {}

    for _, row in line.iterrows():
        if not is_in_service(row):
            continue
        fb, tb = row.get("from_bus"), row.get("to_bus")
        fp, tp = safe_float(row.get("from_phase")), safe_float(row.get("to_phase"))
        if fb is None or tb is None or fp is None or tp is None:
            continue
        n_from = (fb, int(fp))
        n_to = (tb, int(tp))
        phase_adj.setdefault(n_from, set()).add(n_to)
        phase_adj.setdefault(n_to, set()).add(n_from)

    # BFS from sources on all phases
    reachable: set[tuple] = set()
    queue: deque[tuple] = deque()

    # Seed with all (source_bus, phase) combinations known from bus table
    if isinstance(bus.index, pd.MultiIndex):
        for src in source_buses:
            try:
                sub = bus.xs(src, level=0)
                for phase_val in sub.index.get_level_values(0):
                    seed = (src, int(phase_val))
                    if seed not in reachable:
                        reachable.add(seed)
                        queue.append(seed)
            except (KeyError, TypeError):
                continue

    while queue:
        current = queue.popleft()
        for neighbor in phase_adj.get(current, set()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Check loads
    for eid, row in dedup_iter(load):
        if not is_in_service(row):
            continue
        bus_val = row.get("bus")
        phase = safe_float(row.get("from_phase"))
        if bus_val is None or phase is None:
            continue
        ph = int(phase)
        if ph == 0:  # neutral — skip
            continue
        if (bus_val, ph) not in reachable:
            issues.append(issue(
                "critical", "phase_connectivity", "asymmetric_load", eid, "from_phase",
                f"Load {eid} is on phase {ph} at bus {bus_val}, but that phase "
                f"is not reachable from any source.",
                "Verify upstream line phases or correct load phase assignment.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ph_02 — Phase renumbering across transformer
# ---------------------------------------------------------------------------

def _ph_02_phase_renumbering_across_transformer(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    trafo1ph = get_table(net, "trafo1ph")
    line = get_table(net, "line")
    if trafo1ph.empty or line.empty:
        return issues
    if not isinstance(trafo1ph.index, pd.MultiIndex):
        return issues
    if "from_phase" not in trafo1ph.columns or "to_phase" not in trafo1ph.columns:
        return issues

    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
    if bus_level_name is None:
        return issues

    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        try:
            sub = trafo1ph.xs(tid, level=idx_level)
        except Exception:
            continue
        # Get phase connections at each winding
        for _, row in sub.iterrows():
            fp = safe_float(row.get("from_phase"))
            tp = safe_float(row.get("to_phase"))
            if fp is None or tp is None:
                continue
            # Phase renumbering: from_phase and to_phase should be compatible
            # For single-phase trafos, they should typically match or follow known patterns
            if int(fp) != 0 and int(tp) != 0 and int(fp) != int(tp):
                issues.append(issue(
                    "high", "phase_connectivity", "trafo1ph", tid, "from_phase/to_phase",
                    f"Transformer {tid} has phase change: from_phase={int(fp)} → to_phase={int(tp)}.",
                    "Verify this is an intentional delta-wye phase shift or correct the mapping.",
                ))
    return issues


# ---------------------------------------------------------------------------
# ph_03 — Neutral conductor present without ground return
# ---------------------------------------------------------------------------

def _ph_03_neutral_without_ground_return(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    bus = get_table(net, "bus")
    if line.empty or bus.empty:
        return issues
    if "from_phase" not in line.columns:
        return issues
    if "grounded" not in bus.columns:
        return issues

    grounded_buses: set = set()
    if isinstance(bus.index, pd.MultiIndex):
        for idx in bus.index:
            if bus.at[idx, "grounded"]:
                grounded_buses.add(idx[0])

    # Find line endpoints carrying neutral (phase 0)
    checked: set = set()
    for idx, row in line.iterrows():
        eid = elem_id(idx)
        if eid in checked:
            continue
        if not is_in_service(row):
            continue
        fp = safe_float(row.get("from_phase"))
        if fp is None or int(fp) != 0:
            continue
        checked.add(eid)
        fb = row.get("from_bus")
        tb = row.get("to_bus")
        # At least one end should connect to a grounded bus
        fb_grounded = fb in grounded_buses
        tb_grounded = tb in grounded_buses
        if not fb_grounded and not tb_grounded:
            issues.append(issue(
                "high", "phase_connectivity", "line", eid, "neutral",
                f"Line {eid} carries neutral (phase 0) but neither endpoint "
                f"(bus {fb}, {tb}) is grounded.",
                "Add grounding to at least one endpoint or verify neutral connectivity.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ph_04 — Single-phase load on bus but phase not available
# ---------------------------------------------------------------------------

def _ph_04_load_phase_not_on_bus(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    bus = get_table(net, "bus")
    if load.empty or bus.empty:
        return issues
    if "from_phase" not in load.columns or "bus" not in load.columns:
        return issues
    if not isinstance(bus.index, pd.MultiIndex):
        return issues

    # Build bus → available phases map
    bus_phases: dict[Any, set[int]] = {}
    for idx in bus.index:
        bus_id, phase = idx[0], int(idx[1])
        bus_phases.setdefault(bus_id, set()).add(phase)

    for eid, row in dedup_iter(load):
        if not is_in_service(row):
            continue
        bus_val = row.get("bus")
        phase = safe_float(row.get("from_phase"))
        if bus_val is None or phase is None:
            continue
        ph = int(phase)
        available = bus_phases.get(bus_val, set())
        if available and ph not in available:
            issues.append(issue(
                "medium", "phase_connectivity", "asymmetric_load", eid, "from_phase",
                f"Load {eid} on phase {ph} but bus {bus_val} only has phases {sorted(available)}.",
                "Correct load phase assignment to match available bus phases.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ph_05 — Phase ordering anomalies
# ---------------------------------------------------------------------------

def _ph_05_phase_ordering_anomalies(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty or "from_phase" not in line.columns or "to_phase" not in line.columns:
        return issues

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        fp = safe_float(row.get("from_phase"))
        tp = safe_float(row.get("to_phase"))
        if fp is None or tp is None:
            continue
        # Phase swap: from_phase and to_phase should match for direct connections
        if int(fp) != 0 and int(tp) != 0 and int(fp) != int(tp):
            issues.append(issue(
                "low", "phase_connectivity", "line", eid, "from_phase/to_phase",
                f"Line {eid} has phase swap: from_phase={int(fp)} → to_phase={int(tp)}.",
                "Verify this is intentional or correct the phase mapping.",
            ))
    return issues
