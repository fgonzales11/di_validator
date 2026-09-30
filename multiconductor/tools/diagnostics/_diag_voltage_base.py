"""Category 1 — Wrong Voltage Base.

Checks: vb_01 through vb_05.
Phase 1 implements vb_01–vb_04 (scalar comparisons).
Phase 2 adds vb_05 (BFS feeder-path consistency).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ._common import (
    get_table,
    unique_bus_ids,
    elem_id,
    get_bus_vn_kv,
    is_in_service,
    safe_float,
    dedup_iter,
    issue,
    trafo1ph_bus_pair,
)


def check_voltage_base(net: Any, *, include_bfs: bool = True) -> list[dict[str, Any]]:
    """Run all voltage base checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_vb_01_bus_vn_kv_missing_or_zero(net))
    issues.extend(_vb_02_trafo_secondary_vs_bus(net))
    issues.extend(_vb_03_ext_grid_source_mismatch(net))
    issues.extend(_vb_04_unit_confusion(net))
    if include_bfs:
        issues.extend(_vb_05_feeder_path_voltage_consistency(net))
    return issues


# ---------------------------------------------------------------------------
# vb_01 — Bus vn_kv = 0 or NaN
# ---------------------------------------------------------------------------

def _vb_01_bus_vn_kv_missing_or_zero(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty or "vn_kv" not in bus.columns:
        return issues

    for bus_id in unique_bus_ids(bus):
        vn = get_bus_vn_kv(bus, bus_id)
        if vn is None or vn <= 0:
            issues.append(issue(
                "critical", "voltage_base", "bus", bus_id, "vn_kv",
                f"Bus {bus_id} has vn_kv={vn} (missing or non-positive).",
                "Set vn_kv to the correct nominal line-line voltage in kV.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_02 — Bus vn_kv doesn't match transformer winding vn_kv
# ---------------------------------------------------------------------------

def _vb_02_trafo_secondary_vs_bus(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    trafo1ph = get_table(net, "trafo1ph")
    bus = get_table(net, "bus")
    if trafo1ph.empty or bus.empty or "vn_kv" not in trafo1ph.columns:
        return issues
    if not isinstance(trafo1ph.index, pd.MultiIndex):
        return issues

    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
    if bus_level_name is None:
        return issues

    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        try:
            sub = trafo1ph.xs(tid, level=idx_level)
            bus_lv = sub.index.names.index(bus_level_name) if bus_level_name in sub.index.names else 0
            trafo_buses = list(dict.fromkeys(sub.index.get_level_values(bus_lv).tolist()))
        except Exception:
            continue
        for tb in trafo_buses:
            bus_vn = get_bus_vn_kv(bus, tb)
            if bus_vn is None or bus_vn <= 0:
                continue
            # Get transformer winding vn_kv at this bus
            try:
                winding = sub.xs(tb, level=bus_lv)
                winding_vn = winding["vn_kv"]
                winding_vn_val = safe_float(winding_vn.iloc[0] if hasattr(winding_vn, "iloc") else winding_vn)
            except Exception:
                continue
            if winding_vn_val is None or winding_vn_val <= 0:
                continue
            ratio = abs(bus_vn - winding_vn_val) / max(bus_vn, winding_vn_val)
            if ratio > 0.10:
                issues.append(issue(
                    "high", "voltage_base", "trafo1ph", tid, "vn_kv",
                    f"Transformer {tid} winding at bus {tb}: vn_kv={winding_vn_val:.4f} kV "
                    f"but bus vn_kv={bus_vn:.4f} kV (diff {ratio*100:.1f}%).",
                    "Align transformer winding voltage with bus nominal voltage.",
                ))
    return issues


# ---------------------------------------------------------------------------
# vb_03 — ext_grid source voltage mismatch
# ---------------------------------------------------------------------------

def _vb_03_ext_grid_source_mismatch(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for table_name in ("ext_grid", "ext_grid_sequence"):
        eg = get_table(net, table_name)
        if eg.empty:
            continue
        checked = eg
        if table_name == "ext_grid_sequence":
            if isinstance(eg.index, pd.MultiIndex):
                sequence_level = (
                    eg.index.names.index("sequence")
                    if "sequence" in eg.index.names
                    else eg.index.nlevels - 1
                )
                checked = eg.loc[eg.index.get_level_values(sequence_level) == 1]
            elif "sequence" in eg.columns:
                checked = eg.loc[pd.to_numeric(eg["sequence"], errors="coerce") == 1]
        for eid, row in dedup_iter(checked):
            if not is_in_service(row):
                continue
            vm_pu = safe_float(row.get("vm_pu"))
            if vm_pu is not None and abs(vm_pu - 1.0) > 0.10:
                issues.append(issue(
                    "high", "voltage_base", table_name, eid, "vm_pu",
                    f"Source {eid} vm_pu={vm_pu:.4f} deviates >10% from 1.0 pu.",
                    "Verify source voltage magnitude is intended or correct vm_pu.",
                ))
    return issues


# ---------------------------------------------------------------------------
# vb_04 — Likely unit confusion
# ---------------------------------------------------------------------------

def _vb_04_unit_confusion(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty or "vn_kv" not in bus.columns:
        return issues

    for bus_id in unique_bus_ids(bus):
        vn = get_bus_vn_kv(bus, bus_id)
        if vn is None:
            continue
        if 0 < vn < 0.1:
            issues.append(issue(
                "medium", "voltage_base", "bus", bus_id, "vn_kv",
                f"Bus {bus_id} vn_kv={vn:.6f} — possibly Volts stored as kV.",
                "Convert to kV if the intended voltage is in the Volt range.",
            ))
        elif vn > 500:
            issues.append(issue(
                "medium", "voltage_base", "bus", bus_id, "vn_kv",
                f"Bus {bus_id} vn_kv={vn:.1f} — unusually high for distribution.",
                "Verify this is a transmission-level bus or correct the voltage base.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_05 — Voltage base inconsistency along feeder path (Phase 2 — BFS)
# ---------------------------------------------------------------------------

def _vb_05_feeder_path_voltage_consistency(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    line = get_table(net, "line")
    if bus.empty or line.empty:
        return issues

    bus_ids = unique_bus_ids(bus)

    # Collect transformer bus-pairs so we can skip them
    trafo_bus_pairs: set[tuple] = set()
    trafo1ph = get_table(net, "trafo1ph")
    if not trafo1ph.empty and isinstance(trafo1ph.index, pd.MultiIndex):
        idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
        for tid in trafo1ph.index.get_level_values(idx_level).unique():
            pair = trafo1ph_bus_pair(trafo1ph, tid)
            if pair:
                trafo_bus_pairs.add((min(pair), max(pair)))

    # Check each line: if endpoints have different vn_kv and there's no transformer, flag it
    seen: set = set()
    for idx, row in line.iterrows():
        eid = elem_id(idx)
        if eid in seen:
            continue
        seen.add(eid)
        if not is_in_service(row):
            continue
        fb, tb = row.get("from_bus"), row.get("to_bus")
        if fb not in bus_ids or tb not in bus_ids:
            continue
        vn_from = get_bus_vn_kv(bus, fb)
        vn_to = get_bus_vn_kv(bus, tb)
        if vn_from is None or vn_to is None or vn_from <= 0 or vn_to <= 0:
            continue
        pair_key = (min(fb, tb), max(fb, tb))
        if pair_key in trafo_bus_pairs:
            continue
        ratio = abs(vn_from - vn_to) / max(vn_from, vn_to)
        if ratio > 0.01:  # > 1% difference without a transformer
            issues.append(issue(
                "medium", "voltage_base", "line", eid, "from_bus/to_bus",
                f"Line {eid} connects buses with different vn_kv "
                f"({vn_from:.4f} vs {vn_to:.4f} kV) without an intervening transformer.",
                "Insert a transformer or correct bus voltage bases.",
            ))
    return issues
