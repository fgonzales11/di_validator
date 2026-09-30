"""Category 9 — Duplicate / Contradictory Equipment.

Checks: dup_01 through dup_06.
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
)


def check_duplicates(net: Any) -> list[dict[str, Any]]:
    """Run all duplicate / contradictory equipment checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_dup_01_duplicate_element_index(net))
    issues.extend(_dup_02_parallel_lines_contradictory_impedance(net))
    issues.extend(_dup_03_bus_conflicting_vn_kv(net))
    issues.extend(_dup_04_overlapping_transformers(net))
    issues.extend(_dup_05_duplicate_loads(net))
    issues.extend(_dup_06_name_collisions(net))
    return issues


# ---------------------------------------------------------------------------
# dup_01 — Duplicate element index
# ---------------------------------------------------------------------------

_ELEMENT_TABLES = (
    "bus", "line", "trafo", "trafo1ph", "switch",
    "asymmetric_load", "asymmetric_sgen", "asymmetric_shunt",
    "ext_grid", "ext_grid_sequence",
)


def _dup_01_duplicate_element_index(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in _ELEMENT_TABLES:
        table = get_table(net, table_name)
        if table.empty:
            continue

        if isinstance(table.index, pd.MultiIndex):
            # For MultiIndex, duplicates at ALL levels means truly duplicated rows
            dup_mask = table.index.duplicated(keep="first")
            if dup_mask.any():
                dup_indices = table.index[dup_mask].tolist()
                # Report unique element-level duplicates
                seen: set = set()
                for idx_tuple in dup_indices:
                    key = str(idx_tuple)
                    if key in seen:
                        continue
                    seen.add(key)
                    issues.append(issue(
                        "critical", "duplicate_equipment", table_name, str(idx_tuple),
                        "index", f"Duplicate MultiIndex entry {idx_tuple} in {table_name}.",
                        "Remove or merge the duplicate row.",
                    ))
        else:
            dup_mask = table.index.duplicated(keep="first")
            if dup_mask.any():
                for dup_idx in table.index[dup_mask].unique():
                    issues.append(issue(
                        "critical", "duplicate_equipment", table_name, dup_idx,
                        "index", f"Duplicate index {dup_idx} in {table_name}.",
                        "Remove or merge the duplicate element.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# dup_02 — Parallel lines with contradictory impedance
# ---------------------------------------------------------------------------

def _dup_02_parallel_lines_contradictory_impedance(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty or "from_bus" not in line.columns or "to_bus" not in line.columns:
        return issues

    # Group lines by bus pair (order-independent)
    bus_pair_lines: dict[tuple, list[tuple]] = {}  # (min_bus, max_bus) -> [(eid, row), ...]
    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        fb, tb = row.get("from_bus"), row.get("to_bus")
        if fb is None or tb is None:
            continue
        key = (min(fb, tb), max(fb, tb))
        bus_pair_lines.setdefault(key, []).append((eid, row))

    for pair, entries in bus_pair_lines.items():
        if len(entries) < 2:
            continue
        # Compare impedance values across parallel lines
        r_vals = []
        x_vals = []
        for eid, row in entries:
            circuits = get_element_circuits(line, eid)
            if circuits is None or circuits.empty:
                continue
            for col in ("r_ohm_per_km", "r_ohm"):
                if col in circuits.columns:
                    r_vals.append((eid, safe_float(circuits[col].iloc[0])))
                    break
            for col in ("x_ohm_per_km", "x_ohm"):
                if col in circuits.columns:
                    x_vals.append((eid, safe_float(circuits[col].iloc[0])))
                    break

        if len(r_vals) >= 2:
            r_values = [v for _, v in r_vals if v is not None and v > 0]
            if len(r_values) >= 2:
                r_max, r_min = max(r_values), min(r_values)
                if r_min > 0 and (r_max / r_min) > 1.5:
                    eids = [str(e) for e, _ in entries]
                    issues.append(issue(
                        "high", "duplicate_equipment", "line", ",".join(eids),
                        "r_ohm_per_km",
                        f"Parallel lines between buses {pair} have resistance differing by >{50}% "
                        f"(range {r_min:.6f}–{r_max:.6f}).",
                        "Verify both lines are intended parallels or correct impedance values.",
                    ))

    return issues


# ---------------------------------------------------------------------------
# dup_03 — Bus defined with conflicting vn_kv across phases
# ---------------------------------------------------------------------------

def _dup_03_bus_conflicting_vn_kv(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = get_table(net, "bus")
    if bus.empty or "vn_kv" not in bus.columns:
        return issues

    if isinstance(bus.index, pd.MultiIndex):
        grouped = bus.groupby(level=0)["vn_kv"]
        for bus_id, vn_group in grouped:
            unique_vn = vn_group.dropna().unique()
            if len(unique_vn) > 1:
                issues.append(issue(
                    "high", "duplicate_equipment", "bus", bus_id,
                    "vn_kv",
                    f"Bus {bus_id} has conflicting vn_kv across phases: {sorted(unique_vn)}.",
                    "All phases of a bus must share the same nominal voltage.",
                ))
    return issues


# ---------------------------------------------------------------------------
# dup_04 — Overlapping transformers (same bus pair)
# ---------------------------------------------------------------------------

def _dup_04_overlapping_transformers(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    from ._common import trafo1ph_bus_pair

    trafo1ph = get_table(net, "trafo1ph")
    if trafo1ph.empty or not isinstance(trafo1ph.index, pd.MultiIndex):
        return issues

    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    seen_pairs: dict[tuple, list] = {}

    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        pair = trafo1ph_bus_pair(trafo1ph, tid)
        if pair is None:
            continue
        key = (min(pair), max(pair))
        seen_pairs.setdefault(key, []).append(tid)

    for pair, tids in seen_pairs.items():
        if len(tids) > 1:
            issues.append(issue(
                "medium", "duplicate_equipment", "trafo1ph", ",".join(map(str, tids)),
                "bus_pair",
                f"Multiple transformers ({len(tids)}) connect the same bus pair {pair}.",
                "Verify parallel transformers are intentional and have consistent ratings.",
            ))
    return issues


# ---------------------------------------------------------------------------
# dup_05 — Duplicate loads on same bus/phase
# ---------------------------------------------------------------------------

def _dup_05_duplicate_loads(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    if load.empty or "bus" not in load.columns:
        return issues

    has_phase = "from_phase" in load.columns
    bus_phase_loads: dict[tuple, list] = {}

    for idx, row in load.iterrows():
        eid = elem_id(idx)
        if not is_in_service(row):
            continue
        bus_val = row.get("bus")
        phase_val = safe_float(row.get("from_phase")) if has_phase else 0
        key = (bus_val, int(phase_val) if phase_val is not None else -1)
        bus_phase_loads.setdefault(key, []).append(eid)

    for (bus_val, phase_val), eids in bus_phase_loads.items():
        unique_eids = list(dict.fromkeys(eids))
        if len(unique_eids) > 1:
            issues.append(issue(
                "medium", "duplicate_equipment", "asymmetric_load",
                ",".join(map(str, unique_eids)),
                "bus/from_phase",
                f"Multiple loads ({len(unique_eids)}) at bus {bus_val}, phase {phase_val}.",
                "Consolidate into a single load or verify parallel loads are intentional.",
            ))
    return issues


# ---------------------------------------------------------------------------
# dup_06 — Name collisions within a table
# ---------------------------------------------------------------------------

def _dup_06_name_collisions(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in _ELEMENT_TABLES:
        table = get_table(net, table_name)
        if table.empty or "name" not in table.columns:
            continue
        names = table["name"].dropna()
        if names.empty:
            continue
        dup_names = names[names.duplicated(keep=False)]
        if dup_names.empty:
            continue
        for name_val in dup_names.unique():
            issues.append(issue(
                "low", "duplicate_equipment", table_name, name_val,
                "name",
                f"Duplicate name '{name_val}' found in {table_name}.",
                "Use unique names to avoid ambiguity in reports and cross-references.",
            ))
    return issues
