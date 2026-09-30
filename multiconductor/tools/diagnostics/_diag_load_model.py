"""Category 7 — Load Model Problems.

Checks: ld_01 through ld_05.
Phase 1 implements ld_02, ld_05 (power-factor, imbalance).
Phase 3 adds ld_01, ld_03, ld_04 (cross-table aggregation).
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ._common import (
    get_table,
    get_bus_vn_kv,
    is_in_service,
    safe_float,
    dedup_iter,
    issue,
    trafo1ph_bus_pair,
)


def check_load_model(net: Any, *, include_capacity_checks: bool = True) -> list[dict[str, Any]]:
    """Run all load model checks."""
    issues: list[dict[str, Any]] = []
    if include_capacity_checks:
        issues.extend(_ld_01_load_exceeds_transformer_capacity(net))
    issues.extend(_ld_02_power_factor_out_of_range(net))
    if include_capacity_checks:
        issues.extend(_ld_03_load_on_wrong_voltage_level(net))
    issues.extend(_ld_05_unbalanced_loading(net))
    return issues


# ---------------------------------------------------------------------------
# ld_01 — Load kVA exceeds transformer capacity (Phase 3)
# ---------------------------------------------------------------------------

def _ld_01_load_exceeds_transformer_capacity(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    trafo1ph = get_table(net, "trafo1ph")
    if load.empty or trafo1ph.empty:
        return issues
    if "bus" not in load.columns or "p_mw" not in load.columns:
        return issues
    if not isinstance(trafo1ph.index, pd.MultiIndex):
        return issues

    # Sum load p_mw per bus
    load_per_bus: dict[Any, float] = {}
    for _, row in load.iterrows():
        if not is_in_service(row):
            continue
        bus_val = row.get("bus")
        p = safe_float(row.get("p_mw"))
        q = safe_float(row.get("q_mvar")) or 0.0
        if bus_val is not None and p is not None:
            s = math.sqrt(p**2 + q**2)
            load_per_bus[bus_val] = load_per_bus.get(bus_val, 0.0) + s

    # Check each transformer's LV-side loading
    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        pair = trafo1ph_bus_pair(trafo1ph, tid)
        if pair is None:
            continue
        _, lv_bus = pair
        # Get transformer sn_mva
        try:
            sub = trafo1ph.xs(tid, level=idx_level)
            sn_vals = sub["sn_mva"].dropna() if "sn_mva" in sub.columns else pd.Series(dtype=float)
            if sn_vals.empty:
                continue
            sn_mva = safe_float(sn_vals.iloc[0])
        except Exception:
            continue
        if sn_mva is None or sn_mva <= 0:
            continue

        lv_load = load_per_bus.get(lv_bus, 0.0)
        if lv_load > sn_mva * 1.5:
            issues.append(issue(
                "high", "load_model", "trafo1ph", tid, "sn_mva",
                f"Transformer {tid} rated {sn_mva:.4f} MVA but LV bus {lv_bus} "
                f"has {lv_load:.4f} MVA of connected load ({lv_load/sn_mva*100:.0f}%).",
                "Verify load allocation or increase transformer rating.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ld_02 — Power factor out of range
# ---------------------------------------------------------------------------

def _ld_02_power_factor_out_of_range(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    if load.empty or "p_mw" not in load.columns or "q_mvar" not in load.columns:
        return issues

    for eid, row in dedup_iter(load):
        if not is_in_service(row):
            continue
        p = safe_float(row.get("p_mw"))
        q = safe_float(row.get("q_mvar"))
        if p is None or q is None:
            continue
        s = math.sqrt(p**2 + q**2)
        if s < 1e-12:
            continue
        pf = abs(p) / s
        if pf < 0.70:
            issues.append(issue(
                "high", "load_model", "asymmetric_load", eid, "p_mw/q_mvar",
                f"Load {eid} power factor = {pf:.3f} is below 0.70.",
                "Verify reactive power or add power factor correction.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ld_03 — Load on wrong voltage level (Phase 3)
# ---------------------------------------------------------------------------

def _ld_03_load_on_wrong_voltage_level(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    bus = get_table(net, "bus")
    if load.empty or bus.empty or "bus" not in load.columns or "p_mw" not in load.columns:
        return issues

    for eid, row in dedup_iter(load):
        if not is_in_service(row):
            continue
        p = safe_float(row.get("p_mw"))
        bus_val = row.get("bus")
        if p is None or bus_val is None:
            continue
        vn = get_bus_vn_kv(bus, bus_val)
        if vn is None:
            continue
        # Small residential load (< 0.001 MW = 1 kW) on a primary bus (> 4 kV)
        if abs(p) < 0.001 and vn > 4.0:
            issues.append(issue(
                "medium", "load_model", "asymmetric_load", eid, "bus/p_mw",
                f"Load {eid} has p_mw={p:.6f} (< 1 kW) on {vn:.2f} kV bus — "
                f"possible secondary load on primary bus.",
                "Verify load is on the correct bus and voltage level.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ld_05 — Unbalanced loading exceeds threshold
# ---------------------------------------------------------------------------

def _ld_05_unbalanced_loading(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    load = get_table(net, "asymmetric_load")
    if load.empty or "bus" not in load.columns or "p_mw" not in load.columns:
        return issues
    if "from_phase" not in load.columns:
        return issues

    # Aggregate p_mw per bus per phase
    bus_phase_p: dict[Any, dict[int, float]] = {}
    for _, row in load.iterrows():
        if not is_in_service(row):
            continue
        bus_val = row.get("bus")
        phase = safe_float(row.get("from_phase"))
        p = safe_float(row.get("p_mw"))
        if bus_val is None or phase is None or p is None:
            continue
        ph = int(phase)
        if ph == 0:  # skip neutral
            continue
        bus_phase_p.setdefault(bus_val, {}).setdefault(ph, 0.0)
        bus_phase_p[bus_val][ph] += p

    for bus_val, phase_dict in bus_phase_p.items():
        if len(phase_dict) < 2:
            continue
        values = list(phase_dict.values())
        avg = sum(values) / len(values)
        if avg < 1e-12:
            continue
        max_v, min_v = max(values), min(values)
        imbalance = (max_v - min_v) / avg
        if imbalance > 0.20:
            issues.append(issue(
                "low", "load_model", "bus", bus_val, "p_mw",
                f"Bus {bus_val} phase loading imbalance = {imbalance*100:.1f}% "
                f"(phases: {phase_dict}).",
                "Redistribute loads across phases to reduce imbalance.",
            ))
    return issues
