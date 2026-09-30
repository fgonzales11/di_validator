"""Category 2 — Transformer Modeling Errors.

Checks: tx_01 through tx_06.
"""
from __future__ import annotations

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


def check_transformers(net: Any) -> list[dict[str, Any]]:
    """Run all transformer modeling checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_tx_01_zero_or_missing_impedance(net))
    issues.extend(_tx_02_turns_ratio_vs_bus_kv(net))
    issues.extend(_tx_03_sn_mva_out_of_range(net))
    issues.extend(_tx_04_tap_at_extreme(net))
    issues.extend(_tx_05_center_tap_grounding(net))
    issues.extend(_tx_06_xr_ratio(net))
    return issues


# ---------------------------------------------------------------------------
# Helpers for iterating trafo1ph by unique ID
# ---------------------------------------------------------------------------

def _iter_trafo1ph_ids(trafo1ph: pd.DataFrame):
    """Yield unique transformer IDs from the trafo1ph MultiIndex."""
    if trafo1ph.empty or not isinstance(trafo1ph.index, pd.MultiIndex):
        return
    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    for tid in trafo1ph.index.get_level_values(idx_level).unique():
        yield tid


def _get_trafo1ph_sub(trafo1ph: pd.DataFrame, tid: Any) -> pd.DataFrame | None:
    idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    try:
        return trafo1ph.xs(tid, level=idx_level)
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# tx_01 — Zero or missing impedance
# ---------------------------------------------------------------------------

def _tx_01_zero_or_missing_impedance(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in ("trafo", "trafo1ph"):
        table = get_table(net, table_name)
        if table.empty:
            continue
        if table_name == "trafo1ph":
            for tid in _iter_trafo1ph_ids(table):
                sub = _get_trafo1ph_sub(table, tid)
                if sub is None:
                    continue
                first_row = sub.iloc[0]
                if not is_in_service(first_row):
                    continue
                vk = safe_float(first_row.get("vk_percent")) if "vk_percent" in sub.columns else None
                vkr = safe_float(first_row.get("vkr_percent")) if "vkr_percent" in sub.columns else None
                if vk is not None and vk == 0:
                    issues.append(issue(
                        "critical", "transformer_model", table_name, tid, "vk_percent",
                        f"Transformer {tid} has vk_percent=0 (zero short-circuit voltage).",
                        "Set vk_percent to manufacturer nameplate value.",
                    ))
                if vkr is None:
                    issues.append(issue(
                        "critical", "transformer_model", table_name, tid, "vkr_percent",
                        f"Transformer {tid} has missing vkr_percent.",
                        "Set vkr_percent based on nameplate copper losses.",
                    ))
        else:
            for eid, row in dedup_iter(table):
                if not is_in_service(row):
                    continue
                vk = safe_float(row.get("vk_percent")) if "vk_percent" in table.columns else None
                vkr = safe_float(row.get("vkr_percent")) if "vkr_percent" in table.columns else None
                if vk is not None and vk == 0:
                    issues.append(issue(
                        "critical", "transformer_model", table_name, eid, "vk_percent",
                        f"Transformer {eid} has vk_percent=0.",
                        "Set vk_percent to manufacturer nameplate value.",
                    ))
                if vkr is None:
                    issues.append(issue(
                        "critical", "transformer_model", table_name, eid, "vkr_percent",
                        f"Transformer {eid} has missing vkr_percent.",
                        "Set vkr_percent based on nameplate copper losses.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# tx_02 — Turns ratio vs bus kV mismatch
# ---------------------------------------------------------------------------

def _tx_02_turns_ratio_vs_bus_kv(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    trafo1ph = get_table(net, "trafo1ph")
    bus = get_table(net, "bus")
    if trafo1ph.empty or bus.empty or "vn_kv" not in trafo1ph.columns:
        return issues
    if not isinstance(trafo1ph.index, pd.MultiIndex):
        return issues

    bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
    if bus_level_name is None:
        return issues

    for tid in _iter_trafo1ph_ids(trafo1ph):
        sub = _get_trafo1ph_sub(trafo1ph, tid)
        if sub is None:
            continue
        first_row = sub.iloc[0]
        if not is_in_service(first_row):
            continue
        bus_lv = sub.index.names.index(bus_level_name) if bus_level_name in sub.index.names else 0
        trafo_buses = list(dict.fromkeys(sub.index.get_level_values(bus_lv).tolist()))

        for tb in trafo_buses:
            bus_vn = get_bus_vn_kv(bus, tb)
            if bus_vn is None or bus_vn <= 0:
                continue
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
                    "high", "transformer_model", "trafo1ph", tid, "vn_kv",
                    f"Transformer {tid} winding at bus {tb}: vn_kv={winding_vn_val:.4f} kV "
                    f"but bus vn_kv={bus_vn:.4f} kV ({ratio*100:.1f}% mismatch).",
                    "Correct transformer winding voltage or bus nominal voltage.",
                ))
    return issues


# ---------------------------------------------------------------------------
# tx_03 — sn_mva out of range
# ---------------------------------------------------------------------------

def _tx_03_sn_mva_out_of_range(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in ("trafo", "trafo1ph"):
        table = get_table(net, table_name)
        if table.empty or "sn_mva" not in table.columns:
            continue

        if table_name == "trafo1ph":
            for tid in _iter_trafo1ph_ids(table):
                sub = _get_trafo1ph_sub(table, tid)
                if sub is None:
                    continue
                sn = safe_float(sub["sn_mva"].iloc[0]) if "sn_mva" in sub.columns else None
                if sn is None:
                    continue
                if sn <= 0:
                    issues.append(issue(
                        "high", "transformer_model", table_name, tid, "sn_mva",
                        f"Transformer {tid} sn_mva={sn} is non-positive.",
                        "Set sn_mva to the nameplate apparent power rating.",
                    ))
                elif sn > 500:
                    issues.append(issue(
                        "high", "transformer_model", table_name, tid, "sn_mva",
                        f"Transformer {tid} sn_mva={sn} exceeds 500 MVA (unusual for distribution).",
                        "Verify rating or check for unit errors.",
                    ))
        else:
            for eid, row in dedup_iter(table):
                if not is_in_service(row):
                    continue
                sn = safe_float(row.get("sn_mva"))
                if sn is None:
                    continue
                if sn <= 0:
                    issues.append(issue(
                        "high", "transformer_model", table_name, eid, "sn_mva",
                        f"Transformer {eid} sn_mva={sn} is non-positive.",
                        "Set sn_mva to the nameplate apparent power rating.",
                    ))
                elif sn > 500:
                    issues.append(issue(
                        "high", "transformer_model", table_name, eid, "sn_mva",
                        f"Transformer {eid} sn_mva={sn} exceeds 500 MVA.",
                        "Verify rating or check for unit errors.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# tx_04 — Tap position at extreme
# ---------------------------------------------------------------------------

def _tx_04_tap_at_extreme(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in ("trafo", "trafo1ph"):
        table = get_table(net, table_name)
        if table.empty:
            continue

        if table_name == "trafo1ph":
            for tid in _iter_trafo1ph_ids(table):
                sub = _get_trafo1ph_sub(table, tid)
                if sub is None:
                    continue
                first_row = sub.iloc[0]
                if not is_in_service(first_row):
                    continue
                _check_tap_extreme(issues, table_name, tid, first_row)
        else:
            for eid, row in dedup_iter(table):
                if not is_in_service(row):
                    continue
                _check_tap_extreme(issues, table_name, eid, row)
    return issues


def _check_tap_extreme(issues: list, table_name: str, eid: Any, row: pd.Series) -> None:
    tap_pos = safe_float(row.get("tap_pos"))
    tap_min = safe_float(row.get("tap_min"))
    tap_max = safe_float(row.get("tap_max"))
    if tap_pos is None or tap_min is None or tap_max is None:
        return
    if tap_pos == tap_min:
        issues.append(issue(
            "medium", "transformer_model", table_name, eid, "tap_pos",
            f"Transformer {eid} tap is at minimum ({tap_pos}).",
            "This may indicate a voltage regulation issue — verify intentional.",
        ))
    elif tap_pos == tap_max:
        issues.append(issue(
            "medium", "transformer_model", table_name, eid, "tap_pos",
            f"Transformer {eid} tap is at maximum ({tap_pos}).",
            "This may indicate a voltage regulation issue — verify intentional.",
        ))


# ---------------------------------------------------------------------------
# tx_05 — Center-tap / split-phase grounding
# ---------------------------------------------------------------------------

def _tx_05_center_tap_grounding(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    trafo1ph = get_table(net, "trafo1ph")
    bus = get_table(net, "bus")
    if trafo1ph.empty or bus.empty:
        return issues
    if not isinstance(trafo1ph.index, pd.MultiIndex) or "vn_kv" not in trafo1ph.columns:
        return issues

    # Detect 120/240 V split-phase transformers
    for tid in _iter_trafo1ph_ids(trafo1ph):
        sub = _get_trafo1ph_sub(trafo1ph, tid)
        if sub is None or "vn_kv" not in sub.columns:
            continue
        vn_vals = sub["vn_kv"].dropna().tolist()
        # Split-phase identification: one winding ≈ 0.12 kV, another ≈ 0.24 kV
        has_120 = any(0.10 <= v <= 0.15 for v in vn_vals)
        has_240 = any(0.20 <= v <= 0.30 for v in vn_vals)
        if not (has_120 and has_240):
            continue

        # Check for neutral grounding on the LV bus
        pair = trafo1ph_bus_pair(trafo1ph, tid)
        if pair is None:
            continue
        _, lv_bus = pair
        if isinstance(bus.index, pd.MultiIndex) and "grounded" in bus.columns:
            try:
                bus_sub = bus.xs(lv_bus, level=0)
                grounded_vals = bus_sub["grounded"].tolist()
                if not any(grounded_vals):
                    issues.append(issue(
                        "medium", "transformer_model", "trafo1ph", tid, "grounding",
                        f"Split-phase transformer {tid} LV bus {lv_bus} has no grounded phase.",
                        "Add neutral grounding for center-tap (120/240 V) transformer.",
                    ))
            except (KeyError, TypeError):
                pass
    return issues


# ---------------------------------------------------------------------------
# tx_06 — X/R ratio out of typical range
# ---------------------------------------------------------------------------

def _tx_06_xr_ratio(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for table_name in ("trafo", "trafo1ph"):
        table = get_table(net, table_name)
        if table.empty or "vk_percent" not in table.columns or "vkr_percent" not in table.columns:
            continue

        if table_name == "trafo1ph":
            for tid in _iter_trafo1ph_ids(table):
                sub = _get_trafo1ph_sub(table, tid)
                if sub is None:
                    continue
                vk = safe_float(sub["vk_percent"].iloc[0])
                vkr = safe_float(sub["vkr_percent"].iloc[0])
                if vk is None or vkr is None or vkr <= 0 or vk <= 0:
                    continue
                xr = vk / vkr
                if xr < 2.0 or xr > 50.0:
                    issues.append(issue(
                        "low", "transformer_model", table_name, tid, "vk_percent/vkr_percent",
                        f"Transformer {tid} X/R ratio = {xr:.1f} is outside typical range [2, 50].",
                        "Verify impedance data against nameplate test report.",
                    ))
        else:
            for eid, row in dedup_iter(table):
                if not is_in_service(row):
                    continue
                vk = safe_float(row.get("vk_percent"))
                vkr = safe_float(row.get("vkr_percent"))
                if vk is None or vkr is None or vkr <= 0 or vk <= 0:
                    continue
                xr = vk / vkr
                if xr < 2.0 or xr > 50.0:
                    issues.append(issue(
                        "low", "transformer_model", table_name, eid, "vk_percent/vkr_percent",
                        f"Transformer {eid} X/R ratio = {xr:.1f} outside [2, 50].",
                        "Verify impedance data against nameplate test report.",
                    ))
    return issues
