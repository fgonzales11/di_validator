"""Category 5 — Incorrect Impedance Data.

Checks: imp_01 through imp_07.
Phase 1 implements imp_01, imp_06, imp_07 (range checks).
Phase 3 adds imp_02–imp_05 (matrix algebra).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
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


def check_impedance(net: Any, *, include_matrix_checks: bool = True) -> list[dict[str, Any]]:
    """Run all impedance data checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_imp_01_zero_or_negative_self_impedance(net))
    if include_matrix_checks:
        issues.extend(_imp_02_impedance_matrix_not_positive_definite(net))
        issues.extend(_imp_03_impedance_outlier(net))
        issues.extend(_imp_04_zero_sequence_inconsistent(net))
        issues.extend(_imp_05_asymmetric_mutual_impedance(net))
    issues.extend(_imp_06_xr_ratio(net))
    issues.extend(_imp_07_line_length(net))
    return issues


# ---------------------------------------------------------------------------
# Helpers — detect which impedance columns are present
# ---------------------------------------------------------------------------

def _get_r_col(line: pd.DataFrame) -> str | None:
    for col in ("r_ohm_per_km", "r_ohm"):
        if col in line.columns:
            return col
    return None


def _get_x_col(line: pd.DataFrame) -> str | None:
    for col in ("x_ohm_per_km", "x_ohm"):
        if col in line.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# imp_01 — Zero or negative self-impedance
# ---------------------------------------------------------------------------

def _imp_01_zero_or_negative_self_impedance(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty:
        return issues
    r_col = _get_r_col(line)
    x_col = _get_x_col(line)

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        # Check all circuits for this line
        circuits = get_element_circuits(line, eid)
        if circuits is None:
            continue
        for circ_idx, circ_row in circuits.iterrows():
            if r_col:
                r = safe_float(circ_row.get(r_col))
                if r is not None and r <= 0:
                    issues.append(issue(
                        "critical", "impedance_data", "line", eid, r_col,
                        f"Line {eid} circuit has non-positive resistance {r_col}={r}.",
                        "Set resistance to a positive value based on conductor specifications.",
                    ))
            if x_col:
                x = safe_float(circ_row.get(x_col))
                if x is not None and x == 0:
                    issues.append(issue(
                        "critical", "impedance_data", "line", eid, x_col,
                        f"Line {eid} circuit has zero reactance {x_col}={x}.",
                        "Set reactance to a non-zero value for physical conductors.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# imp_02 — Impedance matrix not positive definite (Phase 3)
# ---------------------------------------------------------------------------

def _imp_02_impedance_matrix_not_positive_definite(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty:
        return issues

    # Look for model_type == 'matrix' lines that have impedance matrix data
    if "model_type" not in line.columns:
        return issues

    r_col = _get_r_col(line)
    x_col = _get_x_col(line)
    if not r_col or not x_col:
        return issues

    seen: set = set()
    for idx, row in line.iterrows():
        eid = elem_id(idx)
        if eid in seen:
            continue
        if not is_in_service(row):
            continue
        if str(row.get("model_type", "")).lower() != "matrix":
            continue
        seen.add(eid)

        circuits = get_element_circuits(line, eid)
        if circuits is None or len(circuits) < 2:
            continue

        n = len(circuits)
        z_matrix = np.zeros((n, n), dtype=complex)
        r_vals = circuits[r_col].values if r_col in circuits.columns else np.zeros(n)
        x_vals = circuits[x_col].values if x_col in circuits.columns else np.zeros(n)

        # Build diagonal of Z matrix from self-impedances
        for i in range(n):
            r = safe_float(r_vals[i]) or 0.0
            x = safe_float(x_vals[i]) or 0.0
            z_matrix[i, i] = complex(r, x)

        # Check if real part (R matrix) is positive semi-definite
        r_matrix = z_matrix.real
        try:
            eigenvalues = np.linalg.eigvalsh(r_matrix)
            if np.any(eigenvalues < -1e-10):
                issues.append(issue(
                    "high", "impedance_data", "line", eid, "impedance_matrix",
                    f"Line {eid} impedance matrix has negative eigenvalue(s): "
                    f"min={eigenvalues.min():.6e}.",
                    "Verify mutual impedance data and ensure the Z matrix is physically realizable.",
                ))
        except np.linalg.LinAlgError:
            issues.append(issue(
                "high", "impedance_data", "line", eid, "impedance_matrix",
                f"Line {eid} impedance matrix eigenvalue computation failed.",
                "Check impedance data for NaN or extreme values.",
            ))

    return issues


# ---------------------------------------------------------------------------
# imp_03 — Impedance outlier (per-length) (Phase 3)
# ---------------------------------------------------------------------------

def _imp_03_impedance_outlier(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty:
        return issues

    r_col = _get_r_col(line)
    if not r_col or "length_km" not in line.columns:
        return issues

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        length = safe_float(row.get("length_km"))
        if length is None or length <= 0:
            continue
        r = safe_float(row.get(r_col))
        if r is None or r <= 0:
            continue
        # Normalize to per-km
        if "per_km" in r_col:
            r_per_km = r
        else:
            r_per_km = r / length
        if r_per_km < 0.01 or r_per_km > 10.0:
            issues.append(issue(
                "high", "impedance_data", "line", eid, r_col,
                f"Line {eid} resistance per km = {r_per_km:.4f} Ω/km is outside "
                f"typical range [0.01, 10.0].",
                "Verify conductor type and length, or check for unit errors.",
            ))
    return issues


# ---------------------------------------------------------------------------
# imp_04 — Zero-sequence impedance inconsistent (Phase 3)
# ---------------------------------------------------------------------------

def _imp_04_zero_sequence_inconsistent(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty:
        return issues

    # Check for sequence model lines with z0/z1 data
    if "model_type" not in line.columns:
        return issues

    r_col = _get_r_col(line)
    x_col = _get_x_col(line)
    if not r_col or not x_col:
        return issues

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        if str(row.get("model_type", "")).lower() != "sequence":
            continue
        circuits = get_element_circuits(line, eid)
        if circuits is None or len(circuits) < 2:
            continue
        # Expect circuits indexed by sequence: 0 (zero-seq), 1 (positive-seq)
        r_vals = []
        x_vals = []
        for _, cr in circuits.iterrows():
            r_vals.append(safe_float(cr.get(r_col)) or 0.0)
            x_vals.append(safe_float(cr.get(x_col)) or 0.0)

        if len(r_vals) >= 2 and r_vals[1] > 0:
            z0 = math.sqrt(r_vals[0]**2 + x_vals[0]**2)
            z1 = math.sqrt(r_vals[1]**2 + x_vals[1]**2)
            if z1 > 0:
                ratio = z0 / z1
                if ratio < 1.0 or ratio > 10.0:
                    issues.append(issue(
                        "medium", "impedance_data", "line", eid, "z0/z1",
                        f"Line {eid} z0/z1 ratio = {ratio:.2f} is outside typical range [1, 10].",
                        "Verify zero-sequence impedance data against conductor geometry.",
                    ))
    return issues


# ---------------------------------------------------------------------------
# imp_05 — Asymmetric mutual impedance matrix (Phase 3)
# ---------------------------------------------------------------------------

def _imp_05_asymmetric_mutual_impedance(net: Any) -> list[dict[str, Any]]:
    # Mutual coupling checks require dedicated mutual impedance columns
    # which may not exist in simple models — stub for future expansion
    return []


# ---------------------------------------------------------------------------
# imp_06 — X/R ratio unusually high or low
# ---------------------------------------------------------------------------

def _imp_06_xr_ratio(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty:
        return issues
    r_col = _get_r_col(line)
    x_col = _get_x_col(line)
    if not r_col or not x_col:
        return issues

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        r = safe_float(row.get(r_col))
        x = safe_float(row.get(x_col))
        if r is None or x is None or r <= 0 or x == 0:
            continue
        xr = abs(x) / r
        if xr < 0.2 or xr > 5.0:
            issues.append(issue(
                "low", "impedance_data", "line", eid, "x/r",
                f"Line {eid} X/R ratio = {xr:.2f} is outside typical distribution range [0.2, 5.0].",
                "Verify conductor parameters or check for unusual line characteristics.",
            ))
    return issues


# ---------------------------------------------------------------------------
# imp_07 — Very short or very long lines
# ---------------------------------------------------------------------------

def _imp_07_line_length(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    line = get_table(net, "line")
    if line.empty or "length_km" not in line.columns:
        return issues

    for eid, row in dedup_iter(line):
        if not is_in_service(row):
            continue
        length = safe_float(row.get("length_km"))
        if length is None:
            continue
        if 0 < length < 0.001:
            issues.append(issue(
                "low", "impedance_data", "line", eid, "length_km",
                f"Line {eid} length={length:.6f} km is extremely short (< 1 m).",
                "Verify this isn't a modeling artifact; consider merging adjacent buses.",
            ))
        elif length > 100:
            issues.append(issue(
                "low", "impedance_data", "line", eid, "length_km",
                f"Line {eid} length={length:.1f} km is unusually long for distribution.",
                "Verify length or check for unit errors (meters stored as km).",
            ))
    return issues
