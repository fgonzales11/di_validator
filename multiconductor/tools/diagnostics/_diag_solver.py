"""Optional solver-assisted diagnostics.

These checks are opt-in so the existing static diagnostics surface keeps its
current default behavior. Each category reuses one isolated solve per
``run_diagnostics`` invocation through the shared ``solver_state`` cache.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from multiconductor.pycci.cci_powerflow import run_pf

from ._common import elem_id, get_table, issue

_CACHE_KEY = "solver_assisted"
_DEFAULT_MAX_ITER = 100
_DEFAULT_TOL = 1e-5

_VOLTAGE_HIGH_LIMIT = 1.05
_VOLTAGE_CRITICAL_LIMIT = 1.10
_VOLTAGE_LOW_LIMIT = 0.95
_VOLTAGE_CRITICAL_LOW = 0.90
_THERMAL_HIGH_LIMIT = 100.0
_THERMAL_CRITICAL_LIMIT = 120.0
_UNBALANCE_MEDIUM_LIMIT = 3.0
_UNBALANCE_HIGH_LIMIT = 5.0
_CONDITION_MEDIUM_LIMIT = 1e8
_CONDITION_HIGH_LIMIT = 1e10
_KCL_LOW_LIMIT = 5e-5
_KCL_MEDIUM_LIMIT = 1e-4
_KCL_HIGH_LIMIT = 1e-3


@dataclass(slots=True)
class SolverDiagnosticState:
    work_net: Any
    iterations: int
    max_iterations: int
    residual_max: float
    residual_bus: Any | None
    residual_phase: int | None
    residual_y_index: int | None
    jacobian_condition_number: float


def check_kcl(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _kcl_01_terminal_mismatch(state)


def _kcl_01_terminal_mismatch(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    if not np.isfinite(state.residual_max) or state.residual_max <= _KCL_LOW_LIMIT:
        return []

    severity = "low"
    if state.residual_max > _KCL_HIGH_LIMIT:
        severity = "high"
    elif state.residual_max > _KCL_MEDIUM_LIMIT:
        severity = "medium"

    element_index = state.residual_bus if state.residual_bus is not None else "solver"
    phase = state.residual_phase
    message = (
        f"Terminal KCL mismatch is {state.residual_max:.3e} pu"
        + (
            f" at bus {state.residual_bus} phase {state.residual_phase}."
            if state.residual_bus is not None and state.residual_phase is not None
            else "."
        )
    )
    return [
        issue(
            severity,
            "kcl",
            "bus",
            element_index,
            "current_mismatch_pu",
            message,
            "Inspect branch admittances, source/load injections, and phase connectivity near the flagged terminal.",
            phase=phase,
            evidence={
                "current_mismatch_pu": state.residual_max,
                "threshold_pu": _KCL_LOW_LIMIT,
                "y_index": state.residual_y_index,
            },
        )
    ]


def check_voltage(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _vol_01_voltage_limit_violation(state)


def _vol_01_voltage_limit_violation(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    bus_results = get_table(state.work_net, "res_bus")
    if bus_results.empty or "vm_pu" not in bus_results.columns:
        return []

    findings: list[dict[str, Any]] = []
    for idx, row in bus_results.iterrows():
        bus_id, phase = _bus_phase_from_index(idx)
        if phase == 0:
            continue
        vm_pu = _safe_float(row.get("vm_pu"))
        if vm_pu is None or not np.isfinite(vm_pu):
            continue
        if _VOLTAGE_LOW_LIMIT <= vm_pu <= _VOLTAGE_HIGH_LIMIT:
            continue
        severity = (
            "critical"
            if vm_pu < _VOLTAGE_CRITICAL_LOW or vm_pu > _VOLTAGE_CRITICAL_LIMIT
            else "high"
        )
        findings.append(
            issue(
                severity,
                "voltage",
                "bus",
                bus_id,
                "vm_pu",
                f"Bus {bus_id} phase {phase} solved at {vm_pu:.4f} pu, outside the operating band.",
                "Adjust taps, source settings, reactive support, or loading to restore voltage within limits.",
                phase=phase,
                evidence={
                    "vm_pu": vm_pu,
                    "lower_limit_pu": _VOLTAGE_LOW_LIMIT,
                    "upper_limit_pu": _VOLTAGE_HIGH_LIMIT,
                },
            )
        )
    return findings


def check_thermal(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _thr_01_loading_limit_violation(state)


def _thr_01_loading_limit_violation(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    findings.extend(_loading_findings(get_table(state.work_net, "res_line"), "line"))
    findings.extend(_loading_findings(get_table(state.work_net, "res_trafo"), "trafo"))
    return findings


def check_unbalance(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _unb_01_voltage_unbalance(state)


def _unb_01_voltage_unbalance(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    bus_results = get_table(state.work_net, "res_bus")
    if bus_results.empty or "imbalance_percent" not in bus_results.columns:
        return []

    findings: list[dict[str, Any]] = []
    for bus_id, group in bus_results.groupby(level=0):
        phase_group = group
        if isinstance(group.index, pd.MultiIndex):
            phase_group = group[group.index.get_level_values(1) != 0]
        imbalance_values = pd.to_numeric(phase_group.get("imbalance_percent"), errors="coerce").dropna()
        if imbalance_values.empty:
            continue
        imbalance = float(imbalance_values.max())
        if imbalance is None or not np.isfinite(imbalance) or imbalance <= _UNBALANCE_MEDIUM_LIMIT:
            continue
        severity = "high" if imbalance > _UNBALANCE_HIGH_LIMIT else "medium"
        findings.append(
            issue(
                severity,
                "unbalance",
                "bus",
                bus_id,
                "imbalance_percent",
                f"Bus {bus_id} voltage unbalance is {imbalance:.2f}%.",
                "Rebalance phase loading or generation and verify missing/open conductors are not driving asymmetry.",
                evidence={
                    "imbalance_percent": imbalance,
                    "medium_limit_percent": _UNBALANCE_MEDIUM_LIMIT,
                    "high_limit_percent": _UNBALANCE_HIGH_LIMIT,
                },
            )
        )
    return findings


def check_numerical_conditioning(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _num_01_jacobian_conditioning(state)


def _num_01_jacobian_conditioning(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    cond = state.jacobian_condition_number
    if np.isfinite(cond) and cond <= _CONDITION_MEDIUM_LIMIT:
        return []

    severity = "critical" if not np.isfinite(cond) else "high"
    if np.isfinite(cond) and cond <= _CONDITION_HIGH_LIMIT:
        severity = "medium"
    message = (
        "Solver Jacobian is singular or non-finite."
        if not np.isfinite(cond)
        else f"Solver Jacobian condition number is {cond:.3e}."
    )
    return [
        issue(
            severity,
            "numerical_conditioning",
            "network",
            "solver",
            "jacobian_condition_number",
            message,
            "Check impedance scaling, near-zero admittances, and duplicate/contradictory electrical paths before relying on solved results.",
            evidence={
                "jacobian_condition_number": None if not np.isfinite(cond) else cond,
                "medium_limit": _CONDITION_MEDIUM_LIMIT,
                "high_limit": _CONDITION_HIGH_LIMIT,
            },
        )
    ]


def check_convergence(
    net: Any,
    *,
    solver_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = _get_solver_state(net, solver_state)
    return _cnv_01_iteration_margin(state)


def _cnv_01_iteration_margin(state: SolverDiagnosticState) -> list[dict[str, Any]]:
    if state.max_iterations <= 0:
        return []
    iteration_ratio = state.iterations / state.max_iterations
    if iteration_ratio < 0.75:
        return []

    severity = "medium" if iteration_ratio < 0.9 else "high"
    return [
        issue(
            severity,
            "convergence",
            "network",
            "solver",
            "iterations",
            f"Solver required {state.iterations} iterations out of {state.max_iterations}.",
            "Tighten data quality, reduce ill-conditioned elements, or revisit controls before expecting stable solves under stressed conditions.",
            evidence={
                "iterations": state.iterations,
                "max_iterations": state.max_iterations,
                "residual_max": state.residual_max,
            },
        )
    ]


def _get_solver_state(
    net: Any,
    solver_state: dict[str, Any] | None,
    *,
    max_iterations: int = _DEFAULT_MAX_ITER,
    tol_vmag_pu: float = _DEFAULT_TOL,
    tol_vang_rad: float = _DEFAULT_TOL,
) -> SolverDiagnosticState:
    cache = solver_state if solver_state is not None else {}
    cached = cache.get(_CACHE_KEY)
    if isinstance(cached, SolverDiagnosticState):
        return cached

    work_net = deepcopy(net)
    run_pf(
        work_net,
        tol_vmag_pu=tol_vmag_pu,
        tol_vang_rad=tol_vang_rad,
        MaxIter=max_iterations,
    )

    residual_max, residual_bus, residual_phase, residual_y_index, jacobian_cond = _solver_model_metrics(
        work_net
    )

    state = SolverDiagnosticState(
        work_net=work_net,
        iterations=int(getattr(work_net.model, "iterations", 0)),
        max_iterations=max_iterations,
        residual_max=residual_max,
        residual_bus=residual_bus,
        residual_phase=residual_phase,
        residual_y_index=residual_y_index,
        jacobian_condition_number=jacobian_cond,
    )
    cache[_CACHE_KEY] = state
    return state


def _solver_model_metrics(net: Any) -> tuple[float, Any | None, int | None, int | None, float]:
    model = getattr(net, "model", None)
    if model is None:
        return 0.0, None, None, None, 1.0

    y_pass = _passive_admittance_matrix(model)
    fixed_voltage = np.asarray(getattr(model, "y_fixed_voltage", np.array([], dtype=complex))).reshape(-1)
    if fixed_voltage.size == 0:
        return 0.0, None, None, None, 1.0

    y_fix = np.where(np.isfinite(fixed_voltage) & (fixed_voltage != -1))[0]
    y_nonslack = np.where(fixed_voltage == -1)[0]
    if y_nonslack.size == 0:
        return 0.0, None, None, None, 1.0

    voltage = np.asarray(getattr(model, "E", np.zeros((model.y_size, 1), dtype=complex))).reshape(-1, 1)
    e_fix = np.asarray(model.y_fixed_voltage, dtype=complex)[y_fix].reshape(-1, 1)
    y1 = y_pass[y_nonslack, :]
    y_lg = y1[:, y_fix]
    y_ll = y1[:, y_nonslack]
    i_fix = y_lg @ e_fix
    i_shunt = _net_injection_currents(net, model, voltage)
    residual_complex = (y_ll @ voltage[y_nonslack]) - (i_shunt[y_nonslack] - i_fix)
    residual_vector = np.asarray(residual_complex, dtype=complex).reshape(-1)
    terminal_lookup = _terminal_lookup(net, model.terminal_to_y_lookup)

    if residual_vector.size == 0:
        residual_max = 0.0
        residual_bus = None
        residual_phase = None
        residual_y_index = None
    else:
        magnitudes = np.abs(residual_vector)
        worst_position = int(np.argmax(magnitudes))
        residual_max = float(magnitudes[worst_position])
        residual_y_index = int(y_nonslack[worst_position])
        residual_bus, residual_phase = terminal_lookup.get(residual_y_index, (None, None))

    jacobian_cond = _condition_number(y_ll)
    return residual_max, residual_bus, residual_phase, residual_y_index, jacobian_cond


def _passive_admittance_matrix(model: Any) -> Any:
    y_network = getattr(model, "Y_network")
    y_center_tap = getattr(model, "Y_center_tap_trafo", None)
    if y_center_tap is None:
        y_center_tap = y_network.__class__(y_network.shape, dtype=y_network.dtype)
    return (
        getattr(model, "Y_tran")
        + y_center_tap
        + y_network
        + getattr(model, "Y_ground")
        + getattr(model, "Y_source")
        + getattr(model, "Y_shunt")
        + getattr(model, "Y_switch")
    )


def _net_injection_currents(net: Any, model: Any, voltage: np.ndarray) -> np.ndarray:
    load_terms = _shunt_terms(net, model, voltage, "asymmetric_load", sign=1.0)
    sgen_terms = _shunt_terms(net, model, voltage, "asymmetric_sgen", sign=-1.0)
    y_from = np.hstack([load_terms[0], sgen_terms[0]])
    y_to = np.hstack([load_terms[1], sgen_terms[1]])
    abs_e0 = np.hstack([load_terms[2], sgen_terms[2]])
    s_const_power = np.hstack([load_terms[3], sgen_terms[3]])
    s_const_current = np.hstack([load_terms[4], sgen_terms[4]])
    s_const_impedance = np.hstack([load_terms[5], sgen_terms[5]])

    currents = np.zeros((model.y_size, 1), dtype=np.complex128)
    if len(y_from) == 0:
        return currents

    e_shunt = (voltage[y_from] - voltage[y_to]).reshape(-1)
    e_shunt[np.abs(e_shunt) < 1e-12] = 1.0 + 0.0j
    abs_e = np.abs(e_shunt)
    s_active = (
        s_const_power
        + s_const_current * abs_e / abs_e0
        + s_const_impedance * (abs_e ** 2) / (abs_e0 ** 2)
    )
    i_corr = -np.conj(s_active / (net.sn_mva * 1e6) / e_shunt)
    np.add.at(currents[:, 0], y_from, i_corr)
    np.add.at(currents[:, 0], y_to, -i_corr)
    return currents


def _shunt_terms(
    net: Any,
    model: Any,
    voltage: np.ndarray,
    table_name: str,
    *,
    sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = get_table(net, table_name)
    if table.empty:
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=int)
        return empty_i, empty_i, empty, empty.astype(np.complex128), empty.astype(np.complex128), empty.astype(np.complex128)

    buses = table["bus"].to_numpy(dtype=int)
    from_phases = table["from_phase"].to_numpy(dtype=int)
    to_phases = table["to_phase"].to_numpy(dtype=int)
    y_from = model.terminal_to_y_lookup[buses * 4 + from_phases]
    y_to = model.terminal_to_y_lookup[buses * 4 + to_phases]
    y_isolated = np.asarray(getattr(model, "y_isolated", np.array([], dtype=int)), dtype=int)
    connected = (~np.isin(y_from, y_isolated)) & (y_from >= 0) & (y_to >= 0)
    if not np.any(connected):
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=int)
        return empty_i, empty_i, empty, empty.astype(np.complex128), empty.astype(np.complex128), empty.astype(np.complex128)

    y_from = np.asarray(y_from[connected], dtype=int)
    y_to = np.asarray(y_to[connected], dtype=int)
    abs_e0 = np.abs(voltage[y_from] - voltage[y_to]).reshape(-1)
    abs_e0 = np.where(abs_e0 > 1e-12, abs_e0, 1.0)
    p_mw = table.loc[connected, "p_mw"].to_numpy(dtype=float)
    q_mvar = table.loc[connected, "q_mvar"].to_numpy(dtype=float)
    in_service = _column_array(table.loc[connected], "in_service", True).astype(float)
    s = sign * (p_mw + 1j * q_mvar) * 1e6 * in_service
    k_ip = _column_array(table.loc[connected], "const_i_percent_p", 0.0) / 100.0
    k_zp = _column_array(table.loc[connected], "const_z_percent_p", 0.0) / 100.0
    k_pp = 1.0 - (k_ip + k_zp)
    k_iq = _column_array(table.loc[connected], "const_i_percent_q", 0.0) / 100.0
    k_zq = _column_array(table.loc[connected], "const_z_percent_q", 0.0) / 100.0
    k_pq = 1.0 - (k_iq + k_zq)
    return (
        y_from,
        y_to,
        abs_e0.astype(float),
        k_pp * np.real(s) + 1j * k_pq * np.imag(s),
        k_ip * np.real(s) + 1j * k_iq * np.imag(s),
        k_zp * np.real(s) + 1j * k_zq * np.imag(s),
    )


def _column_array(table: pd.DataFrame, column: str, default: bool | float) -> np.ndarray:
    if column not in table.columns:
        return np.full(len(table), default, dtype=float if isinstance(default, float) else bool)
    values = table[column].to_numpy()
    if isinstance(default, bool):
        return np.asarray(values, dtype=bool)
    return np.nan_to_num(np.asarray(values, dtype=float), nan=float(default))


def _condition_number(matrix: Any) -> float:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 1.0
    try:
        return float(np.linalg.cond(matrix.toarray()))
    except Exception:
        return float("inf")


def _terminal_lookup(net: Any, terminal_to_y_lookup: Any) -> dict[int, tuple[Any, int]]:
    bus = get_table(net, "bus")
    if bus.empty or not isinstance(bus.index, pd.MultiIndex):
        return {}

    lookup: dict[int, tuple[Any, int]] = {}
    terminals = np.asarray(terminal_to_y_lookup, dtype=int)
    for bus_id, phase in bus.index.tolist():
        terminal = int(bus_id) * 4 + int(phase)
        if terminal < 0 or terminal >= len(terminals):
            continue
        y_index = int(terminals[terminal])
        if y_index >= 0:
            lookup[y_index] = (bus_id, int(phase))
    return lookup


def _loading_findings(result_table: pd.DataFrame, element_type: str) -> list[dict[str, Any]]:
    if result_table.empty or "loading_percent" not in result_table.columns:
        return []

    findings: list[dict[str, Any]] = []
    for idx, row in result_table.iterrows():
        loading = _safe_float(row.get("loading_percent"))
        if loading is None or not np.isfinite(loading) or loading <= _THERMAL_HIGH_LIMIT:
            continue
        severity = "critical" if loading > _THERMAL_CRITICAL_LIMIT else "high"
        evidence = {
            "loading_percent": loading,
            "limit_percent": _THERMAL_HIGH_LIMIT,
        }
        if isinstance(idx, tuple) and len(idx) > 1:
            evidence["row_index"] = list(idx)
        findings.append(
            issue(
                severity,
                "thermal",
                element_type,
                elem_id(idx),
                "loading_percent",
                f"{element_type.title()} {elem_id(idx)} solved at {loading:.2f}% loading.",
                "Reduce loading, change topology, or upgrade equipment before accepting the operating point.",
                evidence=evidence,
            )
        )
    return findings


def _bus_phase_from_index(idx: Any) -> tuple[Any, int | None]:
    if isinstance(idx, tuple):
        bus_id = idx[0]
        phase = int(idx[1]) if len(idx) > 1 and idx[1] is not None else None
        return bus_id, phase
    return idx, None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
