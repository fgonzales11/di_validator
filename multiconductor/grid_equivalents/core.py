from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import pickle
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd
from scipy import sparse

from multiconductor.file_io import create_empty_network
from multiconductor.pycci.cci_powerflow import run_pf
from multiconductor.pycci.std_types import copy_std_types
from multiconductor.studies import InvalidStudyInput, TerminalRef


TerminalLike: TypeAlias = int | tuple[int, int] | TerminalRef

GRID_EQUIVALENT_MISMATCH_MAX_TOL = 1e-6
GRID_EQUIVALENT_MISMATCH_NORM_TOL = 1e-6


@dataclass(frozen=True)
class ValidationMetrics:
    current_mismatch_norm: float
    current_mismatch_max: float
    reduced_rank: int
    eliminated_free_nodes: int
    fixed_internal_nodes: int
    is_finite: bool
    is_singular: bool
    passed: bool


class WidelyLinearMatrix(np.ndarray):
    """Complex matrix with an optional conjugate-voltage contribution."""

    conjugate_matrix: np.ndarray

    def __new__(cls, values, conjugate_matrix=None):
        obj = np.asarray(values, dtype=np.complex128).view(cls)
        if conjugate_matrix is None:
            conjugate_matrix = np.zeros_like(obj, dtype=np.complex128)
        obj.conjugate_matrix = np.asarray(conjugate_matrix, dtype=np.complex128)
        if obj.conjugate_matrix.shape != obj.shape:
            raise InvalidStudyInput(
                "conjugate_matrix shape "
                f"{obj.conjugate_matrix.shape} does not match {obj.shape}"
            )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.conjugate_matrix = np.asarray(
            getattr(obj, "conjugate_matrix", np.zeros(self.shape, dtype=np.complex128)),
            dtype=np.complex128,
        )
        if self.conjugate_matrix.shape != self.shape:
            self.conjugate_matrix = np.zeros(self.shape, dtype=np.complex128)

    def __matmul__(self, other):
        vector = np.asarray(other, dtype=np.complex128)
        result = np.ndarray.__matmul__(self.view(np.ndarray), vector)
        if self.conjugate_matrix.size and np.any(self.conjugate_matrix):
            result = result + self.conjugate_matrix @ np.conj(vector)
        return result


@dataclass(frozen=True)
class GridEquivalentModel:
    eq_type: str
    representation: str
    boundary_terminals: tuple[TerminalRef, ...]
    old_to_new_bus_map: dict[int, int]
    y_eq: np.ndarray
    i_offset: np.ndarray
    base_voltage: np.ndarray
    validation: ValidationMetrics
    provenance: dict[str, object]


@dataclass(frozen=True)
class EquivalentResult:
    net: object
    model: GridEquivalentModel

    def solve(self, boundary_conditions=None) -> "EquivalentSolveResult":
        return solve_equivalent(self, boundary_conditions=boundary_conditions)


@dataclass(frozen=True)
class EquivalentSolveResult:
    net: object
    model: GridEquivalentModel
    boundary_table: pd.DataFrame
    residual_norm: float
    residual_max: float


def get_equivalent(
    net,
    eq_type: str,
    boundary_buses: Sequence[TerminalLike],
    internal_buses: Sequence[int] | None = None,
    external_buses: Sequence[int] | None = None,
    validate: bool = True,
) -> EquivalentResult:
    normalized_eq_type = _normalize_eq_type(eq_type)
    boundary_terminals = _normalize_boundary_terminals(net, boundary_buses)
    if not boundary_terminals:
        raise InvalidStudyInput(
            "boundary_buses must resolve to at least one boundary terminal"
        )

    internal_bus_filter = set(internal_buses) if internal_buses is not None else None
    external_bus_filter = set(external_buses) if external_buses is not None else None
    boundary_bus_ids = {terminal.bus for terminal in boundary_terminals}
    _validate_partitions(
        boundary_bus_ids=boundary_bus_ids,
        internal_bus_filter=internal_bus_filter,
        external_bus_filter=external_bus_filter,
    )

    work = deepcopy(net)
    run_pf(work)

    model = work.model
    y_fixed_voltage = np.asarray(model.y_fixed_voltage).reshape(-1)
    fixed_mask = np.isfinite(y_fixed_voltage) & (y_fixed_voltage != -1)
    fixed_y = np.flatnonzero(fixed_mask)
    fixed_voltage = y_fixed_voltage[fixed_y].astype(np.complex128)

    y_pass = (
        model.Y_tran
        + model.Y_network
        + model.Y_ground
        + model.Y_source
        + model.Y_shunt
        + model.Y_switch
    ).tocsr()
    injection = _frozen_injection_vector(work)
    direct_jacobian, conjugate_jacobian, injection_offset = _linearized_injection_terms(work)

    boundary_y, ordered_boundary_terminals = _boundary_y_nodes(work, boundary_terminals)
    boundary_y_set = set(boundary_y.tolist())

    free_internal_y = []
    fixed_internal_y = []
    for y_idx in range(model.y_size):
        if y_idx in boundary_y_set:
            continue
        buses = _y_index_buses(work, y_idx)
        if internal_bus_filter is not None and not buses.intersection(internal_bus_filter):
            continue
        if external_bus_filter is not None and buses and not buses.intersection(external_bus_filter):
            continue
        if not buses and internal_bus_filter is not None:
            continue
        if y_idx in fixed_y:
            fixed_internal_y.append(y_idx)
        else:
            free_internal_y.append(y_idx)

    y_eq, i_offset = _reduce_system(
        y_pass=y_pass,
        direct_jacobian=direct_jacobian,
        conjugate_jacobian=conjugate_jacobian,
        injection_offset=injection_offset,
        boundary_y=boundary_y,
        free_internal_y=np.asarray(free_internal_y, dtype=int),
        fixed_internal_y=np.asarray(fixed_internal_y, dtype=int),
        fixed_voltage=fixed_voltage[np.isin(fixed_y, fixed_internal_y)],
    )

    boundary_voltage = np.asarray(work.model.E[boundary_y]).reshape(-1)
    predicted_boundary_current = y_eq @ boundary_voltage - i_offset
    actual_boundary_current = _boundary_current_from_full_system(
        y_pass=y_pass,
        injection=injection,
        voltage=np.asarray(work.model.E).reshape(-1),
        boundary_y=boundary_y,
    )
    mismatch = predicted_boundary_current - actual_boundary_current
    boundary_validation = ~np.isin(boundary_y, fixed_y)
    if not np.any(boundary_validation):
        boundary_validation = np.zeros_like(boundary_y, dtype=bool)
    validation_slice = mismatch if mismatch.size else mismatch[boundary_validation]
    has_conjugate_term = _has_conjugate_matrix(y_eq)
    expected_rank = int(np.count_nonzero(boundary_validation)) * (2 if has_conjugate_term else 1)
    finite_arrays = _arrays_are_finite(
        y_eq,
        _conjugate_matrix(y_eq),
        i_offset,
        boundary_voltage,
    )
    reduced_rank = int(_effective_rank(y_eq)) if y_eq.size and finite_arrays else 0
    validation_metrics = ValidationMetrics(
        current_mismatch_norm=float(np.linalg.norm(validation_slice)),
        current_mismatch_max=float(np.max(np.abs(validation_slice)) if validation_slice.size else 0.0),
        reduced_rank=reduced_rank,
        eliminated_free_nodes=len(free_internal_y),
        fixed_internal_nodes=len(fixed_internal_y),
        is_finite=finite_arrays,
        is_singular=bool(y_eq.size and reduced_rank < expected_rank),
        passed=False,
    )
    validation_metrics = _finalize_validation(validation_metrics)
    if validate:
        _raise_on_invalid_equivalent(validation_metrics)

    provenance = _build_provenance(
        source_net=net,
        eq_type=normalized_eq_type,
        boundary_terminals=ordered_boundary_terminals,
        validate=validate,
    )
    if has_conjugate_term:
        provenance["widely_linear"] = {
            "y_eq_conjugate": _serialize_complex_array(_conjugate_matrix(y_eq)),
        }

    reduced_net = _build_reduced_net(
        net=net,
        eq_type=normalized_eq_type,
        boundary_terminals=ordered_boundary_terminals,
        y_eq=y_eq,
        i_offset=i_offset,
        boundary_voltage=boundary_voltage,
        validation=validation_metrics,
        provenance=provenance,
    )

    model_result = GridEquivalentModel(
        eq_type=normalized_eq_type,
        representation=normalized_eq_type,
        boundary_terminals=tuple(ordered_boundary_terminals),
        old_to_new_bus_map={bus: bus for bus in boundary_bus_ids},
        y_eq=y_eq,
        i_offset=i_offset,
        base_voltage=boundary_voltage,
        validation=validation_metrics,
        provenance=provenance,
    )
    return EquivalentResult(net=reduced_net, model=model_result)


def solve_equivalent(result_or_net, boundary_conditions=None) -> EquivalentSolveResult:
    work = deepcopy(result_or_net.net if isinstance(result_or_net, EquivalentResult) else result_or_net)
    model = (
        result_or_net.model
        if isinstance(result_or_net, EquivalentResult)
        else _model_from_reduced_net(work)
    )

    voltage = _resolve_boundary_voltage(model, work, boundary_conditions)
    current_target = _resolve_boundary_current(model, boundary_conditions)
    if current_target is None:
        current = model.y_eq @ voltage - model.i_offset
        residual = np.zeros_like(current)
    else:
        current = current_target
        residual = model.y_eq @ voltage - model.i_offset - current

    boundary_table = _build_boundary_results_table(work, model, voltage, current)
    work.res_bus_equivalent = boundary_table
    work.res_grid_equivalent = pd.DataFrame(
        [
            {
                "representation": model.representation,
                "eq_type": model.eq_type,
                "boundary_count": len(model.boundary_terminals),
                "compiled_representation": model.provenance.get("compiled_representation", "multiport"),
                "residual_norm": float(np.linalg.norm(residual)),
                "residual_max": float(np.max(np.abs(residual)) if residual.size else 0.0),
            }
        ]
    )
    return EquivalentSolveResult(
        net=work,
        model=model,
        boundary_table=boundary_table,
        residual_norm=float(np.linalg.norm(residual)),
        residual_max=float(np.max(np.abs(residual)) if residual.size else 0.0),
    )


def _normalize_eq_type(eq_type: str) -> str:
    normalized = str(eq_type).lower()
    if normalized not in {"multiport", "ward", "xward", "rei"}:
        raise InvalidStudyInput(f"Unsupported eq_type {eq_type!r}")
    return normalized


def _model_from_reduced_net(net) -> GridEquivalentModel:
    if "grid_equivalent" not in net or len(net.grid_equivalent) == 0:
        raise InvalidStudyInput(
            "Reduced network does not contain a grid_equivalent payload"
        )
    row = net.grid_equivalent.iloc[0]
    validation_data = row.get("validation", {}) or {}
    provenance = row.get("provenance", {}) or {}
    boundary_terminals = _deserialize_boundary_terminals(row["boundary_terminals"])
    old_to_new_bus_map = {
        terminal.bus: terminal.bus for terminal in boundary_terminals
    }
    if isinstance(provenance, Mapping) and provenance.get("old_to_new_bus_map"):
        old_to_new_bus_map = {
            int(bus): int(new_bus)
            for bus, new_bus in provenance["old_to_new_bus_map"].items()
        }
    y_eq = _with_conjugate_matrix(
        _complex_matrix_from_payload(row),
        _widely_linear_payload_from_provenance(provenance),
    )
    return GridEquivalentModel(
        eq_type=_normalize_eq_type(row["eq_type"]),
        representation=str(row.get("representation", row["eq_type"])),
        boundary_terminals=tuple(boundary_terminals),
        old_to_new_bus_map=old_to_new_bus_map,
        y_eq=y_eq,
        i_offset=_complex_vector_from_payload(row, "i_offset"),
        base_voltage=_complex_vector_from_payload(row, "base_voltage"),
        validation=ValidationMetrics(
            current_mismatch_norm=float(validation_data.get("current_mismatch_norm", 0.0)),
            current_mismatch_max=float(validation_data.get("current_mismatch_max", 0.0)),
            reduced_rank=int(validation_data.get("reduced_rank", 0)),
            eliminated_free_nodes=int(validation_data.get("eliminated_free_nodes", 0)),
            fixed_internal_nodes=int(validation_data.get("fixed_internal_nodes", 0)),
            is_finite=bool(validation_data.get("is_finite", True)),
            is_singular=bool(validation_data.get("is_singular", False)),
            passed=bool(validation_data.get("passed", True)),
        ),
        provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
    )


def _normalize_boundary_terminals(net, boundary_buses: Sequence[TerminalLike]) -> list[TerminalRef]:
    resolved: list[TerminalRef] = []
    bus_index = net.bus.index
    available = {(int(bus), int(phase)) for bus, phase in bus_index}
    for item in boundary_buses:
        if isinstance(item, TerminalRef):
            candidate = item
            if candidate.as_tuple() not in available:
                raise InvalidStudyInput(
                    f"Boundary terminal {candidate.as_tuple()} does not exist"
                )
            resolved.append(candidate)
            continue
        if isinstance(item, tuple):
            if len(item) != 2:
                raise InvalidStudyInput(
                    f"Boundary terminal tuple must be (bus, phase), got {item!r}"
                )
            candidate = TerminalRef(int(item[0]), int(item[1]))
            if candidate.as_tuple() not in available:
                raise InvalidStudyInput(
                    f"Boundary terminal {candidate.as_tuple()} does not exist"
                )
            resolved.append(candidate)
            continue
        bus = int(item)
        bus_rows = [TerminalRef(bus, int(phase)) for _, phase in bus_index if int(_) == bus]
        if not bus_rows:
            raise InvalidStudyInput(f"Boundary bus {bus} does not exist")
        resolved.extend(bus_rows)
    return resolved


def _boundary_y_nodes(net, boundary_terminals: Sequence[TerminalRef]) -> tuple[np.ndarray, list[TerminalRef]]:
    lookup = np.asarray(net.model.terminal_to_y_lookup, dtype=int)
    y_nodes: list[int] = []
    ordered: list[TerminalRef] = []
    seen = set()
    for terminal in boundary_terminals:
        terminal_index = 4 * terminal.bus + terminal.phase
        y_idx = int(lookup[terminal_index])
        if y_idx == -1 or y_idx in seen:
            continue
        seen.add(y_idx)
        y_nodes.append(y_idx)
        ordered.append(terminal)
    if not y_nodes:
        raise InvalidStudyInput(
            "Boundary terminals do not map to any connected Y nodes"
        )
    return np.asarray(y_nodes, dtype=int), ordered


def _y_index_buses(net, y_idx: int) -> set[int]:
    lookup = np.asarray(net.model.terminal_to_y_lookup, dtype=int)
    buses = set()
    for bus, phase in net.bus.index:
        if lookup[4 * int(bus) + int(phase)] == y_idx:
            buses.add(int(bus))
    return buses


def _reduce_system(
    *,
    y_pass: sparse.csr_matrix,
    direct_jacobian: np.ndarray,
    conjugate_jacobian: np.ndarray,
    injection_offset: np.ndarray,
    boundary_y: np.ndarray,
    free_internal_y: np.ndarray,
    fixed_internal_y: np.ndarray,
    fixed_voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_dense = y_pass.toarray().astype(np.complex128)
    a_matrix = y_dense - np.asarray(direct_jacobian, dtype=np.complex128)
    b_matrix = -np.asarray(conjugate_jacobian, dtype=np.complex128)
    d_vector = np.asarray(injection_offset, dtype=np.complex128).reshape(-1)

    a_bb = a_matrix[np.ix_(boundary_y, boundary_y)]
    b_bb = b_matrix[np.ix_(boundary_y, boundary_y)]
    d_b = d_vector[boundary_y]

    fixed_const = np.zeros(len(boundary_y), dtype=np.complex128)
    if fixed_internal_y.size:
        a_bf = a_matrix[np.ix_(boundary_y, fixed_internal_y)]
        b_bf = b_matrix[np.ix_(boundary_y, fixed_internal_y)]
        fixed_const = a_bf @ fixed_voltage + b_bf @ np.conj(fixed_voltage)

    if not free_internal_y.size:
        return _with_conjugate_matrix(a_bb, b_bb), d_b - fixed_const

    a_br = a_matrix[np.ix_(boundary_y, free_internal_y)]
    a_rb = a_matrix[np.ix_(free_internal_y, boundary_y)]
    a_rr = a_matrix[np.ix_(free_internal_y, free_internal_y)]
    b_br = b_matrix[np.ix_(boundary_y, free_internal_y)]
    b_rb = b_matrix[np.ix_(free_internal_y, boundary_y)]
    b_rr = b_matrix[np.ix_(free_internal_y, free_internal_y)]
    d_r = d_vector[free_internal_y]

    rhs_internal = d_r.copy()
    if fixed_internal_y.size:
        a_rf = a_matrix[np.ix_(free_internal_y, fixed_internal_y)]
        b_rf = b_matrix[np.ix_(free_internal_y, fixed_internal_y)]
        rhs_internal = rhs_internal - a_rf @ fixed_voltage - b_rf @ np.conj(fixed_voltage)

    coupled_matrix = np.block(
        [
            [a_rr, b_rr],
            [np.conj(b_rr), np.conj(a_rr)],
        ]
    )
    boundary_coupling = np.block(
        [
            [a_rb, b_rb],
            [np.conj(b_rb), np.conj(a_rb)],
        ]
    )
    boundary_feedback = np.block([a_br, b_br])
    rhs_augmented = np.concatenate([rhs_internal, np.conj(rhs_internal)])

    reduced_augmented = np.block([a_bb, b_bb]) - boundary_feedback @ np.linalg.solve(
        coupled_matrix,
        boundary_coupling,
    )
    constant_term = (
        boundary_feedback @ np.linalg.solve(coupled_matrix, rhs_augmented)
        + fixed_const
        - d_b
    )
    boundary_count = len(boundary_y)
    y_eq = reduced_augmented[:, :boundary_count]
    y_eq_conjugate = reduced_augmented[:, boundary_count:]
    return _with_conjugate_matrix(y_eq, y_eq_conjugate), -constant_term


def _boundary_current_from_full_system(
    *,
    y_pass: sparse.csr_matrix,
    injection: np.ndarray,
    voltage: np.ndarray,
    boundary_y: np.ndarray,
) -> np.ndarray:
    return y_pass[boundary_y, :] @ np.asarray(voltage, dtype=np.complex128).reshape(-1) - np.asarray(
        injection,
        dtype=np.complex128,
    ).reshape(-1)[boundary_y]


def _with_conjugate_matrix(values: np.ndarray, conjugate_matrix: np.ndarray | None = None) -> np.ndarray:
    conjugate = np.asarray(
        np.zeros_like(values, dtype=np.complex128) if conjugate_matrix is None else conjugate_matrix,
        dtype=np.complex128,
    )
    if not np.any(conjugate):
        return np.asarray(values, dtype=np.complex128)
    return WidelyLinearMatrix(values, conjugate)


def _conjugate_matrix(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        getattr(values, "conjugate_matrix", np.zeros(np.asarray(values).shape, dtype=np.complex128)),
        dtype=np.complex128,
    )


def _has_conjugate_matrix(values: np.ndarray) -> bool:
    conjugate = _conjugate_matrix(values)
    return bool(conjugate.size and np.any(conjugate))


def _effective_rank(values: np.ndarray) -> int:
    matrix = np.asarray(values, dtype=np.complex128)
    conjugate = _conjugate_matrix(values)
    if not np.any(conjugate):
        return int(np.linalg.matrix_rank(matrix))
    real_form = np.block(
        [
            [np.real(matrix + conjugate), -np.imag(matrix - conjugate)],
            [np.imag(matrix + conjugate), np.real(matrix - conjugate)],
        ]
    )
    return int(np.linalg.matrix_rank(real_form))


def _widely_linear_payload_from_provenance(provenance: Mapping[str, object]) -> np.ndarray | None:
    payload = provenance.get("widely_linear") if isinstance(provenance, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    y_eq_conjugate = payload.get("y_eq_conjugate")
    if y_eq_conjugate is None:
        return None
    return _deserialize_complex_array(y_eq_conjugate, expected_ndim=2)


def _complex_matrix_from_payload(row) -> np.ndarray:
    if "y_eq" in row and row["y_eq"] is not None:
        return _deserialize_complex_array(row["y_eq"], expected_ndim=2)
    return np.asarray(row["y_eq_real"], dtype=float) + 1j * np.asarray(row["y_eq_imag"], dtype=float)


def _complex_vector_from_payload(row, field: str) -> np.ndarray:
    if field in row and row[field] is not None:
        return _deserialize_complex_array(row[field], expected_ndim=1)
    return np.asarray(row[f"{field}_real"], dtype=float) + 1j * np.asarray(
        row[f"{field}_imag"],
        dtype=float,
    )


def _resolve_boundary_voltage(
    model: GridEquivalentModel,
    net,
    boundary_conditions,
) -> np.ndarray:
    condition_block = _normalize_condition_block(boundary_conditions)
    if "voltage" not in condition_block:
        if "current" not in condition_block:
            return np.asarray(model.base_voltage, dtype=np.complex128).copy()
        if _has_conjugate_matrix(model.y_eq):
            solved_voltage = _solve_widely_linear_voltage(
                model.y_eq,
                model.i_offset + condition_block["current"],
            )
        else:
            solved_voltage, *_ = np.linalg.lstsq(
                model.y_eq,
                model.i_offset + condition_block["current"],
                rcond=None,
            )
        return solved_voltage
    return _merge_terminal_overrides(
        model=model,
        default=np.asarray(model.base_voltage, dtype=np.complex128).copy(),
        override=condition_block["voltage"],
    )


def _resolve_boundary_current(model: GridEquivalentModel, boundary_conditions) -> np.ndarray | None:
    condition_block = _normalize_condition_block(boundary_conditions)
    return condition_block.get("current")


def _normalize_condition_block(boundary_conditions) -> dict[str, object]:
    if boundary_conditions is None:
        return {}
    if isinstance(boundary_conditions, np.ndarray):
        return {"voltage": np.asarray(boundary_conditions, dtype=np.complex128)}
    if isinstance(boundary_conditions, (list, tuple)):
        if len(boundary_conditions) and isinstance(boundary_conditions[0], (complex, float, int, tuple, list, dict)):
            return {"voltage": np.asarray([_parse_complex_value(item) for item in boundary_conditions], dtype=np.complex128)}
    if not isinstance(boundary_conditions, Mapping):
        raise TypeError("boundary_conditions must be None, a vector, or a mapping")
    if "voltage" in boundary_conditions or "current" in boundary_conditions:
        normalized: dict[str, object] = {}
        if "voltage" in boundary_conditions:
            normalized["voltage"] = boundary_conditions["voltage"]
        if "current" in boundary_conditions:
            current = boundary_conditions["current"]
            if isinstance(current, Mapping):
                normalized["current"] = current
            else:
                normalized["current"] = np.asarray(
                    [_parse_complex_value(item) for item in current],
                    dtype=np.complex128,
                )
        return _normalize_vector_or_mapping(normalized)
    return _normalize_vector_or_mapping({"voltage": boundary_conditions})


def _normalize_vector_or_mapping(condition_block: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in condition_block.items():
        if isinstance(value, Mapping):
            normalized[key] = value
        else:
            normalized[key] = np.asarray(
                [_parse_complex_value(item) for item in value],
                dtype=np.complex128,
            )
    return normalized


def _merge_terminal_overrides(
    *,
    model: GridEquivalentModel,
    default: np.ndarray,
    override,
) -> np.ndarray:
    if isinstance(override, Mapping):
        merged = default.copy()
        terminal_positions = {
            terminal.as_tuple(): position
            for position, terminal in enumerate(model.boundary_terminals)
        }
        for raw_terminal, raw_value in override.items():
            terminal = _coerce_terminal_ref(raw_terminal)
            key = terminal.as_tuple()
            if key not in terminal_positions:
                raise InvalidStudyInput(
                    f"Boundary override {key} is not part of this equivalent"
                )
            merged[terminal_positions[key]] = _parse_complex_value(raw_value)
        return merged
    override_array = np.asarray(override, dtype=np.complex128)
    if override_array.shape != default.shape:
        raise InvalidStudyInput(
            f"Boundary vector has shape {override_array.shape}, expected {default.shape}"
        )
    return override_array


def _coerce_terminal_ref(value) -> TerminalRef:
    if isinstance(value, TerminalRef):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return TerminalRef(int(value[0]), int(value[1]))
    raise TypeError(f"Unsupported boundary terminal key {value!r}")


def _parse_complex_value(value) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (float, int, np.floating, np.integer)):
        return complex(float(value), 0.0)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, Mapping):
        if "complex" in value:
            return complex(value["complex"])
        if "real" in value or "imag" in value:
            return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
        if "vm_pu" in value:
            angle = np.deg2rad(float(value.get("va_degree", 0.0)))
            return float(value["vm_pu"]) * np.exp(1j * angle)
    raise TypeError(f"Unsupported complex boundary value {value!r}")


def _frozen_injection_vector(net) -> np.ndarray:
    model = net.model
    e = np.asarray(model.E, dtype=np.complex128).reshape(-1)
    e0 = np.asarray(model.E0, dtype=np.complex128).reshape(-1)
    injection = np.zeros(model.y_size, dtype=np.complex128)
    for table_name, sign in (("asymmetric_load", 1.0), ("asymmetric_sgen", -1.0)):
        table = net[table_name]
        if len(table) == 0:
            continue
        buses = table["bus"].astype(int).to_numpy()
        from_phases = table["from_phase"].astype(int).to_numpy()
        to_phases = table["to_phase"].astype(int).to_numpy()
        y_from = net.model.terminal_to_y_lookup[buses * 4 + from_phases]
        y_to = net.model.terminal_to_y_lookup[buses * 4 + to_phases]
        connected = (y_from != -1) & (y_to != -1)
        if not np.any(connected):
            continue
        y_from = y_from[connected]
        y_to = y_to[connected]
        rows = table.iloc[np.flatnonzero(connected)]
        e_shunt_0 = e0[y_from] - e0[y_to]
        e_shunt = e[y_from] - e[y_to]
        e_shunt = np.where(e_shunt == 0, 1 + 0j, e_shunt)
        abs_e0 = np.where(np.abs(e_shunt_0) == 0, 1.0, np.abs(e_shunt_0))
        abs_e = np.abs(e_shunt)
        s = sign * (rows["p_mw"].to_numpy() + 1j * rows["q_mvar"].to_numpy()) * 1e6 * rows["in_service"].to_numpy()
        k_ip = rows["const_i_percent_p"].to_numpy() / 100.0
        k_zp = rows["const_z_percent_p"].to_numpy() / 100.0
        k_pp = 1.0 - (k_ip + k_zp)
        k_iq = rows["const_i_percent_q"].to_numpy() / 100.0
        k_zq = rows["const_z_percent_q"].to_numpy() / 100.0
        k_pq = 1.0 - (k_iq + k_zq)
        s_const_power = k_pp * np.real(s) + 1j * k_pq * np.imag(s)
        s_const_current = k_ip * np.real(s) + 1j * k_iq * np.imag(s)
        s_const_impedance = k_zp * np.real(s) + 1j * k_zq * np.imag(s)
        s_actual = (
            s_const_power
            + s_const_current * abs_e / abs_e0
            + s_const_impedance * abs_e**2 / abs_e0**2
        )
        i_branch = -np.conj(s_actual / (net.sn_mva * 1e6) / e_shunt)
        np.add.at(injection, y_from, i_branch)
        np.add.at(injection, y_to, -i_branch)
    return injection


def _linearized_injection_terms(net) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = net.model
    e = np.asarray(model.E, dtype=np.complex128).reshape(-1)
    e0 = np.asarray(model.E0, dtype=np.complex128).reshape(-1)
    direct = np.zeros((model.y_size, model.y_size), dtype=np.complex128)
    conjugate = np.zeros((model.y_size, model.y_size), dtype=np.complex128)
    offset = np.zeros(model.y_size, dtype=np.complex128)
    sbase = float(net.sn_mva) * 1e6

    for table_name, sign in (("asymmetric_load", 1.0), ("asymmetric_sgen", -1.0)):
        table = net[table_name]
        if len(table) == 0:
            continue
        buses = table["bus"].astype(int).to_numpy()
        from_phases = table["from_phase"].astype(int).to_numpy()
        to_phases = table["to_phase"].astype(int).to_numpy()
        y_from = model.terminal_to_y_lookup[buses * 4 + from_phases]
        y_to = model.terminal_to_y_lookup[buses * 4 + to_phases]
        connected = (y_from != -1) & (y_to != -1)
        if not np.any(connected):
            continue
        y_from = y_from[connected]
        y_to = y_to[connected]
        rows = table.iloc[np.flatnonzero(connected)]
        branch_voltage_ref = e0[y_from] - e0[y_to]
        branch_voltage = e[y_from] - e[y_to]
        safe_branch_voltage = np.where(
            np.abs(branch_voltage) > 1e-12,
            branch_voltage,
            1.0 + 0.0j,
        )
        abs_branch_ref = np.where(np.abs(branch_voltage_ref) > 1e-12, np.abs(branch_voltage_ref), 1.0)
        abs_branch = np.abs(safe_branch_voltage)
        s = sign * (
            rows["p_mw"].to_numpy() + 1j * rows["q_mvar"].to_numpy()
        ) * 1e6 * rows["in_service"].to_numpy()
        k_ip = rows["const_i_percent_p"].to_numpy() / 100.0
        k_zp = rows["const_z_percent_p"].to_numpy() / 100.0
        k_pp = 1.0 - (k_ip + k_zp)
        k_iq = rows["const_i_percent_q"].to_numpy() / 100.0
        k_zq = rows["const_z_percent_q"].to_numpy() / 100.0
        k_pq = 1.0 - (k_iq + k_zq)
        s_const_power = k_pp * np.real(s) + 1j * k_pq * np.imag(s)
        s_const_current = k_ip * np.real(s) + 1j * k_iq * np.imag(s)
        s_const_impedance = k_zp * np.real(s) + 1j * k_zq * np.imag(s)

        i_const_power = -np.conj(s_const_power / sbase / safe_branch_voltage)
        i_const_current = -np.conj(
            (s_const_current * abs_branch / abs_branch_ref) / sbase / safe_branch_voltage
        )
        i_const_impedance = -np.conj(
            (s_const_impedance * abs_branch**2 / abs_branch_ref**2) / sbase / safe_branch_voltage
        )
        branch_current = i_const_power + i_const_current + i_const_impedance

        direct_branch = (
            i_const_current / (2.0 * safe_branch_voltage)
            + i_const_impedance / safe_branch_voltage
        )
        conjugate_branch = -(
            i_const_power + 0.5 * i_const_current
        ) / np.conj(safe_branch_voltage)
        constant_branch = (
            branch_current
            - direct_branch * safe_branch_voltage
            - conjugate_branch * np.conj(safe_branch_voltage)
        )

        for y_from_idx, y_to_idx, direct_coeff, conjugate_coeff, offset_coeff in zip(
            y_from,
            y_to,
            direct_branch,
            conjugate_branch,
            constant_branch,
            strict=True,
        ):
            for matrix, coefficient in ((direct, direct_coeff), (conjugate, conjugate_coeff)):
                matrix[y_from_idx, y_from_idx] += coefficient
                matrix[y_from_idx, y_to_idx] -= coefficient
                matrix[y_to_idx, y_from_idx] -= coefficient
                matrix[y_to_idx, y_to_idx] += coefficient
            offset[y_from_idx] += offset_coeff
            offset[y_to_idx] -= offset_coeff
    return direct, conjugate, offset


def _build_reduced_net(
    *,
    net,
    eq_type: str,
    boundary_terminals: Sequence[TerminalRef],
    y_eq: np.ndarray,
    i_offset: np.ndarray,
    boundary_voltage: np.ndarray,
    validation: ValidationMetrics,
    provenance: Mapping[str, object],
):
    reduced = create_empty_network(
        name=f"{getattr(net, 'name', '')}_equivalent".strip("_"),
        sn_mva=net.sn_mva,
        rho_ohmm=getattr(net, "rho_ohmm", 100),
        f_hz=net.f_hz,
        add_stdtypes=False,
    )
    for element in ("configuration", "matrix", "sequence", "trafo", "trafo3w"):
        copy_std_types(reduced, net, element=element)

    boundary_bus_ids = sorted({terminal.bus for terminal in boundary_terminals})
    reduced.bus = net.bus.loc[net.bus.index.get_level_values(0).isin(boundary_bus_ids)].copy()
    if hasattr(net, "bus_geodata"):
        reduced.bus_geodata = net.bus_geodata.loc[net.bus_geodata.index.intersection(boundary_bus_ids)].copy()

    reduced.grid_equivalent = pd.DataFrame(
        [
            {
                "name": f"{getattr(net, 'name', '')}_equivalent".strip("_") or None,
                "eq_type": eq_type,
                "representation": eq_type,
                "boundary_terminals": [
                    {"bus": terminal.bus, "phase": terminal.phase, "side": terminal.side}
                    for terminal in boundary_terminals
                ],
                "y_eq": _serialize_complex_array(y_eq),
                "i_offset": _serialize_complex_array(i_offset),
                "base_voltage": _serialize_complex_array(boundary_voltage),
                "validation": {
                    "current_mismatch_norm": validation.current_mismatch_norm,
                    "current_mismatch_max": validation.current_mismatch_max,
                    "reduced_rank": validation.reduced_rank,
                    "eliminated_free_nodes": validation.eliminated_free_nodes,
                    "fixed_internal_nodes": validation.fixed_internal_nodes,
                    "is_finite": validation.is_finite,
                    "is_singular": validation.is_singular,
                    "passed": validation.passed,
                    "tolerances": {
                        "current_mismatch_max": GRID_EQUIVALENT_MISMATCH_MAX_TOL,
                        "current_mismatch_norm": GRID_EQUIVALENT_MISMATCH_NORM_TOL,
                    },
                },
                "provenance": dict(provenance),
            }
        ]
    )
    return reduced


def _build_boundary_results_table(
    net,
    model: GridEquivalentModel,
    voltage: np.ndarray,
    current: np.ndarray,
) -> pd.DataFrame:
    voltage = np.asarray(voltage, dtype=np.complex128).reshape(-1)
    current = np.asarray(current, dtype=np.complex128).reshape(-1)
    s = voltage * np.conj(current) * float(net.sn_mva)

    rows = []
    index = []
    for terminal, v, i, power in zip(model.boundary_terminals, voltage, current, s):
        vn_kv = float(net.bus.at[(terminal.bus, terminal.phase), "vn_kv"])
        i_base_a = float(net.sn_mva) * 1e6 / (vn_kv * 1e3 / np.sqrt(3))
        rows.append(
            {
                "vm_pu": float(np.abs(v)),
                "va_degree": float(np.rad2deg(np.angle(v))),
                "vr_pu": float(np.real(v)),
                "vi_pu": float(np.imag(v)),
                "i_real_pu": float(np.real(i)),
                "i_imag_pu": float(np.imag(i)),
                "i_abs_pu": float(np.abs(i)),
                "i_ka": float(np.abs(i) * i_base_a / 1000.0),
                "p_mw": float(np.real(power)),
                "q_mvar": float(np.imag(power)),
            }
        )
        index.append(terminal.as_tuple())
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["index", "phase"]),
    )


def _validate_partitions(
    *,
    boundary_bus_ids: set[int],
    internal_bus_filter: set[int] | None,
    external_bus_filter: set[int] | None,
) -> None:
    if internal_bus_filter is not None and external_bus_filter is not None:
        overlap = sorted(internal_bus_filter.intersection(external_bus_filter))
        if overlap:
            raise InvalidStudyInput(
                f"internal_buses and external_buses overlap: {overlap}"
            )
    if internal_bus_filter is not None:
        overlap = sorted(boundary_bus_ids.intersection(internal_bus_filter))
        if overlap:
            raise InvalidStudyInput(
                f"boundary_buses cannot also be internal_buses: {overlap}"
            )
    if external_bus_filter is not None:
        overlap = sorted(boundary_bus_ids.intersection(external_bus_filter))
        if overlap:
            raise InvalidStudyInput(
                f"boundary_buses cannot also be external_buses: {overlap}"
            )


def _arrays_are_finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(np.asarray(array).real).all() and np.isfinite(np.asarray(array).imag).all() for array in arrays)


def _solve_widely_linear_voltage(matrix: np.ndarray, current: np.ndarray) -> np.ndarray:
    direct = np.asarray(matrix, dtype=np.complex128)
    conjugate = _conjugate_matrix(matrix)
    target = np.asarray(current, dtype=np.complex128).reshape(-1)
    augmented_matrix = np.block(
        [
            [direct, conjugate],
            [np.conj(conjugate), np.conj(direct)],
        ]
    )
    augmented_target = np.concatenate([target, np.conj(target)])
    solution, *_ = np.linalg.lstsq(augmented_matrix, augmented_target, rcond=None)
    return solution[: direct.shape[1]]


def _finalize_validation(validation: ValidationMetrics) -> ValidationMetrics:
    passed = (
        validation.is_finite
        and not validation.is_singular
        and validation.current_mismatch_max <= GRID_EQUIVALENT_MISMATCH_MAX_TOL
        and validation.current_mismatch_norm <= GRID_EQUIVALENT_MISMATCH_NORM_TOL
    )
    return ValidationMetrics(
        current_mismatch_norm=validation.current_mismatch_norm,
        current_mismatch_max=validation.current_mismatch_max,
        reduced_rank=validation.reduced_rank,
        eliminated_free_nodes=validation.eliminated_free_nodes,
        fixed_internal_nodes=validation.fixed_internal_nodes,
        is_finite=validation.is_finite,
        is_singular=validation.is_singular,
        passed=passed,
    )


def _raise_on_invalid_equivalent(validation: ValidationMetrics) -> None:
    if validation.passed:
        return
    reasons: list[str] = []
    if not validation.is_finite:
        reasons.append("non-finite entries")
    if validation.is_singular:
        reasons.append("rank-deficient reduced admittance")
    if validation.current_mismatch_max > GRID_EQUIVALENT_MISMATCH_MAX_TOL:
        reasons.append(
            f"base-point mismatch max {validation.current_mismatch_max:.3e} exceeds {GRID_EQUIVALENT_MISMATCH_MAX_TOL:.3e}"
        )
    if validation.current_mismatch_norm > GRID_EQUIVALENT_MISMATCH_NORM_TOL:
        reasons.append(
            f"base-point mismatch norm {validation.current_mismatch_norm:.3e} exceeds {GRID_EQUIVALENT_MISMATCH_NORM_TOL:.3e}"
        )
    raise InvalidStudyInput(f"Equivalent validation failed: {', '.join(reasons)}")


def _serialize_complex_array(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.complex128)
    return {
        "real": np.asarray(array.real, dtype=float).tolist(),
        "imag": np.asarray(array.imag, dtype=float).tolist(),
        "shape": list(array.shape),
    }


def _deserialize_complex_array(payload, *, expected_ndim: int) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Complex payload must be a mapping, got {type(payload)!r}")
    real = np.asarray(payload.get("real", []), dtype=float)
    imag = np.asarray(payload.get("imag", []), dtype=float)
    array = real + 1j * imag
    shape = tuple(int(value) for value in payload.get("shape", array.shape))
    if shape:
        array = array.reshape(shape)
    if array.ndim != expected_ndim:
        raise InvalidStudyInput(
            f"Complex payload has ndim={array.ndim}, expected {expected_ndim}"
        )
    return np.asarray(array, dtype=np.complex128)


def _deserialize_boundary_terminals(payload) -> list[TerminalRef]:
    terminals: list[TerminalRef] = []
    for item in payload:
        if isinstance(item, Mapping):
            terminals.append(
                TerminalRef(
                    int(item["bus"]),
                    int(item["phase"]),
                    None if item.get("side") is None else str(item["side"]),
                )
            )
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            terminals.append(TerminalRef(int(item[0]), int(item[1])))
            continue
        raise TypeError(f"Unsupported boundary terminal payload {item!r}")
    return terminals


def _build_provenance(
    *,
    source_net,
    eq_type: str,
    boundary_terminals: Sequence[TerminalRef],
    validate: bool,
) -> dict[str, object]:
    return {
        "source_name": getattr(source_net, "name", "") or None,
        "network_hash": f"sha256:{_network_hash(source_net)}",
        "compiled_representation": "multiport",
        "old_to_new_bus_map": {terminal.bus: terminal.bus for terminal in boundary_terminals},
        "settings": {
            "eq_type": eq_type,
            "validate": bool(validate),
            "mismatch_max_tolerance": GRID_EQUIVALENT_MISMATCH_MAX_TOL,
            "mismatch_norm_tolerance": GRID_EQUIVALENT_MISMATCH_NORM_TOL,
        },
    }


def _network_hash(net) -> str:
    try:
        return hashlib.sha256(pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    except Exception:
        tables = {}
        for name in ("bus", "line", "switch", "ext_grid", "asymmetric_load", "asymmetric_sgen"):
            table = getattr(net, name, None)
            if table is None:
                continue
            try:
                tables[name] = table.to_json(orient="split", default_handler=str)
            except Exception:
                tables[name] = repr(table)
        payload = json.dumps(
            {
                "name": getattr(net, "name", ""),
                "sn_mva": getattr(net, "sn_mva", None),
                "tables": tables,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
