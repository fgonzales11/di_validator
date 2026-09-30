"""Phase-aware rectangular AC optimal power flow for multiconductor networks.

The public OPF surface now solves the native reduced multiconductor model
directly in rectangular voltage coordinates.  The optimizer enforces sparse
terminal KCL rows together with network and device limits; it no longer wraps a
general-purpose outer optimizer around repeated native power-flow solves.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.sparse import csr_matrix, diags, hstack, lil_matrix, vstack

from multiconductor.pycci.cci_powerflow import (
    LoadflowNotConverged,
    _init_pf,
    _raise_for_unsupplied_injections,
    correct_trafo_vnkv,
    run_pf,
)
from multiconductor.pycci.model import (
    _initialize_model,
    get_active_ext_grid_table,
    get_bus_terminal,
    get_ext_grid_terminal,
)
from multiconductor.pycci.pf_results import _generator_results_pf
from multiconductor.pycci.state_estimator import (
    _rectangular_current_block,
    _rectangular_power_block,
    extract_se_matrices,
)
from multiconductor.studies import InvalidStudyInput, StudyNotConverged


SUPPORTED_CONTROLLABLE_TABLES = (
    "asymmetric_sgen",
    "asymmetric_gen",
    "asymmetric_load",
)
SUPPORTED_COST_TABLES = SUPPORTED_CONTROLLABLE_TABLES + (
    "ext_grid",
    "ext_grid_sequence",
)
_VOLTAGE_EPS = 1e-12
_RESIDUAL_TOL = 1e-6
_DISPATCH_REGULARIZATION = 1e-9


class OPFNotConverged(StudyNotConverged):
    """Raised when no feasible AC optimal-power-flow solution is found."""


@dataclass(frozen=True, slots=True)
class OPFResult:
    """Immutable summary returned by :func:`run_opf`."""

    converged: bool
    objective: float
    iterations: int
    message: str
    dispatch: dict[str, pd.DataFrame]
    constraints: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _Variable:
    table: str
    index: Any
    column: str
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class _DispatchTerm:
    table: str
    index: Any
    y_from: int
    y_to: int
    abs_v0: float
    p_fixed: float
    q_fixed: float
    p_var: int | None
    q_var: int | None
    a_p: complex
    a_q: complex
    b_p: complex
    b_q: complex
    c_p: complex
    c_q: complex


@dataclass(frozen=True, slots=True)
class _PowerSelector:
    name: str
    table: str
    index: Any
    kind: str
    circuit: int | None
    power_type: str
    table_rows: tuple[Any, ...] = ()
    source_rows: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _PwlSpec:
    name: str
    selector: _PowerSelector
    column: int
    powers: tuple[float, ...]
    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _SourceInfo:
    active_kind: str
    operator: csr_matrix
    terminals: np.ndarray
    signs: np.ndarray
    row_index: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _BusVoltageLimit:
    name: str
    bus_index: Any
    y_index: int
    lower: float | None
    upper: float | None


@dataclass(frozen=True, slots=True)
class _LineCurrentLimit:
    name: str
    row: int
    side: str
    limit_ka: float
    limit_percent: float


@dataclass(frozen=True, slots=True)
class _TrafoPowerLimit:
    name: str
    row_index: Any
    apparent_limit_percent: float
    rated_power_va: float
    operator_rows: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CapabilityLimit:
    name: str
    table: str
    index: Any
    limit: float
    p_var: int | None
    q_var: int | None
    p_fixed: float
    q_fixed: float


def _ensure_cost_tables(net: Any) -> None:
    if "poly_cost" not in net or not isinstance(net.poly_cost, pd.DataFrame):
        net["poly_cost"] = pd.DataFrame(
            columns=[
                "element",
                "et",
                "phase",
                "circuit",
                "cp0_eur",
                "cp1_eur_per_mw",
                "cp2_eur_per_mw2",
                "cq0_eur",
                "cq1_eur_per_mvar",
                "cq2_eur_per_mvar2",
            ]
        )
    for column in ("phase", "circuit"):
        if column not in net.poly_cost:
            net.poly_cost[column] = np.nan

    if "pwl_cost" not in net or not isinstance(net.pwl_cost, pd.DataFrame):
        net["pwl_cost"] = pd.DataFrame(
            columns=["power_type", "element", "et", "phase", "circuit", "points"]
        )
    for column in ("phase", "circuit"):
        if column not in net.pwl_cost:
            net.pwl_cost[column] = np.nan


def _next_index(frame: pd.DataFrame, index: int | None) -> int:
    if index is not None:
        if index in frame.index:
            raise InvalidStudyInput(f"cost index {index} already exists")
        return int(index)
    if len(frame.index) == 0:
        return 0
    try:
        return int(max(frame.index)) + 1
    except (TypeError, ValueError) as exc:
        raise InvalidStudyInput("cost table must use integer row indexes") from exc


def _validate_cost_target(net: Any, element: int, et: str) -> None:
    if et not in SUPPORTED_COST_TABLES:
        raise InvalidStudyInput(
            f"unsupported cost element type {et!r}; expected one of {SUPPORTED_COST_TABLES}"
        )
    if et not in net or not isinstance(net[et], pd.DataFrame):
        raise InvalidStudyInput(f"network does not contain table {et!r}")
    table = net[et]
    if isinstance(table.index, pd.MultiIndex):
        exists = element in table.index.get_level_values(0)
    else:
        exists = element in table.index
    if not exists:
        raise InvalidStudyInput(f"{et} element {element} does not exist")


def create_poly_cost(
    net: Any,
    element: int,
    et: str,
    *,
    cp0_eur: float = 0.0,
    cp1_eur_per_mw: float = 0.0,
    cp2_eur_per_mw2: float = 0.0,
    cq0_eur: float = 0.0,
    cq1_eur_per_mvar: float = 0.0,
    cq2_eur_per_mvar2: float = 0.0,
    phase: int | None = None,
    circuit: int | None = None,
    index: int | None = None,
) -> int:
    """Create a polynomial active/reactive-power cost row."""

    _ensure_cost_tables(net)
    _validate_cost_target(net, element, et)
    cost_index = _next_index(net.poly_cost, index)
    values = {
        "element": int(element),
        "et": et,
        "phase": np.nan if phase is None else int(phase),
        "circuit": np.nan if circuit is None else int(circuit),
        "cp0_eur": float(cp0_eur),
        "cp1_eur_per_mw": float(cp1_eur_per_mw),
        "cp2_eur_per_mw2": float(cp2_eur_per_mw2),
        "cq0_eur": float(cq0_eur),
        "cq1_eur_per_mvar": float(cq1_eur_per_mvar),
        "cq2_eur_per_mvar2": float(cq2_eur_per_mvar2),
    }
    net.poly_cost.loc[cost_index, list(values)] = list(values.values())
    return cost_index


def _normalise_points(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(points, Sequence) or len(points) == 0:
        raise InvalidStudyInput("piecewise-linear cost requires at least one segment")
    normalised: list[tuple[float, ...]] = []
    width: int | None = None
    for point in points:
        if not isinstance(point, Sequence) or len(point) not in (2, 3):
            raise InvalidStudyInput("PWL points must be (power, cost) or (lower, upper, slope)")
        values = tuple(float(value) for value in point)
        if not all(np.isfinite(values)):
            raise InvalidStudyInput("PWL points must contain only finite values")
        width = len(values) if width is None else width
        if len(values) != width:
            raise InvalidStudyInput("PWL point formats may not be mixed")
        if len(values) == 3 and values[1] <= values[0]:
            raise InvalidStudyInput("PWL segment upper bound must exceed its lower bound")
        normalised.append(values)
    if width == 2:
        powers = [point[0] for point in normalised]
        if any(right <= left for left, right in zip(powers, powers[1:])):
            raise InvalidStudyInput("PWL knot powers must be strictly increasing")
    else:
        normalised.sort(key=lambda item: item[0])
    return tuple(normalised)


def create_pwl_cost(
    net: Any,
    element: int,
    et: str,
    points: Sequence[Sequence[float]],
    *,
    power_type: str = "p",
    phase: int | None = None,
    circuit: int | None = None,
    index: int | None = None,
) -> int:
    """Create a phase-aware piecewise-linear cost row."""

    _ensure_cost_tables(net)
    _validate_cost_target(net, element, et)
    if power_type not in {"p", "q"}:
        raise InvalidStudyInput("power_type must be 'p' or 'q'")
    cost_index = _next_index(net.pwl_cost, index)
    values = {
        "power_type": power_type,
        "element": int(element),
        "et": et,
        "phase": np.nan if phase is None else int(phase),
        "circuit": np.nan if circuit is None else int(circuit),
        "points": _normalise_points(points),
    }
    for column, value in values.items():
        net.pwl_cost.at[cost_index, column] = value
    return cost_index


def _finite_bound(row: pd.Series, column: str) -> float | None:
    value = row.get(column, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _optional_int(value: Any) -> int | None:
    try:
        return None if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def _cost_coefficient(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _row_group_selector(table: pd.DataFrame, element: int, circuit: int | None) -> list[Any]:
    if isinstance(table.index, pd.MultiIndex):
        mask = table.index.get_level_values(0) == element
        selected = table.index[mask]
        if circuit is not None:
            selected = [idx for idx in selected if int(idx[1]) == circuit]
        else:
            selected = list(selected)
        return list(selected)
    if element in table.index:
        return [element]
    return []


def _stage_asymmetric_generators(work: Any) -> dict[Any, Any]:
    """Represent native PV rows as PQ injections during dispatch optimization."""

    mapping: dict[Any, Any] = {}
    if "asymmetric_gen" not in work or len(work.asymmetric_gen) == 0:
        return mapping
    if "q_mvar" not in work.asymmetric_gen:
        work.asymmetric_gen["q_mvar"] = 0.0

    sgen = work.asymmetric_sgen
    first_level = sgen.index.get_level_values(0) if len(sgen) else pd.Index([], dtype=int)
    next_element = int(max(first_level)) + 1 if len(first_level) else 0
    rows: list[dict[str, Any]] = []
    indexes: list[tuple[int, int]] = []
    for offset, (gen_index, row) in enumerate(work.asymmetric_gen.iterrows()):
        staged_index = (next_element + offset, 0)
        mapping[gen_index] = staged_index
        values = {column: np.nan for column in sgen.columns}
        values.update(
            {
                "name": f"__opf_gen__{gen_index}",
                "bus": row.bus,
                "from_phase": row.from_phase,
                "to_phase": row.to_phase,
                "p_mw": row.p_mw,
                "q_mvar": row.get("q_mvar", 0.0),
                "vm_pu": np.nan,
                "const_z_percent_p": 0.0,
                "const_i_percent_p": 0.0,
                "const_z_percent_q": 0.0,
                "const_i_percent_q": 0.0,
                "sn_mva": row.get("sn_mva", np.nan),
                "scaling": row.get("scaling", 1.0),
                "in_service": row.get("in_service", True),
                "type": row.get("type", None),
                "slack": False,
            }
        )
        rows.append(values)
        indexes.append(staged_index)
    if rows:
        staged = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(indexes, names=sgen.index.names),
            columns=sgen.columns,
        )
        work.asymmetric_sgen = pd.concat([sgen, staged])
    for gen_index in mapping:
        work.asymmetric_gen.at[gen_index, "in_service"] = False
    return mapping


def _build_variables(net: Any) -> tuple[list[_Variable], np.ndarray]:
    variables: list[_Variable] = []
    initial: list[float] = []
    for table_name in SUPPORTED_CONTROLLABLE_TABLES:
        if table_name not in net:
            continue
        table = net[table_name]
        for index, row in table.iterrows():
            if not bool(row.get("in_service", True)) or not bool(row.get("controllable", False)):
                continue
            for column, lower_column, upper_column in (
                ("p_mw", "min_p_mw", "max_p_mw"),
                ("q_mvar", "min_q_mvar", "max_q_mvar"),
            ):
                if column not in table and not (table_name == "asymmetric_gen" and column == "q_mvar"):
                    continue
                lower = _finite_bound(row, lower_column)
                upper = _finite_bound(row, upper_column)
                if lower is None or upper is None:
                    raise InvalidStudyInput(
                        f"controllable {table_name} {index!r} requires finite "
                        f"{lower_column}/{upper_column}"
                    )
                if upper < lower:
                    raise InvalidStudyInput(
                        f"{table_name} {index!r} has {upper_column} below {lower_column}"
                    )
                current = float(row.get(column, 0.0))
                if upper == lower:
                    continue
                variables.append(_Variable(table_name, index, column, lower, upper))
                initial.append(float(np.clip(current, lower, upper)))
    return variables, np.asarray(initial, dtype=float)


def _apply_dispatch(
    work: Any,
    variables: Sequence[_Variable],
    x_dispatch: np.ndarray,
    gen_mapping: dict[Any, Any],
) -> None:
    for variable, value in zip(variables, x_dispatch):
        work[variable.table].at[variable.index, variable.column] = float(value)
        if variable.table == "asymmetric_gen":
            staged_index = gen_mapping[variable.index]
            work.asymmetric_sgen.at[staged_index, variable.column] = float(value)


def _prepare_native_model(work: Any) -> None:
    correct_trafo_vnkv(work)
    _initialize_model(work)
    _init_pf(work)
    _raise_for_unsupplied_injections(work, allow_islands=False)


def _power_to_segments(
    points: Iterable[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    values = tuple(tuple(float(v) for v in point) for point in points)
    if not values:
        raise InvalidStudyInput("piecewise-linear cost requires at least one segment")
    if len(values[0]) == 2:
        xp = tuple(point[0] for point in values)
        yp = tuple(point[1] for point in values)
        if len(xp) == 1:
            return (xp[0],), (0.0,), (yp[0],)
        slopes: list[float] = []
        intercepts: list[float] = []
        for left, right, cost_left, cost_right in zip(xp[:-1], xp[1:], yp[:-1], yp[1:]):
            slope = (cost_right - cost_left) / (right - left)
            slopes.append(float(slope))
            intercepts.append(float(cost_left - slope * left))
        slopes.insert(0, slopes[0])
        intercepts.insert(0, float(yp[0] - slopes[0] * xp[0]))
        slopes.append(slopes[-1])
        intercepts.append(float(yp[-1] - slopes[-1] * xp[-1]))
        return xp, tuple(slopes), tuple(intercepts)

    segments = sorted(values, key=lambda item: item[0])
    knots = [segments[0][0]]
    slopes: list[float] = []
    intercepts: list[float] = []
    cumulative = 0.0
    last_upper = segments[0][0]
    for lower, upper, slope in segments:
        if lower > last_upper:
            knots.append(lower)
        slopes.append(float(slope))
        intercepts.append(float(cumulative - slope * lower))
        cumulative += slope * (upper - lower)
        last_upper = upper
        knots.append(upper)
    if slopes:
        slopes.insert(0, slopes[0])
        intercepts.insert(0, intercepts[0])
        slopes.append(slopes[-1])
        intercepts.append(float(cumulative - slopes[-1] * last_upper))
    return tuple(knots), tuple(slopes), tuple(intercepts)


def _objective_from_segments(
    powers: Sequence[float],
    slopes: Sequence[float],
    intercepts: Sequence[float],
    value: float,
) -> float:
    best = -np.inf
    segment_count = min(len(slopes), len(intercepts))
    if segment_count == 0:
        return 0.0
    for slope, intercept in zip(slopes[:segment_count], intercepts[:segment_count]):
        best = max(best, slope * value + intercept)
    return float(best)


def _fixed_dispatch_value(work: Any, table: str, index: Any, column: str) -> float:
    row = work[table].loc[index]
    value = row.get(column, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    return value if np.isfinite(value) else 0.0


class _RectangularOpfModel:
    def __init__(
        self,
        work: Any,
        *,
        variables: Sequence[_Variable],
        x_dispatch0: np.ndarray,
        gen_mapping: dict[Any, Any],
    ) -> None:
        self.work = work
        self.variables = list(variables)
        self.x_dispatch0 = np.asarray(x_dispatch0, dtype=float)
        self.gen_mapping = gen_mapping
        self.model = work.model
        self.se_model = extract_se_matrices(work, rebuild=False)
        self.variable_nodes = np.asarray(self.model.y_nonslack, dtype=int)
        self.variable_lookup = {int(node): pos for pos, node in enumerate(self.variable_nodes)}
        self.state_count = 2 * len(self.variable_nodes)
        self.dispatch_offset = self.state_count
        self.dispatch_lookup = {
            (variable.table, variable.index, variable.column): self.dispatch_offset + pos
            for pos, variable in enumerate(self.variables)
        }
        self._staged_reverse = {value: key for key, value in gen_mapping.items()}
        self._x: np.ndarray | None = None
        self._cache: dict[str, Any] = {}

        Yll = csr_matrix(self.model.Y_tot[self.variable_nodes, :][:, self.variable_nodes])
        self._linear_kcl_jac = hstack(
            [
                csr_matrix(Yll.real),
                csr_matrix(-Yll.imag),
            ],
            format="csr",
        )
        lower = hstack(
            [
                csr_matrix(Yll.imag),
                csr_matrix(Yll.real),
            ],
            format="csr",
        )
        self._linear_kcl_jac = vstack([self._linear_kcl_jac, lower], format="csr")
        empty_matrix = csr_matrix((int(self.model.y_size), int(self.model.y_size)), dtype=complex)
        self._passive_y = csr_matrix(
            getattr(self.model, "Y_tran", empty_matrix)
            + getattr(self.model, "Y_center_tap_trafo", empty_matrix)
            + getattr(self.model, "Y_network", empty_matrix)
            + getattr(self.model, "Y_ground", empty_matrix)
            + getattr(self.model, "Y_shunt", empty_matrix)
            + getattr(self.model, "Y_switch", empty_matrix)
        )
        self.dispatch_terms = self._build_dispatch_terms()
        self.source_info = self._build_source_info()
        self.source_selectors = self._build_source_selectors()
        self.bus_limits = self._build_bus_limits()
        self.line_limits = self._build_line_limits()
        self.trafo_operator, self.trafo_terminals, self.trafo_limits = self._build_trafo_limits()
        self.capability_limits = self._build_capability_limits()
        self.pwl_specs = self._build_pwl_specs()
        self.epigraph_offset = self.dispatch_offset + len(self.variables)
        self.total_size = self.epigraph_offset + len(self.pwl_specs)
        if self.total_size > self.state_count:
            self._linear_kcl_jac = hstack(
                [
                    self._linear_kcl_jac,
                    csr_matrix((self.state_count, self.total_size - self.state_count), dtype=float),
                ],
                format="csr",
            )
        self.kcl_row_names = self._build_kcl_row_names()
        self.ineq_row_names = self._build_ineq_row_names()

    def _build_kcl_row_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for node in self.variable_nodes:
            names.append(f"terminal:{int(node)}:kcl_real")
        for node in self.variable_nodes:
            names.append(f"terminal:{int(node)}:kcl_imag")
        return tuple(names)

    def _build_dispatch_terms(self) -> tuple[_DispatchTerm, ...]:
        terms: list[_DispatchTerm] = []
        isolated = set(np.asarray(self.model.y_isolated, dtype=int).tolist())
        for table_name in ("asymmetric_load", "asymmetric_sgen"):
            table = self.work[table_name]
            if len(table) == 0:
                continue
            for index, row in table.iterrows():
                if not bool(row.get("in_service", True)):
                    continue
                bus = int(row.bus)
                from_phase = int(row.from_phase)
                to_phase = int(row.to_phase)
                y_from = int(self.model.terminal_to_y_lookup[bus * 4 + from_phase])
                y_to = int(self.model.terminal_to_y_lookup[bus * 4 + to_phase])
                if y_from < 0 or y_to < 0 or y_from in isolated or y_to in isolated:
                    continue
                scaling = float(row.get("scaling", 1.0) or 0.0)
                sign = 1.0 if table_name == "asymmetric_load" else -1.0
                k_ip_p = float(row.get("const_i_percent_p", 0.0) or 0.0) / 100.0
                k_iz_p = float(row.get("const_z_percent_p", 0.0) or 0.0) / 100.0
                k_pp = 1.0 - (k_ip_p + k_iz_p)
                k_ip_q = float(row.get("const_i_percent_q", 0.0) or 0.0) / 100.0
                k_iz_q = float(row.get("const_z_percent_q", 0.0) or 0.0) / 100.0
                k_pq = 1.0 - (k_ip_q + k_iz_q)
                v0 = np.asarray(self.model.E0, dtype=complex).reshape(-1)
                abs_v0 = abs(v0[y_from] - v0[y_to])
                if abs_v0 <= _VOLTAGE_EPS:
                    abs_v0 = 1.0

                key_p = (table_name, index, "p_mw")
                key_q = (table_name, index, "q_mvar")
                p_var = self.dispatch_lookup.get(key_p)
                q_var = self.dispatch_lookup.get(key_q)
                p_fixed = _fixed_dispatch_value(self.work, table_name, index, "p_mw")
                q_fixed = _fixed_dispatch_value(self.work, table_name, index, "q_mvar")

                mw_scale = sign * scaling * 1e6 / (self.work.sn_mva * 1e6)
                a_p = -np.conjugate(k_pp * mw_scale)
                a_q = -np.conjugate(1j * k_pq * mw_scale)
                b_p = -np.conjugate(k_ip_p * mw_scale) / abs_v0
                b_q = -np.conjugate(1j * k_ip_q * mw_scale) / abs_v0
                c_p = -np.conjugate(k_iz_p * mw_scale) / (abs_v0**2)
                c_q = -np.conjugate(1j * k_iz_q * mw_scale) / (abs_v0**2)
                terms.append(
                    _DispatchTerm(
                        table=table_name,
                        index=index,
                        y_from=y_from,
                        y_to=y_to,
                        abs_v0=abs_v0,
                        p_fixed=p_fixed,
                        q_fixed=q_fixed,
                        p_var=p_var,
                        q_var=q_var,
                        a_p=a_p,
                        a_q=a_q,
                        b_p=b_p,
                        b_q=b_q,
                        c_p=c_p,
                        c_q=c_q,
                    )
                )
        return tuple(terms)

    def _build_source_info(self) -> _SourceInfo:
        active_kind, active_source = get_active_ext_grid_table(self.work, require_active=False)
        if active_source is None or len(active_source) == 0:
            raise InvalidStudyInput("OPF requires at least one active ext_grid or ext_grid_sequence row")
        row_index = tuple(active_source.index.tolist())
        if self.model.Y_source.nnz > 0 and not bool(self.model.is_ideal_ext_grid):
            terminals = np.asarray(
                [
                    self.model.terminal_to_y_lookup[
                        get_ext_grid_terminal(self.model, int(index[0]), int(index[1]))
                    ]
                    for index in row_index
                ],
                dtype=int,
            )
            operator = csr_matrix(self.model.Y_source[terminals, :])
            signs = np.ones(len(terminals), dtype=float)
            return _SourceInfo(active_kind, operator, terminals, signs, row_index)

        terminals = np.asarray(
            [
                self.model.terminal_to_y_lookup[get_bus_terminal(self.model, int(row.bus), int(row.from_phase))]
                for _, row in active_source.iterrows()
            ],
            dtype=int,
        )
        operator = csr_matrix(self._passive_y[terminals, :])
        signs = -np.ones(len(terminals), dtype=float)
        return _SourceInfo(active_kind, operator, terminals, signs, row_index)

    def _build_source_selectors(self) -> dict[tuple[str, Any, str], _PowerSelector]:
        selectors: dict[tuple[str, Any, str], _PowerSelector] = {}
        if self.source_info.active_kind not in {"ext_grid", "ext_grid_sequence"}:
            return selectors
        table = self.work[self.source_info.active_kind]
        for index in table.index:
            row = int(self.source_info.row_index.index(index))
            for power_type in ("p", "q"):
                selectors[(self.source_info.active_kind, index, power_type)] = _PowerSelector(
                    name=f"{self.source_info.active_kind}:{index}:{power_type}",
                    table=self.source_info.active_kind,
                    index=index,
                    kind="source",
                    circuit=index[1] if isinstance(index, tuple) and len(index) > 1 else None,
                    power_type=power_type,
                    table_rows=(),
                    source_rows=(row,),
                )
        elements = table.index.get_level_values(0).unique() if isinstance(table.index, pd.MultiIndex) else table.index
        for element in elements:
            selected = [
                pos
                for pos, idx in enumerate(self.source_info.row_index)
                if (idx[0] if isinstance(idx, tuple) else idx) == element
            ]
            for power_type in ("p", "q"):
                selectors[(self.source_info.active_kind, int(element), power_type)] = _PowerSelector(
                    name=f"{self.source_info.active_kind}:{int(element)}:{power_type}",
                    table=self.source_info.active_kind,
                    index=int(element),
                    kind="source",
                    circuit=None,
                    power_type=power_type,
                    table_rows=(),
                    source_rows=tuple(selected),
                )
        return selectors

    def _build_bus_limits(self) -> tuple[_BusVoltageLimit, ...]:
        limits: list[_BusVoltageLimit] = []
        for index, row in self.work.bus.iterrows():
            lower = _finite_bound(row, "min_vm_pu")
            upper = _finite_bound(row, "max_vm_pu")
            if lower is None and upper is None:
                continue
            y_index = int(self.model.terminal_to_y_lookup[get_bus_terminal(self.model, int(index[0]), int(index[1]))])
            if y_index < 0:
                continue
            limits.append(
                _BusVoltageLimit(
                    name=f"bus:{index}:vm",
                    bus_index=index,
                    y_index=y_index,
                    lower=lower,
                    upper=upper,
                )
            )
        return tuple(limits)

    def _build_line_limits(self) -> tuple[_LineCurrentLimit, ...]:
        limits: list[_LineCurrentLimit] = []
        line_info = self.model.line_info
        for line_id, active_circuits in zip(line_info["indices"], line_info.get("active_circuits", [])):
            group = self.work.line.xs(line_id, level=0, drop_level=False)
            row0 = group.iloc[0]
            std_type = self.work.std_types[row0["model_type"]][row0["std_type"]]
            current_limits = np.asarray(std_type.get("max_i_ka", []), dtype=float).reshape(-1)
            if current_limits.size == 1 and len(active_circuits) > 1:
                current_limits = np.repeat(current_limits, max(active_circuits) + 1)
            for circuit_id in active_circuits:
                if circuit_id >= len(current_limits) or not np.isfinite(current_limits[circuit_id]) or current_limits[circuit_id] <= 0:
                    continue
                row_index = (int(line_id), int(circuit_id))
                table_row = group.loc[row_index] if row_index in group.index else row0
                max_loading_percent = _finite_bound(table_row, "max_loading_percent")
                if max_loading_percent is None:
                    max_loading_percent = 100.0
                if max_loading_percent <= 0:
                    continue
                row = self.se_model.line_rows.get((int(line_id), int(circuit_id)))
                if row is None:
                    continue
                limit_ka = float(current_limits[circuit_id]) * float(max_loading_percent) / 100.0
                for side in ("from", "to"):
                    limits.append(
                        _LineCurrentLimit(
                            name=f"line:{(int(line_id), int(circuit_id))}:{side}_ampacity_max",
                            row=int(row),
                            side=side,
                            limit_ka=limit_ka,
                            limit_percent=float(max_loading_percent),
                        )
                    )
        return tuple(limits)

    def _build_trafo_limits(
        self,
    ) -> tuple[csr_matrix | None, np.ndarray | None, tuple[_TrafoPowerLimit, ...]]:
        rows: list[dict[str, Any]] = []
        row_i: list[int] = []
        row_j: list[int] = []
        row_data: list[complex] = []
        terminals: list[int] = []
        row_cursor = 0
        for index, circuit in self.model.trafo_circuits.items():
            element_index = int(index[0]) if isinstance(index, tuple) else int(index)
            if element_index not in self.work.trafo1ph.index.get_level_values(0):
                continue
            row_table = self.work.trafo1ph.xs(element_index, level=0, drop_level=False)
            max_loading_percent = _finite_bound(row_table.iloc[0], "max_loading_percent")
            if max_loading_percent is None:
                max_loading_percent = 100.0
            if max_loading_percent <= 0:
                continue
            y_send_rec = np.asarray(circuit["y_send_rec"], dtype=int)
            Ybr = np.asarray(circuit["Ybr"], dtype=complex)
            for side_position, (bus, pn) in enumerate(zip(circuit["buses"], circuit["pns"])):
                if not np.isfinite(pn) or pn <= 0:
                    continue
                terminal_row = 2 * side_position
                operator_rows: list[int] = []
                for branch_row in (terminal_row, terminal_row + 1):
                    for column, node in enumerate(y_send_rec):
                        value = Ybr[branch_row, column]
                        if value != 0:
                            row_i.append(row_cursor)
                            row_j.append(int(node))
                            row_data.append(value)
                    terminals.append(int(y_send_rec[branch_row]))
                    operator_rows.append(row_cursor)
                    row_cursor += 1
                rows.append(
                    {
                        "name": f"trafo:{(element_index, int(bus), int(index[1]))}:loading_max",
                        "row_index": (element_index, int(bus), int(index[1])),
                        "limit_percent": float(max_loading_percent),
                        "rated_power_va": float(pn),
                        "operator_rows": tuple(operator_rows),
                    }
                )
        if not rows:
            return None, None, ()
        operator = csr_matrix((row_data, (row_i, row_j)), shape=(row_cursor, int(self.model.y_size)))
        limits = tuple(
            _TrafoPowerLimit(
                name=row["name"],
                row_index=row["row_index"],
                apparent_limit_percent=row["limit_percent"],
                rated_power_va=row["rated_power_va"],
                operator_rows=row["operator_rows"],
            )
            for row in rows
        )
        return operator, np.asarray(terminals, dtype=int), limits

    def _build_capability_limits(self) -> tuple[_CapabilityLimit, ...]:
        limits: list[_CapabilityLimit] = []
        for table_name in SUPPORTED_CONTROLLABLE_TABLES:
            table = self.work.get(table_name)
            if table is None:
                continue
            for index, row in table.iterrows():
                if not bool(row.get("in_service", True)):
                    continue
                rating = _finite_bound(row, "sn_mva")
                if rating is None or rating <= 0:
                    continue
                limits.append(
                    _CapabilityLimit(
                        name=f"{table_name}:{index}:capability",
                        table=table_name,
                        index=index,
                        limit=rating,
                        p_var=self.dispatch_lookup.get((table_name, index, "p_mw")),
                        q_var=self.dispatch_lookup.get((table_name, index, "q_mvar")),
                        p_fixed=_fixed_dispatch_value(self.work, table_name, index, "p_mw"),
                        q_fixed=_fixed_dispatch_value(self.work, table_name, index, "q_mvar"),
                    )
                )
        return tuple(limits)

    def _selector_for_cost_row(self, row: pd.Series, power_type: str) -> _PowerSelector:
        element = int(row.element)
        table = str(row.et)
        circuit = _optional_int(row.get("circuit", np.nan))
        if table in {"ext_grid", "ext_grid_sequence"}:
            key = (table, element if circuit is None else (element, circuit), power_type)
            selector = self.source_selectors.get(key)
            if selector is None:
                raise InvalidStudyInput(f"cost references missing {table} element {element}")
            return selector

        table_frame = self.work[table]
        rows = _row_group_selector(table_frame, element, circuit)
        if not rows:
            raise InvalidStudyInput(f"cost references missing {table} element {element}")
        return _PowerSelector(
            name=f"{table}:{element}:{power_type}",
            table=table,
            index=element,
            kind="dispatch",
            circuit=circuit,
            power_type=power_type,
            table_rows=tuple(rows),
        )

    def _build_pwl_specs(self) -> tuple[_PwlSpec, ...]:
        specs: list[_PwlSpec] = []
        for _, row in self.work.pwl_cost.iterrows():
            power_type = str(row.power_type).lower()
            selector = self._selector_for_cost_row(row, power_type)
            powers, slopes, intercepts = _power_to_segments(row.points)
            column = self.dispatch_offset + len(self.variables) + len(specs)
            specs.append(
                _PwlSpec(
                    name=f"pwl:{row.et}:{int(row.element)}:{power_type}:{len(specs)}",
                    selector=selector,
                    column=column,
                    powers=powers,
                    slopes=slopes,
                    intercepts=intercepts,
                )
            )
        return tuple(specs)

    def _build_ineq_row_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for limit in self.bus_limits:
            if limit.lower is not None:
                names.append(f"{limit.name}_min")
            if limit.upper is not None:
                names.append(f"{limit.name}_max")
        for limit in self.line_limits:
            names.append(limit.name)
        for limit in self.trafo_limits:
            names.append(limit.name)
        for limit in self.capability_limits:
            names.append(limit.name)
        active_source = self.work.get(self.source_info.active_kind)
        if active_source is not None:
            seen: set[tuple[str, Any, str]] = set()
            for index, row in active_source.iterrows():
                lower_p = _finite_bound(row, "min_p_mw")
                upper_p = _finite_bound(row, "max_p_mw")
                lower_q = _finite_bound(row, "min_q_mvar")
                upper_q = _finite_bound(row, "max_q_mvar")
                aggregate_keys = [
                    (self.source_info.active_kind, index, "p", lower_p, upper_p),
                    (self.source_info.active_kind, index, "q", lower_q, upper_q),
                ]
                for table_name, selector_index, power_type, lower, upper in aggregate_keys:
                    selector_key = (table_name, selector_index, power_type)
                    if selector_key in seen:
                        continue
                    seen.add(selector_key)
                    if lower is not None:
                        names.append(f"{table_name}:{selector_index}:{power_type}_min")
                    if upper is not None:
                        names.append(f"{table_name}:{selector_index}:{power_type}_max")
        for spec in self.pwl_specs:
            for segment in range(min(len(spec.slopes), len(spec.intercepts))):
                names.append(f"{spec.name}:segment:{segment}")
        return tuple(names)

    def _full_voltage(self, x: np.ndarray) -> np.ndarray:
        voltage = np.asarray(self.model.E0, dtype=complex).reshape(-1).copy()
        count = len(self.variable_nodes)
        voltage[self.variable_nodes] = x[:count] + 1j * x[count : 2 * count]
        return voltage

    def _dispatch_value(self, x: np.ndarray, table: str, index: Any, column: str) -> float:
        position = self.dispatch_lookup.get((table, index, column))
        if position is None:
            return _fixed_dispatch_value(self.work, table, index, column)
        return float(x[position])

    def _term_dispatch(self, term: _DispatchTerm, x: np.ndarray) -> tuple[float, float]:
        p = float(x[term.p_var]) if term.p_var is not None else term.p_fixed
        q = float(x[term.q_var]) if term.q_var is not None else term.q_fixed
        return p, q

    def _zip_current(
        self,
        term: _DispatchTerm,
        voltage: complex,
        p: float,
        q: float,
    ) -> tuple[complex, tuple[complex, complex], tuple[complex, complex]]:
        vr = float(voltage.real)
        vi = float(voltage.imag)
        radius_sq = max(vr * vr + vi * vi, _VOLTAGE_EPS**2)
        radius = sqrt(radius_sq)
        radius_cubed = radius_sq * radius

        coeff_a = term.a_p * p + term.a_q * q
        coeff_b = term.b_p * p + term.b_q * q
        coeff_c = term.c_p * p + term.c_q * q

        a_r = float(coeff_a.real)
        a_i = float(coeff_a.imag)
        b_r = float(coeff_b.real)
        b_i = float(coeff_b.imag)
        c_r = float(coeff_c.real)
        c_i = float(coeff_c.imag)

        current_power = coeff_a * voltage / radius_sq
        current_current = coeff_b * (voltage / radius)
        current_impedance = coeff_c * voltage
        current = current_power + current_current + current_impedance

        num_r = a_r * vr - a_i * vi
        num_i = a_r * vi + a_i * vr
        d_power_dvr = complex(
            (a_r * radius_sq - num_r * 2.0 * vr) / (radius_sq**2),
            (a_i * radius_sq - num_i * 2.0 * vr) / (radius_sq**2),
        )
        d_power_dvi = complex(
            (-a_i * radius_sq - num_r * 2.0 * vi) / (radius_sq**2),
            (a_r * radius_sq - num_i * 2.0 * vi) / (radius_sq**2),
        )

        du_dvr = vi * vi / radius_cubed
        du_dvi = -vr * vi / radius_cubed
        dv_dvr = -vr * vi / radius_cubed
        dv_dvi = vr * vr / radius_cubed
        d_current_dvr = complex(
            b_r * du_dvr - b_i * dv_dvr,
            b_r * dv_dvr + b_i * du_dvr,
        )
        d_current_dvi = complex(
            b_r * du_dvi - b_i * dv_dvi,
            b_r * dv_dvi + b_i * du_dvi,
        )

        d_impedance_dvr = complex(c_r, c_i)
        d_impedance_dvi = complex(-c_i, c_r)

        d_voltage = (
            d_power_dvr + d_current_dvr + d_impedance_dvr,
            d_power_dvi + d_current_dvi + d_impedance_dvi,
        )
        d_dispatch = (
            term.a_p / np.conjugate(voltage) + term.b_p * voltage / radius + term.c_p * voltage,
            term.a_q / np.conjugate(voltage) + term.b_q * voltage / radius + term.c_q * voltage,
        )
        return current, d_voltage, d_dispatch

    def _selector_value(self, x: np.ndarray, selector: _PowerSelector) -> tuple[float, csr_matrix]:
        voltage = self._cache["voltage"]
        if selector.kind == "source":
            key = "source_power"
            if key not in self._cache:
                p_jac, q_jac, power = _rectangular_power_block(
                    self.source_info.operator,
                    self.source_info.terminals,
                    voltage,
                    self.variable_nodes,
                )
                real = np.asarray(power.real).reshape(-1)
                imag = np.asarray(power.imag).reshape(-1)
                self._cache[key] = {
                    "p": p_jac.tocsr(),
                    "q": q_jac.tocsr(),
                    "real": real,
                    "imag": imag,
                }
            values = self._cache[key]
            rows = np.asarray(selector.source_rows, dtype=int)
            sign = self.source_info.signs[rows]
            amount = float(np.dot(sign, values["real" if selector.power_type == "p" else "imag"][rows]))
            jacobian = values["p" if selector.power_type == "p" else "q"][rows, :]
            if len(rows) > 1:
                jacobian = csr_matrix(sign.reshape(1, -1) @ jacobian)
            else:
                jacobian = csr_matrix(jacobian.multiply(sign[0]))
            if self.total_size > self.state_count:
                jacobian = hstack(
                    [
                        jacobian,
                        csr_matrix((1, self.total_size - self.state_count), dtype=float),
                    ],
                    format="csr",
                )
            return amount, jacobian

        amount = 0.0
        grad = np.zeros(self.total_size, dtype=float)
        for row_index in selector.table_rows:
            if selector.power_type == "p":
                amount += self._dispatch_value(x, selector.table, row_index, "p_mw")
                position = self.dispatch_lookup.get((selector.table, row_index, "p_mw"))
                if position is not None:
                    grad[position] += 1.0
            else:
                amount += self._dispatch_value(x, selector.table, row_index, "q_mvar")
                position = self.dispatch_lookup.get((selector.table, row_index, "q_mvar"))
                if position is not None:
                    grad[position] += 1.0
        return float(amount), csr_matrix(grad.reshape(1, -1))

    def _evaluate(self, x: np.ndarray) -> None:
        values = np.asarray(x, dtype=float).reshape(-1)
        if self._x is not None and np.array_equal(values, self._x):
            return
        self._x = values.copy()
        self._cache = {}
        voltage = self._full_voltage(values)
        self._cache["voltage"] = voltage

        node_current = np.zeros(int(self.model.y_size), dtype=complex)
        nonlinear = lil_matrix((self.state_count, self.total_size), dtype=float)
        for term in self.dispatch_terms:
            p, q = self._term_dispatch(term, values)
            local_voltage = voltage[term.y_from] - voltage[term.y_to]
            if abs(local_voltage) <= _VOLTAGE_EPS:
                local_voltage = _VOLTAGE_EPS + 0j
            current, (d_dvr, d_dvi), (d_dp, d_dq) = self._zip_current(term, local_voltage, p, q)
            node_current[term.y_from] += current
            node_current[term.y_to] -= current
            for terminal, sign in ((term.y_from, -1.0), (term.y_to, 1.0)):
                row = self.variable_lookup.get(int(terminal))
                if row is None:
                    continue
                for local_terminal, local_sign in ((term.y_from, 1.0), (term.y_to, -1.0)):
                    column = self.variable_lookup.get(int(local_terminal))
                    if column is None:
                        continue
                    coeff = sign * local_sign
                    nonlinear[row, column] += coeff * d_dvr.real
                    nonlinear[row, len(self.variable_nodes) + column] += coeff * d_dvi.real
                    nonlinear[len(self.variable_nodes) + row, column] += coeff * d_dvr.imag
                    nonlinear[len(self.variable_nodes) + row, len(self.variable_nodes) + column] += coeff * d_dvi.imag
                if term.p_var is not None:
                    nonlinear[row, term.p_var] += sign * d_dp.real
                    nonlinear[len(self.variable_nodes) + row, term.p_var] += sign * d_dp.imag
                if term.q_var is not None:
                    nonlinear[row, term.q_var] += sign * d_dq.real
                    nonlinear[len(self.variable_nodes) + row, term.q_var] += sign * d_dq.imag

        kcl_complex = self.model.Y_tot[self.variable_nodes, :] @ voltage.reshape(-1, 1) - node_current[self.variable_nodes].reshape(-1, 1)
        self._cache["kcl_complex"] = kcl_complex.reshape(-1)
        self._cache["kcl_residual"] = np.hstack([kcl_complex.real.reshape(-1), kcl_complex.imag.reshape(-1)])
        self._cache["kcl_jacobian"] = self._linear_kcl_jac + nonlinear.tocsr()

        source_frames = {}
        for key, selector in self.source_selectors.items():
            source_frames[key] = self._selector_value(values, selector)
        self._cache["source_frames"] = source_frames

    def objective(self, x: np.ndarray) -> float:
        self._evaluate(x)
        return self._objective_value(x, include_regularization=True)

    def reported_objective(self, x: np.ndarray) -> float:
        self._evaluate(x)
        return self._objective_value(x, include_regularization=False)

    def _objective_value(self, x: np.ndarray, *, include_regularization: bool) -> float:
        total = 0.0
        for _, row in self.work.poly_cost.iterrows():
            for power_type, columns in (
                ("p", ("cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")),
                ("q", ("cq0_eur", "cq1_eur_per_mvar", "cq2_eur_per_mvar2")),
            ):
                selector = self._selector_for_cost_row(row, power_type)
                value, _ = self._selector_value(x, selector)
                total += _cost_coefficient(row, columns[0])
                total += _cost_coefficient(row, columns[1]) * value
                total += _cost_coefficient(row, columns[2]) * value * value
        for spec in self.pwl_specs:
            total += float(x[spec.column])
        if include_regularization and self.variables:
            dispatch_slice = x[self.dispatch_offset : self.dispatch_offset + len(self.variables)]
            total += 0.5 * _DISPATCH_REGULARIZATION * float(np.dot(dispatch_slice, dispatch_slice))
        if not np.isfinite(total):
            raise InvalidStudyInput("OPF objective evaluated to a non-finite value")
        return float(total)

    def objective_jac(self, x: np.ndarray) -> np.ndarray:
        self._evaluate(x)
        grad = np.zeros(self.total_size, dtype=float)
        for _, row in self.work.poly_cost.iterrows():
            for power_type, columns in (
                ("p", ("cp1_eur_per_mw", "cp2_eur_per_mw2")),
                ("q", ("cq1_eur_per_mvar", "cq2_eur_per_mvar2")),
            ):
                selector = self._selector_for_cost_row(row, power_type)
                value, jacobian = self._selector_value(x, selector)
                factor = _cost_coefficient(row, columns[0]) + 2.0 * _cost_coefficient(row, columns[1]) * value
                grad += factor * np.asarray(jacobian.toarray()).reshape(-1)
        for spec in self.pwl_specs:
            grad[spec.column] += 1.0
        if self.variables:
            dispatch_slice = x[self.dispatch_offset : self.dispatch_offset + len(self.variables)]
            grad[self.dispatch_offset : self.dispatch_offset + len(self.variables)] += (
                _DISPATCH_REGULARIZATION * dispatch_slice
            )
        return grad

    def kcl_constraint(self, x: np.ndarray) -> np.ndarray:
        self._evaluate(x)
        return np.asarray(self._cache["kcl_residual"], dtype=float)

    def kcl_jacobian(self, x: np.ndarray) -> csr_matrix:
        self._evaluate(x)
        return csr_matrix(self._cache["kcl_jacobian"])

    def inequality_constraint(self, x: np.ndarray) -> np.ndarray:
        values, _ = self._inequality_rows(x)
        return values

    def inequality_jacobian(self, x: np.ndarray) -> csr_matrix:
        _, jacobian = self._inequality_rows(x)
        return jacobian

    def _inequality_rows(self, x: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
        self._evaluate(x)
        voltage = self._cache["voltage"]
        rows: list[float] = []
        jac_rows = lil_matrix((len(self.ineq_row_names), self.total_size), dtype=float)
        row_cursor = 0

        for limit in self.bus_limits:
            value = voltage[limit.y_index]
            magnitude = abs(value)
            column = self.variable_lookup.get(int(limit.y_index))
            if limit.lower is not None:
                rows.append(float(limit.lower - magnitude))
                if column is not None and magnitude > _VOLTAGE_EPS:
                    jac_rows[row_cursor, column] = -value.real / magnitude
                    jac_rows[row_cursor, len(self.variable_nodes) + column] = -value.imag / magnitude
                row_cursor += 1
            if limit.upper is not None:
                rows.append(float(magnitude - limit.upper))
                if column is not None and magnitude > _VOLTAGE_EPS:
                    jac_rows[row_cursor, column] = value.real / magnitude
                    jac_rows[row_cursor, len(self.variable_nodes) + column] = value.imag / magnitude
                row_cursor += 1

        current_cache = self._line_current_cache()
        for limit in self.line_limits:
            side_key = "from" if limit.side == "from" else "to"
            current = current_cache[side_key]["value"][limit.row]
            rows.append(float(current - limit.limit_ka))
            jac_rows[row_cursor, : self.state_count] = current_cache[side_key]["jac"][limit.row, :]
            row_cursor += 1

        if self.trafo_operator is not None and self.trafo_terminals is not None:
            p_jac, q_jac, power = _rectangular_power_block(
                self.trafo_operator,
                self.trafo_terminals,
                voltage,
                self.variable_nodes,
            )
            apparent = np.sqrt(power.real**2 + power.imag**2)
            for limit in self.trafo_limits:
                side_rows = np.asarray(limit.operator_rows, dtype=int)
                side_loading = (
                    apparent[side_rows] * self.work.sn_mva * 1e6 / limit.rated_power_va * 100.0
                )
                active_side = int(side_rows[int(np.argmax(side_loading))])
                actual_loading = float(side_loading.max())
                rows.append(actual_loading - limit.apparent_limit_percent)
                if apparent[active_side] > _VOLTAGE_EPS:
                    jac = (
                        power.real[active_side] / apparent[active_side] * p_jac[active_side, :]
                        + power.imag[active_side] / apparent[active_side] * q_jac[active_side, :]
                    ) * (self.work.sn_mva * 1e6 / limit.rated_power_va * 100.0)
                    jac_rows[row_cursor, : self.state_count] = jac
                row_cursor += 1

        for limit in self.capability_limits:
            p = float(x[limit.p_var]) if limit.p_var is not None else limit.p_fixed
            q = float(x[limit.q_var]) if limit.q_var is not None else limit.q_fixed
            actual = float(np.hypot(p, q))
            rows.append(actual - limit.limit)
            if actual > _VOLTAGE_EPS:
                if limit.p_var is not None:
                    jac_rows[row_cursor, limit.p_var] = p / actual
                if limit.q_var is not None:
                    jac_rows[row_cursor, limit.q_var] = q / actual
            row_cursor += 1

        active_source = self.work.get(self.source_info.active_kind)
        if active_source is not None:
            seen: set[tuple[str, Any, str]] = set()
            for index, row in active_source.iterrows():
                for power_type, lower_name, upper_name in (
                    ("p", "min_p_mw", "max_p_mw"),
                    ("q", "min_q_mvar", "max_q_mvar"),
                ):
                    selector_key = (self.source_info.active_kind, index, power_type)
                    if selector_key in seen:
                        continue
                    seen.add(selector_key)
                    selector = self.source_selectors[selector_key]
                    value, jacobian = self._selector_value(x, selector)
                    lower = _finite_bound(row, lower_name)
                    upper = _finite_bound(row, upper_name)
                    if lower is not None:
                        rows.append(float(lower - value))
                        jac_rows[row_cursor, :] = -jacobian
                        row_cursor += 1
                    if upper is not None:
                        rows.append(float(value - upper))
                        jac_rows[row_cursor, :] = jacobian
                        row_cursor += 1

        for spec in self.pwl_specs:
            value, jacobian = self._selector_value(x, spec.selector)
            segment_count = min(len(spec.slopes), len(spec.intercepts))
            for segment in range(segment_count):
                rows.append(float(spec.slopes[segment] * value + spec.intercepts[segment] - x[spec.column]))
                jac_rows[row_cursor, :] = spec.slopes[segment] * jacobian
                jac_rows[row_cursor, spec.column] = -1.0
                row_cursor += 1

        return np.asarray(rows, dtype=float), jac_rows.tocsr()

    def _line_current_cache(self) -> dict[str, dict[str, Any]]:
        key = "line_current"
        if key in self._cache:
            return self._cache[key]
        voltage = self._cache["voltage"]
        d_ir_from, d_ii_from, current_from = _rectangular_current_block(
            self.se_model.Yf, voltage, self.variable_nodes
        )
        d_ir_to, d_ii_to, current_to = _rectangular_current_block(
            self.se_model.Yt, voltage, self.variable_nodes
        )
        scales_from = self.model.Ibase_y[self.se_model.f] / 1000.0
        scales_to = self.model.Ibase_y[self.se_model.t] / 1000.0

        def magnitude(current: np.ndarray, d_ir: csr_matrix, d_ii: csr_matrix, scale: np.ndarray):
            current = np.asarray(current).reshape(-1)
            value = np.abs(current) * scale
            jacobian = lil_matrix((len(current), self.state_count), dtype=float)
            for row, (item, factor) in enumerate(zip(current, scale)):
                magnitude_value = abs(item)
                if magnitude_value <= _VOLTAGE_EPS:
                    continue
                jacobian[row, :] = factor * (
                    item.real / magnitude_value * d_ir[row, :]
                    + item.imag / magnitude_value * d_ii[row, :]
                )
            return {"value": value, "jac": jacobian.tocsr()}

        cache = {
            "from": magnitude(current_from, d_ir_from, d_ii_from, scales_from),
            "to": magnitude(current_to, d_ir_to, d_ii_to, scales_to),
        }
        self._cache[key] = cache
        return cache

    def initial_point(self, init: str, dispatch_seed: np.ndarray | None = None) -> np.ndarray:
        point = np.zeros(self.total_size, dtype=float)
        dispatch_seed = (
            np.asarray(dispatch_seed, dtype=float).reshape(-1)
            if dispatch_seed is not None
            else np.asarray(self.x_dispatch0, dtype=float).reshape(-1)
        )
        if init == "pf":
            initial_work = deepcopy(self.work)
            _apply_dispatch(initial_work, self.variables, dispatch_seed, self.gen_mapping)
            try:
                run_pf(initial_work, run_control=False)
                voltage = np.asarray(initial_work.model.E, dtype=complex).reshape(-1)
            except LoadflowNotConverged:
                voltage = np.asarray(self.model.E0, dtype=complex).reshape(-1)
        else:
            voltage = np.asarray(self.model.E0, dtype=complex).reshape(-1)
        point[: len(self.variable_nodes)] = voltage[self.variable_nodes].real
        point[len(self.variable_nodes) : self.state_count] = voltage[self.variable_nodes].imag
        point[self.dispatch_offset : self.dispatch_offset + len(self.variables)] = dispatch_seed
        self._evaluate(point)
        for spec in self.pwl_specs:
            value, _ = self._selector_value(point, spec.selector)
            point[spec.column] = _objective_from_segments(
                spec.powers,
                spec.slopes,
                spec.intercepts,
                value,
            )
        return point

    def bounds(self) -> Bounds:
        lower = np.full(self.total_size, -np.inf, dtype=float)
        upper = np.full(self.total_size, np.inf, dtype=float)
        for pos, variable in enumerate(self.variables):
            column = self.dispatch_offset + pos
            lower[column] = variable.lower
            upper[column] = variable.upper
        return Bounds(lower, upper, keep_feasible=False)

    def apply_solution(self, x: np.ndarray) -> None:
        _apply_dispatch(
            self.work,
            self.variables,
            np.asarray(x[self.dispatch_offset : self.dispatch_offset + len(self.variables)], dtype=float),
            self.gen_mapping,
        )

    def constraint_frame(self, x: np.ndarray, *, dual_eq: np.ndarray | None, dual_ineq: np.ndarray | None) -> pd.DataFrame:
        self._evaluate(x)
        voltage = self._cache["voltage"]
        rows: list[dict[str, Any]] = []
        for position, name in enumerate(self.kcl_row_names):
            residual = float(self._cache["kcl_residual"][position])
            rows.append(
                {
                    "constraint": name,
                    "table": "terminal",
                    "element": int(self.variable_nodes[position % len(self.variable_nodes)]),
                    "kind": "kcl",
                    "actual": residual,
                    "limit": 0.0,
                    "residual": abs(residual),
                    "dual": np.nan if dual_eq is None or position >= len(dual_eq) else float(dual_eq[position]),
                    "dual_status": "equality",
                    "satisfied": bool(np.isfinite(residual) and abs(residual) <= _RESIDUAL_TOL),
                }
            )

        row_cursor = 0
        for limit in self.bus_limits:
            value = abs(voltage[limit.y_index])
            if limit.lower is not None:
                residual = float(limit.lower - value)
                rows.append(
                    {
                        "constraint": f"{limit.name}_min",
                        "table": "bus",
                        "element": limit.bus_index,
                        "kind": "vm_min",
                        "actual": value,
                        "limit": limit.lower,
                        "residual": residual,
                        "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                        "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                        "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                    }
                )
                row_cursor += 1
            if limit.upper is not None:
                residual = float(value - limit.upper)
                rows.append(
                    {
                        "constraint": f"{limit.name}_max",
                        "table": "bus",
                        "element": limit.bus_index,
                        "kind": "vm_max",
                        "actual": value,
                        "limit": limit.upper,
                        "residual": residual,
                        "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                        "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                        "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                    }
                )
                row_cursor += 1

        current_cache = self._line_current_cache()
        for limit in self.line_limits:
            side_key = "from" if limit.side == "from" else "to"
            actual = float(current_cache[side_key]["value"][limit.row])
            residual = float(actual - limit.limit_ka)
            rows.append(
                {
                    "constraint": limit.name,
                    "table": "line",
                    "element": limit.row,
                    "kind": "ampacity_max",
                    "actual": actual,
                    "limit": limit.limit_ka,
                    "residual": residual,
                    "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                    "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                    "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                }
            )
            row_cursor += 1

        if self.trafo_operator is not None and self.trafo_terminals is not None:
            p_jac, q_jac, power = _rectangular_power_block(
                self.trafo_operator,
                self.trafo_terminals,
                voltage,
                self.variable_nodes,
            )
            apparent = np.sqrt(power.real**2 + power.imag**2)
            for limit in self.trafo_limits:
                side_rows = np.asarray(limit.operator_rows, dtype=int)
                side_loading = (
                    apparent[side_rows] * self.work.sn_mva * 1e6 / limit.rated_power_va * 100.0
                )
                actual = float(side_loading.max())
                residual = actual - limit.apparent_limit_percent
                rows.append(
                    {
                        "constraint": limit.name,
                        "table": "trafo1ph",
                        "element": limit.row_index,
                        "kind": "loading_max",
                        "actual": actual,
                        "limit": limit.apparent_limit_percent,
                        "residual": residual,
                        "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                        "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                        "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                    }
                )
                row_cursor += 1

        for limit in self.capability_limits:
            p = float(x[limit.p_var]) if limit.p_var is not None else limit.p_fixed
            q = float(x[limit.q_var]) if limit.q_var is not None else limit.q_fixed
            actual = float(np.hypot(p, q))
            residual = actual - limit.limit
            rows.append(
                {
                    "constraint": limit.name,
                    "table": limit.table,
                    "element": limit.index,
                    "kind": "capability",
                    "actual": actual,
                    "limit": limit.limit,
                    "residual": residual,
                    "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                    "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                    "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                }
            )
            row_cursor += 1

        active_source = self.work.get(self.source_info.active_kind)
        if active_source is not None:
            seen: set[tuple[str, Any, str]] = set()
            for index, row in active_source.iterrows():
                for power_type, lower_name, upper_name in (
                    ("p", "min_p_mw", "max_p_mw"),
                    ("q", "min_q_mvar", "max_q_mvar"),
                ):
                    selector_key = (self.source_info.active_kind, index, power_type)
                    if selector_key in seen:
                        continue
                    seen.add(selector_key)
                    selector = self.source_selectors[selector_key]
                    actual, _ = self._selector_value(x, selector)
                    lower = _finite_bound(row, lower_name)
                    upper = _finite_bound(row, upper_name)
                    if lower is not None:
                        residual = lower - actual
                        rows.append(
                            {
                                "constraint": f"{self.source_info.active_kind}:{index}:{power_type}_min",
                                "table": self.source_info.active_kind,
                                "element": index,
                                "kind": f"source_{power_type}_min",
                                "actual": actual,
                                "limit": lower,
                                "residual": residual,
                                "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                                "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                                "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                            }
                        )
                        row_cursor += 1
                    if upper is not None:
                        residual = actual - upper
                        rows.append(
                            {
                                "constraint": f"{self.source_info.active_kind}:{index}:{power_type}_max",
                                "table": self.source_info.active_kind,
                                "element": index,
                                "kind": f"source_{power_type}_max",
                                "actual": actual,
                                "limit": upper,
                                "residual": residual,
                                "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                                "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                                "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                            }
                        )
                        row_cursor += 1

        for spec in self.pwl_specs:
            value, _ = self._selector_value(x, spec.selector)
            segment_count = min(len(spec.slopes), len(spec.intercepts))
            for segment in range(segment_count):
                actual = spec.slopes[segment] * value + spec.intercepts[segment]
                residual = actual - float(x[spec.column])
                rows.append(
                    {
                        "constraint": f"{spec.name}:segment:{segment}",
                        "table": "pwl_cost",
                        "element": spec.name,
                        "kind": "pwl_epigraph",
                        "actual": actual,
                        "limit": float(x[spec.column]),
                        "residual": residual,
                        "dual": np.nan if dual_ineq is None else float(dual_ineq[row_cursor]),
                        "dual_status": "active" if abs(residual) <= _RESIDUAL_TOL else "inactive",
                        "satisfied": bool(np.isfinite(residual) and residual <= _RESIDUAL_TOL),
                    }
                )
                row_cursor += 1

        rows.append(
            {
                "constraint": "native_ac_power_flow",
                "table": "network",
                "element": None,
                "kind": "power_flow_converged",
                "actual": 0.0,
                "limit": 0.0,
                "residual": 0.0,
                "dual": np.nan,
                "dual_status": "validation",
                "satisfied": bool(getattr(self.work.model, "solved", False)),
            }
        )
        return pd.DataFrame(rows).set_index("constraint")


def _copy_opf_results(net: Any, work: Any, gen_mapping: dict[Any, Any]) -> dict[str, pd.DataFrame]:
    dispatch: dict[str, pd.DataFrame] = {}
    if "asymmetric_load" in work:
        net["res_asymmetric_load_opf"] = work.asymmetric_load[["p_mw", "q_mvar"]].copy()
        dispatch["asymmetric_load"] = net["res_asymmetric_load_opf"]
    if "asymmetric_sgen" in work:
        frame = work.asymmetric_sgen.copy().drop(index=list(gen_mapping.values()), errors="ignore")
        net["res_asymmetric_sgen_opf"] = frame[["p_mw", "q_mvar"]].copy()
        dispatch["asymmetric_sgen"] = net["res_asymmetric_sgen_opf"]
    if "asymmetric_gen" in work:
        _generator_results_pf(work)
        result = work.res_asymmetric_gen.copy()
        for original_index, staged_index in gen_mapping.items():
            result.at[original_index, "p_mw"] = work.asymmetric_gen.at[original_index, "p_mw"]
            result.at[original_index, "q_mvar"] = work.asymmetric_gen.at[original_index, "q_mvar"]
        net["res_asymmetric_gen_opf"] = result
        dispatch["asymmetric_gen"] = result

    for result_name in ("res_bus", "res_line", "res_trafo"):
        if result_name in work:
            net[f"{result_name}_opf"] = work[result_name].copy()

    active_kind, active_source = get_active_ext_grid_table(work, require_active=False)
    source_rows: list[dict[str, Any]] = []
    source_index: list[tuple[str, Any]] = []
    if active_source is not None and len(active_source):
        voltage = np.asarray(work.model.E, dtype=complex).reshape(-1)
        if work.model.Y_source.nnz > 0 and not bool(work.model.is_ideal_ext_grid):
            terminals = np.asarray(
                [
                    work.model.terminal_to_y_lookup[
                        get_ext_grid_terminal(work.model, int(index[0]), int(index[1]))
                    ]
                    for index in active_source.index
                ],
                dtype=int,
            )
            operator = csr_matrix(work.model.Y_source[terminals, :])
            signs = np.ones(len(terminals), dtype=float)
        else:
            terminals = np.asarray(
                [
                    work.model.terminal_to_y_lookup[
                        get_bus_terminal(work.model, int(row.bus), int(row.from_phase))
                    ]
                    for _, row in active_source.iterrows()
                ],
                dtype=int,
            )
            operator = csr_matrix(
                work.model.Y_tran
                + getattr(work.model, "Y_network", csr_matrix((work.model.y_size, work.model.y_size)))
                + getattr(work.model, "Y_ground", csr_matrix((work.model.y_size, work.model.y_size)))
            )[terminals, :]
            signs = -np.ones(len(terminals), dtype=float)
        _, _, power = _rectangular_power_block(operator, terminals, voltage, np.asarray(work.model.y_nonslack, dtype=int))
        power = np.asarray(power).reshape(-1)
        for position, index in enumerate(active_source.index):
            p = float(signs[position] * power.real[position])
            q = float(signs[position] * power.imag[position])
            source_index.append((active_kind, index))
            source_rows.append({"p_mw": p, "q_mvar": q})
    source_result = pd.DataFrame(
        source_rows,
        index=pd.MultiIndex.from_tuples(source_index, names=["element_type", "element"]),
    )
    net["res_ext_grid_opf"] = source_result
    dispatch["ext_grid"] = source_result
    return dispatch


def run_opf(
    net: Any,
    *,
    init: str = "pf",
    solver_options: dict[str, Any] | None = None,
) -> OPFResult:
    """Run a continuous phase-aware AC optimal power flow."""

    if init not in {"pf", "flat"}:
        raise InvalidStudyInput("init must be 'pf' or 'flat'")
    _ensure_cost_tables(net)
    if len(net.poly_cost) == 0 and len(net.pwl_cost) == 0:
        raise InvalidStudyInput("OPF requires at least one polynomial or PWL cost")

    work = deepcopy(net)
    _ensure_cost_tables(work)
    variables, x_dispatch0 = _build_variables(work)
    gen_mapping = _stage_asymmetric_generators(work)
    _prepare_native_model(work)
    opf = _RectangularOpfModel(
        work,
        variables=variables,
        x_dispatch0=x_dispatch0,
        gen_mapping=gen_mapping,
    )

    if not variables and not opf.pwl_specs:
        try:
            run_pf(work, run_control=False)
        except LoadflowNotConverged as exc:
            net["OPF_converged"] = False
            raise OPFNotConverged("fixed dispatch is not AC-feasible") from exc
        point = np.zeros(opf.total_size, dtype=float)
        point[: len(opf.variable_nodes)] = np.asarray(work.model.E, dtype=complex).reshape(-1)[opf.variable_nodes].real
        point[len(opf.variable_nodes) : opf.state_count] = np.asarray(work.model.E, dtype=complex).reshape(-1)[opf.variable_nodes].imag
        ineq_values = opf.inequality_constraint(point) if opf.ineq_row_names else np.asarray([], dtype=float)
        if len(ineq_values) and float(np.max(ineq_values)) > _RESIDUAL_TOL:
            net["OPF_converged"] = False
            raise OPFNotConverged("fixed dispatch is not AC-feasible")
        constraints = opf.constraint_frame(point, dual_eq=None, dual_ineq=None)
        dispatch = _copy_opf_results(net, work, gen_mapping)
        net["res_opf_constraint"] = constraints
        objective = opf.reported_objective(point)
        net["res_cost"] = pd.DataFrame([{"objective": objective, "iterations": 0, "algorithm": "fixed", "message": "fixed feasible dispatch"}])
        net["OPF_converged"] = True
        return OPFResult(True, objective, 0, "fixed feasible dispatch", dispatch, constraints)

    lower_dispatch = np.asarray([variable.lower for variable in variables], dtype=float)
    upper_dispatch = np.asarray([variable.upper for variable in variables], dtype=float)
    midpoint_dispatch = 0.5 * (lower_dispatch + upper_dispatch) if len(variables) else np.asarray([], dtype=float)
    dispatch_seeds: list[np.ndarray] = []
    for seed in (x_dispatch0, lower_dispatch, upper_dispatch, midpoint_dispatch):
        seed = np.asarray(seed, dtype=float)
        if any(np.array_equal(seed, existing) for existing in dispatch_seeds):
            continue
        dispatch_seeds.append(seed.copy())
    options = {
        "maxiter": 300,
        "gtol": 1e-8,
        "xtol": 1e-10,
        "barrier_tol": 1e-10,
        "verbose": 0,
    }
    if solver_options:
        options.update(solver_options)

    feasible_candidates: list[
        tuple[float, Any, np.ndarray, np.ndarray, float, float]
    ] = []
    last_result = None
    last_kcl = np.inf
    last_ineq = np.inf
    for dispatch_seed in dispatch_seeds:
        x0 = opf.initial_point(init, dispatch_seed)

        # Source-strength admittances can be many orders of magnitude larger
        # than feeder admittances.  Optimizing the raw rectangular coordinates
        # consequently makes a physically meaningful source-voltage movement
        # look smaller than trust-constr's step resolution.  Scale columns by
        # the KCL Jacobian and center them at the independently solved start;
        # dispatch variables are scaled by their capability range.  The model,
        # constraints, derivatives, and reported residuals remain in native
        # units -- this is only a well-conditioned optimizer coordinate system.
        kcl_jacobian0 = opf.kcl_jacobian(x0)
        column_max = np.asarray(abs(kcl_jacobian0).max(axis=0).toarray()).reshape(-1)
        coordinate_scale = np.ones(opf.total_size, dtype=float)
        coordinate_scale[: opf.state_count] = 1.0 / np.maximum(
            column_max[: opf.state_count], 1.0
        )
        if variables:
            dispatch_span = upper_dispatch - lower_dispatch
            dispatch_scale = np.where(
                np.isfinite(dispatch_span) & (dispatch_span > 1e-9),
                np.maximum(dispatch_span, 0.1),
                np.maximum(
                    np.abs(x0[opf.dispatch_offset : opf.epigraph_offset]), 1.0
                ),
            )
            coordinate_scale[opf.dispatch_offset : opf.epigraph_offset] = dispatch_scale
        if opf.pwl_specs:
            coordinate_scale[opf.epigraph_offset :] = np.maximum(
                np.abs(x0[opf.epigraph_offset :]), 1.0
            )
        coordinate_matrix = diags(coordinate_scale, format="csr")
        scaled_kcl_jacobian0 = kcl_jacobian0 @ coordinate_matrix
        row_max = np.asarray(abs(scaled_kcl_jacobian0).max(axis=1).toarray()).reshape(-1)
        kcl_row_scale = 1.0 / np.maximum(row_max, 1.0)
        row_matrix = diags(kcl_row_scale, format="csr")

        def to_physical(scaled: np.ndarray) -> np.ndarray:
            return x0 + coordinate_scale * np.asarray(scaled, dtype=float)

        def scaled_objective(scaled: np.ndarray) -> float:
            return opf.objective(to_physical(scaled))

        def scaled_objective_jac(scaled: np.ndarray) -> np.ndarray:
            return opf.objective_jac(to_physical(scaled)) * coordinate_scale

        def scaled_kcl(scaled: np.ndarray) -> np.ndarray:
            return kcl_row_scale * opf.kcl_constraint(to_physical(scaled))

        def scaled_kcl_jac(scaled: np.ndarray) -> csr_matrix:
            return row_matrix @ opf.kcl_jacobian(to_physical(scaled)) @ coordinate_matrix

        scaled_constraints: list[NonlinearConstraint] = [
            NonlinearConstraint(scaled_kcl, 0.0, 0.0, jac=scaled_kcl_jac)
        ]
        if opf.ineq_row_names:

            def scaled_inequality(scaled: np.ndarray) -> np.ndarray:
                return opf.inequality_constraint(to_physical(scaled))

            def scaled_inequality_jac(scaled: np.ndarray) -> csr_matrix:
                return opf.inequality_jacobian(to_physical(scaled)) @ coordinate_matrix

            scaled_constraints.append(
                NonlinearConstraint(
                    scaled_inequality,
                    -np.inf,
                    0.0,
                    jac=scaled_inequality_jac,
                )
            )

        physical_bounds = opf.bounds()
        scaled_bounds = Bounds(
            (physical_bounds.lb - x0) / coordinate_scale,
            (physical_bounds.ub - x0) / coordinate_scale,
            keep_feasible=False,
        )
        try:
            result = minimize(
                scaled_objective,
                np.zeros(opf.total_size, dtype=float),
                jac=scaled_objective_jac,
                method="trust-constr",
                bounds=scaled_bounds,
                constraints=tuple(scaled_constraints),
                options=options,
            )
        except LoadflowNotConverged as exc:
            net["OPF_converged"] = False
            raise OPFNotConverged("rectangular AC OPF setup failed") from exc

        physical_x = to_physical(result.x)
        kcl_residual = opf.kcl_constraint(physical_x)
        kcl_violation = float(np.max(np.abs(kcl_residual))) if len(kcl_residual) else 0.0
        ineq_values = (
            opf.inequality_constraint(physical_x)
            if opf.ineq_row_names
            else np.asarray([], dtype=float)
        )
        ineq_violation = float(np.max(ineq_values)) if len(ineq_values) else 0.0
        last_result = result
        last_kcl = kcl_violation
        last_ineq = ineq_violation
        if kcl_violation <= _RESIDUAL_TOL and ineq_violation <= _RESIDUAL_TOL:
            feasible_candidates.append(
                (
                    float(opf.reported_objective(physical_x)),
                    result,
                    physical_x,
                    kcl_row_scale,
                    kcl_violation,
                    ineq_violation,
                )
            )

    if not feasible_candidates or last_result is None:
        net["OPF_converged"] = False
        raise OPFNotConverged(
            "rectangular AC OPF failed: "
            f"{last_result.message if last_result is not None else 'no candidate'}; "
            f"max_kcl={last_kcl:.6g}; max_ineq={last_ineq:.6g}"
        )

    _, result, physical_x, kcl_row_scale, kcl_violation, ineq_violation = min(
        feasible_candidates, key=lambda item: item[0]
    )

    opf.apply_solution(physical_x)
    try:
        run_pf(work, run_control=False)
    except LoadflowNotConverged as exc:
        net["OPF_converged"] = False
        raise OPFNotConverged("optimized dispatch failed final AC validation") from exc

    dual_eq = None
    dual_ineq = None
    if getattr(result, "v", None):
        if len(result.v) >= 1:
            dual_eq = (
                np.asarray(result.v[0], dtype=float).reshape(-1) * kcl_row_scale
            )
        if len(result.v) >= 2:
            dual_ineq = np.asarray(result.v[1], dtype=float).reshape(-1)
    constraint_frame = opf.constraint_frame(
        physical_x, dual_eq=dual_eq, dual_ineq=dual_ineq
    )
    if not bool(constraint_frame["satisfied"].all()):
        net["OPF_converged"] = False
        worst = pd.to_numeric(constraint_frame["residual"], errors="coerce").fillna(0.0).max()
        raise OPFNotConverged(f"optimized dispatch violates an AC network constraint (max residual {worst:.6g})")

    dispatch = _copy_opf_results(net, work, gen_mapping)
    objective = float(opf.reported_objective(physical_x))
    net["res_opf_constraint"] = constraint_frame
    net["res_cost"] = pd.DataFrame(
        [
            {
                "objective": objective,
                "iterations": int(result.nit),
                "algorithm": "trust-constr",
                "message": str(result.message),
            }
        ]
    )
    net["OPF_converged"] = True
    return OPFResult(
        converged=True,
        objective=objective,
        iterations=int(result.nit),
        message=str(result.message),
        dispatch=dispatch,
        constraints=constraint_frame,
    )


__all__ = [
    "OPFNotConverged",
    "OPFResult",
    "create_poly_cost",
    "create_pwl_cost",
    "run_opf",
]
