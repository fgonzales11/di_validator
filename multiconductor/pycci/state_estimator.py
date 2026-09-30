"""Sparse weighted-least-squares state estimation for multiconductor networks.

The native multiconductor engine does not use ``bus * phase`` as its solved
node index.  Ideal grounds, closed switches, and element terminals are reduced
to electrical nodes through ``net.model.terminal_to_y_lookup``.  This module
therefore exposes both a low-level sparse WLS solver and a network adapter that
uses the initialized native model directly.

All low-level voltage, power, and current values are per-unit.  Voltage angles
are radians.  ``measurements_from_dataframe`` converts the public network table
units (MW, MVAr, kA, degrees) into those internal units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, diags, hstack, vstack
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import MatrixRankWarning, splu, spsolve
from scipy.stats import chi2

from multiconductor.studies import InvalidStudyInput, MulticonductorStudyError


class StateEstimationError(MulticonductorStudyError):
    """Base class for state-estimation failures."""


class UnobservableNetworkError(StateEstimationError):
    """Raised when the measurement Jacobian cannot observe every state."""

    def __init__(
        self,
        message: str,
        *,
        measurement_count: int | None = None,
        state_count: int | None = None,
        structural_rank: int | None = None,
        numeric_rank: int | None = None,
        zero_columns: list[int] | None = None,
    ) -> None:
        super().__init__(message)
        self.measurement_count = measurement_count
        self.state_count = state_count
        self.structural_rank = structural_rank
        self.numeric_rank = numeric_rank
        self.zero_columns = zero_columns or []


class StateEstimationNotConverged(StateEstimationError):
    """Raised when requested by a caller and the WLS iterations do not converge."""


@dataclass(frozen=True)
class ObservabilityReport:
    measurement_count: int
    state_count: int
    structural_rank: int
    numeric_rank: int | None
    zero_columns: tuple[int, ...] = ()


@dataclass
class Measurement:
    """A scalar per-unit measurement and its one-standard-deviation error."""

    value: float
    sigma: float
    name: str | None = None
    source_index: object | None = None


@dataclass(frozen=True)
class _OrderedMeasurement:
    kind: str
    index: int
    side: str
    measurement: Measurement


class StateEstimationInput:
    """Container for phase-aware scalar measurements in reduced-node space.

    ``node`` arguments are indices in the native admittance matrix.  Branch
    indices address rows returned by :func:`extract_se_matrices`; ``side`` is
    either ``"from"`` or ``"to"``.
    """

    def __init__(self):
        self._p_flow: list[tuple[int, Measurement, str]] = []
        self._q_flow: list[tuple[int, Measurement, str]] = []
        self._p_inj: list[tuple[int, Measurement]] = []
        self._q_inj: list[tuple[int, Measurement]] = []
        self._i_flow: list[tuple[int, Measurement, str]] = []
        self._i_angle: list[tuple[int, Measurement, str]] = []
        self._i_real: list[tuple[int, Measurement, str]] = []
        self._i_imag: list[tuple[int, Measurement, str]] = []
        self._vm: list[tuple[int, Measurement]] = []
        self._va: list[tuple[int, Measurement]] = []
        self._v_real: list[tuple[int, Measurement]] = []
        self._v_imag: list[tuple[int, Measurement]] = []

    @staticmethod
    def _validated(index, value, sigma, side=None, name=None, source_index=None):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or int(index) < 0:
            raise ValueError("Measurement index must be a non-negative integer.")
        if not np.isfinite(value):
            raise ValueError("Measurement value must be finite.")
        if not np.isfinite(sigma) or float(sigma) <= 0:
            raise ValueError("Measurement sigma must be finite and greater than zero.")
        if side is not None and str(side).lower() not in {"from", "to"}:
            raise ValueError("Measurement side must be 'from' or 'to'.")
        return int(index), Measurement(float(value), float(sigma), name, source_index)

    def _add_node(self, target, node, value, sigma, name=None, source_index=None):
        index, measurement = self._validated(node, value, sigma, name=name, source_index=source_index)
        target.append((index, measurement))
        return self

    def _add_branch(self, target, branch_row, value, sigma, side="from", name=None, source_index=None):
        index, measurement = self._validated(
            branch_row, value, sigma, side=side, name=name, source_index=source_index
        )
        target.append((index, measurement, str(side).lower()))
        return self

    def add_p_flow(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._p_flow, branch_row, value, sigma, side, name, source_index)

    def add_q_flow(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._q_flow, branch_row, value, sigma, side, name, source_index)

    def add_p_inj(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._p_inj, node, value, sigma, name, source_index)

    def add_q_inj(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._q_inj, node, value, sigma, name, source_index)

    def add_i_flow(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._i_flow, branch_row, value, sigma, side, name, source_index)

    def add_i_angle(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._i_angle, branch_row, value, sigma, side, name, source_index)

    def add_i_real(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._i_real, branch_row, value, sigma, side, name, source_index)

    def add_i_imag(self, branch_row, value, sigma, side="from", name=None, source_index=None):
        return self._add_branch(self._i_imag, branch_row, value, sigma, side, name, source_index)

    def add_vm(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._vm, node, value, sigma, name, source_index)

    def add_va(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._va, node, value, sigma, name, source_index)

    def add_v_real(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._v_real, node, value, sigma, name, source_index)

    def add_v_imag(self, node, value, sigma, name=None, source_index=None):
        return self._add_node(self._v_imag, node, value, sigma, name, source_index)

    @property
    def p_flow_idx(self):
        return np.asarray([row for row, _, _ in self._p_flow], dtype=int)

    @property
    def q_flow_idx(self):
        return np.asarray([row for row, _, _ in self._q_flow], dtype=int)

    @property
    def p_inj_idx(self):
        return np.asarray([node for node, _ in self._p_inj], dtype=int)

    @property
    def q_inj_idx(self):
        return np.asarray([node for node, _ in self._q_inj], dtype=int)

    @property
    def i_flow_idx(self):
        return np.asarray([row for row, _, _ in self._i_flow], dtype=int)

    @property
    def vm_m_idx(self):
        return np.asarray([node for node, _ in self._vm], dtype=int)

    def ordered_measurements(self):
        groups = (
            ("p_flow", self._p_flow, True),
            ("p_inj", self._p_inj, False),
            ("q_flow", self._q_flow, True),
            ("q_inj", self._q_inj, False),
            ("i_flow", self._i_flow, True),
            ("vm", self._vm, False),
            ("va", self._va, False),
            ("v_real", self._v_real, False),
            ("v_imag", self._v_imag, False),
            ("i_angle", self._i_angle, True),
            ("i_real", self._i_real, True),
            ("i_imag", self._i_imag, True),
        )
        result = []
        for kind, values, is_branch in groups:
            if is_branch:
                result.extend(_OrderedMeasurement(kind, row, side, measurement) for row, measurement, side in values)
            else:
                result.extend(_OrderedMeasurement(kind, node, "", measurement) for node, measurement in values)
        return result

    def consolidate(self):
        ordered = self.ordered_measurements()
        if not ordered:
            raise ValueError("No measurements added to StateEstimationInput.")
        z = np.asarray([item.measurement.value for item in ordered], dtype=float)
        sigma = np.asarray([item.measurement.sigma for item in ordered], dtype=float)
        return z, sigma

    def without_positions(self, positions):
        positions = set(int(position) for position in positions)
        result = StateEstimationInput()
        for position, item in enumerate(self.ordered_measurements()):
            if position in positions:
                continue
            method = getattr(result, f"add_{item.kind}")
            kwargs = {
                "name": item.measurement.name,
                "source_index": item.measurement.source_index,
            }
            if item.side:
                kwargs["side"] = item.side
            method(item.index, item.measurement.value, item.measurement.sigma, **kwargs)
        return result


_MEASUREMENT_UNIT_BY_KIND = {
    "p": "mw",
    "q": "mvar",
    "i": "ka",
    "im": "ka",
    "ia": "degree",
    "v": "pu",
    "vm": "pu",
    "va": "degree",
    "vr": "pu",
    "vi": "pu",
}

_STATE_ESTIMATION_RESULT_UNITS = {
    "p_flow": "pu",
    "q_flow": "pu",
    "p_inj": "pu",
    "q_inj": "pu",
    "i_flow": "pu",
    "i_angle": "rad",
    "i_real": "pu",
    "i_imag": "pu",
    "vm": "pu",
    "va": "rad",
    "v_real": "pu",
    "v_imag": "pu",
}

_BUS_ELEMENT_TYPES = {"bus", "node"}
_LINE_ELEMENT_TYPES = {"line"}
_TRAFO_ELEMENT_TYPES = {"trafo", "trafo1ph", "transformer"}
_SOURCE_ELEMENT_TYPES = {"ext_grid", "ext_grid_sequence", "source"}
_SWITCH_ELEMENT_TYPES = {"switch"}
_ZERO_INJECTION_ELEMENT_TYPES = {"zero_injection", "zero-injection", "zero injection"}


def _canonical_measurement_element_type(element_type):
    normalized = str(element_type).strip().lower()
    if normalized in _BUS_ELEMENT_TYPES:
        return "bus"
    if normalized in _LINE_ELEMENT_TYPES:
        return "line"
    if normalized in _TRAFO_ELEMENT_TYPES:
        return "trafo1ph"
    if normalized in _SOURCE_ELEMENT_TYPES:
        return "ext_grid"
    if normalized in _SWITCH_ELEMENT_TYPES:
        return "switch"
    if normalized in _ZERO_INJECTION_ELEMENT_TYPES:
        return "zero_injection"
    return normalized


def _canonical_branch_side(element_type, side):
    element_type = _canonical_measurement_element_type(element_type)
    if side is None or pd.isna(side) or not str(side).strip():
        defaults = {
            "line": "from",
            "trafo1ph": "hv",
            "ext_grid": "bus",
            "switch": "bus",
        }
        return defaults.get(element_type)
    normalized = str(side).strip().lower()
    aliases = {
        "line": {
            "from": "from",
            "to": "to",
            "sending": "from",
            "receiving": "to",
        },
        "trafo1ph": {
            "hv": "hv",
            "lv": "lv",
            "from": "hv",
            "to": "lv",
        },
        "ext_grid": {
            "bus": "bus",
            "grid": "bus",
            "network": "bus",
            "from": "bus",
            "source": "source",
            "internal": "source",
            "to": "source",
        },
        "switch": {
            "bus": "bus",
            "from": "bus",
            "element": "element",
            "to": "element",
        },
    }
    mapping = aliases.get(element_type)
    if mapping is None or normalized not in mapping:
        if element_type in {"line", "trafo1ph", "ext_grid", "switch"}:
            raise InvalidStudyInput(
                f"Unsupported {element_type} measurement side {side!r}."
            )
        return normalized
    return mapping[normalized]


def _internal_branch_side(element_type, side):
    canonical = _canonical_branch_side(element_type, side)
    if canonical in {None, "from", "hv", "bus"}:
        return "from"
    if canonical in {"to", "lv", "source", "element"}:
        return "to"
    raise InvalidStudyInput(f"Unsupported branch side {side!r}.")


def _state_measurement_terminal(element_type, element, *, phase=None, circuit=None, side=None):
    element_type = _canonical_measurement_element_type(element_type)
    if element_type == "bus":
        if phase is None:
            return None
        return f"bus:{int(element)}:{int(phase)}"
    if element_type == "zero_injection":
        if phase is None:
            return None
        return f"zero_injection:{int(element)}:{int(phase)}"
    if element_type == "line":
        if circuit is None:
            return None
        return f"line:{int(element)}:{int(circuit)}:{_canonical_branch_side(element_type, side)}"
    if element_type == "trafo1ph":
        if circuit is None:
            return None
        return f"trafo1ph:{int(element)}:{int(circuit)}:{_canonical_branch_side(element_type, side)}"
    if element_type == "ext_grid":
        if circuit is None:
            return None
        return f"ext_grid:{int(element)}:{int(circuit)}:{_canonical_branch_side(element_type, side)}"
    if element_type == "switch":
        if circuit is None:
            return None
        return f"switch:{int(element)}:{int(circuit)}:{_canonical_branch_side(element_type, side)}"
    return None


def _parse_measurement_terminal(value):
    if value is None or pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    parts = [part.strip() for part in text.split(":")]
    kind = _canonical_measurement_element_type(parts[0])
    try:
        if kind in {"bus", "zero_injection"}:
            if len(parts) != 3:
                raise InvalidStudyInput(
                    f"Terminal {text!r} must be '<bus|zero_injection>:<element>:<phase>'."
                )
            return {
                "element_type": kind,
                "element": _integer_identifier(parts[1], "Terminal element"),
                "phase": _integer_identifier(parts[2], "Terminal phase"),
            }
        if kind in {"line", "trafo1ph", "ext_grid", "switch"}:
            if len(parts) != 4:
                raise InvalidStudyInput(
                    f"Terminal {text!r} must be '<element_type>:<element>:<circuit>:<side>'."
                )
            return {
                "element_type": kind,
                "element": _integer_identifier(parts[1], "Terminal element"),
                "circuit": _integer_identifier(parts[2], "Terminal circuit"),
                "side": _canonical_branch_side(kind, parts[3]),
            }
    except ValueError as exc:
        raise InvalidStudyInput(str(exc)) from exc
    raise InvalidStudyInput(f"Unsupported terminal target {text!r}.")


def _merged_row_target(row):
    target = {
        "element_type": _canonical_measurement_element_type(row.get("element_type", "")),
        "element": _integer_identifier(row["element"], "Element"),
        "side": row.get("side"),
        "phase": row.get("phase", np.nan),
        "circuit": row.get("circuit", np.nan),
    }
    terminal_target = _parse_measurement_terminal(row.get("terminal"))
    for key, terminal_value in terminal_target.items():
        existing = target.get(key)
        if key in {"phase", "circuit"} and pd.isna(existing):
            target[key] = terminal_value
            continue
        if key == "side" and (existing is None or pd.isna(existing) or not str(existing).strip()):
            target[key] = terminal_value
            continue
        if key == "element_type" and existing == terminal_value:
            continue
        if key == "element" and existing == terminal_value:
            continue
        if existing is None:
            target[key] = terminal_value
            continue
        if key in {"phase", "circuit"} and not pd.isna(existing):
            existing_value = _integer_identifier(existing, f"Row {key}")
            if existing_value != terminal_value:
                raise InvalidStudyInput(
                    f"Measurement row conflicts with terminal metadata for {key}."
                )
            target[key] = existing_value
            continue
        if key == "side":
            existing_value = _canonical_branch_side(target["element_type"], existing)
            if existing_value != terminal_value:
                raise InvalidStudyInput(
                    "Measurement row conflicts with terminal metadata for side."
                )
            target[key] = existing_value
            continue
        if existing != terminal_value:
            raise InvalidStudyInput(
                f"Measurement row conflicts with terminal metadata for {key}."
            )
    if not pd.isna(target["phase"]):
        target["phase"] = _integer_identifier(target["phase"], "Phase")
    else:
        target["phase"] = None
    if not pd.isna(target["circuit"]):
        target["circuit"] = _integer_identifier(target["circuit"], "Circuit")
    else:
        target["circuit"] = None
    if target["side"] is not None and not pd.isna(target["side"]):
        target["side"] = _canonical_branch_side(target["element_type"], target["side"])
    else:
        target["side"] = _canonical_branch_side(target["element_type"], None)
    return target


def _safe_vnorm(V):
    magnitude = np.abs(V)
    result = np.ones_like(V, dtype=complex)
    nonzero = magnitude > 1e-12
    result[nonzero] = V[nonzero] / magnitude[nonzero]
    return result


def dSbus_dV(Ybus, V):
    """Partial derivatives of nodal complex power with respect to Vm and Va."""
    V = np.asarray(V, dtype=complex).reshape(-1)
    Ibus = np.asarray(Ybus @ V).reshape(-1)
    diagV = diags(V)
    diagIbus = diags(Ibus)
    diagVnorm = diags(_safe_vnorm(V))
    dS_dVm = diagV @ (Ybus @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
    dS_dVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conjugate()
    return dS_dVm.tocsr(), dS_dVa.tocsr()


def _branch_derivatives(Ybranch, terminal_nodes, V):
    V = np.asarray(V, dtype=complex).reshape(-1)
    terminal_nodes = np.asarray(terminal_nodes, dtype=int)
    rows = np.arange(len(terminal_nodes))
    current = np.asarray(Ybranch @ V).reshape(-1)
    diagV = diags(V)
    diagVnorm = diags(_safe_vnorm(V))
    diagVterminal = diags(V[terminal_nodes])
    diagI = diags(current)
    selector = csr_matrix((np.ones(len(rows)), (rows, terminal_nodes)), shape=(len(rows), len(V)))
    dS_dVa = 1j * (diagI.conjugate() @ selector @ diagV - diagVterminal @ (Ybranch @ diagV).conjugate())
    dS_dVm = diagVterminal @ (Ybranch @ diagVnorm).conjugate() + diagI.conjugate() @ selector @ diagVnorm
    dI_dVa = Ybranch @ (1j * diagV)
    dI_dVm = Ybranch @ diagVnorm
    S = V[terminal_nodes] * np.conjugate(current)
    return dS_dVm.tocsr(), dS_dVa.tocsr(), dI_dVm.tocsr(), dI_dVa.tocsr(), S, current


def dSbr_dV(Yf, Yt, V, f, t):
    """Partial derivatives of from- and to-side complex branch powers."""
    f_values = _branch_derivatives(Yf, f, V)
    t_values = _branch_derivatives(Yt, t, V)
    dSf_dVm, dSf_dVa, _, _, Sf, _ = f_values
    dSt_dVm, dSt_dVa, _, _, St, _ = t_values
    return dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St


def dIbr_dV(Yf, Yt, V):
    """Partial derivatives of complex branch currents."""
    V = np.asarray(V, dtype=complex).reshape(-1)
    diagV = diags(V)
    diagVnorm = diags(_safe_vnorm(V))
    dIf_dVa = (Yf @ (1j * diagV)).tocsr()
    dIf_dVm = (Yf @ diagVnorm).tocsr()
    dIt_dVa = (Yt @ (1j * diagV)).tocsr()
    dIt_dVm = (Yt @ diagVnorm).tocsr()
    return dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, np.asarray(Yf @ V), np.asarray(Yt @ V)


def _selected_branch_matrix(Yf, Yt, f, t, records):
    if not records:
        return csr_matrix((0, Yf.shape[1]), dtype=complex), np.empty(0, dtype=int)
    rows = np.asarray([row for row, _, _ in records], dtype=int)
    if np.any(rows >= Yf.shape[0]):
        raise ValueError("Branch measurement row is outside the extracted branch matrix.")
    use_to = np.asarray([side == "to" for _, _, side in records])
    from_weight = diags((~use_to).astype(float))
    to_weight = diags(use_to.astype(float))
    matrix = from_weight @ Yf[rows, :] + to_weight @ Yt[rows, :]
    terminals = np.where(use_to, np.asarray(t)[rows], np.asarray(f)[rows])
    return matrix.tocsr(), terminals.astype(int)


def Jacobian_SE(Ybus, Yf, Yt, V, f, t, inputs, pvpq, vm_nodes=None):
    """Return the sparse analytic measurement Jacobian and prediction vector.

    Columns are ``[Va[pvpq], Vm[vm_nodes]]``.  ``vm_nodes`` defaults to all
    nodes for compatibility with the historical low-level API.
    """
    V = np.asarray(V, dtype=complex).reshape(-1)
    n = len(V)
    angle_nodes = np.asarray(pvpq, dtype=int)
    magnitude_nodes = np.arange(n, dtype=int) if vm_nodes is None else np.asarray(vm_nodes, dtype=int)
    blocks = []
    predictions = []

    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
    Sbus = V * np.conjugate(np.asarray(Ybus @ V).reshape(-1))

    def add_power_flow(records, component):
        if not records:
            return
        matrix, terminals = _selected_branch_matrix(Yf, Yt, f, t, records)
        dS_dVm_b, dS_dVa_b, _, _, values, _ = _branch_derivatives(matrix, terminals, V)
        derivative = dS_dVa_b.real if component == "real" else dS_dVa_b.imag
        derivative_vm = dS_dVm_b.real if component == "real" else dS_dVm_b.imag
        blocks.append(hstack([derivative[:, angle_nodes], derivative_vm[:, magnitude_nodes]], format="csr"))
        predictions.append(values.real if component == "real" else values.imag)

    add_power_flow(inputs._p_flow, "real")

    if inputs._p_inj:
        nodes = inputs.p_inj_idx
        blocks.append(hstack([dS_dVa[nodes, :][:, angle_nodes].real, dS_dVm[nodes, :][:, magnitude_nodes].real], format="csr"))
        predictions.append(Sbus[nodes].real)

    add_power_flow(inputs._q_flow, "imag")

    if inputs._q_inj:
        nodes = inputs.q_inj_idx
        blocks.append(hstack([dS_dVa[nodes, :][:, angle_nodes].imag, dS_dVm[nodes, :][:, magnitude_nodes].imag], format="csr"))
        predictions.append(Sbus[nodes].imag)

    def add_current(records, component):
        if not records:
            return
        matrix, terminals = _selected_branch_matrix(Yf, Yt, f, t, records)
        _, _, dI_dVm, dI_dVa, _, current = _branch_derivatives(matrix, terminals, V)
        magnitude = np.abs(current)
        if component == "magnitude":
            unit_conjugate = np.zeros_like(current, dtype=complex)
            nonzero = magnitude > 1e-12
            unit_conjugate[nonzero] = np.conjugate(current[nonzero]) / magnitude[nonzero]
            left = diags(unit_conjugate)
            jac_va = (left @ dI_dVa).real
            jac_vm = (left @ dI_dVm).real
            values = magnitude
        elif component == "angle":
            inverse = np.zeros_like(current, dtype=complex)
            nonzero = magnitude > 1e-12
            inverse[nonzero] = 1.0 / current[nonzero]
            left = diags(inverse)
            jac_va = (left @ dI_dVa).imag
            jac_vm = (left @ dI_dVm).imag
            values = np.angle(current)
        elif component == "real":
            jac_va, jac_vm, values = dI_dVa.real, dI_dVm.real, current.real
        else:
            jac_va, jac_vm, values = dI_dVa.imag, dI_dVm.imag, current.imag
        blocks.append(hstack([jac_va[:, angle_nodes], jac_vm[:, magnitude_nodes]], format="csr"))
        predictions.append(values)

    add_current(inputs._i_flow, "magnitude")

    def add_voltage(records, component):
        if not records:
            return
        nodes = np.asarray([node for node, _ in records], dtype=int)
        count = len(nodes)
        angle_map = {int(node): column for column, node in enumerate(angle_nodes)}
        magnitude_map = {int(node): column for column, node in enumerate(magnitude_nodes)}
        row_index = []
        column_index = []
        data = []
        values = np.empty(count, dtype=float)
        for row, node in enumerate(nodes):
            voltage = V[node]
            vm = abs(voltage)
            va = np.angle(voltage)
            if component == "magnitude":
                values[row] = vm
                if node in magnitude_map:
                    row_index.append(row)
                    column_index.append(len(angle_nodes) + magnitude_map[node])
                    data.append(1.0)
            elif component == "angle":
                values[row] = va
                if node in angle_map:
                    row_index.append(row)
                    column_index.append(angle_map[node])
                    data.append(1.0)
            elif component == "real":
                values[row] = voltage.real
                if node in angle_map:
                    row_index.append(row)
                    column_index.append(angle_map[node])
                    data.append(-voltage.imag)
                if node in magnitude_map:
                    row_index.append(row)
                    column_index.append(len(angle_nodes) + magnitude_map[node])
                    data.append(np.cos(va))
            else:
                values[row] = voltage.imag
                if node in angle_map:
                    row_index.append(row)
                    column_index.append(angle_map[node])
                    data.append(voltage.real)
                if node in magnitude_map:
                    row_index.append(row)
                    column_index.append(len(angle_nodes) + magnitude_map[node])
                    data.append(np.sin(va))
        blocks.append(csr_matrix((data, (row_index, column_index)), shape=(count, len(angle_nodes) + len(magnitude_nodes))))
        predictions.append(values)

    add_voltage(inputs._vm, "magnitude")
    add_voltage(inputs._va, "angle")
    add_voltage(inputs._v_real, "real")
    add_voltage(inputs._v_imag, "imag")
    add_current(inputs._i_angle, "angle")
    add_current(inputs._i_real, "real")
    add_current(inputs._i_imag, "imag")

    if not blocks:
        inputs.consolidate()
    return vstack(blocks, format="csr"), np.concatenate(predictions)


def _rectangular_power_block(operator, terminal_nodes, voltage, variable_nodes):
    """Return P/Q values and rectangular-coordinate derivatives."""
    operator = csr_matrix(operator)
    voltage = np.asarray(voltage, dtype=complex).reshape(-1)
    terminal_nodes = np.asarray(terminal_nodes, dtype=int)
    variable_nodes = np.asarray(variable_nodes, dtype=int)
    row_count = len(terminal_nodes)
    selector = csr_matrix(
        (np.ones(row_count), (np.arange(row_count), terminal_nodes)),
        shape=(row_count, len(voltage)),
    )
    current = np.asarray(operator @ voltage).reshape(-1)
    vr = voltage[terminal_nodes].real
    vi = voltage[terminal_nodes].imag
    ir = current.real
    ii = current.imag
    conductance = operator.real.tocsr()
    susceptance = operator.imag.tocsr()

    dP_dVr = diags(ir) @ selector + diags(vr) @ conductance + diags(vi) @ susceptance
    dP_dVi = diags(ii) @ selector - diags(vr) @ susceptance + diags(vi) @ conductance
    dQ_dVr = -diags(ii) @ selector + diags(vi) @ conductance - diags(vr) @ susceptance
    dQ_dVi = diags(ir) @ selector - diags(vi) @ susceptance - diags(vr) @ conductance
    power = voltage[terminal_nodes] * np.conjugate(current)
    p_jacobian = hstack(
        [dP_dVr[:, variable_nodes], dP_dVi[:, variable_nodes]], format="csr"
    )
    q_jacobian = hstack(
        [dQ_dVr[:, variable_nodes], dQ_dVi[:, variable_nodes]], format="csr"
    )
    return p_jacobian, q_jacobian, power


def _rectangular_current_block(operator, voltage, variable_nodes):
    """Return current values and rectangular-coordinate derivative blocks."""
    operator = csr_matrix(operator)
    voltage = np.asarray(voltage, dtype=complex).reshape(-1)
    variable_nodes = np.asarray(variable_nodes, dtype=int)
    current = np.asarray(operator @ voltage).reshape(-1)
    conductance = operator.real.tocsr()
    susceptance = operator.imag.tocsr()
    dIr = hstack(
        [conductance[:, variable_nodes], -susceptance[:, variable_nodes]], format="csr"
    )
    dIi = hstack(
        [susceptance[:, variable_nodes], conductance[:, variable_nodes]], format="csr"
    )
    return dIr, dIi, current


def _Jacobian_SE_rectangular(Ybus, Yf, Yt, voltage, f, t, inputs, variable_nodes):
    """Sparse measurement Jacobian for ``[Vr, Vi]`` variable states."""
    voltage = np.asarray(voltage, dtype=complex).reshape(-1)
    variable_nodes = np.asarray(variable_nodes, dtype=int)
    state_count = 2 * len(variable_nodes)
    variable_lookup = {int(node): column for column, node in enumerate(variable_nodes)}
    blocks = []
    predictions = []

    def add_flow_power(records, component):
        if not records:
            return
        operator, terminals = _selected_branch_matrix(Yf, Yt, f, t, records)
        p_jacobian, q_jacobian, power = _rectangular_power_block(
            operator, terminals, voltage, variable_nodes
        )
        blocks.append(p_jacobian if component == "p" else q_jacobian)
        predictions.append(power.real if component == "p" else power.imag)

    add_flow_power(inputs._p_flow, "p")

    if inputs._p_inj:
        nodes = inputs.p_inj_idx
        p_jacobian, _, power = _rectangular_power_block(
            Ybus[nodes, :], nodes, voltage, variable_nodes
        )
        blocks.append(p_jacobian)
        predictions.append(power.real)

    add_flow_power(inputs._q_flow, "q")

    if inputs._q_inj:
        nodes = inputs.q_inj_idx
        _, q_jacobian, power = _rectangular_power_block(
            Ybus[nodes, :], nodes, voltage, variable_nodes
        )
        blocks.append(q_jacobian)
        predictions.append(power.imag)

    def add_current(records, component):
        if not records:
            return
        operator, _ = _selected_branch_matrix(Yf, Yt, f, t, records)
        dIr, dIi, current = _rectangular_current_block(operator, voltage, variable_nodes)
        magnitude = np.abs(current)
        if component == "magnitude":
            inverse_magnitude = np.zeros_like(magnitude)
            nonzero = magnitude > 1e-12
            inverse_magnitude[nonzero] = 1.0 / magnitude[nonzero]
            jacobian = diags(current.real * inverse_magnitude) @ dIr + diags(
                current.imag * inverse_magnitude
            ) @ dIi
            values = magnitude
        elif component == "angle":
            inverse_squared = np.zeros_like(magnitude)
            nonzero = magnitude > 1e-12
            inverse_squared[nonzero] = 1.0 / magnitude[nonzero] ** 2
            jacobian = diags(inverse_squared) @ (
                diags(current.real) @ dIi - diags(current.imag) @ dIr
            )
            values = np.angle(current)
        elif component == "real":
            jacobian, values = dIr, current.real
        else:
            jacobian, values = dIi, current.imag
        blocks.append(jacobian.tocsr())
        predictions.append(values)

    add_current(inputs._i_flow, "magnitude")

    def add_voltage(records, component):
        if not records:
            return
        row_indices = []
        column_indices = []
        data = []
        values = np.empty(len(records), dtype=float)
        for row, (node, _) in enumerate(records):
            node = int(node)
            value = voltage[node]
            magnitude = abs(value)
            column = variable_lookup.get(node)
            if component == "magnitude":
                values[row] = magnitude
                if column is not None and magnitude > 1e-12:
                    row_indices.extend((row, row))
                    column_indices.extend((column, len(variable_nodes) + column))
                    data.extend((value.real / magnitude, value.imag / magnitude))
            elif component == "angle":
                values[row] = np.angle(value)
                if column is not None and magnitude > 1e-12:
                    row_indices.extend((row, row))
                    column_indices.extend((column, len(variable_nodes) + column))
                    data.extend((-value.imag / magnitude**2, value.real / magnitude**2))
            elif component == "real":
                values[row] = value.real
                if column is not None:
                    row_indices.append(row)
                    column_indices.append(column)
                    data.append(1.0)
            else:
                values[row] = value.imag
                if column is not None:
                    row_indices.append(row)
                    column_indices.append(len(variable_nodes) + column)
                    data.append(1.0)
        blocks.append(
            csr_matrix((data, (row_indices, column_indices)), shape=(len(records), state_count))
        )
        predictions.append(values)

    add_voltage(inputs._vm, "magnitude")
    add_voltage(inputs._va, "angle")
    add_voltage(inputs._v_real, "real")
    add_voltage(inputs._v_imag, "imag")
    add_current(inputs._i_angle, "angle")
    add_current(inputs._i_real, "real")
    add_current(inputs._i_imag, "imag")

    if not blocks:
        inputs.consolidate()
    return vstack(blocks, format="csr"), np.concatenate(predictions)


@dataclass
class _SolverDiagnostics:
    iterations: int
    error: float
    objective: float
    prediction: np.ndarray
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    jacobian: csr_matrix
    observability: ObservabilityReport


def _check_observability(jacobian, state_count):
    measurement_count = int(jacobian.shape[0])
    rank = int(structural_rank(jacobian))
    zero_columns: list[int] = []
    numeric_rank: int | None = None

    if measurement_count < state_count:
        raise UnobservableNetworkError(
            f"Unobservable state: {measurement_count} measurements for {state_count} states.",
            measurement_count=measurement_count,
            state_count=state_count,
            structural_rank=rank,
        )
    if rank < state_count:
        raise UnobservableNetworkError(
            f"Unobservable state: measurement Jacobian structural rank {rank} is below {state_count}.",
            measurement_count=measurement_count,
            state_count=state_count,
            structural_rank=rank,
        )
    if state_count <= 300:
        numeric_rank = int(np.linalg.matrix_rank(jacobian.toarray()))
        if numeric_rank < state_count:
            raise UnobservableNetworkError(
                f"Unobservable state: measurement Jacobian rank {numeric_rank} is below {state_count}.",
                measurement_count=measurement_count,
                state_count=state_count,
                structural_rank=rank,
                numeric_rank=numeric_rank,
            )
        return ObservabilityReport(measurement_count, state_count, rank, numeric_rank)

    column_norm = np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).reshape(-1))
    zero_columns = np.flatnonzero(column_norm <= np.finfo(float).eps).astype(int).tolist()
    if zero_columns:
        raise UnobservableNetworkError(
            "Unobservable state: measurement Jacobian has a zero column.",
            measurement_count=measurement_count,
            state_count=state_count,
            structural_rank=rank,
            zero_columns=zero_columns,
        )
    scaled = (jacobian @ diags(1.0 / column_norm)).tocsr()
    gain = (scaled.T @ scaled).tocsc()
    try:
        factor = splu(gain)
    except RuntimeError as exc:
        raise UnobservableNetworkError(
            "Unobservable state: measurement Jacobian is numerically rank deficient.",
            measurement_count=measurement_count,
            state_count=state_count,
            structural_rank=rank,
        ) from exc
    pivots = np.abs(factor.U.diagonal())
    pivot_limit = np.finfo(float).eps * max(gain.shape) * max(float(pivots.max()), 1.0)
    if np.any(pivots <= pivot_limit):
        raise UnobservableNetworkError(
            "Unobservable state: measurement Jacobian is numerically rank deficient.",
            measurement_count=measurement_count,
            state_count=state_count,
            structural_rank=rank,
        )
    return ObservabilityReport(measurement_count, state_count, rank, numeric_rank, tuple(zero_columns))


def _measurement_residual(inputs, observed, predicted):
    """Return measurement residuals with circular quantities wrapped."""
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    ordered = inputs.ordered_measurements()
    if len(residual) != len(ordered):
        raise StateEstimationError("Measurement and prediction row ordering are inconsistent.")
    angular = [
        position
        for position, item in enumerate(ordered)
        if item.kind in {"va", "i_angle"}
    ]
    if angular:
        residual[angular] = (residual[angular] + np.pi) % (2.0 * np.pi) - np.pi
    return residual


def _normalized_residuals(jacobian, residual, sigma, *, exact_limit=2500):
    """Return residuals normalized by their post-fit covariance.

    Small and medium sets use the exact leverage diagonal. Large sets use a
    deterministic Hutchinson estimate so Data Port-scale bad-data rejection
    remains sparse and bounded in memory.
    """
    whitened_residual = np.asarray(residual, dtype=float) / np.asarray(sigma, dtype=float)
    measurement_count = jacobian.shape[0]
    whitened_jacobian = jacobian.multiply((1.0 / np.asarray(sigma))[:, None]).tocsr()
    gain = (whitened_jacobian.T @ whitened_jacobian).tocsc()
    try:
        factor = splu(gain)
    except RuntimeError:
        return whitened_residual
    if measurement_count <= exact_limit:
        covariance_floor = 1e-12
        leverage = np.empty(measurement_count, dtype=float)
        chunk_size = 128
        for start in range(0, measurement_count, chunk_size):
            stop = min(start + chunk_size, measurement_count)
            rows = whitened_jacobian[start:stop, :].toarray()
            solved = factor.solve(rows.T)
            leverage[start:stop] = np.sum(rows * solved.T, axis=1)
    else:
        probe_count = 64
        covariance_floor = 1.0 / probe_count
        rng = np.random.default_rng(0)
        probes = rng.choice((-1.0, 1.0), size=(measurement_count, probe_count))
        solved = factor.solve(np.asarray(whitened_jacobian.T @ probes))
        projected = np.asarray(whitened_jacobian @ solved)
        leverage = np.mean(probes * projected, axis=1)
    covariance_diagonal = np.maximum(
        1.0 - np.clip(leverage, 0.0, 1.0),
        covariance_floor,
    )
    return whitened_residual / np.sqrt(covariance_diagonal)


def _solve_se_rectangular(
    Ybus,
    Yf,
    Yt,
    f,
    t,
    se_input,
    variable_nodes,
    V0,
    *,
    tol=1e-8,
    max_iter=50,
    objective_tol=1e-10,
    initial_damping=1e-3,
):
    """Safeguarded sparse LM solve in rectangular voltage coordinates."""
    variable_nodes = np.asarray(variable_nodes, dtype=int)
    voltage = np.asarray(V0, dtype=complex).reshape(-1).copy()
    z, sigma = se_input.consolidate()
    jacobian, prediction = _Jacobian_SE_rectangular(
        Ybus, Yf, Yt, voltage, f, t, se_input, variable_nodes
    )
    state_count = 2 * len(variable_nodes)
    observability = _check_observability(jacobian, state_count)
    inv_sigma = 1.0 / sigma
    residual = _measurement_residual(se_input, z, prediction)
    objective = 0.5 * float(np.dot(residual * inv_sigma, residual * inv_sigma))
    damping = initial_damping
    converged = False
    error = np.inf
    iterations = 0

    for iteration in range(1, max_iter + 1):
        whitened_jacobian = jacobian.multiply(inv_sigma[:, None]).tocsr()
        whitened_residual = residual * inv_sigma
        gain = (whitened_jacobian.T @ whitened_jacobian).tocsc()
        rhs = np.asarray(whitened_jacobian.T @ whitened_residual).reshape(-1)
        gain_diagonal = np.maximum(np.abs(gain.diagonal()), 1e-12)
        accepted = False
        for _ in range(14):
            matrix = gain + diags(damping * gain_diagonal, format="csc")
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    step = np.asarray(spsolve(matrix, rhs)).reshape(-1)
                except (MatrixRankWarning, RuntimeError) as exc:
                    raise UnobservableNetworkError(
                        "Unobservable state: singular weighted gain matrix."
                    ) from exc
            if not np.isfinite(step).all():
                raise UnobservableNetworkError("Unobservable state: non-finite WLS update.")
            candidate = voltage.copy()
            count = len(variable_nodes)
            candidate[variable_nodes] = (
                voltage[variable_nodes].real
                + step[:count]
                + 1j * (voltage[variable_nodes].imag + step[count:])
            )
            candidate_jacobian, candidate_prediction = _Jacobian_SE_rectangular(
                Ybus, Yf, Yt, candidate, f, t, se_input, variable_nodes
            )
            candidate_residual = _measurement_residual(se_input, z, candidate_prediction)
            candidate_objective = 0.5 * float(
                np.dot(candidate_residual * inv_sigma, candidate_residual * inv_sigma)
            )
            if candidate_objective <= objective * (1.0 + 1e-12):
                objective_delta = abs(objective - candidate_objective)
                voltage = candidate
                jacobian = candidate_jacobian
                prediction = candidate_prediction
                residual = candidate_residual
                objective = candidate_objective
                error = float(np.linalg.norm(step, np.inf))
                iterations = iteration
                damping = max(damping / 3.0, 1e-15)
                accepted = True
                converged = error <= tol or objective_delta <= float(objective_tol)
                break
            damping *= 10.0
        if not accepted or converged:
            break

    return voltage, converged, _SolverDiagnostics(
        iterations=iterations,
        error=float(error),
        objective=float(objective),
        prediction=np.asarray(prediction, dtype=float),
        residuals=residual,
        standardized_residuals=_normalized_residuals(jacobian, residual, sigma),
        jacobian=jacobian,
        observability=observability,
    )


def solve_se_lm(
    Ybus,
    Yf,
    Yt,
    f,
    t,
    se_input,
    ref,
    pq,
    pv,
    tol=1e-8,
    max_iter=50,
    V0=None,
    fixed_voltage=None,
    objective_tol=1e-10,
    initial_damping=1e-3,
    return_diagnostics=False,
):
    """Solve sparse nonlinear WLS with a safeguarded LM iteration."""
    Ybus = csr_matrix(Ybus)
    Yf = csr_matrix(Yf)
    Yt = csr_matrix(Yt)
    n = Ybus.shape[0]
    angle_nodes = np.unique(np.r_[np.asarray(pv, dtype=int), np.asarray(pq, dtype=int)])
    if fixed_voltage is None:
        magnitude_nodes = np.arange(n, dtype=int)
    else:
        fixed_voltage = np.asarray(fixed_voltage, dtype=int)
        magnitude_nodes = np.setdiff1d(np.arange(n, dtype=int), fixed_voltage, assume_unique=False)
    state_count = len(angle_nodes) + len(magnitude_nodes)
    if state_count == 0:
        raise ValueError("No variable voltage states were supplied.")

    V = np.ones(n, dtype=complex) if V0 is None else np.asarray(V0, dtype=complex).reshape(-1).copy()
    if len(V) != n:
        raise ValueError(f"V0 has {len(V)} entries; expected {n}.")
    Va = np.angle(V)
    Vm = np.abs(V)
    V = Vm * np.exp(1j * Va)
    if fixed_voltage is not None and V0 is not None:
        V[fixed_voltage] = np.asarray(V0, dtype=complex).reshape(-1)[fixed_voltage]

    z, sigma = se_input.consolidate()
    H, h = Jacobian_SE(Ybus, Yf, Yt, V, f, t, se_input, angle_nodes, magnitude_nodes)
    if len(h) != len(z):
        raise StateEstimationError("Measurement and Jacobian row ordering are inconsistent.")
    observability = _check_observability(H, state_count)

    inv_sigma = 1.0 / sigma
    residual = _measurement_residual(se_input, z, h)
    objective = 0.5 * float(np.dot(residual * inv_sigma, residual * inv_sigma))
    damping = None
    converged = False
    error = np.inf
    iterations = 0

    for iteration in range(1, max_iter + 1):
        Hw = H.multiply(inv_sigma[:, None]).tocsr()
        rw = residual * inv_sigma
        gain = (Hw.T @ Hw).tocsc()
        rhs = np.asarray(Hw.T @ rw).reshape(-1)
        gain_diagonal = np.maximum(np.abs(gain.diagonal()), 1e-12)
        if damping is None:
            damping = initial_damping

        accepted = False
        trial_step = None
        for _ in range(14):
            matrix = gain + diags(damping * gain_diagonal, format="csc")
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    trial_step = np.asarray(spsolve(matrix, rhs)).reshape(-1)
                except (MatrixRankWarning, RuntimeError) as exc:
                    raise UnobservableNetworkError("Unobservable state: singular weighted gain matrix.") from exc
            if not np.isfinite(trial_step).all():
                raise UnobservableNetworkError("Unobservable state: non-finite WLS update.")

            candidate_va = Va.copy()
            candidate_vm = Vm.copy()
            candidate_va[angle_nodes] += trial_step[: len(angle_nodes)]
            candidate_vm[magnitude_nodes] += trial_step[len(angle_nodes) :]
            if np.any(candidate_vm[magnitude_nodes] <= 0):
                damping *= 10.0
                continue
            candidate = candidate_vm * np.exp(1j * candidate_va)
            if fixed_voltage is not None:
                candidate[fixed_voltage] = V[fixed_voltage]
            candidate_H, candidate_h = Jacobian_SE(
                Ybus, Yf, Yt, candidate, f, t, se_input, angle_nodes, magnitude_nodes
            )
            candidate_residual = _measurement_residual(se_input, z, candidate_h)
            candidate_objective = 0.5 * float(
                np.dot(candidate_residual * inv_sigma, candidate_residual * inv_sigma)
            )
            if candidate_objective <= objective * (1.0 + 1e-12):
                Va, Vm, V = candidate_va, candidate_vm, candidate
                H, h, residual = candidate_H, candidate_h, candidate_residual
                objective = candidate_objective
                damping = max(damping / 3.0, 1e-15)
                error = float(np.linalg.norm(trial_step, np.inf))
                accepted = True
                iterations = iteration
                converged = error <= tol
                break
            damping *= 10.0

        if not accepted:
            break
        if converged:
            break

    standardized = _normalized_residuals(H, residual, sigma)
    diagnostics = _SolverDiagnostics(
        iterations=iterations,
        error=float(error),
        objective=float(objective),
        prediction=np.asarray(h, dtype=float),
        residuals=residual,
        standardized_residuals=standardized,
        jacobian=H,
        observability=observability,
    )
    if return_diagnostics:
        return V, converged, diagnostics
    return V, float(error), converged


@dataclass
class NetworkStateEstimationModel:
    Ybus: csr_matrix
    Yf: csr_matrix
    Yt: csr_matrix
    f: np.ndarray
    t: np.ndarray
    fixed_nodes: np.ndarray
    variable_nodes: np.ndarray
    initial_voltage: np.ndarray
    line_row_count: int
    line_rows: dict[tuple[int, int], int] = field(default_factory=dict)
    trafo_rows: dict[tuple[int, int], int] = field(default_factory=dict)
    source_rows: dict[tuple[int, int], int] = field(default_factory=dict)
    switch_rows: dict[tuple[int, int], int] = field(default_factory=dict)


def _ensure_native_model(net, rebuild=False):
    required = {"Y_tot", "E0", "y_fix", "y_nonslack", "line_info"}
    model = getattr(net, "model", None)
    if not rebuild and model is not None and required.issubset(model.keys()):
        return model
    from .cci_powerflow import _init_pf, correct_trafo_vnkv
    from .model import _initialize_model

    correct_trafo_vnkv(net)
    _initialize_model(net)
    _init_pf(net)
    return net.model


def extract_se_matrices(net, rebuild=False):
    """Extract sparse WLS matrices from the native reduced terminal model."""
    model = _ensure_native_model(net, rebuild=rebuild)
    node_count = int(model.y_size)
    rows_f = []
    cols_f = []
    data_f = []
    rows_t = []
    cols_t = []
    data_t = []
    f_nodes = []
    t_nodes = []
    line_rows = {}
    trafo_rows = {}
    source_rows = {}
    switch_rows = {}
    output_row = 0

    for line_index, y_send_rec, primitive, active_circuits in zip(
        model.line_info["indices"],
        model.line_info["y_send_rec"],
        model.line_info["Yprimitive"],
        model.line_info["active_circuits"],
    ):
        y_send_rec = np.asarray(y_send_rec, dtype=int)
        primitive = np.asarray(primitive, dtype=complex)
        conductor_count = len(y_send_rec) // 2
        for local in range(conductor_count):
            circuit = int(active_circuits[local]) if local < len(active_circuits) else local
            line_rows[(int(line_index), circuit)] = output_row
            _append_branch_operator_row(
                rows_f,
                cols_f,
                data_f,
                rows_t,
                cols_t,
                data_t,
                f_nodes,
                t_nodes,
                output_row,
                primitive,
                y_send_rec,
                local,
                conductor_count + local,
            )
            output_row += 1

    line_row_count = output_row

    for (trafo_index, circuit), circuit_data in getattr(model, "trafo_circuits", {}).items():
        operator = np.asarray(circuit_data["Ybr"], dtype=complex)
        local_nodes = np.asarray(circuit_data["y_send_rec"], dtype=int)
        if operator.shape[0] < 4 or len(local_nodes) < 4:
            continue
        trafo_rows[(int(trafo_index), int(circuit))] = output_row
        _append_branch_operator_row(
            rows_f,
            cols_f,
            data_f,
            rows_t,
            cols_t,
            data_t,
            f_nodes,
            t_nodes,
            output_row,
            operator,
            local_nodes,
            0,
            2,
        )
        output_row += 1

    source_kind, active_source = _active_source_rows(net)
    if active_source is not None and len(active_source):
        sbase = float(net.sn_mva) * 1e6
        slack_vn = float(model.terminal_vn_kv[model.terminal_is_slack][0])
        zbase = (slack_vn * 1e3 / np.sqrt(3)) ** 2 / sbase
        if source_kind == "ext_grid_sequence":
            alpha = np.exp(1j * 2.0 * np.pi / 3.0)
            transform = np.array(
                [[1.0, 1.0, 1.0], [1.0, alpha**2, alpha], [1.0, alpha, alpha**2]],
                dtype=complex,
            )
            sequence_inverse = np.linalg.inv(transform)
        for source_index, source_group in active_source.groupby(level=0, sort=False):
            source_group = source_group.sort_index()
            zdata = (
                source_group["r_ohm"].to_numpy(dtype=np.float64)
                + 1j * source_group["x_ohm"].to_numpy(dtype=np.float64)
            )
            if not np.all(np.isfinite(zdata)) or np.any(np.abs(zdata) <= 0.0):
                message = (
                    "ext_grid_sequence requires finite, non-zero source impedance."
                    if source_kind == "ext_grid_sequence"
                    else "Active ext_grid rows require finite, non-zero source impedance."
                )
                raise InvalidStudyInput(message)
            if source_kind == "ext_grid_sequence":
                ysq = np.linalg.inv(np.diag(zdata))
                yph = transform @ ysq @ sequence_inverse
            else:
                yph = np.linalg.inv(np.diag(zdata))
            bus_id = int(source_group["bus"].iloc[0])
            phases = source_group["from_phase"].astype(int).to_numpy()
            grid_terms = np.asarray(
                [_bus_y_index(net, bus_id, int(phase)) for phase in phases],
                dtype=int,
            )
            source_terminals = np.asarray(
                [
                    int(
                        model.terminal_to_y_lookup[
                            int(model.ext_grid_index_start + int(source_index) * 4 + int(circuit))
                        ]
                    )
                    for _, circuit in source_group.index
                ],
                dtype=int,
            )
            operator = np.block([[yph, -yph], [-yph, yph]]) * zbase
            local_nodes = np.concatenate([grid_terms, source_terminals])
            row_count = len(source_group)
            for local, (_, circuit) in enumerate(source_group.index):
                source_rows[(int(source_index), int(circuit))] = output_row
                _append_branch_operator_row(
                    rows_f,
                    cols_f,
                    data_f,
                    rows_t,
                    cols_t,
                    data_t,
                    f_nodes,
                    t_nodes,
                    output_row,
                    operator,
                    local_nodes,
                    local,
                    row_count + local,
                )
                output_row += 1

    switch_table = getattr(net, "switch", None)
    if switch_table is not None and len(switch_table):
        sbase = float(net.sn_mva) * 1e6
        for (switch_index, circuit), row in switch_table.iterrows():
            if not bool(row.get("closed", False)):
                continue
            if str(row.get("et", "")).strip().lower() != "b":
                continue
            resistance = float(row.get("r_ohm", 0.0))
            if not np.isfinite(resistance) or resistance <= 0.0:
                continue
            phase = _integer_identifier(row["phase"], "Switch phase")
            bus = _integer_identifier(row["bus"], "Switch bus")
            element = _integer_identifier(row["element"], "Switch element")
            try:
                bus_node = _bus_y_index(net, bus, phase)
                element_node = _bus_y_index(net, element, phase)
            except ValueError:
                continue
            bus_terminal = bus * 4 + phase
            vn_kv = float(model.terminal_vn_kv[bus_terminal])
            zbase = (vn_kv * 1e3 / np.sqrt(3)) ** 2 / sbase
            admittance = zbase / resistance
            operator = np.array([[admittance, -admittance], [-admittance, admittance]], dtype=complex)
            local_nodes = np.asarray([bus_node, element_node], dtype=int)
            switch_rows[(int(switch_index), int(circuit))] = output_row
            _append_branch_operator_row(
                rows_f,
                cols_f,
                data_f,
                rows_t,
                cols_t,
                data_t,
                f_nodes,
                t_nodes,
                output_row,
                operator,
                local_nodes,
                0,
                1,
            )
            output_row += 1

    shape = (output_row, node_count)
    Yf = csr_matrix((data_f, (rows_f, cols_f)), shape=shape, dtype=complex)
    Yt = csr_matrix((data_t, (rows_t, cols_t)), shape=shape, dtype=complex)
    initial = np.asarray(model.E0, dtype=complex).reshape(-1).copy()
    fixed = np.asarray(model.y_fix, dtype=int)
    fixed_values = np.asarray(model.y_fixed_voltage, dtype=complex).reshape(-1)
    initial[fixed] = fixed_values[fixed]
    return NetworkStateEstimationModel(
        Ybus=csr_matrix(model.Y_tot),
        Yf=Yf,
        Yt=Yt,
        f=np.asarray(f_nodes, dtype=int),
        t=np.asarray(t_nodes, dtype=int),
        fixed_nodes=fixed,
        variable_nodes=np.asarray(model.y_nonslack, dtype=int),
        initial_voltage=initial,
        line_row_count=line_row_count,
        line_rows=line_rows,
        trafo_rows=trafo_rows,
        source_rows=source_rows,
        switch_rows=switch_rows,
    )


def _integer_identifier(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer identifier.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} must be an integer identifier.")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer identifier.") from exc
    elif isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
    else:
        raise ValueError(f"{label} must be an integer identifier.")
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer identifier.")
    return int(numeric)


def _active_source_rows(net):
    ext_grid = (
        net["ext_grid"].loc[net["ext_grid"]["in_service"].astype(bool)]
        if len(net["ext_grid"])
        else net["ext_grid"]
    )
    ext_grid_sequence = (
        net["ext_grid_sequence"].loc[net["ext_grid_sequence"]["in_service"].astype(bool)]
        if len(net["ext_grid_sequence"])
        else net["ext_grid_sequence"]
    )
    if len(ext_grid) and len(ext_grid_sequence):
        raise InvalidStudyInput("Only one active external-grid table is supported per power-flow solve.")
    if len(ext_grid_sequence) and len(ext_grid_sequence) != 3:
        raise InvalidStudyInput("Active ext_grid_sequence rows must contain all three sequence components.")
    if len(ext_grid):
        return "ext_grid", ext_grid.sort_index()
    if len(ext_grid_sequence):
        return "ext_grid_sequence", ext_grid_sequence.sort_index()
    return None, None


def _append_branch_operator_row(
    rows_f,
    cols_f,
    data_f,
    rows_t,
    cols_t,
    data_t,
    f_nodes,
    t_nodes,
    row_index,
    operator,
    local_nodes,
    from_row,
    to_row,
):
    operator = np.asarray(operator, dtype=complex)
    local_nodes = np.asarray(local_nodes, dtype=int)
    for column, node in enumerate(local_nodes):
        from_value = operator[from_row, column]
        to_value = operator[to_row, column]
        if from_value != 0:
            rows_f.append(row_index)
            cols_f.append(int(node))
            data_f.append(from_value)
        if to_value != 0:
            rows_t.append(row_index)
            cols_t.append(int(node))
            data_t.append(to_value)
    f_nodes.append(int(local_nodes[from_row]))
    t_nodes.append(int(local_nodes[to_row]))


def _source_circuit_from_phase(frame, source_index, phase):
    rows = frame.xs(source_index, level=0)
    matches = rows.loc[rows["from_phase"].astype(int) == int(phase)]
    if len(matches) != 1:
        raise InvalidStudyInput(
            f"Source {source_index} phase {phase} does not resolve to exactly one circuit."
        )
    return int(matches.index[0])


def _resolve_source_branch_row(net, model, element, circuit, phase):
    kind, frame = _active_source_rows(net)
    if frame is None or len(frame) == 0:
        raise InvalidStudyInput("No active ext_grid rows are available.")
    source_index = _integer_identifier(element, "Source element")
    if source_index not in set(frame.index.get_level_values(0).astype(int)):
        raise InvalidStudyInput(f"Source {source_index} does not exist.")
    if circuit is None and phase is None:
        raise InvalidStudyInput("Source measurements require a phase or circuit.")
    resolved_circuit = (
        _source_circuit_from_phase(frame, source_index, phase)
        if circuit is None
        else _integer_identifier(circuit, "Source circuit")
    )
    if phase is not None:
        phase_rows = frame.xs(source_index, level=0)
        row = phase_rows.loc[resolved_circuit]
        if _integer_identifier(row["from_phase"], "Source phase") != int(phase):
            raise InvalidStudyInput(
                f"Source {source_index} circuit {resolved_circuit} does not match phase {phase}."
            )
    if (source_index, resolved_circuit) not in model.source_rows:
        if kind == "ext_grid_sequence":
            raise InvalidStudyInput(
                f"Active ext_grid_sequence row {(source_index, resolved_circuit)!r} is not an observable source branch."
            )
        raise InvalidStudyInput(f"Unknown source/circuit measurement target {(source_index, resolved_circuit)!r}.")
    return resolved_circuit, model.source_rows[(source_index, resolved_circuit)]


def _resolve_trafo_branch_row(net, model, element, circuit):
    trafo_index = _integer_identifier(element, "Transformer element")
    if circuit is None:
        raise InvalidStudyInput("Transformer measurements require a circuit.")
    resolved_circuit = _integer_identifier(circuit, "Transformer circuit")
    trafo_rows = net.trafo1ph
    matches = (
        trafo_rows.index.get_level_values("index").astype(int) == trafo_index
    ) & (
        trafo_rows.index.get_level_values("circuit").astype(int) == resolved_circuit
    )
    if not np.any(matches):
        raise InvalidStudyInput(
            f"Unknown transformer/circuit measurement target {(trafo_index, resolved_circuit)!r}."
        )
    if (trafo_index, resolved_circuit) not in model.trafo_rows:
        raise InvalidStudyInput(
            f"Transformer {(trafo_index, resolved_circuit)!r} is not an active observable branch."
        )
    return resolved_circuit, model.trafo_rows[(trafo_index, resolved_circuit)]


def _resolve_line_branch_row(model, element, circuit):
    line_index = _integer_identifier(element, "Line element")
    if circuit is None:
        rows = [row_index for (line, _), row_index in model.line_rows.items() if line == line_index]
        if len(rows) != 1:
            raise InvalidStudyInput(f"Line {line_index} measurements require a circuit.")
        return None, rows[0]
    resolved_circuit = _integer_identifier(circuit, "Line circuit")
    try:
        return resolved_circuit, model.line_rows[(line_index, resolved_circuit)]
    except KeyError as exc:
        raise InvalidStudyInput(
            f"Unknown line/circuit measurement target {(line_index, resolved_circuit)!r}."
        ) from exc


def _resolve_switch_branch_row(net, model, element, circuit, phase):
    switch_index = _integer_identifier(element, "Switch element")
    switch_rows = net.switch
    first_level = set(switch_rows.index.get_level_values(0).astype(int))
    if switch_index not in first_level:
        raise InvalidStudyInput(f"Switch {switch_index} does not exist.")
    if circuit is None and phase is None:
        raise InvalidStudyInput("Switch measurements require a phase or circuit.")
    resolved_circuit = (
        _integer_identifier(phase, "Switch phase/circuit")
        if circuit is None
        else _integer_identifier(circuit, "Switch circuit")
    )
    try:
        switch_row = switch_rows.loc[(switch_index, resolved_circuit)]
    except KeyError as exc:
        raise InvalidStudyInput(
            f"Unknown switch/circuit measurement target {(switch_index, resolved_circuit)!r}."
        ) from exc

    et = str(switch_row["et"]).strip().lower()
    if et == "b":
        if not bool(switch_row.get("closed", False)):
            raise InvalidStudyInput("Open bus-bus switches are not measurable state-estimation branches.")
        if float(switch_row.get("r_ohm", 0.0)) <= 0.0:
            raise InvalidStudyInput(
                "Ideal closed bus-bus switches are merged electrical nodes; branch flow measurements are unsupported."
            )
        if (switch_index, resolved_circuit) not in model.switch_rows:
            raise InvalidStudyInput(
                f"Switch {(switch_index, resolved_circuit)!r} is not an active observable branch."
            )
        return resolved_circuit, model.switch_rows[(switch_index, resolved_circuit)], None

    if not bool(switch_row.get("closed", False)):
        raise InvalidStudyInput("Open bus-element switches are not measurable state-estimation branches.")

    bus = _integer_identifier(switch_row["bus"], "Switch bus")
    phase_number = _integer_identifier(switch_row["phase"], "Switch phase")
    target_element = _integer_identifier(switch_row["element"], "Switch element target")
    if et == "l":
        line_matches = net.line.loc[target_element]
        if isinstance(line_matches, pd.Series):
            line_matches = line_matches.to_frame().T
            line_matches.index = pd.Index([resolved_circuit], name="circuit")
        for candidate_circuit, row in line_matches.iterrows():
            if int(row["from_bus"]) == bus and int(row["from_phase"]) == phase_number:
                branch_row = model.line_rows.get((target_element, int(candidate_circuit)))
                if branch_row is not None:
                    return int(candidate_circuit), branch_row, "from"
            if int(row["to_bus"]) == bus and int(row["to_phase"]) == phase_number:
                branch_row = model.line_rows.get((target_element, int(candidate_circuit)))
                if branch_row is not None:
                    return int(candidate_circuit), branch_row, "to"
        raise InvalidStudyInput(
            f"Switch {(switch_index, resolved_circuit)!r} does not resolve to an active line terminal."
        )
    if et == "t":
        trafo_candidates = net.trafo1ph
        matches = trafo_candidates.index.get_level_values("index").astype(int) == target_element
        matches &= trafo_candidates.index.get_level_values("circuit").astype(int) == resolved_circuit
        matches &= trafo_candidates.index.get_level_values("bus").astype(int) == bus
        if not np.any(matches):
            raise InvalidStudyInput(
                f"Switch {(switch_index, resolved_circuit)!r} does not resolve to an active transformer terminal."
            )
        branch_row = model.trafo_rows.get((target_element, resolved_circuit))
        if branch_row is None:
            raise InvalidStudyInput(
                f"Transformer {(target_element, resolved_circuit)!r} is not an active observable branch."
            )
        buses = list(getattr(net.model, "trafo_circuits", {}).get((target_element, resolved_circuit), {}).get("buses", ()))
        if len(buses) < 2:
            raise InvalidStudyInput(
                f"Transformer {(target_element, resolved_circuit)!r} does not expose both measurable sides."
            )
        if bus == int(buses[0]):
            side = "from"
        elif bus == int(buses[1]):
            side = "to"
        else:
            raise InvalidStudyInput(
                f"Switch {(switch_index, resolved_circuit)!r} bus {bus} does not match transformer {(target_element, resolved_circuit)!r}."
            )
        return resolved_circuit, branch_row, side
    raise InvalidStudyInput(f"Unsupported switch measurement element type {et!r}.")


def _validated_bus_phase(net, bus, phase):
    bus_number = _integer_identifier(bus, "Bus")
    phase_number = _integer_identifier(phase, "Bus phase")
    if phase_number not in {0, 1, 2, 3}:
        raise ValueError("Bus phase must be one of 0 (neutral), 1 (A), 2 (B), or 3 (C).")
    bus_index = net.bus.index
    if isinstance(bus_index, pd.MultiIndex):
        if (bus_number, phase_number) not in bus_index:
            raise ValueError(f"Bus {bus_number} phase {phase_number} does not exist.")
    elif bus_number not in bus_index:
        raise ValueError(f"Bus {bus_number} does not exist.")
    return bus_number, phase_number


def _bus_y_index(net, bus, phase):
    bus, phase = _validated_bus_phase(net, bus, phase)
    terminal = bus * 4 + phase
    lookup = np.asarray(net.model.terminal_to_y_lookup, dtype=int)
    if terminal < 0 or terminal >= len(lookup) or lookup[terminal] < 0:
        raise ValueError(f"Bus {bus} phase {phase} is not an active electrical node.")
    return int(lookup[terminal])


def create_state_measurement(
    net,
    measurement_type,
    element_type,
    element,
    value,
    std_dev,
    *,
    phase=None,
    circuit=None,
    side=None,
    name=None,
    index=None,
    terminal=None,
    units=None,
    status=None,
):
    """Append a phase-aware row to ``net.measurement``.

    ``phase`` addresses a bus conductor (0=N, 1=A, 2=B, 3=C). ``circuit``
    addresses a line's MultiIndex circuit. This helper complements the older
    generic ``create_measurement`` API, whose schema has no phase identity.
    """
    element_type = _canonical_measurement_element_type(element_type)
    if element_type in {"bus", "zero_injection"} and phase is None:
        raise ValueError("Bus state-estimation measurements require phase.")
    if element_type in {"line", "trafo1ph", "ext_grid", "switch"} and circuit is None:
        raise ValueError(f"{element_type} state-estimation measurements require circuit.")
    if element_type not in {"bus", "line", "trafo1ph", "ext_grid", "switch", "zero_injection"}:
        raise ValueError(f"Unsupported state-estimation element type {element_type!r}.")
    if not np.isfinite(value):
        raise ValueError("Measurement value must be finite.")
    if not np.isfinite(std_dev) or float(std_dev) <= 0:
        raise ValueError("Measurement standard deviation must be finite and greater than zero.")
    element = _integer_identifier(element, "Element")
    if element_type in {"bus", "zero_injection"}:
        element, phase = _validated_bus_phase(net, element, phase)
    elif element_type == "line":
        circuit = _integer_identifier(circuit, "Line circuit")
        if isinstance(net.line.index, pd.MultiIndex):
            if (element, circuit) not in net.line.index:
                raise ValueError(f"Line {element} circuit {circuit} does not exist.")
        elif element not in net.line.index:
            raise ValueError(f"Line {element} does not exist.")
    elif element_type == "trafo1ph":
        circuit = _integer_identifier(circuit, "Transformer circuit")
        matches = (
            net.trafo1ph.index.get_level_values("index").astype(int) == element
        ) & (
            net.trafo1ph.index.get_level_values("circuit").astype(int) == circuit
        )
        if not np.any(matches):
            raise ValueError(f"Transformer {element} circuit {circuit} does not exist.")
        side = _canonical_branch_side(element_type, side)
    elif element_type == "ext_grid":
        circuit = _integer_identifier(circuit, "Source circuit")
        source_kind, active_source = _active_source_rows(net)
        if active_source is None or len(active_source) == 0:
            raise InvalidStudyInput("No active ext_grid rows are available.")
        if (element, circuit) not in active_source.index:
            raise ValueError(f"Source {element} circuit {circuit} does not exist.")
        if phase is not None:
            phase = _integer_identifier(phase, "Source phase")
        side = _canonical_branch_side(element_type, side)
    elif element_type == "switch":
        circuit = _integer_identifier(circuit, "Switch circuit")
        if (element, circuit) not in net.switch.index:
            raise ValueError(f"Switch {element} circuit {circuit} does not exist.")
        if phase is None:
            phase = _integer_identifier(net.switch.at[(element, circuit), "phase"], "Switch phase")
        else:
            phase = _integer_identifier(phase, "Switch phase")
        side = _canonical_branch_side(element_type, side)

    columns = [
        "name",
        "measurement_type",
        "element_type",
        "element",
        "value",
        "std_dev",
        "side",
        "phase",
        "circuit",
        "terminal",
        "units",
        "status",
    ]
    if "measurement" not in net or net.measurement is None:
        net["measurement"] = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in net.measurement.columns:
                net.measurement[column] = np.nan
    for column in ("name", "measurement_type", "element_type", "side", "terminal", "units", "status"):
        if column in net.measurement.columns:
            net.measurement[column] = net.measurement[column].astype(object)
    if index is None:
        numeric_indices = pd.to_numeric(pd.Index(net.measurement.index), errors="coerce")
        finite_indices = numeric_indices[np.isfinite(numeric_indices)]
        index = int(finite_indices.max()) + 1 if len(finite_indices) else 0
    if index in net.measurement.index:
        raise ValueError(f"Measurement index {index!r} already exists.")
    net.measurement.loc[index, columns] = [
        name,
        str(measurement_type).strip().lower(),
        element_type,
        element,
        float(value),
        float(std_dev),
        side,
        phase,
        circuit,
        terminal or _state_measurement_terminal(
            element_type,
            element,
            phase=phase,
            circuit=circuit,
            side=side,
        ),
        units or _MEASUREMENT_UNIT_BY_KIND.get(str(measurement_type).strip().lower()),
        status,
    ]
    return index


def measurements_from_dataframe(net, dataframe=None, model=None):
    """Convert ``net.measurement`` rows to reduced-node per-unit inputs.

    Supported types are ``p``, ``q``, ``i``, ``vm``, ``va``, ``vr``, and
    ``vi``.  Bus voltage measurements require a ``phase`` or ``circuit``
    column.  Bus P/Q without a phase is expanded only when the row itself is
    phase-specific; aggregate multi-phase injections are intentionally rejected
    because they are not a scalar nodal measurement.
    """
    model = extract_se_matrices(net) if model is None else model
    frame = getattr(net, "measurement", None) if dataframe is None else dataframe
    if frame is None or len(frame) == 0:
        raise ValueError("No measurements found in network measurement table.")
    result = StateEstimationInput()
    for source_index, row in frame.iterrows():
        kind = str(row.get("measurement_type", "")).strip().lower()
        target = _merged_row_target(row)
        element_type = target["element_type"]
        element = target["element"]
        value = float(row["value"])
        sigma = float(row["std_dev"])
        phase = target["phase"]
        circuit = target["circuit"]
        side = target["side"]
        name = row.get("name")

        if element_type in {"bus", "zero_injection"}:
            if phase is None:
                raise InvalidStudyInput(f"Bus measurement {source_index!r} requires a phase column.")
            if element_type == "zero_injection" and kind in {"p", "q"} and abs(value) > 1e-12:
                raise InvalidStudyInput(
                    f"Zero-injection measurement {source_index!r} must use a zero value."
                )
            node = _bus_y_index(net, element, phase)
            if kind == "p":
                result.add_p_inj(node, value / float(net.sn_mva), sigma / float(net.sn_mva), name, source_index)
            elif kind == "q":
                result.add_q_inj(node, value / float(net.sn_mva), sigma / float(net.sn_mva), name, source_index)
            elif kind in {"v", "vm"}:
                result.add_vm(node, value, sigma, name, source_index)
            elif kind == "va":
                result.add_va(node, np.deg2rad(value), np.deg2rad(sigma), name, source_index)
            elif kind == "vr":
                result.add_v_real(node, value, sigma, name, source_index)
            elif kind == "vi":
                result.add_v_imag(node, value, sigma, name, source_index)
            else:
                raise InvalidStudyInput(f"Unsupported bus measurement type {kind!r}.")
            continue

        implicit_side = None
        if element_type == "line":
            _, branch_row = _resolve_line_branch_row(model, element, circuit)
            internal_side = _internal_branch_side(element_type, side)
        elif element_type == "trafo1ph":
            _, branch_row = _resolve_trafo_branch_row(net, model, element, circuit)
            internal_side = _internal_branch_side(element_type, side)
        elif element_type == "ext_grid":
            _, branch_row = _resolve_source_branch_row(net, model, element, circuit, phase)
            internal_side = _internal_branch_side(element_type, side)
        elif element_type == "switch":
            _, branch_row, implicit_side = _resolve_switch_branch_row(net, model, element, circuit, phase)
            internal_side = implicit_side or _internal_branch_side(element_type, side)
        else:
            raise InvalidStudyInput(f"Unsupported state-estimation element type {element_type!r}.")
        if kind == "p":
            result.add_p_flow(branch_row, value / float(net.sn_mva), sigma / float(net.sn_mva), internal_side, name, source_index)
        elif kind == "q":
            result.add_q_flow(branch_row, value / float(net.sn_mva), sigma / float(net.sn_mva), internal_side, name, source_index)
        elif kind in {"i", "im"}:
            terminal = model.f[branch_row] if internal_side == "from" else model.t[branch_row]
            base_ka = float(net.model.Ibase_y[terminal]) / 1000.0
            result.add_i_flow(branch_row, value / base_ka, sigma / base_ka, internal_side, name, source_index)
        elif kind == "ia":
            result.add_i_angle(branch_row, np.deg2rad(value), np.deg2rad(sigma), internal_side, name, source_index)
        else:
            raise InvalidStudyInput(f"Unsupported branch measurement type {kind!r}.")
    return result


def measurements_from_power_flow(
    net,
    *,
    sigma_p=1e-3,
    sigma_q=1e-3,
    sigma_vm=5e-4,
    sigma_va=None,
    sigma_vpmu=5e-4,
    include_line_flows=False,
    noise=False,
    seed=0,
):
    """Create deterministic synthetic measurements from the current PF state.

    This helper is intended for validation and simulation.  Production state
    estimation should consume actual ``net.measurement`` data.
    """
    if not getattr(getattr(net, "model", None), "solved", False):
        raise ValueError("Run multiconductor power flow before generating synthetic measurements.")
    model = extract_se_matrices(net, rebuild=False)
    truth = np.asarray(net.model.E, dtype=complex).reshape(-1)
    injection = truth * np.conjugate(np.asarray(model.Ybus @ truth).reshape(-1))
    rng = np.random.default_rng(seed)

    def measured(value, sigma):
        return float(value + (rng.normal(0.0, sigma) if noise else 0.0))

    result = StateEstimationInput()
    for node in model.variable_nodes:
        result.add_p_inj(int(node), measured(injection[node].real, sigma_p), sigma_p, name=f"pseudo_p_{node}")
        result.add_q_inj(int(node), measured(injection[node].imag, sigma_q), sigma_q, name=f"pseudo_q_{node}")
        result.add_vm(int(node), measured(abs(truth[node]), sigma_vm), sigma_vm, name=f"vm_{node}")
        result.add_v_real(int(node), measured(truth[node].real, sigma_vpmu), sigma_vpmu, name=f"vr_{node}")
        result.add_v_imag(int(node), measured(truth[node].imag, sigma_vpmu), sigma_vpmu, name=f"vi_{node}")
        if sigma_va is not None:
            result.add_va(int(node), measured(np.angle(truth[node]), sigma_va), sigma_va, name=f"va_{node}")
    if include_line_flows:
        If = np.asarray(model.Yf @ truth).reshape(-1)
        Sf = truth[model.f] * np.conjugate(If)
        for row in range(model.line_row_count):
            result.add_p_flow(row, measured(Sf[row].real, sigma_p), sigma_p, name=f"p_line_{row}")
            result.add_q_flow(row, measured(Sf[row].imag, sigma_q), sigma_q, name=f"q_line_{row}")
    return result


@dataclass
class StateEstimationResult:
    voltage: np.ndarray
    converged: bool
    iterations: int
    error: float
    objective: float
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    bad_measurement_indices: list[object]
    bad_measurement_positions: list[int]
    fixed_nodes: np.ndarray
    measurement_count: int
    state_count: int
    observability: ObservabilityReport
    measurement_results: pd.DataFrame
    res_bus: pd.DataFrame
    res_line: pd.DataFrame
    res_trafo: pd.DataFrame

    @property
    def V_est(self):
        return self.voltage


def _estimated_result_tables(net, voltage):
    from .pf_results import _bus_results_pf, _line_results_pf, _trafo_results_pf

    saved_voltage = net.model.E
    saved_tables = {name: net.get(name) for name in ("res_bus", "res_line", "res_trafo")}
    try:
        net.model.E = np.asarray(voltage, dtype=complex).reshape(-1, 1)
        _bus_results_pf(net)
        _line_results_pf(net)
        _trafo_results_pf(net)
        return net.res_bus.copy(deep=True), net.res_line.copy(deep=True), net.res_trafo.copy(deep=True)
    finally:
        net.model.E = saved_voltage
        for name, table in saved_tables.items():
            if table is None:
                net.pop(name, None)
            else:
                net[name] = table


def _measurement_results_table(original_measurements, active_measurements, diagnostics, rejected_positions):
    original_ordered = original_measurements.ordered_measurements()
    active_lookup = {position: index for index, position in enumerate(
        original_index
        for original_index in range(len(original_ordered))
        if original_index not in set(int(pos) for pos in rejected_positions)
    )}
    rows = []
    for position, item in enumerate(original_ordered):
        active_position = active_lookup.get(position)
        accepted = active_position is not None
        rows.append(
            {
                "position": position,
                "measurement_type": item.kind,
                "target_index": item.index,
                "side": item.side or None,
                "name": item.measurement.name,
                "source_index": item.measurement.source_index,
                "observed": item.measurement.value,
                "sigma": item.measurement.sigma,
                "predicted": (
                    float(diagnostics.prediction[active_position])
                    if accepted and active_position < len(diagnostics.prediction)
                    else np.nan
                ),
                "residual": (
                    float(diagnostics.residuals[active_position])
                    if accepted and active_position < len(diagnostics.residuals)
                    else np.nan
                ),
                "standardized_residual": (
                    float(diagnostics.standardized_residuals[active_position])
                    if accepted and active_position < len(diagnostics.standardized_residuals)
                    else np.nan
                ),
                "normalized_residual": (
                    float(diagnostics.standardized_residuals[active_position])
                    if accepted and active_position < len(diagnostics.standardized_residuals)
                    else np.nan
                ),
                "accepted": accepted,
                "status": "accepted" if accepted else "rejected_bad_data",
                "rejection_reason": None if accepted else "largest_normalized_residual",
                "units": _STATE_ESTIMATION_RESULT_UNITS.get(item.kind, "pu"),
            }
        )
    return pd.DataFrame(rows)


def run_state_estimation(
    net,
    measurements=None,
    *,
    tol=1e-8,
    max_iter=50,
    objective_tol=1e-10,
    bad_data_threshold=None,
    max_bad_data=0,
    rebuild_model=False,
    raise_on_failure=False,
):
    """Estimate the full unbalanced multiconductor voltage state.

    Fixed source phasors and exact grounds are removed from the optimization and
    copied back exactly.  When ``bad_data_threshold`` is supplied, the largest
    standardized residual above that threshold is removed and the estimator is
    rerun, up to ``max_bad_data`` times.
    """
    if not np.isfinite(tol) or float(tol) <= 0:
        raise ValueError("tol must be finite and greater than zero.")
    if not np.isfinite(objective_tol) or float(objective_tol) <= 0:
        raise ValueError("objective_tol must be finite and greater than zero.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if (
        isinstance(max_bad_data, bool)
        or not isinstance(max_bad_data, (int, np.integer))
        or max_bad_data < 0
    ):
        raise ValueError("max_bad_data must be a non-negative integer.")
    if bad_data_threshold is not None and (
        not np.isfinite(bad_data_threshold) or float(bad_data_threshold) <= 0
    ):
        raise ValueError("bad_data_threshold must be finite and greater than zero.")
    model = extract_se_matrices(net, rebuild=rebuild_model)
    active = measurements_from_dataframe(net, model=model) if measurements is None else measurements
    if not isinstance(active, StateEstimationInput):
        raise TypeError("measurements must be a StateEstimationInput or None.")
    original_measurements = active
    rejected_positions = []
    diagnostics = None
    voltage = None
    converged = False

    for attempt in range(max_bad_data + 1):
        voltage, converged, diagnostics = _solve_se_rectangular(
            model.Ybus,
            model.Yf,
            model.Yt,
            model.f,
            model.t,
            active,
            variable_nodes=model.variable_nodes,
            tol=tol,
            max_iter=max_iter,
            objective_tol=objective_tol,
            V0=model.initial_voltage,
        )
        if bad_data_threshold is None or attempt >= max_bad_data:
            break
        local_index = int(np.argmax(np.abs(diagnostics.standardized_residuals)))
        if abs(diagnostics.standardized_residuals[local_index]) <= float(bad_data_threshold):
            break
        original_positions = [
            index
            for index in range(len(original_measurements.ordered_measurements()))
            if index not in rejected_positions
        ]
        rejected_positions.append(original_positions[local_index])
        active = original_measurements.without_positions(rejected_positions)

    if raise_on_failure and not converged:
        raise StateEstimationNotConverged(
            f"State estimation did not converge after {diagnostics.iterations} iterations."
        )
    res_bus, res_line, res_trafo = _estimated_result_tables(net, voltage)
    res_measurement = _measurement_results_table(
        original_measurements,
        active,
        diagnostics,
        rejected_positions,
    )
    net["res_bus_est"] = res_bus
    net["res_line_est"] = res_line
    net["res_trafo_est"] = res_trafo
    net["res_measurement_est"] = res_measurement
    measurement_count = len(active.ordered_measurements())
    state_count = 2 * len(model.variable_nodes)
    dof = max(measurement_count - state_count, 0)
    statistic = 2.0 * diagnostics.objective
    p_value = float(chi2.sf(statistic, dof)) if dof > 0 else float("nan")
    ordered_original = original_measurements.ordered_measurements()
    rejected_indices = [
        ordered_original[position].measurement.source_index
        if ordered_original[position].measurement.source_index is not None
        else position
        for position in rejected_positions
    ]
    result = StateEstimationResult(
        voltage=np.asarray(voltage, dtype=complex),
        converged=bool(converged),
        iterations=diagnostics.iterations,
        error=diagnostics.error,
        objective=diagnostics.objective,
        chi_square=statistic,
        degrees_of_freedom=dof,
        p_value=p_value,
        residuals=diagnostics.residuals,
        standardized_residuals=diagnostics.standardized_residuals,
        bad_measurement_indices=rejected_indices,
        bad_measurement_positions=rejected_positions,
        fixed_nodes=model.fixed_nodes.copy(),
        measurement_count=measurement_count,
        state_count=state_count,
        observability=diagnostics.observability,
        measurement_results=res_measurement,
        res_bus=res_bus,
        res_line=res_line,
        res_trafo=res_trafo,
    )
    net["state_estimation"] = result
    return result


def estimate(net, measurements=None, **kwargs):
    """Convenience alias matching the higher-level estimator verb."""
    return run_state_estimation(net, measurements=measurements, **kwargs)


def node_index(bus, conductor, nc):
    """Flat index helper retained for standalone synthetic examples."""
    return int(bus) * int(nc) + int(conductor)


def build_mc_Ybus(nb, nc, branch_list):
    """Build sparse matrices for small standalone multiconductor examples."""
    node_count = int(nb) * int(nc)
    rows = []
    cols = []
    values = []
    yf_rows = []
    yf_cols = []
    yf_values = []
    yt_rows = []
    yt_cols = []
    yt_values = []
    f = []
    t = []
    output_row = 0
    for branch in branch_list:
        send = np.asarray([node_index(branch["from_bus"], c, nc) for c in range(nc)], dtype=int)
        receive = np.asarray([node_index(branch["to_bus"], c, nc) for c in range(nc)], dtype=int)
        series = np.asarray(branch["Y_series"], dtype=complex)
        shunt_f = np.asarray(branch.get("Y_shunt_f", np.zeros_like(series)), dtype=complex)
        shunt_t = np.asarray(branch.get("Y_shunt_t", np.zeros_like(series)), dtype=complex)
        for i in range(nc):
            for j in range(nc):
                contributions = (
                    (send[i], send[j], series[i, j] + shunt_f[i, j]),
                    (receive[i], receive[j], series[i, j] + shunt_t[i, j]),
                    (send[i], receive[j], -series[i, j]),
                    (receive[i], send[j], -series[i, j]),
                )
                for row, column, value in contributions:
                    if value != 0:
                        rows.append(row)
                        cols.append(column)
                        values.append(value)
                from_values = ((send[j], series[i, j] + shunt_f[i, j]), (receive[j], -series[i, j]))
                to_values = ((send[j], -series[i, j]), (receive[j], series[i, j] + shunt_t[i, j]))
                for column, value in from_values:
                    if value != 0:
                        yf_rows.append(output_row + i)
                        yf_cols.append(column)
                        yf_values.append(value)
                for column, value in to_values:
                    if value != 0:
                        yt_rows.append(output_row + i)
                        yt_cols.append(column)
                        yt_values.append(value)
            f.append(send[i])
            t.append(receive[i])
        output_row += nc
    Ybus = csc_matrix((values, (rows, cols)), shape=(node_count, node_count))
    Yf = csc_matrix((yf_values, (yf_rows, yf_cols)), shape=(output_row, node_count))
    Yt = csc_matrix((yt_values, (yt_rows, yt_cols)), shape=(output_row, node_count))
    return Ybus, Yf, Yt, np.asarray(f, dtype=int), np.asarray(t, dtype=int)


__all__ = [
    "Jacobian_SE",
    "Measurement",
    "NetworkStateEstimationModel",
    "ObservabilityReport",
    "StateEstimationError",
    "StateEstimationInput",
    "StateEstimationNotConverged",
    "StateEstimationResult",
    "estimate",
    "UnobservableNetworkError",
    "build_mc_Ybus",
    "create_state_measurement",
    "dIbr_dV",
    "dSbr_dV",
    "dSbus_dV",
    "extract_se_matrices",
    "measurements_from_dataframe",
    "measurements_from_power_flow",
    "node_index",
    "run_state_estimation",
    "solve_se_lm",
]
