"""Native phase-domain short-circuit calculations.

This module operates on the multiconductor CCI nodal model.  It intentionally
does not convert a network to a balanced pandapower network: phase coupling,
phase selection, open conductors, and switch state therefore remain visible to
the calculation.

The implementation is a voltage-source/Thevenin calculation.  Passive network
impedances are taken from ``net.model.Y_tot`` and the explicit external-grid
impedance.  Explicitly rated converter current sources are added as Norton
injections.  Loads influence a Type-C calculation through the pre-fault voltage
but are not independently converted into fault-current sources.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from multiconductor.pycci.cci_powerflow import _init_pf
from multiconductor.pycci.model import (
    _initialize_model,
    get_active_ext_grid_table,
    get_ext_grid_terminal,
)
from multiconductor.shortcircuit.kappa import _kappa
from multiconductor.studies import InvalidStudyInput, StudyNotConverged


logger = logging.getLogger(__name__)


_FAULT_ALIASES = {
    "3PH": "LLL",
    "LLL": "LLL",
    "2PH": "LL",
    "LL": "LL",
    "1PH": "LG",
    "SLG": "LG",
    "LG": "LG",
    "2PH-G": "LLG",
    "2PHG": "LLG",
    "LLG": "LLG",
}
_EXPECTED_PHASE_COUNT = {"LLL": 3, "LL": 2, "LG": 1, "LLG": 2}
_DEFAULT_PHASES = {"LLL": (1, 2, 3), "LL": (1, 2), "LG": (1,), "LLG": (1, 2)}
_PHASE_NAMES = {1: "a", 2: "b", 3: "c"}


class ShortCircuitSourceError(InvalidStudyInput):
    """Raised when no finite, explicit short-circuit source is available."""


class ShortCircuitCalculationError(StudyNotConverged):
    """Raised when the phase-domain short-circuit system cannot be solved."""


def _normalize_fault(value: str) -> str:
    token = str(value).strip().upper().replace("_", "-").replace(" ", "")
    try:
        return _FAULT_ALIASES[token]
    except KeyError as exc:
        raise NotImplementedError(
            "Only LLL, LL, LG and LLG short-circuit faults are implemented"
        ) from exc


@dataclass(frozen=True)
class FaultSpec:
    """Definition of one phase-domain fault.

    Parameters are expressed in physical units.  ``phases`` uses the
    multiconductor phase labels 1, 2 and 3.  For LG and LLG faults,
    ``r_fault_ohm + j*x_fault_ohm`` is the common fault-point-to-ground
    impedance.  For LL it is the total phase-to-phase fault impedance.  For
    LLL it is the per-phase impedance to the floating common fault point.
    """

    fault: str = "LLL"
    bus: int | None = None
    line: int | None = None
    location: float | None = None
    phases: tuple[int, ...] | None = None
    case: str = "max"
    r_fault_ohm: float = 0.0
    x_fault_ohm: float = 0.0
    r_ohm: float | None = None
    x_ohm: float | None = None
    ip: bool = False
    ith: bool = False
    tk_s: float = 1.0
    use_pre_fault_voltage: bool = False
    voltage_factor: float | None = None

    def __post_init__(self) -> None:
        fault = _normalize_fault(self.fault)
        case = str(self.case).lower()
        if case not in {"max", "min"}:
            raise ValueError('case must be either "max" or "min"')

        phases = self.phases if self.phases is not None else _DEFAULT_PHASES[fault]
        phases = tuple(int(phase) for phase in phases)
        if len(phases) != _EXPECTED_PHASE_COUNT[fault]:
            raise ValueError(
                f"{fault} requires {_EXPECTED_PHASE_COUNT[fault]} selected phase(s), "
                f"got {phases}"
            )
        if len(set(phases)) != len(phases) or any(phase not in (1, 2, 3) for phase in phases):
            raise ValueError("fault phases must be distinct phase labels chosen from 1, 2 and 3")
        r_fault_ohm = (
            float(self.r_fault_ohm) if self.r_ohm is None else float(self.r_ohm)
        )
        x_fault_ohm = (
            float(self.x_fault_ohm) if self.x_ohm is None else float(self.x_ohm)
        )
        if self.r_ohm is not None and self.r_fault_ohm != 0.0 and not np.isclose(
            self.r_fault_ohm, self.r_ohm
        ):
            raise ValueError("r_ohm and legacy r_fault_ohm specify different values")
        if self.x_ohm is not None and self.x_fault_ohm != 0.0 and not np.isclose(
            self.x_fault_ohm, self.x_ohm
        ):
            raise ValueError("x_ohm and legacy x_fault_ohm specify different values")
        if not np.isfinite(r_fault_ohm) or not np.isfinite(x_fault_ohm):
            raise ValueError("fault impedance must be finite")
        if r_fault_ohm < 0:
            raise ValueError("r_ohm must be non-negative")
        if self.tk_s <= 0 or not np.isfinite(self.tk_s):
            raise ValueError("tk_s must be a finite positive value")
        if self.voltage_factor is not None and (
            self.voltage_factor <= 0 or not np.isfinite(self.voltage_factor)
        ):
            raise ValueError("voltage_factor must be a finite positive value")

        object.__setattr__(self, "fault", fault)
        object.__setattr__(self, "case", case)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "r_fault_ohm", r_fault_ohm)
        object.__setattr__(self, "x_fault_ohm", x_fault_ohm)
        object.__setattr__(self, "r_ohm", r_fault_ohm)
        object.__setattr__(self, "x_ohm", x_fault_ohm)
        if self.bus is not None:
            object.__setattr__(self, "bus", int(self.bus))
        if self.line is not None:
            object.__setattr__(self, "line", int(self.line))
            if self.line < 0:
                raise ValueError("line must be a non-negative element index")
            location = 0.5 if self.location is None else float(self.location)
            if not np.isfinite(location) or not 0.0 <= location <= 1.0:
                raise ValueError("line-fault location must be between 0.0 and 1.0")
            object.__setattr__(self, "location", location)
        elif self.location is not None:
            raise ValueError("location can only be specified together with line")
        if self.bus is not None and self.line is not None:
            raise ValueError("FaultSpec must select either a bus or a line location, not both")


def _clone_network(net):
    clone = net.__class__({})
    if hasattr(net, "_allow_invalid_attributes"):
        clone._setattr("_allow_invalid_attributes", net._allow_invalid_attributes)
    for key, value in net.items():
        if key == "model" or str(key).startswith("res_"):
            continue
        if isinstance(value, pd.DataFrame):
            clone[key] = value.copy(deep=True)
        elif key in {
            "name",
            "f_hz",
            "sn_mva",
            "rho_ohmm",
            "std_types",
            "user_pf_options",
            "version",
            "format_version",
        }:
            clone[key] = copy.deepcopy(value)
    return clone


def _split_line_for_fault(net, line: int, location: float):
    if line not in net.line.index.get_level_values(0):
        raise KeyError(f"fault line {line} does not exist")
    original_rows = net.line.xs(line, level=0, drop_level=False)
    if original_rows.empty or not original_rows["in_service"].astype(bool).all():
        raise ValueError(f"fault line {line} must be fully in service")
    from_buses = original_rows["from_bus"].astype(int).unique()
    to_buses = original_rows["to_bus"].astype(int).unique()
    lengths = pd.to_numeric(original_rows["length_km"], errors="coerce").unique()
    if len(from_buses) != 1 or len(to_buses) != 1 or len(lengths) != 1:
        raise ValueError("all conductor rows of a faulted line must share buses and length")
    from_bus, to_bus = int(from_buses[0]), int(to_buses[0])
    if location == 0.0:
        return net, from_bus, None
    if location == 1.0:
        return net, to_bus, None

    aux = _clone_network(net)
    bus_id = int(aux.bus.index.get_level_values(0).max()) + 1
    bus_rows = aux.bus.xs(from_bus, level=0, drop_level=False).copy()
    bus_rows.index = pd.MultiIndex.from_tuples(
        [(bus_id, int(phase)) for _, phase in bus_rows.index], names=aux.bus.index.names
    )
    bus_rows["name"] = f"line_{line}_fault_{location:.6f}"
    bus_rows["grounded"] = False
    bus_rows["grounding_r_ohm"] = np.nan
    bus_rows["grounding_x_ohm"] = np.nan
    aux.bus = pd.concat([aux.bus, bus_rows])

    max_line = int(aux.line.index.get_level_values(0).max()) if len(aux.line) else -1
    segment_from, segment_to = max_line + 1, max_line + 2
    segment_frames = []
    for segment_id, length_fraction, first_segment in (
        (segment_from, location, True),
        (segment_to, 1.0 - location, False),
    ):
        segment = original_rows.copy()
        segment.index = pd.MultiIndex.from_tuples(
            [(segment_id, int(circuit)) for _, circuit in segment.index],
            names=aux.line.index.names,
        )
        segment["length_km"] = float(lengths[0]) * length_fraction
        if first_segment:
            segment["to_bus"] = bus_id
        else:
            segment["from_bus"] = bus_id
        if "name" in segment:
            segment["name"] = segment["name"].map(
                lambda value: f"{value or f'line_{line}'}__fault_segment"
            )
        segment_frames.append(segment)
    aux.line.loc[original_rows.index, "in_service"] = False
    aux.line = pd.concat([aux.line, *segment_frames])

    if isinstance(aux.get("switch"), pd.DataFrame) and not aux.switch.empty:
        line_switches = (aux.switch["et"].astype(str).str.lower() == "l") & (
            pd.to_numeric(aux.switch["element"], errors="coerce") == line
        )
        from_side = line_switches & (pd.to_numeric(aux.switch["bus"], errors="coerce") == from_bus)
        to_side = line_switches & (pd.to_numeric(aux.switch["bus"], errors="coerce") == to_bus)
        aux.switch.loc[from_side, "element"] = segment_from
        aux.switch.loc[to_side, "element"] = segment_to
    return aux, bus_id, (segment_from, segment_to)


def _transfer_line_fault_results(net, aux, spec: FaultSpec, aux_bus: int, segments) -> None:
    bus_results = aux.res_bus_sc.copy()
    bus_results["fault_line"] = spec.line
    bus_results["fault_location"] = spec.location
    net.res_bus_sc = bus_results
    fault_results = bus_results.xs(aux_bus, level=0).copy()
    fault_results.index.name = "phase"
    fault_results["element_type"] = "line"
    fault_results["element"] = spec.line
    fault_results["location"] = spec.location
    net.res_fault_sc = fault_results

    for table_name in (
        "res_switch_sc",
        "res_trafo_sc",
        "res_ext_grid_sc",
        "res_ext_grid_sequence_sc",
        "res_asymmetric_sgen_sc",
        "res_asymmetric_gen_sc",
        "res_sgen_sc",
        "res_gen_sc",
    ):
        if table_name in aux:
            table = aux[table_name]
            target_index = None
            if table_name == "res_switch_sc":
                target_index = net.switch.index
            elif table_name == "res_trafo_sc":
                target_index = net.trafo1ph.index
            elif table_name in {"res_ext_grid_sc", "res_ext_grid_sequence_sc"}:
                source_name = table_name.removeprefix("res_").removesuffix("_sc")
                target_index = net[source_name].index
            elif table_name in {"res_asymmetric_sgen_sc", "res_sgen_sc"}:
                target_index = net.asymmetric_sgen.index
            elif table_name in {"res_asymmetric_gen_sc", "res_gen_sc"}:
                target_index = net.asymmetric_gen.index
            net[table_name] = table.reindex(target_index).copy() if target_index is not None else table.copy()

    if "res_line_sc" not in aux:
        return
    line_results = aux.res_line_sc.reindex(net.line.index).copy()
    if segments is not None:
        segment_from, segment_to = segments
        for original_index in net.line.xs(spec.line, level=0, drop_level=False).index:
            circuit = int(original_index[1])
            from_index = (segment_from, circuit)
            to_index = (segment_to, circuit)
            if from_index not in aux.res_line_sc.index or to_index not in aux.res_line_sc.index:
                continue
            for column in (
                "ikss_from_ka",
                "ikss_from_degree",
                "p_from_mw",
                "q_from_mvar",
                "vm_from_pu",
                "va_from_degree",
            ):
                line_results.loc[original_index, column] = aux.res_line_sc.loc[from_index, column]
            for column in (
                "ikss_to_ka",
                "ikss_to_degree",
                "p_to_mw",
                "q_to_mvar",
                "vm_to_pu",
                "va_to_degree",
            ):
                line_results.loc[original_index, column] = aux.res_line_sc.loc[to_index, column]
            line_results.loc[original_index, "ikss_ka"] = max(
                line_results.loc[original_index, "ikss_from_ka"],
                line_results.loc[original_index, "ikss_to_ka"],
            )
            for column in ("ip_ka", "ith_ka"):
                values = [aux.res_line_sc.loc[from_index, column], aux.res_line_sc.loc[to_index, column]]
                if not all(pd.isna(value) for value in values):
                    line_results.loc[original_index, column] = np.nanmax(values)
            line_results.loc[original_index, "fault_bus"] = aux_bus
    line_results["fault_line"] = spec.line
    line_results["fault_location"] = spec.location
    net.res_line_sc = line_results


def _calc_sc_at_line_location(
    net,
    spec: FaultSpec,
    *,
    lv_tol_percent: float,
    branch_results: bool,
    return_all_currents: bool,
) -> None:
    if spec.line is None or spec.location is None:
        raise ValueError("line and location are required for a line fault")
    aux, aux_bus, segments = _split_line_for_fault(net, spec.line, spec.location)
    if spec.use_pre_fault_voltage:
        original_model = net.get("model")
        if original_model is None or not bool(getattr(original_model, "solved", False)):
            raise ValueError(
                "use_pre_fault_voltage=True requires a converged multiconductor power flow"
            )
        from multiconductor.pycci.cci_powerflow import run_pf

        run_pf(aux)
    aux_spec = replace(spec, bus=aux_bus, line=None, location=None)
    calc_sc_native(
        aux,
        aux_spec,
        bus=aux_bus,
        lv_tol_percent=lv_tol_percent,
        branch_results=branch_results,
        return_all_currents=return_all_currents,
    )
    _transfer_line_fault_results(net, aux, spec, aux_bus, segments)


def _active_source_table(net) -> tuple[str, pd.DataFrame]:
    try:
        name, active = get_active_ext_grid_table(net, require_active=True)
    except InvalidStudyInput as exc:
        raise ShortCircuitSourceError(str(exc)) from exc
    if name is None or active is None or active.empty:
        raise ShortCircuitSourceError(
            "native short-circuit calculation requires an in-service external grid "
            "with explicit source impedance"
        )
    return name, active


def _validate_source_strength(net) -> None:
    name, source = _active_source_table(net)
    for column in ("r_ohm", "x_ohm"):
        if column not in source:
            raise ShortCircuitSourceError(
                f"{name} is missing required source-impedance column {column!r}"
            )
    resistance = pd.to_numeric(source["r_ohm"], errors="coerce").to_numpy()
    reactance = pd.to_numeric(source["x_ohm"], errors="coerce").to_numpy()
    impedance = resistance + 1j * reactance
    if not np.all(np.isfinite(impedance)) or np.any(np.abs(impedance) == 0):
        raise ShortCircuitSourceError(
            "every external-grid phase must define a finite, non-zero r_ohm/x_ohm "
            "source impedance; ideal-source fallback is intentionally disabled"
        )
    if np.any(resistance < 0) or np.any(reactance < 0):
        raise ShortCircuitSourceError("external-grid resistance and reactance must be non-negative")
    if "to_phase" in source and any(int(phase) != 0 for phase in source["to_phase"]):
        raise ShortCircuitSourceError(
            "the native short-circuit source currently supports grounded-wye sources only"
        )
    for source_id, group in source.groupby(level=0, sort=False):
        phases = tuple(int(phase) for phase in group["from_phase"])
        if len(phases) != len(set(phases)):
            raise ShortCircuitSourceError(
                f"external-grid source {source_id} repeats a phase row"
            )
        if any(phase not in (1, 2, 3) for phase in phases):
            raise ShortCircuitSourceError(
                f"external-grid source {source_id} contains an unsupported phase"
            )
        if group["bus"].astype(int).nunique() != 1:
            raise ShortCircuitSourceError(
                f"external-grid source {source_id} rows must share one bus"
            )


def _initialize_fault_model(net) -> None:
    """Build CCI matrices while honoring open line switches.

    The base CCI connectivity builder does not currently disconnect ``et='l'``
    switches.  Short-circuit studies conservatively remove the whole associated
    multiconductor line when any of its monitored switch rows is open.
    """

    restore = None
    if isinstance(net.get("switch"), pd.DataFrame) and not net.switch.empty:
        open_line_switches = net.switch.loc[
            (net.switch["et"].astype(str).str.lower() == "l")
            & ~net.switch["closed"].map(lambda value: bool(value))
        ]
        if not open_line_switches.empty:
            line_ids = set(pd.to_numeric(open_line_switches["element"], errors="coerce").dropna().astype(int))
            mask = net.line.index.get_level_values(0).isin(line_ids)
            restore = net.line.loc[mask, "in_service"].copy()
            net.line.loc[mask, "in_service"] = False
    try:
        _initialize_model(net)
    finally:
        if restore is not None:
            net.line.loc[restore.index, "in_service"] = restore


def _bus_ids(net, bus: int | Sequence[int] | None) -> list[int]:
    available = [int(value) for value in net.bus.index.get_level_values(0).unique()]
    if bus is None:
        return available
    if isinstance(bus, tuple) and len(bus) == 2 and all(np.isscalar(value) for value in bus):
        requested = [int(bus[0])]
    elif np.isscalar(bus):
        requested = [int(bus)]
    else:
        requested = []
        for value in bus:
            requested.append(int(value[0] if isinstance(value, tuple) else value))
    requested = list(dict.fromkeys(requested))
    missing = sorted(set(requested).difference(available))
    if missing:
        raise KeyError(f"short-circuit bus(es) do not exist: {missing}")
    return requested


def _voltage_factor(net, bus: int, spec: FaultSpec, lv_tol_percent: float) -> float:
    if spec.use_pre_fault_voltage:
        return 1.0
    if spec.voltage_factor is not None:
        return float(spec.voltage_factor)
    vn_kv = float(net.bus.loc[(bus, spec.phases[0]), "vn_kv"])
    if spec.case == "min":
        return 0.95 if vn_kv <= 1.0 else 1.0
    if vn_kv <= 1.0 and float(lv_tol_percent) == 6.0:
        return 1.05
    return 1.10


def _fault_impedance_pu(net, bus: int, phase: int, spec: FaultSpec) -> complex:
    vn_kv = float(net.bus.loc[(bus, phase), "vn_kv"])
    z_base_ohm = (vn_kv * 1e3 / np.sqrt(3.0)) ** 2 / (float(net.sn_mva) * 1e6)
    return complex(spec.r_fault_ohm, spec.x_fault_ohm) / z_base_ohm


def _thermal_factor(kappa: np.ndarray, frequency_hz: float, tk_s: float) -> np.ndarray:
    m = np.zeros_like(kappa, dtype=float)
    valid = (kappa > 1.0 + 1e-12) & (kappa < 1.99)
    if np.any(valid):
        log_term = np.log(kappa[valid] - 1.0)
        m[valid] = (
            np.exp(4.0 * frequency_hz * tk_s * log_term) - 1.0
        ) / (2.0 * frequency_hz * tk_s * log_term)
    return np.sqrt(np.maximum(1.0 + m, 0.0))


def _converter_current_injections(
    net,
    nonslack: np.ndarray,
    position: dict[int, int],
    prefault: np.ndarray,
    spec: FaultSpec,
) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    """Build explicit converter Norton injections in per-unit nodal current."""

    injection = np.zeros(len(nonslack), dtype=complex)
    result_tables: dict[str, pd.DataFrame] = {}
    for table_name, result_name in (
        ("asymmetric_sgen", "res_asymmetric_sgen_sc"),
        ("asymmetric_gen", "res_asymmetric_gen_sc"),
    ):
        table = net.get(table_name)
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        result = pd.DataFrame(index=table.index.copy())
        result["current_source"] = False
        result["injected_ka"] = 0.0
        result["injected_degree"] = np.nan
        result["status"] = "not-current-source"
        current_source = table.get(
            "current_source", pd.Series(False, index=table.index, dtype=bool)
        ).map(lambda value: False if pd.isna(value) else bool(value))
        in_service = table.get(
            "in_service", pd.Series(True, index=table.index, dtype=bool)
        ).map(lambda value: False if pd.isna(value) else bool(value))
        selected = table.loc[current_source & in_service]
        for row_index, row in selected.iterrows():
            result.loc[row_index, "current_source"] = True
            if spec.case == "min" and not spec.use_pre_fault_voltage:
                result.loc[row_index, "status"] = "excluded-min-case"
                continue
            bus = int(row["bus"])
            from_phase = int(row["from_phase"])
            to_phase = int(row.get("to_phase", 0))
            from_node = int(net.model.terminal_to_y_lookup[bus * 4 + from_phase])
            to_node = int(net.model.terminal_to_y_lookup[bus * 4 + to_phase])
            if from_node not in position:
                raise ShortCircuitCalculationError(
                    f"current source {table_name}{row_index} is on a disconnected/fixed phase"
                )

            direct_current = None
            for column in ("max_ik_ka", "sc_current_ka", "ikss_ka"):
                if column in table and pd.notna(row.get(column)):
                    direct_current = float(row[column])
                    break
            if direct_current is None:
                sn_mva = float(row.get("sn_mva", np.nan))
                k = float(row.get("k", np.nan))
                if not np.isfinite(sn_mva) or sn_mva <= 0:
                    raise ValueError(
                        f"sn_mva must be positive for current-source {table_name}{row_index}"
                    )
                if not np.isfinite(k) or k < 0:
                    raise ValueError(
                        f"k must define nominal-to-short-circuit current for {table_name}{row_index}"
                    )
                vn_kv = float(net.bus.loc[(bus, from_phase), "vn_kv"])
                connection_kv = vn_kv / np.sqrt(3.0) if to_phase == 0 else vn_kv
                direct_current = k * sn_mva / connection_kv
                if spec.use_pre_fault_voltage:
                    kappa_limit = float(row.get("kappa", np.nan))
                    if not np.isfinite(kappa_limit) or kappa_limit < 0:
                        raise ValueError(
                            "kappa must define the Type-C converter current cap for "
                            f"{table_name}{row_index}"
                        )
                    rated_current = sn_mva / connection_kv
                    direct_current = min(direct_current, kappa_limit * rated_current)
            if not np.isfinite(direct_current) or direct_current < 0:
                raise ValueError("converter short-circuit current must be finite and non-negative")

            angle_value = row.get("current_angle_degree", np.nan)
            if pd.notna(angle_value) and np.isfinite(float(angle_value)):
                angle_degree = float(angle_value)
            else:
                from_voltage = prefault[from_node]
                to_voltage = prefault[to_node] if to_node >= 0 else 0.0
                angle_degree = float(np.angle(from_voltage - to_voltage, deg=True) - 90.0)
            current_ka = direct_current * np.exp(1j * np.deg2rad(angle_degree))
            from_base_ka = float(net.model.Ibase_y[from_node]) / 1000.0
            injection[position[from_node]] += current_ka / from_base_ka
            if to_node in position:
                to_base_ka = float(net.model.Ibase_y[to_node]) / 1000.0
                injection[position[to_node]] -= current_ka / to_base_ka
            result.loc[row_index, "injected_ka"] = abs(current_ka)
            result.loc[row_index, "injected_degree"] = np.angle(current_ka, deg=True)
            result.loc[row_index, "status"] = (
                "type-c-cap" if spec.use_pre_fault_voltage else "injected"
            )
        result_tables[result_name] = result
    return injection, result_tables


def _empty_bus_results(net, ip: bool, ith: bool) -> pd.DataFrame:
    result = pd.DataFrame(index=net.bus.index.copy())
    result["fault_bus"] = pd.Series(index=result.index, dtype="Int64")
    result["fault"] = None
    result["fault_phases"] = None
    result["case"] = None
    result["faulted"] = False
    for column in ("ikss_ka", "ikss_degree", "skss_mw", "rk_ohm", "xk_ohm"):
        result[column] = 0.0
    result["vm_fault_pu"] = np.nan
    result["va_fault_degree"] = np.nan
    result["ip_ka"] = np.nan if not ip else 0.0
    result["ith_ka"] = np.nan if not ith else 0.0
    for phase_name in ("a", "b", "c"):
        result[f"ikss_{phase_name}_ka"] = 0.0
        result[f"ip_{phase_name}_ka"] = np.nan if not ip else 0.0
        result[f"ith_{phase_name}_ka"] = np.nan if not ith else 0.0
    return result


def _empty_line_results(net, ip: bool, ith: bool) -> pd.DataFrame:
    result = pd.DataFrame(index=net.line.index.copy())
    for column in (
        "ikss_ka",
        "ikss_from_ka",
        "ikss_from_degree",
        "ikss_to_ka",
        "ikss_to_degree",
        "p_from_mw",
        "q_from_mvar",
        "p_to_mw",
        "q_to_mvar",
        "vm_from_pu",
        "va_from_degree",
        "vm_to_pu",
        "va_to_degree",
    ):
        result[column] = 0.0
    result["ip_ka"] = np.nan if not ip else 0.0
    result["ith_ka"] = np.nan if not ith else 0.0
    result["fault_bus"] = pd.Series(index=result.index, dtype="Int64")
    return result


def _line_results_for_fault(net, prefault: np.ndarray, postfault: np.ndarray, bus: int,
                            kappa: float, spec: FaultSpec) -> pd.DataFrame:
    result = _empty_line_results(net, spec.ip, spec.ith)
    delta_voltage = postfault - prefault
    sbase_mva = float(net.sn_mva)

    info = net.model.line_info
    for line_id, count, primitive, nodes in zip(
        info["indices"], info["num_circuits"], info["Yprimitive"], info["y_send_rec"]
    ):
        nodes = np.asarray(nodes, dtype=int)
        contribution = np.asarray(primitive @ delta_voltage[nodes]).reshape(-1)
        total_voltage = postfault[nodes]
        rows = list(net.line.xs(line_id, level=0, drop_level=False).index)
        for position, row_index in enumerate(rows[:count]):
            from_node = nodes[position]
            to_node = nodes[count + position]
            from_current = contribution[position] * net.model.Ibase_y[from_node] / 1000.0
            to_current = contribution[count + position] * net.model.Ibase_y[to_node] / 1000.0
            from_magnitude = float(abs(from_current))
            to_magnitude = float(abs(to_current))
            result.loc[row_index, "ikss_from_ka"] = from_magnitude
            result.loc[row_index, "ikss_from_degree"] = float(np.angle(from_current, deg=True))
            result.loc[row_index, "ikss_to_ka"] = to_magnitude
            result.loc[row_index, "ikss_to_degree"] = float(np.angle(to_current, deg=True))
            result.loc[row_index, "ikss_ka"] = max(from_magnitude, to_magnitude)
            s_from = total_voltage[position] * np.conj(contribution[position]) * sbase_mva
            s_to = total_voltage[count + position] * np.conj(contribution[count + position]) * sbase_mva
            result.loc[row_index, ["p_from_mw", "q_from_mvar"]] = [s_from.real, s_from.imag]
            result.loc[row_index, ["p_to_mw", "q_to_mvar"]] = [s_to.real, s_to.imag]
            result.loc[row_index, "vm_from_pu"] = abs(total_voltage[position])
            result.loc[row_index, "va_from_degree"] = np.angle(total_voltage[position], deg=True)
            result.loc[row_index, "vm_to_pu"] = abs(total_voltage[count + position])
            result.loc[row_index, "va_to_degree"] = np.angle(total_voltage[count + position], deg=True)
            if spec.ip:
                result.loc[row_index, "ip_ka"] = np.sqrt(2.0) * kappa * result.loc[row_index, "ikss_ka"]
            if spec.ith:
                thermal = _thermal_factor(
                    np.asarray([kappa]), float(getattr(net, "f_hz", 50.0)), spec.tk_s
                )[0]
                result.loc[row_index, "ith_ka"] = thermal * result.loc[row_index, "ikss_ka"]
            result.loc[row_index, "fault_bus"] = bus
    return result


def _trafo_results_for_fault(net, prefault: np.ndarray, postfault: np.ndarray, bus: int,
                             kappa: float, spec: FaultSpec) -> pd.DataFrame:
    columns = [
        "ikss_ka",
        "ikss_from_ka",
        "ikss_from_degree",
        "ikss_to_ka",
        "ikss_to_degree",
        "ip_ka",
        "ith_ka",
        "p_mw",
        "q_mvar",
        "vm_fault_pu",
        "va_fault_degree",
        "side",
        "fault_bus",
    ]
    result = pd.DataFrame(index=net.trafo1ph.index.copy(), columns=columns)
    for column in (
        "ikss_ka",
        "ikss_from_ka",
        "ikss_from_degree",
        "ikss_to_ka",
        "ikss_to_degree",
        "p_mw",
        "q_mvar",
        "vm_fault_pu",
        "va_fault_degree",
    ):
        result[column] = 0.0
    result["ip_ka"] = np.nan if not spec.ip else 0.0
    result["ith_ka"] = np.nan if not spec.ith else 0.0
    result["fault_bus"] = bus
    delta_voltage = postfault - prefault
    thermal = _thermal_factor(
        np.asarray([kappa]), float(getattr(net, "f_hz", 50.0)), spec.tk_s
    )[0]
    for (trafo_id, circuit), info in net.model.trafo_circuits.items():
        nodes = np.asarray(info["y_send_rec"], dtype=int)
        contribution = np.asarray(info["Ybr"] @ delta_voltage[nodes]).reshape(-1)
        winding_buses = [int(value) for value in info["buses"]]
        winding_vn = [
            float(net.bus.xs(winding_bus, level=0)["vn_kv"].iloc[0])
            for winding_bus in winding_buses
        ]
        hv_position = int(np.argmax(winding_vn))
        for winding_position, winding_bus in enumerate(winding_buses):
            row_index = (int(trafo_id), winding_bus, int(circuit))
            if row_index not in result.index:
                continue
            from_offset = 2 * winding_position
            to_offset = from_offset + 1
            from_current = (
                contribution[from_offset] * net.model.Ibase_y[nodes[from_offset]] / 1000.0
            )
            to_current = contribution[to_offset] * net.model.Ibase_y[nodes[to_offset]] / 1000.0
            magnitude = max(abs(from_current), abs(to_current))
            result.loc[row_index, "ikss_from_ka"] = abs(from_current)
            result.loc[row_index, "ikss_from_degree"] = np.angle(from_current, deg=True)
            result.loc[row_index, "ikss_to_ka"] = abs(to_current)
            result.loc[row_index, "ikss_to_degree"] = np.angle(to_current, deg=True)
            result.loc[row_index, "ikss_ka"] = magnitude
            result.loc[row_index, "ip_ka"] = (
                np.sqrt(2.0) * kappa * magnitude if spec.ip else np.nan
            )
            result.loc[row_index, "ith_ka"] = thermal * magnitude if spec.ith else np.nan
            voltage = postfault[nodes[from_offset]] - postfault[nodes[to_offset]]
            power = voltage * np.conj(contribution[from_offset]) * float(net.sn_mva)
            result.loc[row_index, "p_mw"] = power.real
            result.loc[row_index, "q_mvar"] = power.imag
            result.loc[row_index, "vm_fault_pu"] = abs(voltage)
            result.loc[row_index, "va_fault_degree"] = np.angle(voltage, deg=True)
            result.loc[row_index, "side"] = "hv" if winding_position == hv_position else "lv"
    return result


def _source_results_for_fault(
    net,
    prefault: np.ndarray,
    postfault: np.ndarray,
    bus: int,
    kappa: float,
    spec: FaultSpec,
) -> tuple[str, pd.DataFrame]:
    """Attribute incremental fault current to each explicit source conductor."""

    source_name, active_source = _active_source_table(net)
    result = pd.DataFrame(index=net[source_name].index.copy())
    result["ikss_ka"] = 0.0
    result["ikss_degree"] = np.nan
    result["ip_ka"] = np.nan if not spec.ip else 0.0
    result["ith_ka"] = np.nan if not spec.ith else 0.0
    result["fault_bus"] = bus
    result["in_service"] = False

    delta_voltage = np.asarray(postfault - prefault, dtype=complex).reshape(-1)
    incremental_nodal_current = np.asarray(
        net.model.Y_source @ delta_voltage, dtype=complex
    ).reshape(-1)
    thermal = _thermal_factor(
        np.asarray([kappa]), float(getattr(net, "f_hz", 50.0)), spec.tk_s
    )[0]
    lookup = np.asarray(net.model.terminal_to_y_lookup, dtype=int)
    for source_index, _row in active_source.iterrows():
        element, circuit = source_index
        terminal = get_ext_grid_terminal(net.model, int(element), int(circuit))
        if terminal >= len(lookup):
            continue
        y_index = int(lookup[terminal])
        if y_index < 0:
            continue
        current = (
            incremental_nodal_current[y_index]
            * float(net.model.Ibase_y[y_index])
            / 1000.0
        )
        magnitude = float(abs(current))
        result.loc[source_index, "ikss_ka"] = magnitude
        result.loc[source_index, "ikss_degree"] = float(np.angle(current, deg=True))
        result.loc[source_index, "ip_ka"] = (
            np.sqrt(2.0) * kappa * magnitude if spec.ip else np.nan
        )
        result.loc[source_index, "ith_ka"] = thermal * magnitude if spec.ith else np.nan
        result.loc[source_index, "in_service"] = True
    return f"res_{source_name}_sc", result


def _switch_results_for_fault(net, line_result: pd.DataFrame, prefault: np.ndarray,
                              postfault: np.ndarray, bus: int, spec: FaultSpec) -> pd.DataFrame:
    columns = ["ikss_ka", "ip_ka", "ith_ka", "fault_bus"]
    result = pd.DataFrame(index=net.switch.index.copy(), columns=columns, dtype=float)
    result["ikss_ka"] = 0.0
    result["ip_ka"] = np.nan if not spec.ip else 0.0
    result["ith_ka"] = np.nan if not spec.ith else 0.0
    result["fault_bus"] = bus

    for switch_index, switch in net.switch.iterrows():
        if not bool(switch.get("closed", True)):
            continue
        element_type = str(switch.get("et", "")).lower()
        phase = int(switch.get("phase", switch_index[-1]))
        if element_type == "l":
            line_id = int(switch["element"])
            try:
                line_rows = net.line.xs(line_id, level=0, drop_level=False)
            except KeyError:
                continue
            matches = line_rows.loc[
                (line_rows["from_phase"].astype(int) == phase)
                | (line_rows["to_phase"].astype(int) == phase)
            ]
            if matches.empty:
                continue
            line_index = matches.index[0]
            side = "from" if int(switch["bus"]) == int(matches.iloc[0]["from_bus"]) else "to"
            magnitude = float(line_result.loc[line_index, f"ikss_{side}_ka"])
            result.loc[switch_index, "ikss_ka"] = magnitude
            if spec.ip:
                result.loc[switch_index, "ip_ka"] = float(line_result.loc[line_index, "ip_ka"])
            if spec.ith:
                result.loc[switch_index, "ith_ka"] = float(line_result.loc[line_index, "ith_ka"])
        elif element_type == "b" and float(switch.get("r_ohm", 0.0)) > 0.0:
            from_node = int(net.model.terminal_to_y_lookup[int(switch["bus"]) * 4 + phase])
            to_node = int(net.model.terminal_to_y_lookup[int(switch["element"]) * 4 + phase])
            delta_v_pu = (postfault[from_node] - prefault[from_node]) - (
                postfault[to_node] - prefault[to_node]
            )
            vn_kv = float(net.bus.loc[(int(switch["bus"]), phase), "vn_kv"])
            magnitude = abs(delta_v_pu) * (vn_kv / np.sqrt(3.0)) / float(switch["r_ohm"])
            result.loc[switch_index, "ikss_ka"] = magnitude
    return result


def _aggregate_results(frames: list[pd.DataFrame], case: str) -> pd.DataFrame:
    if len(frames) == 1:
        return frames[0]
    stacked = pd.concat(frames, keys=range(len(frames)), names=["_case"])
    index_levels = list(range(1, stacked.index.nlevels))

    def choose(group: pd.DataFrame) -> pd.Series:
        currents = group["ikss_ka"].astype(float)
        position = currents.idxmax() if case == "max" else currents.idxmin()
        return group.loc[position]

    aggregated = stacked.groupby(level=index_levels, sort=False, group_keys=False).apply(choose)
    aggregated.index = frames[0].index
    return aggregated


def _all_current_results(frames: list[pd.DataFrame], buses: Sequence[int]) -> pd.DataFrame:
    result = pd.concat(frames, keys=buses, names=["fault_bus"])
    order = list(range(1, result.index.nlevels)) + [0]
    return result.reorder_levels(order).sort_index()


def calc_sc_native(
    net,
    fault_spec: FaultSpec,
    *,
    bus: int | Sequence[int] | None = None,
    lv_tol_percent: float = 10.0,
    branch_results: bool = False,
    return_all_currents: bool = False,
) -> None:
    """Run a native multiconductor short-circuit calculation in place."""

    if not isinstance(net.bus.index, pd.MultiIndex):
        raise TypeError("calc_sc_native requires a multiconductor network with a bus MultiIndex")
    if fault_spec.line is not None:
        if bus is not None:
            raise ValueError("bus cannot be combined with a line/location FaultSpec")
        _calc_sc_at_line_location(
            net,
            fault_spec,
            lv_tol_percent=lv_tol_percent,
            branch_results=branch_results,
            return_all_currents=return_all_currents,
        )
        return
    _validate_source_strength(net)

    requested_bus = fault_spec.bus if bus is None else bus
    fault_buses = _bus_ids(net, requested_bus)
    if requested_bus is None:
        eligible = [
            fault_bus
            for fault_bus in fault_buses
            if all((fault_bus, phase) in net.bus.index for phase in fault_spec.phases)
        ]
        skipped = sorted(set(fault_buses).difference(eligible))
        if skipped:
            logger.warning(
                "skipping buses without all phases required for %s %s: %s",
                fault_spec.fault,
                fault_spec.phases,
                skipped,
            )
        fault_buses = eligible
    if not fault_buses:
        raise ValueError("no bus contains all phases required by the fault specification")

    saved_prefault = None
    if fault_spec.use_pre_fault_voltage:
        model = net.get("model")
        if model is None or not bool(getattr(model, "solved", False)) or not hasattr(model, "E"):
            raise ValueError(
                "use_pre_fault_voltage=True requires a converged multiconductor power flow"
            )
        saved_prefault = np.asarray(model.E, dtype=complex).reshape(-1).copy()

    _initialize_fault_model(net)
    _init_pf(net)
    if saved_prefault is not None:
        if saved_prefault.shape != np.asarray(net.model.E0).reshape(-1).shape:
            raise ValueError("the pre-fault model topology no longer matches the current network")
        base_prefault = saved_prefault
    else:
        base_prefault = np.asarray(net.model.E0, dtype=complex).reshape(-1)

    nonslack = np.asarray(net.model.y_nonslack, dtype=int)
    position = {int(node): offset for offset, node in enumerate(nonslack)}
    ynn = csc_matrix(net.model.Y_tot[nonslack, :][:, nonslack])
    try:
        solver = splu(ynn)
    except RuntimeError as exc:
        raise ShortCircuitCalculationError(
            "passive phase-domain admittance matrix is singular; check grounding and islands"
        ) from exc

    bus_result = _empty_bus_results(net, fault_spec.ip, fault_spec.ith)
    line_frames: list[pd.DataFrame] = []
    switch_frames: list[pd.DataFrame] = []
    trafo_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    source_result_name: str | None = None
    converter_results: dict[str, pd.DataFrame] = {}

    for fault_bus in fault_buses:
        missing_rows = [
            (fault_bus, phase) for phase in fault_spec.phases if (fault_bus, phase) not in net.bus.index
        ]
        if missing_rows:
            raise KeyError(f"fault phases are not present at bus {fault_bus}: {missing_rows}")
        fault_nodes = np.asarray(
            [net.model.terminal_to_y_lookup[fault_bus * 4 + phase] for phase in fault_spec.phases],
            dtype=int,
        )
        if np.any(fault_nodes < 0):
            raise ShortCircuitCalculationError(
                f"fault bus {fault_bus} contains disconnected phase terminals"
            )
        try:
            fault_positions = np.asarray([position[int(node)] for node in fault_nodes], dtype=int)
        except KeyError as exc:
            isolated = set(np.asarray(getattr(net.model, "y_isolated", []), dtype=int))
            if any(int(node) in isolated for node in fault_nodes):
                raise ShortCircuitCalculationError(
                    f"fault bus {fault_bus} is isolated from every source"
                ) from exc
            raise ShortCircuitSourceError(
                "fault is on a fixed-voltage node; define a finite external-grid source impedance"
            ) from exc

        response = np.empty((len(nonslack), len(fault_nodes)), dtype=complex)
        for column, fault_position in enumerate(fault_positions):
            rhs = np.zeros(len(nonslack), dtype=complex)
            rhs[fault_position] = 1.0
            response[:, column] = solver.solve(rhs)
        z_fault_bus = response[fault_positions, :]

        prefault = base_prefault * _voltage_factor(net, fault_bus, fault_spec, lv_tol_percent)
        converter_injection, converter_results = _converter_current_injections(
            net, nonslack, position, prefault, fault_spec
        )
        driving_voltage = prefault.copy()
        if np.any(converter_injection):
            driving_voltage[nonslack] += solver.solve(converter_injection)
        v_prefault = driving_voltage[fault_nodes]
        zf_pu = _fault_impedance_pu(net, fault_bus, fault_spec.phases[0], fault_spec)
        grounded = fault_spec.fault in {"LG", "LLG"}
        try:
            if grounded:
                fault_matrix = z_fault_bus + zf_pu * np.ones_like(z_fault_bus)
                fault_current_pu = np.linalg.solve(fault_matrix, v_prefault)
            else:
                series_z = zf_pu / 2.0 if fault_spec.fault == "LL" else zf_pu
                coefficients = np.block(
                    [
                        [z_fault_bus + np.eye(len(fault_nodes)) * series_z,
                         np.ones((len(fault_nodes), 1), dtype=complex)],
                        [np.ones((1, len(fault_nodes)), dtype=complex),
                         np.zeros((1, 1), dtype=complex)],
                    ]
                )
                rhs = np.concatenate([v_prefault, np.zeros(1, dtype=complex)])
                fault_current_pu = np.linalg.solve(coefficients, rhs)[:-1]
        except np.linalg.LinAlgError as exc:
            raise ShortCircuitCalculationError(
                f"fault equations are singular for {fault_spec.fault} at bus {fault_bus}"
            ) from exc

        postfault = driving_voltage.copy()
        postfault[nonslack] -= response @ fault_current_pu
        current_ka_complex = (
            fault_current_pu * np.asarray(net.model.Ibase_y[fault_nodes], dtype=float) / 1000.0
        )
        current_ka = np.abs(current_ka_complex)
        diagonal_z = np.diag(z_fault_bus)
        impedance_for_peak = diagonal_z
        if fault_spec.fault == "LLL":
            impedance_for_peak = np.divide(
                v_prefault,
                fault_current_pu,
                out=diagonal_z.copy(),
                where=np.abs(fault_current_pu) > 1e-12,
            )
        rx = np.maximum(impedance_for_peak.real, 0.0) / np.maximum(
            np.abs(impedance_for_peak.imag), 1e-12
        )
        kappa = np.clip(_kappa(rx), 1.0, 2.0)
        peak_ka = np.sqrt(2.0) * kappa * current_ka
        thermal_ka = current_ka * _thermal_factor(
            kappa, float(getattr(net, "f_hz", 50.0)), fault_spec.tk_s
        )

        bus_rows = net.bus.index.get_level_values(0) == fault_bus
        bus_result.loc[bus_rows, "fault_bus"] = fault_bus
        bus_result.loc[bus_rows, "fault"] = fault_spec.fault
        bus_result.loc[bus_rows, "fault_phases"] = pd.Series(
            [fault_spec.phases] * int(bus_rows.sum()), index=bus_result.index[bus_rows], dtype=object
        )
        bus_result.loc[bus_rows, "case"] = fault_spec.case
        for offset, (phase, node) in enumerate(zip(fault_spec.phases, fault_nodes)):
            row = (fault_bus, phase)
            vn_kv = float(net.bus.loc[row, "vn_kv"])
            z_base = (vn_kv * 1e3 / np.sqrt(3.0)) ** 2 / (float(net.sn_mva) * 1e6)
            bus_result.loc[row, "faulted"] = True
            bus_result.loc[row, "ikss_ka"] = current_ka[offset]
            bus_result.loc[row, "ikss_degree"] = np.angle(current_ka_complex[offset], deg=True)
            bus_result.loc[row, "skss_mw"] = np.sqrt(3.0) * vn_kv * current_ka[offset]
            bus_result.loc[row, "rk_ohm"] = impedance_for_peak[offset].real * z_base
            bus_result.loc[row, "xk_ohm"] = impedance_for_peak[offset].imag * z_base
            bus_result.loc[row, "ip_ka"] = peak_ka[offset] if fault_spec.ip else np.nan
            bus_result.loc[row, "ith_ka"] = thermal_ka[offset] if fault_spec.ith else np.nan
            phase_name = _PHASE_NAMES[phase]
            bus_result.loc[bus_rows, f"ikss_{phase_name}_ka"] = current_ka[offset]
            if fault_spec.ip:
                bus_result.loc[bus_rows, f"ip_{phase_name}_ka"] = peak_ka[offset]
            if fault_spec.ith:
                bus_result.loc[bus_rows, f"ith_{phase_name}_ka"] = thermal_ka[offset]
        for phase in net.bus.index.get_level_values(1)[bus_rows]:
            node = int(net.model.terminal_to_y_lookup[fault_bus * 4 + int(phase)])
            if node >= 0:
                bus_result.loc[(fault_bus, phase), "vm_fault_pu"] = abs(postfault[node])
                bus_result.loc[(fault_bus, phase), "va_fault_degree"] = np.angle(
                    postfault[node], deg=True
                )

        line_result = _line_results_for_fault(
            net, prefault, postfault, fault_bus, float(np.max(kappa)), fault_spec
        )
        line_frames.append(line_result)
        switch_frames.append(
            _switch_results_for_fault(net, line_result, prefault, postfault, fault_bus, fault_spec)
        )
        trafo_frames.append(
            _trafo_results_for_fault(
                net, prefault, postfault, fault_bus, float(np.max(kappa)), fault_spec
            )
        )
        source_result_name, source_result = _source_results_for_fault(
            net,
            prefault,
            postfault,
            fault_bus,
            float(np.max(kappa)),
            fault_spec,
        )
        source_frames.append(source_result)

    net.res_bus_sc = bus_result
    for result_name, result in converter_results.items():
        net[result_name] = result
        if result_name == "res_asymmetric_sgen_sc":
            net.res_sgen_sc = result.copy()
        elif result_name == "res_asymmetric_gen_sc":
            net.res_gen_sc = result.copy()
    if source_result_name is not None:
        if return_all_currents:
            net[source_result_name] = _all_current_results(source_frames, fault_buses)
        else:
            net[source_result_name] = _aggregate_results(source_frames, fault_spec.case)
    if branch_results:
        if return_all_currents:
            net.res_line_sc = _all_current_results(line_frames, fault_buses)
            net.res_switch_sc = _all_current_results(switch_frames, fault_buses)
            net.res_trafo_sc = _all_current_results(trafo_frames, fault_buses)
        else:
            net.res_line_sc = _aggregate_results(line_frames, fault_spec.case)
            net.res_switch_sc = _aggregate_results(switch_frames, fault_spec.case)
            net.res_trafo_sc = _aggregate_results(trafo_frames, fault_spec.case)


def fault_spec_from_options(
    *,
    fault: str,
    bus: int | None,
    phases: Iterable[int] | None,
    case: str,
    r_fault_ohm: float,
    x_fault_ohm: float,
    ip: bool,
    ith: bool,
    tk_s: float,
    use_pre_fault_voltage: bool,
) -> FaultSpec:
    """Build a :class:`FaultSpec` from the compatibility ``calc_sc`` options."""

    return FaultSpec(
        fault=fault,
        bus=bus,
        phases=None if phases is None else tuple(phases),
        case=case,
        r_fault_ohm=r_fault_ohm,
        x_fault_ohm=x_fault_ohm,
        ip=ip,
        ith=ith,
        tk_s=tk_s,
        use_pre_fault_voltage=use_pre_fault_voltage,
    )


def merge_fault_spec(spec: FaultSpec, **overrides) -> FaultSpec:
    """Return ``spec`` with non-``None`` compatibility overrides applied."""

    return replace(spec, **{name: value for name, value in overrides.items() if value is not None})
