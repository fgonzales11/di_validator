from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from multiconductor.pycci.cci_powerflow import LoadflowNotConverged, run_pf
from multiconductor.pycci.std_types import load_std_type
from multiconductor.studies import (
    InvalidStudyInput,
    StudyStatus,
    TerminalRef,
)


@dataclass(frozen=True, slots=True)
class ContingencyAction:
    """One topology or availability change applied to a study case."""

    table: str
    index: Any
    circuit: int | None = None
    bus: int | None = None
    column: str | None = None
    value: Any = False

    def resolved_column(self) -> str:
        if self.column is not None:
            return self.column
        return "closed" if self.table == "switch" else "in_service"

    def resolved_value(self) -> Any:
        return False if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class ContingencyCase:
    """A named set of contingency actions."""

    actions: tuple[ContingencyAction, ...]
    case_id: str
    name: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ContingencyCaseResult:
    """Per-case immutable execution summary."""

    case: ContingencyCase
    status: StudyStatus
    converged: bool
    unsupplied_p_mw: float
    unsupplied_q_mvar: float
    bus_results: pd.DataFrame
    line_results: pd.DataFrame
    trafo_results: pd.DataFrame
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ContingencyResult:
    """Immutable contingency-study bundle returned by :func:`run_contingency`."""

    cases: tuple[ContingencyCaseResult, ...]
    case_table: pd.DataFrame
    bus_table: pd.DataFrame
    line_table: pd.DataFrame
    trafo_table: pd.DataFrame

    def get_case(self, case_id: str) -> ContingencyCaseResult:
        for case in self.cases:
            if case.case.case_id == case_id:
                return case
        raise KeyError(case_id)


_ELEMENT_ALIASES = {
    "line": "line",
    "line1ph": "line",
    "trafo": "trafo1ph",
    "trafo1ph": "trafo1ph",
    "switch": "switch",
    "ext_grid": "ext_grid",
    "ext_grid_sequence": "ext_grid_sequence",
    "source": "ext_grid",
    "source_sequence": "ext_grid_sequence",
    "asymmetric_sgen": "asymmetric_sgen",
    "sgen": "asymmetric_sgen",
    "asymmetric_gen": "asymmetric_gen",
    "gen": "asymmetric_gen",
}


def generate_nminus1_cases(
    net,
    *,
    element_types: Sequence[str] = ("line", "trafo1ph"),
) -> list[ContingencyCase]:
    normalized_types = tuple(_canonical_table_name(name) for name in element_types)
    generated: list[ContingencyCase] = []
    for table_name in normalized_types:
        if table_name not in net or getattr(net, table_name).empty:
            continue
        frame = getattr(net, table_name)
        if isinstance(frame.index, pd.MultiIndex):
            grouped = frame.groupby(level=0, sort=False)
            for element_index, rows in grouped:
                actions = (ContingencyAction(table=table_name, index=element_index),)
                generated.append(
                    ContingencyCase(
                        actions=actions,
                        case_id=f"{table_name}:{element_index}",
                        name=f"{table_name} {element_index} out",
                    )
                )
        else:
            for element_index in frame.index.tolist():
                generated.append(
                    ContingencyCase(
                        actions=(ContingencyAction(table=table_name, index=element_index),),
                        case_id=f"{table_name}:{element_index}",
                        name=f"{table_name} {element_index} out",
                    )
                )
    return generated


def run_contingency(
    net,
    *,
    cases: Sequence[ContingencyCase | Mapping[str, Any]] | None = None,
    element_types: Sequence[str] = ("line", "trafo1ph"),
    run_control: bool = False,
    continue_on_failure: bool = True,
    write_to_net: bool = True,
    min_vm_pu: float = 0.95,
    max_vm_pu: float = 1.05,
    max_loading_percent: float = 100.0,
    pf_options: Mapping[str, Any] | None = None,
) -> ContingencyResult:
    """Execute contingency cases on snapshot copies of ``net``."""

    if cases is None:
        cases = generate_nminus1_cases(net, element_types=element_types)
    else:
        cases = tuple(_normalize_case(case, position) for position, case in enumerate(cases))

    pf_kwargs = dict(pf_options or {})
    pf_kwargs["run_control"] = run_control
    pf_kwargs["allow_islands"] = True

    case_results: list[ContingencyCaseResult] = []
    for case in cases:
        try:
            case_results.append(
                _run_case(
                    net,
                    case,
                    min_vm_pu=min_vm_pu,
                    max_vm_pu=max_vm_pu,
                    max_loading_percent=max_loading_percent,
                    pf_kwargs=pf_kwargs,
                )
            )
        except InvalidStudyInput as exc:
            if not continue_on_failure:
                raise
            case_results.append(_failed_case(case, StudyStatus.INVALID, str(exc)))
        except LoadflowNotConverged as exc:
            if not continue_on_failure:
                raise
            case_results.append(_failed_case(case, StudyStatus.NONCONVERGED, str(exc)))
        except Exception as exc:
            if not continue_on_failure:
                raise
            case_results.append(_failed_case(case, StudyStatus.ERROR, str(exc)))

    result = _assemble_result(case_results)
    if write_to_net:
        net["res_contingency_case"] = result.case_table
        net["res_contingency_bus"] = result.bus_table
        net["res_contingency_line"] = result.line_table
        net["res_contingency_trafo"] = result.trafo_table
    return result


def _run_case(
    source_net,
    case: ContingencyCase,
    *,
    min_vm_pu: float,
    max_vm_pu: float,
    max_loading_percent: float,
    pf_kwargs: Mapping[str, Any],
) -> ContingencyCaseResult:
    case_net = copy.deepcopy(source_net)
    for action in case.actions:
        _apply_action(case_net, action)

    if not _has_active_source(case_net):
        raise LoadflowNotConverged("no active source remains in service")

    run_pf(case_net, **pf_kwargs)

    bus_results = _case_bus_results(case_net, min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    line_results = _case_line_results(case_net, max_loading_percent=max_loading_percent)
    trafo_results = _case_trafo_results(case_net, max_loading_percent=max_loading_percent)
    isolated_terminals = _isolated_terminals(case_net)
    unsupplied_p_mw, unsupplied_q_mvar = _unsupplied_load(case_net, isolated_terminals)
    status = StudyStatus.ISLANDED if isolated_terminals else StudyStatus.CONVERGED

    return ContingencyCaseResult(
        case=case,
        status=status,
        converged=bool(getattr(case_net.model, "solved", False)),
        unsupplied_p_mw=unsupplied_p_mw,
        unsupplied_q_mvar=unsupplied_q_mvar,
        bus_results=bus_results,
        line_results=line_results,
        trafo_results=trafo_results,
    )


def _failed_case(case: ContingencyCase, status: StudyStatus, error: str) -> ContingencyCaseResult:
    empty_bus = pd.DataFrame(
        columns=[
            "vm_pu",
            "va_degree",
            "p_mw",
            "q_mvar",
            "imbalance_percent",
            "under_voltage",
            "over_voltage",
            "voltage_violation",
            "is_isolated",
        ]
    )
    empty_line = pd.DataFrame(
        columns=[
            "i_from_ka",
            "ia_from_degree",
            "i_to_ka",
            "ia_to_degree",
            "i_ka",
            "p_from_mw",
            "p_to_mw",
            "q_from_mvar",
            "q_to_mvar",
            "pl_mw",
            "ql_mvar",
            "vm_from_pu",
            "vm_to_pu",
            "va_from_degree",
            "va_to_degree",
            "loading_percent",
            "rating_ka",
            "loading_violation",
        ]
    )
    empty_trafo = pd.DataFrame(
        columns=[
            "p_mw",
            "q_mvar",
            "i_ka",
            "vm_pu",
            "va_degree",
            "pl_mw",
            "ql_mvar",
            "loading_percent",
            "loading_violation",
        ]
    )
    return ContingencyCaseResult(
        case=case,
        status=status,
        converged=False,
        unsupplied_p_mw=0.0,
        unsupplied_q_mvar=0.0,
        bus_results=empty_bus,
        line_results=empty_line,
        trafo_results=empty_trafo,
        error=error,
    )


def _assemble_result(case_results: Sequence[ContingencyCaseResult]) -> ContingencyResult:
    case_rows = []
    bus_tables = []
    line_tables = []
    trafo_tables = []

    for result in case_results:
        case_rows.append(
            {
                "case_id": result.case.case_id,
                "name": result.case.name,
                "description": result.case.description,
                "status": result.status.value,
                "converged": result.converged,
                "unsupplied_p_mw": result.unsupplied_p_mw,
                "unsupplied_q_mvar": result.unsupplied_q_mvar,
                "voltage_violation_count": int(
                    result.bus_results.get("voltage_violation", pd.Series(dtype=bool)).sum()
                ),
                "line_violation_count": int(
                    result.line_results.get("loading_violation", pd.Series(dtype=bool)).sum()
                ),
                "trafo_violation_count": int(
                    result.trafo_results.get("loading_violation", pd.Series(dtype=bool)).sum()
                ),
                "error": result.error,
            }
        )
        bus_tables.append(_prepend_case_id(result.bus_results, result.case.case_id))
        line_tables.append(_prepend_case_id(result.line_results, result.case.case_id))
        trafo_tables.append(_prepend_case_id(result.trafo_results, result.case.case_id))

    case_table = (
        pd.DataFrame(case_rows).set_index("case_id")
        if case_rows
        else pd.DataFrame(
            columns=[
                "name",
                "description",
                "status",
                "converged",
                "unsupplied_p_mw",
                "unsupplied_q_mvar",
                "voltage_violation_count",
                "line_violation_count",
                "trafo_violation_count",
                "error",
            ]
        )
    )
    return ContingencyResult(
        cases=tuple(case_results),
        case_table=case_table,
        bus_table=_concat_case_tables(bus_tables),
        line_table=_concat_case_tables(line_tables),
        trafo_table=_concat_case_tables(trafo_tables),
    )


def _prepend_case_id(frame: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    tuples = [
        (case_id,) + (idx if isinstance(idx, tuple) else (idx,))
        for idx in frame.index.tolist()
    ]
    names = [
        "case_id",
        *(frame.index.names if isinstance(frame.index, pd.MultiIndex) else [frame.index.name or "index"]),
    ]
    frame.index = pd.MultiIndex.from_tuples(tuples, names=names)
    return frame


def _concat_case_tables(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None]
    if not non_empty:
        return pd.DataFrame()
    populated = [frame for frame in non_empty if not frame.empty]
    if not populated:
        return non_empty[0].copy()
    return pd.concat(populated, axis=0).sort_index()


def _normalize_case(case: ContingencyCase | Mapping[str, Any], position: int) -> ContingencyCase:
    if isinstance(case, ContingencyCase):
        return case
    if not isinstance(case, Mapping):
        raise InvalidStudyInput(f"unsupported contingency case type: {type(case)!r}")

    if "actions" in case:
        actions = tuple(_normalize_action(action) for action in case["actions"])
    else:
        actions = (_normalize_action(case),)

    case_id = str(case.get("case_id") or case.get("name") or f"case_{position}")
    return ContingencyCase(
        actions=actions,
        case_id=case_id,
        name=case.get("name"),
        description=case.get("description"),
    )


def _normalize_action(action: ContingencyAction | Mapping[str, Any]) -> ContingencyAction:
    if isinstance(action, ContingencyAction):
        return action
    if not isinstance(action, Mapping):
        raise InvalidStudyInput(f"unsupported contingency action type: {type(action)!r}")

    if "table" not in action or "index" not in action:
        raise InvalidStudyInput("contingency actions require 'table' and 'index'")

    return ContingencyAction(
        table=_canonical_table_name(str(action["table"])),
        index=action["index"],
        circuit=action.get("circuit"),
        bus=action.get("bus"),
        column=action.get("column"),
        value=action.get("value"),
    )


def _canonical_table_name(name: str) -> str:
    try:
        return _ELEMENT_ALIASES[name]
    except KeyError as exc:
        raise InvalidStudyInput(f"unsupported contingency element table '{name}'") from exc


def _apply_action(net, action: ContingencyAction) -> None:
    if action.table not in net:
        raise InvalidStudyInput(f"network has no '{action.table}' table")
    frame = getattr(net, action.table)
    if frame.empty:
        raise InvalidStudyInput(f"network table '{action.table}' is empty")

    column = action.resolved_column()
    if column not in frame.columns:
        raise InvalidStudyInput(f"table '{action.table}' has no '{column}' column")

    if isinstance(frame.index, pd.MultiIndex):
        mask = np.ones(len(frame), dtype=bool)
        if "index" in frame.index.names:
            mask &= frame.index.get_level_values("index") == action.index
        else:
            mask &= np.array([idx[0] == action.index for idx in frame.index.tolist()], dtype=bool)
        if action.circuit is not None and "circuit" in frame.index.names:
            mask &= frame.index.get_level_values("circuit") == action.circuit
        if action.bus is not None and "bus" in frame.index.names:
            mask &= frame.index.get_level_values("bus") == action.bus
        if not mask.any():
            raise InvalidStudyInput(
                f"no rows matched {action.table}:{action.index} "
                f"(circuit={action.circuit!r}, bus={action.bus!r})"
            )
        frame.loc[mask, column] = action.resolved_value()
        return

    if action.index not in frame.index:
        raise InvalidStudyInput(f"{action.table}:{action.index} does not exist")
    frame.at[action.index, column] = action.resolved_value()


def _case_bus_results(net, *, min_vm_pu: float, max_vm_pu: float) -> pd.DataFrame:
    frame = net.res_bus.copy()
    non_ground = frame.index.get_level_values("phase") != 0
    frame["under_voltage"] = non_ground & (frame["vm_pu"] < min_vm_pu)
    frame["over_voltage"] = non_ground & (frame["vm_pu"] > max_vm_pu)
    frame["voltage_violation"] = frame["under_voltage"] | frame["over_voltage"]
    frame["is_isolated"] = False
    isolated = _isolated_terminals(net)
    for terminal in isolated:
        key = (terminal.bus, terminal.phase)
        if key in frame.index:
            frame.at[key, "is_isolated"] = True
    return frame


def _case_line_results(net, *, max_loading_percent: float) -> pd.DataFrame:
    if not hasattr(net, "res_line") or net.res_line.empty:
        return pd.DataFrame(
            columns=[
                "i_from_ka",
                "ia_from_degree",
                "i_to_ka",
                "ia_to_degree",
                "i_ka",
                "p_from_mw",
                "p_to_mw",
                "q_from_mvar",
                "q_to_mvar",
                "pl_mw",
                "ql_mvar",
                "vm_from_pu",
                "vm_to_pu",
                "va_from_degree",
                "va_to_degree",
                "loading_percent",
                "rating_ka",
                "loading_violation",
            ]
        )
    frame = net.res_line.copy()
    ratings = pd.Series(index=frame.index, dtype=float)
    for index, row in net.line.iterrows():
        ratings.at[index] = _line_rating_ka(net, row)
    frame["rating_ka"] = ratings
    frame["loading_percent"] = np.where(
        frame["rating_ka"] > 0,
        frame["i_ka"] / frame["rating_ka"] * 100.0,
        np.nan,
    )
    frame["loading_violation"] = frame["loading_percent"] > max_loading_percent
    return frame


def _case_trafo_results(net, *, max_loading_percent: float) -> pd.DataFrame:
    if not hasattr(net, "res_trafo") or net.res_trafo.empty:
        return pd.DataFrame(
            columns=[
                "p_mw",
                "q_mvar",
                "i_ka",
                "vm_pu",
                "va_degree",
                "pl_mw",
                "ql_mvar",
                "loading_percent",
                "loading_violation",
            ]
        )
    frame = net.res_trafo.copy()
    frame["loading_violation"] = frame["loading_percent"] > max_loading_percent
    return frame


def _line_rating_ka(net, line_row: pd.Series) -> float:
    std_type = line_row.get("std_type")
    model_type = line_row.get("model_type")
    if pd.isna(std_type) or pd.isna(model_type):
        return np.nan
    try:
        std_data = load_std_type(net, std_type, str(model_type))
    except Exception:
        return np.nan
    max_i_ka = std_data.get("max_i_ka")
    if isinstance(max_i_ka, (list, tuple, np.ndarray)):
        circuit = line_row.name[1] if isinstance(line_row.name, tuple) and len(line_row.name) > 1 else 0
        if circuit < len(max_i_ka):
            return float(max_i_ka[circuit])
        return np.nan
    return float(max_i_ka) if max_i_ka is not None else np.nan


def _isolated_terminals(net) -> set[TerminalRef]:
    isolated_y = set(int(value) for value in np.asarray(getattr(net.model, "y_isolated", []), dtype=int).tolist())
    if not isolated_y:
        return set()
    terminals: set[TerminalRef] = set()
    lookup = np.asarray(net.model.terminal_to_y_lookup, dtype=int)
    for bus, phase in net.bus.index.tolist():
        if int(phase) == 0:
            continue
        terminal = int(bus) * 4 + int(phase)
        y_index = lookup[terminal]
        if int(y_index) in isolated_y:
            terminals.add(TerminalRef(bus=int(bus), phase=int(phase)))
    return terminals


def _unsupplied_load(net, isolated: Iterable[TerminalRef]) -> tuple[float, float]:
    isolated_lookup = {(terminal.bus, terminal.phase) for terminal in isolated}
    if not isolated_lookup or net.asymmetric_load.empty:
        return 0.0, 0.0

    unsupplied_p_mw = 0.0
    unsupplied_q_mvar = 0.0
    for _, row in net.asymmetric_load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        if (int(row["bus"]), int(row["from_phase"])) not in isolated_lookup:
            continue
        scaling = float(row.get("scaling", 1.0) or 1.0)
        unsupplied_p_mw += float(row["p_mw"]) * scaling
        unsupplied_q_mvar += float(row["q_mvar"]) * scaling
    return unsupplied_p_mw, unsupplied_q_mvar


def _has_active_source(net) -> bool:
    for table_name in ("ext_grid", "ext_grid_sequence"):
        table = net.get(table_name)
        if table is None or len(table) == 0:
            continue
        if bool(table.get("in_service", pd.Series(dtype=bool)).fillna(False).astype(bool).any()):
            return True
    return False
