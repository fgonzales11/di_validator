from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from multiconductor.studies import InvalidStudyInput


@dataclass(frozen=True)
class ProfileBinding:
    """Map one profile series onto one network table/column selection."""

    table: str
    variable: str
    profile: str
    element: object | None = None
    phase: int | None = None
    circuit: int | None = None
    mode: str = "absolute"
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"absolute", "scale", "delta"}:
            raise InvalidStudyInput(
                "ProfileBinding.mode must be one of 'absolute', 'scale', or 'delta'"
            )


@runtime_checkable
class DataSourceProtocol(Protocol):
    def get_time_steps(self) -> list[object]:
        ...

    def get_time_step_values(self, time_step: object) -> dict[object, object]:
        ...

    def identity_hash(self) -> str:
        ...


class DataSource(ABC):
    """Abstract base class for canonical time-series profile sources."""

    @abstractmethod
    def get_time_steps(self) -> list[object]:
        """Return the ordered time-step identifiers."""

    @abstractmethod
    def get_time_step_values(self, time_step: object) -> dict[object, object]:
        """Return the profile values for a single time step."""

    @abstractmethod
    def identity_hash(self) -> str:
        """Return a stable digest for checkpoint/resume validation."""


def _ordered_unique(values: pd.Series) -> list[object]:
    seen: set[object] = set()
    ordered: list[object] = []
    for value in values.tolist():
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _hash_frame(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    payload = normalized.to_json(orient="split", date_format="iso", default_handler=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DFData(DataSource):
    """Data source backed by a pandas DataFrame.

    Supports two layouts:

    - Wide: one row per time step, one column per profile.
    - Long: one row per time step/profile/value triple.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        time_column: str = "TIME_STEP",
        profile_column: str | None = None,
        value_column: str = "VALUE",
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if time_column not in frame.columns:
            raise KeyError(f"frame must include {time_column!r}")
        self._frame = frame.copy()
        self._time_column = time_column
        self._profile_column = profile_column
        self._value_column = value_column
        self._time_steps = _ordered_unique(self._frame[time_column])
        self._step_map = self._build_step_map()

    def _build_step_map(self) -> dict[object, dict[object, object]]:
        if self._profile_column is None:
            profile_columns = [
                column for column in self._frame.columns if column != self._time_column
            ]
            if not profile_columns:
                raise InvalidStudyInput("wide DFData requires at least one profile column")
            step_map: dict[object, dict[object, object]] = {}
            for _, row in self._frame.iterrows():
                time_step = row[self._time_column]
                values = {
                    column: row[column]
                    for column in profile_columns
                }
                step_map[time_step] = values
            return step_map

        if self._profile_column not in self._frame.columns:
            raise KeyError(f"frame must include {self._profile_column!r}")
        if self._value_column not in self._frame.columns:
            raise KeyError(f"frame must include {self._value_column!r}")

        step_map = {}
        for _, row in self._frame.iterrows():
            time_step = row[self._time_column]
            profile = row[self._profile_column]
            step_map.setdefault(time_step, {})[profile] = row[self._value_column]
        return step_map

    def get_time_steps(self) -> list[object]:
        return list(self._time_steps)

    def get_time_step_values(self, time_step: object) -> dict[object, object]:
        return dict(self._step_map.get(time_step, {}))

    def identity_hash(self) -> str:
        return _hash_frame(self._frame)


class CSVData(DFData):
    """CSV-backed canonical data source.

    Parameters mirror :class:`DFData` after reading the CSV.
    """

    def __init__(
        self,
        csv_path: str = "ts.csv",
        *,
        time_column: str = "TIME_STEP",
        profile_column: str | None = None,
        value_column: str = "VALUE",
    ) -> None:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        super().__init__(
            frame,
            time_column=time_column,
            profile_column=profile_column,
            value_column=value_column,
        )
        self._csv_path = csv_path


class MC_CSVDataSource(DataSource):
    """Legacy CSV-backed three-phase load/generation source.

    This preserves the original notebook-facing schema and lookup contract.
    """

    REQUIRED_COLUMNS = [
        "REPORTED_DTTM",
        "MEASURE_VALUE_P_MW_APHASE",
        "MEASURE_VALUE_P_MW_BPHASE",
        "MEASURE_VALUE_P_MW_CPHASE",
        "MEASURE_VALUE_Q_MVAR_APHASE",
        "MEASURE_VALUE_Q_MVAR_BPHASE",
        "MEASURE_VALUE_Q_MVAR_CPHASE",
        "TYPE",
        "NAME",
    ]

    def __init__(self, csv_path: str = "ts.csv"):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        raw = pd.read_csv(csv_path)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise InvalidStudyInput(
                f"CSV is missing required columns: {', '.join(missing)}"
            )

        self._raw = raw.copy()
        self._time_steps = _ordered_unique(raw["REPORTED_DTTM"])
        self._df = raw.set_index(["REPORTED_DTTM", "NAME"])

    def get_time_steps(self) -> list[object]:
        return list(self._time_steps)

    def get_time_step_values(self, time_step: object) -> dict[object, object]:
        try:
            subset = self._df.loc[time_step]
        except KeyError:
            return {}

        result: dict[object, object] = {}
        if isinstance(subset, pd.Series):
            name = subset.name
            result[(name, subset["TYPE"])] = {
                "p_mw_a": subset["MEASURE_VALUE_P_MW_APHASE"],
                "p_mw_b": subset["MEASURE_VALUE_P_MW_BPHASE"],
                "p_mw_c": subset["MEASURE_VALUE_P_MW_CPHASE"],
                "q_mvar_a": subset["MEASURE_VALUE_Q_MVAR_APHASE"],
                "q_mvar_b": subset["MEASURE_VALUE_Q_MVAR_BPHASE"],
                "q_mvar_c": subset["MEASURE_VALUE_Q_MVAR_CPHASE"],
            }
            return result

        for name, row in subset.iterrows():
            result[(name, row["TYPE"])] = {
                "p_mw_a": row["MEASURE_VALUE_P_MW_APHASE"],
                "p_mw_b": row["MEASURE_VALUE_P_MW_BPHASE"],
                "p_mw_c": row["MEASURE_VALUE_P_MW_CPHASE"],
                "q_mvar_a": row["MEASURE_VALUE_Q_MVAR_APHASE"],
                "q_mvar_b": row["MEASURE_VALUE_Q_MVAR_BPHASE"],
                "q_mvar_c": row["MEASURE_VALUE_Q_MVAR_CPHASE"],
            }
        return result

    def get_element_names(self, element_type: str) -> list[object]:
        mask = self._df["TYPE"] == element_type
        return list(self._df.loc[mask].index.get_level_values("NAME").unique())

    def identity_hash(self) -> str:
        return _hash_frame(self._raw)


def bindings_identity_hash(bindings: list[ProfileBinding] | None) -> str:
    if bindings is None:
        return ""
    payload = json.dumps(
        [
            {
                "table": binding.table,
                "variable": binding.variable,
                "profile": binding.profile,
                "element": binding.element,
                "phase": binding.phase,
                "circuit": binding.circuit,
                "mode": binding.mode,
                "scale_factor": binding.scale_factor,
            }
            for binding in bindings
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
