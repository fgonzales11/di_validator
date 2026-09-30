"""Protection-device base classes and current measurement helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable
import weakref

import numpy as np
import pandas as pd

from multiconductor.studies import InvalidStudyInput

_PROTECTION_COLUMNS = [
    "name",
    "element_type",
    "element",
    "phase",
    "circuit",
    "side",
    "device_type",
    "curve_id",
    "pickup_current_a",
    "instantaneous_current_a",
    "time_dial",
    "time_grading_s",
    "ct_ratio",
    "opening_time_s",
    "curve_kind",
    "in_service",
    "settings",
    "tripped",
]
_PROTECTION_CURVE_COLUMNS = [
    "curve_id",
    "curve_type",
    "points",
    "metadata",
]
_DEVICE_REGISTRY: dict[int, dict[int, "ProtectionDevice"]] = {}
_REGISTRY_FINALIZERS: dict[int, weakref.finalize] = {}


def _registry(net) -> dict[int, "ProtectionDevice"]:
    key = id(net)
    if key not in _DEVICE_REGISTRY:
        _DEVICE_REGISTRY[key] = {}
        try:
            _REGISTRY_FINALIZERS[key] = weakref.finalize(
                net,
                lambda registry_key=key: (
                    _DEVICE_REGISTRY.pop(registry_key, None),
                    _REGISTRY_FINALIZERS.pop(registry_key, None),
                ),
            )
        except TypeError:
            pass
    return _DEVICE_REGISTRY[key]


def _ensure_protection_table(net) -> pd.DataFrame:
    table = net.get("protection")
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(columns=_PROTECTION_COLUMNS)
    if "object" in table:
        for index, device in table["object"].dropna().items():
            _registry(net)[int(index)] = device
        table = table.drop(columns=["object"])
    if "switch_index" in table:
        table = table.drop(columns=["switch_index"])
    for column in _PROTECTION_COLUMNS:
        if column not in table:
            table[column] = pd.Series(dtype=object)
    net["protection"] = table
    return table


def _ensure_protection_curve_table(net) -> pd.DataFrame:
    table = net.get("protection_curve")
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(columns=_PROTECTION_CURVE_COLUMNS)
    if "object" in table:
        table = table.drop(columns=["object"])
    for column in _PROTECTION_CURVE_COLUMNS:
        if column not in table:
            table[column] = pd.Series(dtype=object)
    net["protection_curve"] = table
    return table


def register_curve(
    net,
    curve_id: str,
    curve_type: str,
    points,
    metadata,
) -> str:
    table = _ensure_protection_curve_table(net)
    normalized_points = [
        [float(pair[0]), float(pair[1])] for pair in (points or [])
    ]
    normalized_metadata = dict(metadata or {})
    row = pd.Series(
        {
            "curve_id": curve_id,
            "curve_type": str(curve_type),
            "points": normalized_points,
            "metadata": normalized_metadata,
        }
    )
    existing = table.index[table["curve_id"] == curve_id]
    if len(existing):
        table.loc[existing[0], row.index] = row.values
    else:
        next_index = max((int(value) for value in table.index), default=-1) + 1
        for column, value in row.items():
            table.at[next_index, column] = value
    return curve_id


def protection_devices(net) -> list[tuple[int, "ProtectionDevice"]]:
    """Return runtime devices, rebuilding them from normalized rows if needed."""

    table = _ensure_protection_table(net)
    registry = _registry(net)
    for index, row in table.iterrows():
        index = int(index)
        if index not in registry:
            from multiconductor.protection.devices import device_from_record

            registry[index] = device_from_record(net, index, row)
    return [(int(index), registry[int(index)]) for index in table.index]


def _switch_rows(net, switch_index: Hashable) -> pd.DataFrame:
    if isinstance(switch_index, tuple):
        try:
            return net.switch.loc[[switch_index]]
        except KeyError as exc:
            raise KeyError(f"switch {switch_index!r} does not exist") from exc
    if isinstance(net.switch.index, pd.MultiIndex):
        try:
            return net.switch.xs(switch_index, level=0, drop_level=False)
        except KeyError as exc:
            raise KeyError(f"switch {switch_index!r} does not exist") from exc
    try:
        return net.switch.loc[[switch_index]]
    except KeyError as exc:
        raise KeyError(f"switch {switch_index!r} does not exist") from exc


def _result_rows(result: pd.DataFrame, switch_index: Hashable) -> pd.DataFrame:
    if isinstance(switch_index, tuple):
        if switch_index in result.index:
            return result.loc[[switch_index]]
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels > len(switch_index):
            try:
                return result.xs(switch_index, level=list(range(len(switch_index))), drop_level=False)
            except KeyError:
                pass
        raise KeyError(f"no result is available for switch {switch_index!r}")
    if isinstance(result.index, pd.MultiIndex):
        try:
            return result.xs(switch_index, level=0, drop_level=False)
        except KeyError as exc:
            raise KeyError(f"no result is available for switch {switch_index!r}") from exc
    try:
        return result.loc[[switch_index]]
    except KeyError as exc:
        raise KeyError(f"no result is available for switch {switch_index!r}") from exc


def _line_current_for_switch(net, switch_index: Hashable, scenario: str) -> float:
    switches = _switch_rows(net, switch_index)
    result_name = "res_line_sc" if scenario == "sc" else "res_line"
    result = net.get(result_name)
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError(
            f"{result_name} is unavailable; run the corresponding network calculation first"
        )
    currents: list[float] = []
    for switch_key, switch in switches.iterrows():
        if str(switch.get("et", "")).lower() != "l":
            continue
        line_id = int(switch["element"])
        phase = int(switch.get("phase", switch_key[-1] if isinstance(switch_key, tuple) else 1))
        try:
            line_rows = net.line.xs(line_id, level=0, drop_level=False)
        except (KeyError, TypeError):
            line_rows = net.line.loc[[line_id]]
        if "from_phase" in line_rows:
            candidates = line_rows.loc[
                (line_rows["from_phase"].astype(int) == phase)
                | (line_rows["to_phase"].astype(int) == phase)
            ]
            if not candidates.empty:
                line_index = candidates.index[0]
                line = candidates.iloc[0]
            else:
                continue
        else:
            line_index = line_rows.index[0]
            line = line_rows.iloc[0]
        side = "from" if int(switch["bus"]) == int(line["from_bus"]) else "to"
        if scenario == "sc":
            column = f"ikss_{side}_ka" if f"ikss_{side}_ka" in result else "ikss_ka"
        else:
            column = f"i_{side}_ka" if f"i_{side}_ka" in result else "i_ka"
        value = result.loc[line_index, column]
        if isinstance(value, pd.Series):
            value = pd.to_numeric(value, errors="coerce")
            value = value[np.isfinite(value)]
            if value.empty:
                continue
            currents.append(float(np.max(np.abs(value))))
        else:
            currents.append(float(value))
    if not currents:
        raise ValueError(
            f"switch {switch_index!r} has no supported line-current measurement"
        )
    return float(np.nanmax(np.abs(currents)))


def switch_current_ka(net, switch_index: Hashable, scenario: str = "sc") -> float:
    """Return the maximum phase current measured by a switch protection device."""

    scenario = str(scenario).lower()
    if scenario not in {"sc", "pp"}:
        raise InvalidStudyInput('scenario must be either "sc" or "pp"')
    result_name = "res_switch_sc" if scenario == "sc" else "res_switch"
    result = net.get(result_name)
    if isinstance(result, pd.DataFrame) and not result.empty:
        column = "ikss_ka" if scenario == "sc" else "i_ka"
        if column in result:
            values = pd.to_numeric(_result_rows(result, switch_index)[column], errors="coerce")
            finite = values[np.isfinite(values)]
            if not finite.empty:
                return float(np.max(np.abs(finite)))
    return _line_current_for_switch(net, switch_index, scenario)


class ProtectionDevice(ABC):
    """Base class for switch-associated multiconductor protection devices."""

    def __init__(
        self,
        net,
        switch_index: Hashable,
        *,
        ct_ratio: float = 1.0,
        opening_time_s: float | None = None,
        curve_kind: str | None = None,
        index: int | None = None,
        in_service: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        _register: bool = True,
    ) -> None:
        _switch_rows(net, switch_index)
        table = _ensure_protection_table(net)
        if overwrite and not table.empty:
            switch_element = switch_index[0] if isinstance(switch_index, tuple) else switch_index
            same_switch = pd.to_numeric(table["element"], errors="coerce") == int(switch_element)
            removed = [int(value) for value in table.index[same_switch]]
            table.drop(index=table.index[same_switch], inplace=True)
            for removed_index in removed:
                _registry(net).pop(removed_index, None)
        if index is None:
            numeric = [int(value) for value in table.index if isinstance(value, (int, np.integer))]
            index = max(numeric, default=-1) + 1
        if _register and index in table.index:
            raise UserWarning(f"protection device index {index} already exists")

        self.index = int(index)
        self.switch_index = switch_index
        self.in_service = bool(in_service)
        self.name = name
        self.ct_ratio = float(ct_ratio)
        if not np.isfinite(self.ct_ratio) or self.ct_ratio <= 0.0:
            raise ValueError("ct_ratio must be a finite positive value")
        self.opening_time_s = (
            None if opening_time_s is None else float(opening_time_s)
        )
        if self.opening_time_s is not None and (
            not np.isfinite(self.opening_time_s) or self.opening_time_s < 0.0
        ):
            raise ValueError("opening_time_s must be a finite non-negative value")
        self.curve_kind = None if curve_kind is None else str(curve_kind).strip()
        self.activation_parameter = "i_ka"
        self.tripped = False

        if _register:
            switch_rows = _switch_rows(net, switch_index)
            switch_element = switch_index[0] if isinstance(switch_index, tuple) else switch_index
            switch_circuit = switch_index[1] if isinstance(switch_index, tuple) else np.nan
            switch_phase = (
                int(switch_rows.iloc[0]["phase"])
                if isinstance(switch_index, tuple) and "phase" in switch_rows
                else np.nan
            )
            curve_id = register_curve(
                net,
                self.default_curve_id(),
                type(self).__name__,
                self.curve_points(),
                self.curve_metadata(),
            )
            self.curve_id = curve_id
            record = {
                "name": name,
                "element_type": "switch",
                "element": int(switch_element),
                "phase": switch_phase,
                "circuit": switch_circuit,
                "side": None,
                "device_type": type(self).__name__,
                "curve_id": curve_id,
                "pickup_current_a": 1000.0
                * float(getattr(self, "pickup_ka", getattr(self, "rated_i_a", np.nan))),
                "instantaneous_current_a": 1000.0
                * float(getattr(self, "instantaneous_ka", np.nan) or np.nan),
                "time_dial": getattr(self, "time_multiplier_s", np.nan),
                "time_grading_s": getattr(
                    self, "time_delay_s", getattr(self, "delay_s", np.nan)
                ),
                "ct_ratio": self.ct_ratio,
                "opening_time_s": self.opening_time_s,
                "curve_kind": self.curve_kind,
                "in_service": self.in_service,
                "settings": self.to_settings(),
                "tripped": False,
            }
            for column, value in record.items():
                table.at[self.index, column] = value
            _registry(net)[self.index] = self
        else:
            self.curve_id = getattr(self, "curve_id", None)

    def default_curve_id(self) -> str:
        return f"{type(self).__name__}:{self.index}"

    def curve_points(self) -> list[list[float]]:
        return []

    def curve_metadata(self) -> dict:
        return {
            "device_type": type(self).__name__,
            "curve_kind": self.curve_kind,
            "opening_time_s": self.opening_time_s,
        }

    def to_settings(self) -> dict:
        if isinstance(self.switch_index, tuple):
            switch_index = [
                int(value) if isinstance(value, (int, np.integer)) else value
                for value in self.switch_index
            ]
        else:
            switch_index = (
                int(self.switch_index)
                if isinstance(self.switch_index, (int, np.integer))
                else self.switch_index
            )
        return {
            "switch_index": switch_index,
            "ct_ratio": self.ct_ratio,
            "opening_time_s": self.opening_time_s,
            "curve_kind": self.curve_kind,
        }

    def reset_device(self) -> None:
        self.tripped = False

    def has_tripped(self) -> bool:
        return bool(self.tripped)

    def status_to_net(self, net) -> None:
        rows = _switch_rows(net, self.switch_index)
        net.switch.loc[rows.index, "closed"] = not self.tripped

    @abstractmethod
    def trip_time(self, current_ka: float) -> tuple[bool, float, str | None]:
        """Return ``(trip, time_s, stage)`` for a measured current."""

    def protection_function(self, net, scenario: str = "sc") -> dict:
        current_ka = switch_current_ka(net, self.switch_index, scenario) / self.ct_ratio
        trip, operating_time_s, stage = self.trip_time(current_ka)
        opening_time_s = 0.0 if self.opening_time_s is None else self.opening_time_s
        total_time_s = (
            float(operating_time_s) + opening_time_s
            if trip and np.isfinite(operating_time_s)
            else float(operating_time_s)
        )
        self.tripped = bool(trip)
        table = _ensure_protection_table(net)
        if self.index in table.index:
            table.loc[self.index, "tripped"] = self.tripped
        return {
            "device_index": self.index,
            "switch_id": self.switch_index,
            "protection_type": type(self).__name__,
            "trip_melt": self.tripped,
            "trip": self.tripped,
            "stage": stage,
            "activation_parameter": self.activation_parameter,
            "activation_parameter_value": current_ka,
            "operating_time_s": float(operating_time_s),
            "opening_time_s": opening_time_s,
            "trip_melt_time_s": float(operating_time_s),
            "trip_time_s": total_time_s,
        }
