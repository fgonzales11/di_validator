"""Fuse and overcurrent-relay protection devices."""

from __future__ import annotations

import json
from typing import Hashable, Iterable

import numpy as np
import pandas as pd

from multiconductor.protection.base import ProtectionDevice, _switch_rows, switch_current_ka


_CURVES = {
    "standard_inverse": (0.14, 0.02),
    "very_inverse": (13.5, 1.0),
    "extremely_inverse": (80.0, 2.0),
    "long_inverse": (120.0, 1.0),
}


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not valid or not np.isfinite(value):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} value")
    return value


class Fuse(ProtectionDevice):
    """Fuse with a user-supplied or generic log-log time-current curve.

    ``characteristic`` contains ``(current_a, time_s)`` pairs.  A generic
    engineering curve is used when only ``rated_i_a`` is supplied; it is useful
    for studies but is not a manufacturer-certified characteristic.
    """

    def __init__(
        self,
        net,
        switch_index: Hashable,
        fuse_type: str | None = None,
        rated_i_a: float = 0.0,
        characteristic: Iterable[tuple[float, float]] | None = None,
        *,
        current_a: Iterable[float] | None = None,
        time_s: Iterable[float] | None = None,
        opening_time_s: float | None = 0.0,
        curve_kind: str | None = "generic",
        in_service: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        index: int | None = None,
        _register: bool = True,
        **_kwargs,
    ) -> None:
        self.fuse_type = fuse_type or "generic"
        std_type = net.get("std_types", {}).get("fuse", {}).get(self.fuse_type)
        if std_type is not None:
            rated_i_a = float(std_type.get("i_rated_a", rated_i_a))
            if characteristic is None and current_a is None:
                if np.ndim(std_type.get("t_avg", 0)):
                    current_a, time_s = std_type["x_avg"], std_type["t_avg"]
                elif np.ndim(std_type.get("t_min", 0)):
                    current_a, time_s = std_type["x_min"], std_type["t_min"]
        self.rated_i_a = _positive(rated_i_a, "rated_i_a")

        if characteristic is not None:
            points = np.asarray(list(characteristic), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("characteristic must contain (current_a, time_s) pairs")
            currents, times = points[:, 0], points[:, 1]
        elif current_a is not None or time_s is not None:
            if current_a is None or time_s is None:
                raise ValueError("current_a and time_s must be supplied together")
            currents = np.asarray(list(current_a), dtype=float)
            times = np.asarray(list(time_s), dtype=float)
        else:
            multipliers = np.asarray([1.25, 2.0, 4.0, 10.0, 20.0])
            currents = self.rated_i_a * multipliers
            times = np.asarray([3600.0, 100.0, 10.0, 0.1, 0.01])
        if len(currents) < 2 or len(currents) != len(times):
            raise ValueError("fuse characteristic requires at least two current/time points")
        if (
            not np.all(np.isfinite(currents))
            or not np.all(np.isfinite(times))
            or np.any(currents <= 0)
            or np.any(times <= 0)
            or np.any(np.diff(currents) <= 0)
        ):
            raise ValueError("fuse currents must increase strictly and all curve values must be positive")
        self.current_a = currents
        self.time_s = times
        super().__init__(
            net,
            switch_index,
            opening_time_s=opening_time_s,
            curve_kind=curve_kind,
            index=index,
            in_service=in_service,
            overwrite=overwrite,
            name=name,
            _register=_register,
        )

    def to_settings(self) -> dict:
        settings = super().to_settings()
        settings.update(
            {
                "fuse_type": self.fuse_type,
                "rated_i_a": self.rated_i_a,
                "characteristic": [
                    [float(current), float(time)]
                    for current, time in zip(self.current_a, self.time_s)
                ],
            }
        )
        return settings

    @property
    def i_start_a(self) -> float:
        return float(self.current_a[0])

    @property
    def i_stop_a(self) -> float:
        return float(self.current_a[-1])

    def trip_time(self, current_ka: float) -> tuple[bool, float, str | None]:
        current_a = abs(float(current_ka)) * 1000.0
        if current_a < self.i_start_a:
            return False, np.inf, None
        if current_a > self.i_stop_a:
            return True, 0.0, "high-current"
        time = np.exp(
            np.interp(np.log(current_a), np.log(self.current_a), np.log(self.time_s))
        )
        return True, float(time), "melting"


class DTOC(ProtectionDevice):
    """Definite-time overcurrent relay with an optional high-set stage."""

    def __init__(
        self,
        net,
        switch_index: Hashable,
        pickup_ka: float,
        delay_s: float,
        *,
        instantaneous_ka: float | None = None,
        instantaneous_delay_s: float = 0.0,
        opening_time_s: float | None = None,
        curve_kind: str | None = "relay_operating",
        in_service: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        index: int | None = None,
        _register: bool = True,
    ) -> None:
        self.pickup_ka = _positive(pickup_ka, "pickup_ka")
        self.delay_s = _positive(delay_s, "delay_s", allow_zero=True)
        self.instantaneous_ka = (
            None if instantaneous_ka is None else _positive(instantaneous_ka, "instantaneous_ka")
        )
        self.instantaneous_delay_s = _positive(
            instantaneous_delay_s, "instantaneous_delay_s", allow_zero=True
        )
        if self.instantaneous_ka is not None and self.instantaneous_ka < self.pickup_ka:
            raise ValueError("instantaneous_ka must be greater than or equal to pickup_ka")
        self.oc_relay_type = "DTOC"
        self.I_g = self.pickup_ka
        self.I_gg = self.instantaneous_ka
        self.t_g = self.delay_s
        self.t_gg = self.instantaneous_delay_s
        super().__init__(
            net,
            switch_index,
            opening_time_s=opening_time_s,
            curve_kind=curve_kind,
            index=index,
            in_service=in_service,
            overwrite=overwrite,
            name=name,
            _register=_register,
        )

    def to_settings(self) -> dict:
        settings = super().to_settings()
        settings.update(
            {
                "pickup_ka": self.pickup_ka,
                "delay_s": self.delay_s,
                "instantaneous_ka": self.instantaneous_ka,
                "instantaneous_delay_s": self.instantaneous_delay_s,
            }
        )
        return settings

    def trip_time(self, current_ka: float) -> tuple[bool, float, str | None]:
        current_ka = abs(float(current_ka))
        if self.instantaneous_ka is not None and current_ka >= self.instantaneous_ka:
            return True, self.instantaneous_delay_s, "instantaneous"
        if current_ka >= self.pickup_ka:
            return True, self.delay_s, "definite-time"
        return False, np.inf, None


class IDMT(ProtectionDevice):
    """IEC inverse-definite-minimum-time overcurrent relay."""

    def __init__(
        self,
        net,
        switch_index: Hashable,
        pickup_ka: float,
        time_multiplier_s: float,
        *,
        curve: str = "standard_inverse",
        time_delay_s: float = 0.0,
        opening_time_s: float | None = None,
        curve_kind: str | None = "relay_operating",
        in_service: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        index: int | None = None,
        _register: bool = True,
    ) -> None:
        self.pickup_ka = _positive(pickup_ka, "pickup_ka")
        self.time_multiplier_s = _positive(time_multiplier_s, "time_multiplier_s")
        self.curve = str(curve).lower().replace(" ", "_").replace("-", "_")
        if self.curve not in _CURVES:
            raise ValueError(f"curve must be one of {sorted(_CURVES)}")
        self.time_delay_s = _positive(time_delay_s, "time_delay_s", allow_zero=True)
        self.oc_relay_type = "IDMT"
        self.curve_type = self.curve
        self.I_s = self.pickup_ka
        self.tms = self.time_multiplier_s
        self.t_grade = self.time_delay_s
        self.k, self.alpha = _CURVES[self.curve]
        super().__init__(
            net,
            switch_index,
            opening_time_s=opening_time_s,
            curve_kind=curve_kind,
            index=index,
            in_service=in_service,
            overwrite=overwrite,
            name=name,
            _register=_register,
        )

    def to_settings(self) -> dict:
        settings = super().to_settings()
        settings.update(
            {
                "pickup_ka": self.pickup_ka,
                "time_multiplier_s": self.time_multiplier_s,
                "curve": self.curve,
                "time_delay_s": self.time_delay_s,
            }
        )
        return settings

    def trip_time(self, current_ka: float) -> tuple[bool, float, str | None]:
        multiple = abs(float(current_ka)) / self.pickup_ka
        if multiple <= 1.0:
            return False, np.inf, None
        time = (
            self.time_multiplier_s * self.k / (multiple**self.alpha - 1.0)
            + self.time_delay_s
        )
        return True, float(time), "inverse-time"


class IDTOC(ProtectionDevice):
    """Combined inverse-time and definite/high-set overcurrent relay."""

    def __init__(
        self,
        net,
        switch_index: Hashable,
        pickup_ka: float,
        time_multiplier_s: float,
        definite_pickup_ka: float,
        definite_delay_s: float,
        *,
        instantaneous_ka: float | None = None,
        instantaneous_delay_s: float = 0.0,
        curve: str = "standard_inverse",
        time_delay_s: float = 0.0,
        opening_time_s: float | None = None,
        curve_kind: str | None = "relay_operating",
        in_service: bool = True,
        overwrite: bool = False,
        name: str | None = None,
        index: int | None = None,
        _register: bool = True,
    ) -> None:
        self.pickup_ka = _positive(pickup_ka, "pickup_ka")
        self.time_multiplier_s = _positive(time_multiplier_s, "time_multiplier_s")
        self.definite_pickup_ka = _positive(definite_pickup_ka, "definite_pickup_ka")
        self.definite_delay_s = _positive(definite_delay_s, "definite_delay_s", allow_zero=True)
        self.instantaneous_ka = (
            None if instantaneous_ka is None else _positive(instantaneous_ka, "instantaneous_ka")
        )
        self.instantaneous_delay_s = _positive(
            instantaneous_delay_s, "instantaneous_delay_s", allow_zero=True
        )
        self.curve = str(curve).lower().replace(" ", "_").replace("-", "_")
        if self.curve not in _CURVES:
            raise ValueError(f"curve must be one of {sorted(_CURVES)}")
        self.time_delay_s = _positive(time_delay_s, "time_delay_s", allow_zero=True)
        if self.definite_pickup_ka < self.pickup_ka:
            raise ValueError("definite_pickup_ka must be greater than or equal to pickup_ka")
        if self.instantaneous_ka is not None and self.instantaneous_ka < self.definite_pickup_ka:
            raise ValueError(
                "instantaneous_ka must be greater than or equal to definite_pickup_ka"
            )
        self.oc_relay_type = "IDTOC"
        self.curve_type = self.curve
        self.I_s = self.pickup_ka
        self.I_g = self.definite_pickup_ka
        self.I_gg = self.instantaneous_ka
        self.tms = self.time_multiplier_s
        self.t_grade = self.time_delay_s
        self.t_g = self.definite_delay_s
        self.t_gg = self.instantaneous_delay_s
        self.k, self.alpha = _CURVES[self.curve]
        super().__init__(
            net,
            switch_index,
            opening_time_s=opening_time_s,
            curve_kind=curve_kind,
            index=index,
            in_service=in_service,
            overwrite=overwrite,
            name=name,
            _register=_register,
        )

    def to_settings(self) -> dict:
        settings = super().to_settings()
        settings.update(
            {
                "pickup_ka": self.pickup_ka,
                "time_multiplier_s": self.time_multiplier_s,
                "definite_pickup_ka": self.definite_pickup_ka,
                "definite_delay_s": self.definite_delay_s,
                "instantaneous_ka": self.instantaneous_ka,
                "instantaneous_delay_s": self.instantaneous_delay_s,
                "curve": self.curve,
                "time_delay_s": self.time_delay_s,
            }
        )
        return settings

    def trip_time(self, current_ka: float) -> tuple[bool, float, str | None]:
        current_ka = abs(float(current_ka))
        candidates: list[tuple[float, str]] = []
        if self.instantaneous_ka is not None and current_ka >= self.instantaneous_ka:
            candidates.append((self.instantaneous_delay_s, "instantaneous"))
        if current_ka >= self.definite_pickup_ka:
            candidates.append((self.definite_delay_s, "definite-time"))
        multiple = current_ka / self.pickup_ka
        if multiple > 1.0:
            inverse_time = (
                self.time_multiplier_s * self.k / (multiple**self.alpha - 1.0)
                + self.time_delay_s
            )
            candidates.append((float(inverse_time), "inverse-time"))
        if not candidates:
            return False, np.inf, None
        time, stage = min(candidates, key=lambda item: item[0])
        return True, float(time), stage


def create_fuse(net, switch_index: Hashable, **kwargs) -> Fuse:
    return Fuse(net, switch_index, **kwargs)


def create_dtoc(net, switch_index: Hashable, **kwargs) -> DTOC:
    return DTOC(net, switch_index, **kwargs)


def create_idmt(net, switch_index: Hashable, **kwargs) -> IDMT:
    return IDMT(net, switch_index, **kwargs)


def create_idtoc(net, switch_index: Hashable, **kwargs) -> IDTOC:
    return IDTOC(net, switch_index, **kwargs)


def create_overcurrent_relay(net, switch_index: Hashable, relay_type: str, **kwargs):
    relay_type = str(relay_type).upper()
    classes = {"DTOC": DTOC, "IDMT": IDMT, "IDTOC": IDTOC}
    try:
        relay_class = classes[relay_type]
    except KeyError as exc:
        raise ValueError("relay_type must be DTOC, IDMT or IDTOC") from exc
    return relay_class(net, switch_index, **kwargs)


def _switch_rated_current_ka(net, switch_index: Hashable) -> float:
    switches = _switch_rows(net, switch_index)
    ratings = []
    for _, switch in switches.iterrows():
        if str(switch.get("et", "")).lower() != "l":
            continue
        line_id = int(switch["element"])
        if isinstance(net.line.index, pd.MultiIndex):
            line_rows = net.line.xs(line_id, level=0, drop_level=False)
        else:
            line_rows = net.line.loc[[line_id]]
        row = line_rows.iloc[0]
        if "max_i_ka" in line_rows:
            values = pd.to_numeric(line_rows["max_i_ka"], errors="coerce")
            ratings.extend(values[np.isfinite(values) & (values > 0)].tolist())
            continue
        std = net.get("std_types", {}).get(str(row["model_type"]), {}).get(row["std_type"], {})
        value = std.get("max_i_ka", std.get("Imax"))
        if value is not None:
            array = np.asarray(value, dtype=float).reshape(-1)
            ratings.extend(array[np.isfinite(array) & (array > 0)].tolist())
    if not ratings:
        raise ValueError(
            "automatic relay pickup requires a positive line current rating; "
            "supply pickup_current_manual or explicit pickup_ka values"
        )
    return float(min(ratings))


def OCRelay(net, switch_index: Hashable, oc_relay_type: str, time_settings=None,
            pickup_current_manual=None, **kwargs):
    """Compatibility factory for pandapower's ``OCRelay`` constructor.

    New code should prefer the explicit DTOC, IDMT and IDTOC constructors.
    Manual pickup values may be a mapping, Series, or one-row DataFrame.
    """

    relay_type = str(oc_relay_type).upper()
    manual = pickup_current_manual
    if isinstance(manual, pd.DataFrame):
        if "switch_id" in manual and (manual["switch_id"] == switch_index).any():
            manual = manual.loc[manual["switch_id"] == switch_index].iloc[0]
        else:
            manual = manual.iloc[0]
    if manual is not None and not isinstance(manual, (dict, pd.Series)):
        raise TypeError("pickup_current_manual must be a mapping, Series or DataFrame")
    manual = {} if manual is None else dict(manual)
    setting_map = {}
    if isinstance(time_settings, pd.DataFrame):
        if "switch_id" in time_settings and (time_settings["switch_id"] == switch_index).any():
            setting_map = dict(
                time_settings.loc[time_settings["switch_id"] == switch_index].iloc[0]
            )
        elif not time_settings.empty:
            setting_map = dict(time_settings.iloc[0])
        settings = []
    else:
        settings = [] if time_settings is None else list(time_settings)
    overload_factor = float(kwargs.pop("overload_factor", 1.2))
    ct_current_factor = float(kwargs.pop("ct_current_factor", 1.25))
    safety_factor = float(kwargs.pop("safety_factor", 1.0))
    inverse_overload_factor = float(kwargs.pop("inverse_overload_factor", 1.2))
    kwargs.pop("sc_fraction", None)
    if "curve_type" in kwargs:
        kwargs.setdefault("curve", kwargs.pop("curve_type"))

    rated_current = None
    if not manual:
        rated_current = _switch_rated_current_ka(net, switch_index)
    automatic_definite = (
        None if rated_current is None else rated_current * overload_factor * ct_current_factor
    )
    automatic_inverse = (
        None if rated_current is None else rated_current * inverse_overload_factor
    )

    if relay_type == "DTOC":
        kwargs.setdefault(
            "pickup_ka",
            manual.get("I_g", automatic_definite),
        )
        if "instantaneous_ka" not in kwargs:
            kwargs["instantaneous_ka"] = manual.get("I_gg")
            if kwargs["instantaneous_ka"] is None:
                kwargs["instantaneous_ka"] = switch_current_ka(
                    net, switch_index, "sc"
                ) * safety_factor
        if len(settings) >= 2:
            kwargs.setdefault("instantaneous_delay_s", settings[0])
            kwargs.setdefault("delay_s", settings[1])
        kwargs.setdefault("instantaneous_delay_s", setting_map.get("t_gg", 0.0))
        if "delay_s" not in kwargs and "t_g" in setting_map:
            kwargs["delay_s"] = setting_map["t_g"]
    elif relay_type == "IDMT":
        kwargs.setdefault(
            "pickup_ka", manual.get("I_s", automatic_inverse)
        )
        if len(settings) >= 2:
            kwargs.setdefault("time_multiplier_s", settings[0])
            kwargs.setdefault("time_delay_s", settings[1])
        if "time_multiplier_s" not in kwargs and "tms" in setting_map:
            kwargs["time_multiplier_s"] = setting_map["tms"]
        kwargs.setdefault("time_delay_s", setting_map.get("t_grade", 0.0))
    elif relay_type == "IDTOC":
        kwargs.setdefault(
            "pickup_ka", manual.get("I_s", automatic_inverse)
        )
        kwargs.setdefault(
            "definite_pickup_ka",
            manual.get("I_g", automatic_definite),
        )
        if "instantaneous_ka" not in kwargs:
            kwargs["instantaneous_ka"] = manual.get("I_gg")
            if kwargs["instantaneous_ka"] is None:
                kwargs["instantaneous_ka"] = switch_current_ka(
                    net, switch_index, "sc"
                ) * safety_factor
        if len(settings) >= 5:
            kwargs.setdefault("instantaneous_delay_s", settings[0])
            kwargs.setdefault("definite_delay_s", settings[1])
            kwargs.setdefault("time_multiplier_s", settings[3])
            kwargs.setdefault("time_delay_s", settings[4])
        kwargs.setdefault("instantaneous_delay_s", setting_map.get("t_gg", 0.0))
        if "definite_delay_s" not in kwargs and "t_g" in setting_map:
            kwargs["definite_delay_s"] = setting_map["t_g"]
        if "time_multiplier_s" not in kwargs and "tms" in setting_map:
            kwargs["time_multiplier_s"] = setting_map["tms"]
        kwargs.setdefault("time_delay_s", setting_map.get("t_grade", 0.0))
    return create_overcurrent_relay(net, switch_index, relay_type, **kwargs)


def create_oc_relay(
    net,
    switch_index: Hashable,
    relay_type: str | None = None,
    *,
    oc_relay_type: str | None = None,
    time_settings=None,
    pickup_current_manual=None,
    **kwargs,
):
    """Create a relay using either the explicit or pandapower-style API."""

    selected_type = relay_type or oc_relay_type
    if selected_type is None:
        raise TypeError("relay_type or oc_relay_type is required")
    if time_settings is not None or pickup_current_manual is not None or oc_relay_type is not None:
        return OCRelay(
            net,
            switch_index,
            selected_type,
            time_settings=time_settings,
            pickup_current_manual=pickup_current_manual,
            **kwargs,
        )
    return create_overcurrent_relay(net, switch_index, selected_type, **kwargs)


def device_from_record(net, index: int, row: pd.Series) -> ProtectionDevice:
    """Rebuild a runtime device from one normalized protection-table row."""

    settings = row.get("settings", {})
    if isinstance(settings, str):
        settings = json.loads(settings)
    if not isinstance(settings, dict):
        raise ValueError(f"protection row {index} has invalid serialized settings")
    settings = dict(settings)
    switch_index = settings.pop("switch_index", None)
    if isinstance(switch_index, list):
        switch_index = tuple(switch_index)
    if switch_index is None:
        element = int(row["element"])
        circuit = row.get("circuit", np.nan)
        switch_index = element if pd.isna(circuit) else (element, int(circuit))
    classes = {"Fuse": Fuse, "DTOC": DTOC, "IDMT": IDMT, "IDTOC": IDTOC}
    try:
        device_class = classes[str(row["device_type"])]
    except KeyError as exc:
        raise ValueError(
            f"protection row {index} has unsupported device_type {row['device_type']!r}"
        ) from exc
    device = device_class(
        net,
        switch_index,
        index=int(index),
        in_service=bool(row.get("in_service", True)),
        name=row.get("name"),
        _register=False,
        **settings,
    )
    device.tripped = bool(row.get("tripped", False))
    return device
