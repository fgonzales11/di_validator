"""Network protector element model for multiconductor networks.

The multiconductor solver does not have a native network-protector branch
element. This module stores protectors in ``net.network_protector`` and, when
requested, projects each protector into phase-level bus-bus switches so the
existing topology and power-flow machinery sees the open/closed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    from multiconductor.create import create_switch as _create_mc_switch
except ImportError:  # pragma: no cover - only used outside this repo/runtime
    _create_mc_switch = None


NETWORK_PROTECTOR_COLUMNS = [
    "name",
    "from_bus",
    "to_bus",
    "phases",
    "closed",
    "in_service",
    "r_ohm",
    "rated_current_a",
    "reverse_power_kw",
    "trip_delay_s",
    "reclose_delay_s",
    "control_mode",
    "switch_index",
    "metadata",
]

RES_NETWORK_PROTECTOR_COLUMNS = [
    "closed",
    "p_mw",
    "q_mvar",
    "i_ka",
    "loading_percent",
    "tripped",
]

__all__ = [
    "NetworkProtector",
    "NETWORK_PROTECTOR_COLUMNS",
    "RES_NETWORK_PROTECTOR_COLUMNS",
    "add_network_protector",
    "ensure_network_protector_tables",
    "get_network_protector",
    "insert_network_protectors_on_transformer_secondaries",
    "set_network_protector_closed",
    "set_network_protector_in_service",
    "sync_network_protector_switches",
]


def _normalise_phases(phases: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(phases, int):
        normalised = (phases,)
    else:
        normalised = tuple(int(phase) for phase in phases)

    if not normalised:
        raise ValueError("Network protector must include at least one phase.")
    if any(phase < 0 for phase in normalised):
        raise ValueError(f"Network protector phases must be non-negative: {normalised}")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"Network protector phases must be unique: {normalised}")
    return normalised


def _next_index(table: pd.DataFrame) -> int:
    if table.empty:
        return 0
    return int(max(table.index)) + 1


def _bus_exists(net: Any, bus: int) -> bool:
    if "bus" not in net:
        return False

    bus_table = net["bus"]
    if isinstance(bus_table.index, pd.MultiIndex):
        return bus in set(bus_table.index.get_level_values(0))
    return bus in set(bus_table.index)


def _bus_phase_exists(net: Any, bus: int, phase: int) -> bool:
    bus_table = net["bus"]
    if not isinstance(bus_table.index, pd.MultiIndex):
        return _bus_exists(net, bus)

    level_names = list(bus_table.index.names)
    bus_level = level_names.index("index") if "index" in level_names else 0
    phase_level = level_names.index("phase") if "phase" in level_names else 1
    return (bus, phase) in {
        (idx[bus_level], idx[phase_level]) for idx in bus_table.index
    }


def _index_level(index: pd.Index, preferred_name: str, default: int) -> int:
    if isinstance(index, pd.MultiIndex) and preferred_name in index.names:
        return index.names.index(preferred_name)
    return default


def _same_int(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return left == right


def _next_bus_index(net: Any) -> int:
    bus_table = net["bus"]
    if bus_table.empty:
        return 0

    if isinstance(bus_table.index, pd.MultiIndex):
        bus_level = _index_level(bus_table.index, "index", 0)
        values = bus_table.index.get_level_values(bus_level)
    else:
        values = bus_table.index
    return int(max(int(value) for value in values)) + 1


def _clone_rows_by_index_level(
    table: pd.DataFrame,
    source_value: int,
    new_value: int,
    *,
    level_name: str = "index",
) -> pd.DataFrame:
    if isinstance(table.index, pd.MultiIndex):
        level = _index_level(table.index, level_name, 0)
        selector = [_same_int(value, source_value) for value in table.index.get_level_values(level)]
        cloned = table.loc[selector].copy()
        if cloned.empty:
            return cloned

        new_tuples = []
        for raw_idx in cloned.index:
            idx_values = list(raw_idx)
            idx_values[level] = int(new_value)
            new_tuples.append(tuple(idx_values))
        cloned.index = pd.MultiIndex.from_tuples(new_tuples, names=table.index.names)
        return cloned

    if source_value not in table.index:
        return table.iloc[0:0].copy()

    cloned = table.loc[[source_value]].copy()
    cloned.index = pd.Index([new_value], name=table.index.name)
    return cloned


def _bus_name(net: Any, bus: int) -> str:
    bus_table = net["bus"]
    if "name" not in bus_table.columns:
        return f"bus_{bus}"

    if isinstance(bus_table.index, pd.MultiIndex):
        bus_level = _index_level(bus_table.index, "index", 0)
        rows = bus_table.loc[bus_table.index.get_level_values(bus_level) == bus]
        if rows.empty:
            return f"bus_{bus}"
        value = rows["name"].iloc[0]
    else:
        if bus not in bus_table.index:
            return f"bus_{bus}"
        value = bus_table.loc[bus, "name"]

    return str(value) if pd.notna(value) else f"bus_{bus}"


def _bus_phases(net: Any, bus: int) -> tuple[int, ...]:
    bus_table = net["bus"]
    if not isinstance(bus_table.index, pd.MultiIndex):
        return (1, 2, 3)

    bus_level = _index_level(bus_table.index, "index", 0)
    phase_level = _index_level(bus_table.index, "phase", 1)
    phases = [
        int(raw_idx[phase_level])
        for raw_idx in bus_table.index
        if _same_int(raw_idx[bus_level], bus)
    ]
    return tuple(sorted(set(phases)))


def _bus_nominal_voltage_kv(net: Any, bus: int) -> float | None:
    bus_table = net["bus"]
    if "vn_kv" not in bus_table.columns:
        return None

    if isinstance(bus_table.index, pd.MultiIndex):
        bus_level = _index_level(bus_table.index, "index", 0)
        rows = bus_table.loc[bus_table.index.get_level_values(bus_level) == bus]
        values = rows["vn_kv"] if not rows.empty else []
    else:
        if bus not in bus_table.index:
            return None
        values = [bus_table.loc[bus, "vn_kv"]]

    for value in values:
        try:
            voltage = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(voltage):
            return voltage
    return None


def _clone_bus(net: Any, source_bus: int, *, name: str | None = None) -> int:
    if not _bus_exists(net, source_bus):
        raise ValueError(f"Cannot clone missing bus {source_bus}.")

    new_bus = _next_bus_index(net)
    bus_table = net["bus"]
    cloned = _clone_rows_by_index_level(bus_table, source_bus, new_bus)
    if cloned.empty:
        raise ValueError(f"Cannot clone bus {source_bus}; no bus rows found.")

    if name is not None and "name" in cloned.columns:
        cloned["name"] = name

    net["bus"] = pd.concat([bus_table, cloned])
    return new_bus


def _clone_bus_geodata(net: Any, source_bus: int, new_bus: int) -> None:
    for table_name in ("bus_geo", "bus_geodata"):
        if table_name not in net:
            continue

        table = net[table_name]
        if isinstance(table, pd.DataFrame):
            cloned = _clone_rows_by_index_level(table, source_bus, new_bus)
            if not cloned.empty:
                net[table_name] = pd.concat([table, cloned])
        elif isinstance(table, dict) and source_bus in table:
            table[new_bus] = table[source_bus]


def _trafo_index_values(trafo_table: pd.DataFrame) -> list[int]:
    if not isinstance(trafo_table.index, pd.MultiIndex):
        return list(dict.fromkeys(int(value) for value in trafo_table.index))

    index_level = _index_level(trafo_table.index, "index", 0)
    return list(dict.fromkeys(int(value) for value in trafo_table.index.get_level_values(index_level)))


def _trafo_rows(trafo_table: pd.DataFrame, trafo_index: int) -> pd.DataFrame:
    if not isinstance(trafo_table.index, pd.MultiIndex):
        return trafo_table.loc[[trafo_index]]

    index_level = _index_level(trafo_table.index, "index", 0)
    selector = [_same_int(value, trafo_index) for value in trafo_table.index.get_level_values(index_level)]
    return trafo_table.loc[selector]


def _trafo_bus_values(trafo_rows: pd.DataFrame) -> list[int]:
    if not isinstance(trafo_rows.index, pd.MultiIndex):
        if {"hv_bus", "lv_bus"}.issubset(trafo_rows.columns):
            buses = trafo_rows[["hv_bus", "lv_bus"]].to_numpy().ravel()
            return list(dict.fromkeys(int(value) for value in buses))
        return []

    bus_level = _index_level(trafo_rows.index, "bus", 1)
    return list(dict.fromkeys(int(value) for value in trafo_rows.index.get_level_values(bus_level)))


def _rows_for_trafo_bus(trafo_rows: pd.DataFrame, bus: int) -> pd.DataFrame:
    if not isinstance(trafo_rows.index, pd.MultiIndex):
        if "lv_bus" in trafo_rows.columns:
            return trafo_rows.loc[trafo_rows["lv_bus"].map(lambda value: _same_int(value, bus))]
        return trafo_rows.iloc[0:0]

    bus_level = _index_level(trafo_rows.index, "bus", 1)
    selector = [_same_int(value, bus) for value in trafo_rows.index.get_level_values(bus_level)]
    return trafo_rows.loc[selector]


def _minimum_winding_voltage(rows: pd.DataFrame) -> float | None:
    if "vn_kv" not in rows.columns:
        return None

    values = []
    for value in rows["vn_kv"]:
        try:
            voltage = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(voltage):
            values.append(voltage)
    return min(values) if values else None


def _secondary_trafo_bus(net: Any, trafo_index: int) -> tuple[int, pd.DataFrame]:
    trafo_table = net["trafo1ph"]
    if not isinstance(trafo_table.index, pd.MultiIndex):
        raise ValueError("Expected trafo1ph to use a MultiIndex with an index and bus level.")

    rows = _trafo_rows(trafo_table, trafo_index)
    bus_candidates = []
    for bus in _trafo_bus_values(rows):
        bus_rows = _rows_for_trafo_bus(rows, bus)
        winding_voltage = _minimum_winding_voltage(bus_rows)
        bus_voltage = _bus_nominal_voltage_kv(net, bus)
        voltage = winding_voltage if winding_voltage is not None else bus_voltage
        bus_candidates.append((float("inf") if voltage is None else voltage, bus, bus_rows))

    if not bus_candidates:
        raise ValueError(f"Transformer {trafo_index} has no bus terminals.")

    _, secondary_bus, secondary_rows = min(bus_candidates, key=lambda item: (item[0], item[1]))
    return secondary_bus, secondary_rows


def _secondary_phases(net: Any, secondary_bus: int, secondary_rows: pd.DataFrame) -> tuple[int, ...]:
    phases: set[int] = set()
    for column in ("from_phase", "to_phase"):
        if column not in secondary_rows.columns:
            continue
        for value in secondary_rows[column]:
            try:
                phase = int(value)
            except (TypeError, ValueError):
                continue
            if phase != 0:
                phases.add(phase)

    if not phases:
        phases = {phase for phase in _bus_phases(net, secondary_bus) if phase != 0}

    return tuple(sorted(phases))


def _replace_trafo_bus(net: Any, trafo_index: int, old_bus: int, new_bus: int) -> None:
    trafo_table = net["trafo1ph"].copy()
    if not isinstance(trafo_table.index, pd.MultiIndex):
        raise ValueError("Expected trafo1ph to use a MultiIndex with an index and bus level.")

    index_level = _index_level(trafo_table.index, "index", 0)
    bus_level = _index_level(trafo_table.index, "bus", 1)
    new_index = []
    changed = 0

    for raw_idx in trafo_table.index:
        idx_values = list(raw_idx)
        if _same_int(idx_values[index_level], trafo_index) and _same_int(idx_values[bus_level], old_bus):
            idx_values[bus_level] = int(new_bus)
            changed += 1
        new_index.append(tuple(idx_values))

    if not changed:
        raise ValueError(f"Transformer {trafo_index} has no rows on bus {old_bus}.")

    trafo_table.index = pd.MultiIndex.from_tuples(new_index, names=trafo_table.index.names)
    net["trafo1ph"] = trafo_table


def _existing_protector_transformers(net: Any) -> set[int]:
    if "network_protector" not in net:
        return set()

    table = net["network_protector"]
    if table.empty or "metadata" not in table.columns:
        return set()

    existing = set()
    for metadata in table["metadata"]:
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("inserted_on") != "trafo1ph_secondary":
            continue
        try:
            existing.add(int(metadata["trafo1ph_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def ensure_network_protector_tables(net: Any) -> None:
    """Ensure network-protector element and result tables exist on ``net``."""
    if "network_protector" not in net:
        net["network_protector"] = pd.DataFrame(columns=NETWORK_PROTECTOR_COLUMNS)
        net["network_protector"].index.name = "index"

    if "res_network_protector" not in net:
        net["res_network_protector"] = pd.DataFrame(
            columns=RES_NETWORK_PROTECTOR_COLUMNS
        )
        net["res_network_protector"].index.name = "index"


@dataclass
class NetworkProtector:
    """Phase-aware network protector for a secondary network connection.

    ``from_bus`` is the transformer/source side and ``to_bus`` is the network
    side. By default, adding the element also creates a matching multiconductor
    bus-bus switch bank in ``net.switch``.
    """

    from_bus: int
    to_bus: int
    phases: tuple[int, ...] = (1, 2, 3)
    name: str | None = None
    closed: bool = True
    in_service: bool = True
    r_ohm: float = 0.0
    rated_current_a: float | None = None
    reverse_power_kw: float | None = None
    trip_delay_s: float = 0.0
    reclose_delay_s: float | None = None
    control_mode: str = "reverse_power"
    switch_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.from_bus = int(self.from_bus)
        self.to_bus = int(self.to_bus)
        self.phases = _normalise_phases(self.phases)
        self.closed = bool(self.closed)
        self.in_service = bool(self.in_service)
        self.r_ohm = float(self.r_ohm)
        self.trip_delay_s = float(self.trip_delay_s)
        if self.reclose_delay_s is not None:
            self.reclose_delay_s = float(self.reclose_delay_s)
        if self.rated_current_a is not None:
            self.rated_current_a = float(self.rated_current_a)
        if self.reverse_power_kw is not None:
            self.reverse_power_kw = float(self.reverse_power_kw)
        if self.switch_index is not None:
            self.switch_index = int(self.switch_index)

    def validate_for_network(self, net: Any) -> None:
        """Raise ``ValueError`` if this protector cannot be attached to ``net``."""
        missing_buses = [
            bus for bus in (self.from_bus, self.to_bus) if not _bus_exists(net, bus)
        ]
        if missing_buses:
            raise ValueError(f"Network protector references missing bus(es): {missing_buses}")

        missing_phases = [
            (bus, phase)
            for bus in (self.from_bus, self.to_bus)
            for phase in self.phases
            if not _bus_phase_exists(net, bus, phase)
        ]
        if missing_phases:
            raise ValueError(
                "Network protector references missing bus-phase terminals: "
                f"{missing_phases}"
            )

    @property
    def effective_closed(self) -> bool:
        """Switch state seen by the solver."""
        return self.closed and self.in_service

    def to_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "from_bus": self.from_bus,
            "to_bus": self.to_bus,
            "phases": self.phases,
            "closed": self.closed,
            "in_service": self.in_service,
            "r_ohm": self.r_ohm,
            "rated_current_a": self.rated_current_a,
            "reverse_power_kw": self.reverse_power_kw,
            "trip_delay_s": self.trip_delay_s,
            "reclose_delay_s": self.reclose_delay_s,
            "control_mode": self.control_mode,
            "switch_index": self.switch_index,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(cls, row: pd.Series) -> "NetworkProtector":
        switch_index = row.get("switch_index")
        if pd.isna(switch_index):
            switch_index = None
        return cls(
            name=row.get("name"),
            from_bus=int(row["from_bus"]),
            to_bus=int(row["to_bus"]),
            phases=tuple(row["phases"]),
            closed=bool(row.get("closed", True)),
            in_service=bool(row.get("in_service", True)),
            r_ohm=float(row.get("r_ohm", 0.0)),
            rated_current_a=row.get("rated_current_a"),
            reverse_power_kw=row.get("reverse_power_kw"),
            trip_delay_s=float(row.get("trip_delay_s", 0.0)),
            reclose_delay_s=row.get("reclose_delay_s"),
            control_mode=row.get("control_mode", "reverse_power"),
            switch_index=switch_index,
            metadata=row.get("metadata") or {},
        )

    def add_to_network(
        self,
        net: Any,
        *,
        index: int | None = None,
        create_switches: bool = True,
        overwrite: bool = False,
    ) -> int:
        """Add this network protector to ``net`` and return its element index."""
        return add_network_protector(
            net,
            self,
            index=index,
            create_switches=create_switches,
            overwrite=overwrite,
        )


def add_network_protector(
    net: Any,
    protector: NetworkProtector,
    *,
    index: int | None = None,
    create_switches: bool = True,
    overwrite: bool = False,
) -> int:
    """Add a ``NetworkProtector`` to a pandapower/multiconductor network."""
    ensure_network_protector_tables(net)
    protector.validate_for_network(net)

    table = net["network_protector"]
    if index is None:
        index = _next_index(table)
    index = int(index)
    if index in table.index and not overwrite:
        raise ValueError(f"network_protector index {index} already exists.")

    if create_switches and protector.switch_index is None:
        if _create_mc_switch is None:
            raise RuntimeError("multiconductor.create.create_switch is not available.")
        protector.switch_index = _create_mc_switch(
            net,
            bus=protector.from_bus,
            phase=protector.phases,
            element=protector.to_bus,
            et="b",
            type="NP",
            closed=protector.effective_closed,
            name=protector.name,
            r_ohm=protector.r_ohm,
        )

    table.loc[index, NETWORK_PROTECTOR_COLUMNS] = protector.to_row()
    net["network_protector"] = table

    net["res_network_protector"].loc[index, RES_NETWORK_PROTECTOR_COLUMNS] = {
        "closed": protector.effective_closed,
        "p_mw": pd.NA,
        "q_mvar": pd.NA,
        "i_ka": pd.NA,
        "loading_percent": pd.NA,
        "tripped": False,
    }
    sync_network_protector_switches(net, index)
    return index


def insert_network_protectors_on_transformer_secondaries(
    net: Any,
    *,
    transformer_indices: Iterable[int] | None = None,
    name_prefix: str = "NP",
    closed: bool = True,
    in_service: bool = True,
    r_ohm: float = 0.0,
    rated_current_a: float | None = None,
    reverse_power_kw: float | None = None,
    trip_delay_s: float = 0.0,
    reclose_delay_s: float | None = None,
    control_mode: str = "reverse_power",
    create_switches: bool = True,
    skip_existing: bool = True,
    copy_geodata: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> list[int]:
    """Insert network protectors on the secondary side of ``trafo1ph`` elements.

    For each selected transformer, the helper identifies the lowest-voltage
    winding as the secondary side, clones that bus as a transformer-side
    terminal, rewires the transformer secondary rows to the clone, and adds a
    ``NetworkProtector`` bus-bus switch from the cloned transformer terminal
    back to the original secondary network bus.

    The operation is idempotent by default. Existing protectors with metadata
    marking the same ``trafo1ph`` index are skipped when ``skip_existing`` is
    true.
    """
    if create_switches and _create_mc_switch is None:
        raise RuntimeError("multiconductor.create.create_switch is not available.")

    ensure_network_protector_tables(net)
    if "trafo1ph" not in net or net["trafo1ph"].empty:
        return []

    trafo_table = net["trafo1ph"]
    if not isinstance(trafo_table.index, pd.MultiIndex):
        raise ValueError("Expected trafo1ph to use a MultiIndex with an index and bus level.")

    requested = None
    if transformer_indices is not None:
        requested = {int(index) for index in transformer_indices}

    existing = _existing_protector_transformers(net) if skip_existing else set()
    inserted_indices: list[int] = []
    common_metadata = dict(metadata or {})

    for trafo_index in _trafo_index_values(trafo_table):
        if requested is not None and trafo_index not in requested:
            continue
        if trafo_index in existing:
            continue

        secondary_bus, secondary_rows = _secondary_trafo_bus(net, trafo_index)
        phases = _secondary_phases(net, secondary_bus, secondary_rows)
        if not phases:
            raise ValueError(f"Transformer {trafo_index} secondary bus {secondary_bus} has no usable phases.")

        original_bus_name = _bus_name(net, secondary_bus)
        protector_bus = _clone_bus(
            net,
            secondary_bus,
            name=f"{original_bus_name} {name_prefix} source",
        )
        if copy_geodata:
            _clone_bus_geodata(net, secondary_bus, protector_bus)

        _replace_trafo_bus(net, trafo_index, secondary_bus, protector_bus)

        row_name = secondary_rows["name"].iloc[0] if "name" in secondary_rows.columns else trafo_index
        protector_metadata = {
            **common_metadata,
            "inserted_on": "trafo1ph_secondary",
            "trafo1ph_index": int(trafo_index),
            "original_secondary_bus": int(secondary_bus),
            "protector_source_bus": int(protector_bus),
            "secondary_phases": tuple(int(phase) for phase in phases),
        }
        protector = NetworkProtector(
            from_bus=protector_bus,
            to_bus=secondary_bus,
            phases=phases,
            name=f"{name_prefix}-{trafo_index}",
            closed=closed,
            in_service=in_service,
            r_ohm=r_ohm,
            rated_current_a=rated_current_a,
            reverse_power_kw=reverse_power_kw,
            trip_delay_s=trip_delay_s,
            reclose_delay_s=reclose_delay_s,
            control_mode=control_mode,
            metadata={
                **protector_metadata,
                "trafo1ph_name": None if pd.isna(row_name) else str(row_name),
            },
        )
        inserted_indices.append(add_network_protector(net, protector, create_switches=create_switches))

    return inserted_indices


def get_network_protector(net: Any, index: int) -> NetworkProtector:
    """Return a ``NetworkProtector`` object from ``net.network_protector``."""
    ensure_network_protector_tables(net)
    if index not in net["network_protector"].index:
        raise KeyError(f"network_protector index {index} does not exist.")
    return NetworkProtector.from_row(net["network_protector"].loc[index])


def sync_network_protector_switches(net: Any, index: int | None = None) -> None:
    """Push network-protector open/closed states into linked switch rows."""
    ensure_network_protector_tables(net)
    if "switch" not in net:
        return

    protector_table = net["network_protector"]
    if index is not None:
        protector_table = protector_table.loc[[index]]

    for _, row in protector_table.iterrows():
        switch_index = row.get("switch_index")
        if pd.isna(switch_index):
            continue

        switch_index = int(switch_index)
        if isinstance(net["switch"].index, pd.MultiIndex):
            if switch_index not in net["switch"].index.get_level_values(0):
                continue
            selector = net["switch"].index.get_level_values(0) == switch_index
            net["switch"].loc[selector, "closed"] = bool(row["closed"] and row["in_service"])
            net["switch"].loc[selector, "r_ohm"] = float(row["r_ohm"])
        elif switch_index in net["switch"].index:
            net["switch"].loc[switch_index, "closed"] = bool(
                row["closed"] and row["in_service"]
            )
            net["switch"].loc[switch_index, "r_ohm"] = float(row["r_ohm"])


def set_network_protector_closed(net: Any, index: int, closed: bool) -> None:
    """Set a protector's commanded status and update linked switch rows."""
    ensure_network_protector_tables(net)
    if index not in net["network_protector"].index:
        raise KeyError(f"network_protector index {index} does not exist.")

    net["network_protector"].loc[index, "closed"] = bool(closed)
    if index in net["res_network_protector"].index:
        in_service = bool(net["network_protector"].loc[index, "in_service"])
        net["res_network_protector"].loc[index, "closed"] = bool(closed) and in_service
    sync_network_protector_switches(net, index)


def set_network_protector_in_service(net: Any, index: int, in_service: bool) -> None:
    """Set a protector's service flag and update linked switch rows."""
    ensure_network_protector_tables(net)
    if index not in net["network_protector"].index:
        raise KeyError(f"network_protector index {index} does not exist.")

    net["network_protector"].loc[index, "in_service"] = bool(in_service)
    closed = bool(net["network_protector"].loc[index, "closed"])
    if index in net["res_network_protector"].index:
        net["res_network_protector"].loc[index, "closed"] = closed and bool(in_service)
    sync_network_protector_switches(net, index)
