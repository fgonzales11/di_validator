import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import NAMESPACE_OID, uuid4, uuid5
import xml.etree.ElementTree as ET

import pandas as pd

from .file_io import create_empty_network
from .create import (
    create_asymmetric_load,
    create_asymmetric_sgen,
    create_bus,
    create_ext_grid,
    create_line,
    create_switch,
    create_transformer_3ph,
)
from .pycci.std_types import create_std_type, std_type_exists

logger = logging.getLogger(__name__)

CIM_NS = "http://iec.ch/TC57/CIM100#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _resolve_from_cim_callable():
    """Return pandapower CIM import callable across API variants."""

    try:
        from pandapower.converter.cim import from_cim  # type: ignore
    except Exception as exc:
        raise ImportError("pandapower CIM converter is not available") from exc

    fn = getattr(from_cim, "from_cim", None)
    if callable(fn):
        return fn
    if callable(from_cim):
        return from_cim
    raise ImportError("pandapower CIM converter API is unsupported: from_cim is not callable")


@dataclass
class CIMExportConfig:
    base_name: str = "multiconductor"
    include_transformers: bool = True
    include_switches: bool = True
    include_generators: bool = True
    include_loads: bool = True
    include_lines: bool = True


@dataclass
class CGMESExportConfig:
    base_name: str = "multiconductor"
    output_dir: str = "."
    profiles: Tuple[str, ...] = ("EQ", "TP", "SSH", "SV", "DL")
    include_transformers: bool = True
    include_switches: bool = True
    include_generators: bool = True
    include_loads: bool = True
    include_lines: bool = True
    include_diagram_layout: bool = True
    include_sv: bool = True
    include_ssh: bool = True


@dataclass
class CGMESImportConfig:
    csv_dir: Optional[str] = None


class CIMAdapter:
    """Import/export CIM XML to and from multiconductor networks.

    This adapter uses pandapower's CIM converter for import when available.
    Export emits a minimal CIM100 RDF document with common distribution elements.
    """

    def __init__(
        self,
        cgmes_version: str = "3.0",
        convert_line_to_switch: bool = False,
        line_r_limit: float = 0.1,
        line_x_limit: float = 0.1,
    ) -> None:
        self.cgmes_version = cgmes_version
        self.convert_line_to_switch = convert_line_to_switch
        self.line_r_limit = line_r_limit
        self.line_x_limit = line_x_limit

    def import_cim(
        self,
        cim_path: str,
        *,
        encoding: Optional[str] = None,
        **kwargs,
    ):
        if not os.path.exists(cim_path):
            raise FileNotFoundError(f"CIM file not found: {cim_path}")

        from_cim_fn = _resolve_from_cim_callable()

        pp_net = from_cim_fn(
            file_list=[cim_path],
            encoding=encoding,
            convert_line_to_switch=self.convert_line_to_switch,
            line_r_limit=self.line_r_limit,
            line_x_limit=self.line_x_limit,
            cgmes_version=self.cgmes_version,
            **kwargs,
        )

        return self._pandapower_to_multiconductor(pp_net)

    def import_cgmes(
        self,
        file_list: List[str],
        *,
        config: Optional[CGMESImportConfig] = None,
        encoding: Optional[str] = None,
        **kwargs,
    ):
        if not file_list:
            raise ValueError("file_list must contain at least one CGMES file")

        normalized = _normalize_file_list(file_list)
        if not normalized:
            raise FileNotFoundError("No CGMES files found for import")

        from_cim_fn = _resolve_from_cim_callable()

        pp_net = from_cim_fn(
            file_list=normalized,
            encoding=encoding,
            convert_line_to_switch=self.convert_line_to_switch,
            line_r_limit=self.line_r_limit,
            line_x_limit=self.line_x_limit,
            cgmes_version=self.cgmes_version,
            **kwargs,
        )

        net = self._pandapower_to_multiconductor(pp_net)

        if config and config.csv_dir:
            _apply_csv_enrichment(net, config.csv_dir)

        return net

    def export_cim(
        self,
        net,
        cim_path: str,
        *,
        config: Optional[CIMExportConfig] = None,
    ) -> None:
        if config is None:
            config = CIMExportConfig()

        ET.register_namespace("cim", CIM_NS)
        ET.register_namespace("rdf", RDF_NS)
        rdf = ET.Element(f"{{{RDF_NS}}}RDF")

        def _new_id() -> str:
            return f"_{uuid4()}".upper()

        def _add_identified_object(element, name: str, mrid: Optional[str] = None) -> str:
            oid = mrid or _new_id()
            element.set(f"{{{RDF_NS}}}ID", oid)
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.mRID").text = oid
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.name").text = name
            return oid

        def _stable_id(prefix: str, key: str) -> str:
            return f"_{prefix}_{uuid5(NAMESPACE_OID, key).hex.upper()}"

        def _stable_id(prefix: str, key: str) -> str:
            return f"_{prefix}_{uuid5(NAMESPACE_OID, key).hex.upper()}"

        def _bus_rows() -> Dict[int, pd.Series]:
            bus_rows: Dict[int, pd.Series] = {}
            if not hasattr(net, "bus"):
                return bus_rows
            for bus_idx in sorted(set(idx[0] for idx in net.bus.index)):
                try:
                    row = net.bus.xs(bus_idx, level=0).iloc[0]
                except Exception:
                    row = net.bus.loc[(bus_idx, 0)] if (bus_idx, 0) in net.bus.index else net.bus.loc[bus_idx]
                bus_rows[bus_idx] = row
            return bus_rows

        def _sum_value(value: object) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (list, tuple)):
                return sum(float(v) for v in value)
            try:
                if pd.isna(value):
                    return 0.0
            except Exception:
                pass
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _sum_series(series: Iterable[object]) -> float:
            total = 0.0
            for value in series:
                total += _sum_value(value)
            return total

        version = ET.SubElement(rdf, f"{{{CIM_NS}}}IEC61970CIMVersion")
        _add_identified_object(version, "cim_version")
        ET.SubElement(version, f"{{{CIM_NS}}}IEC61970CIMVersion.version").text = "IEC61970CIM100"

        feeder = ET.SubElement(rdf, f"{{{CIM_NS}}}Feeder")
        feeder_id = _add_identified_object(feeder, config.base_name)

        base_voltage_ids: Dict[float, str] = {}
        for bus_idx, row in _bus_rows().items():
            vn_kv = float(row.get("vn_kv", 0.0) or 0.0)
            if vn_kv <= 0:
                continue
            if vn_kv not in base_voltage_ids:
                base_voltage = ET.SubElement(rdf, f"{{{CIM_NS}}}BaseVoltage")
                name = f"BaseV_{vn_kv:.4f}"
                base_voltage_id = _add_identified_object(base_voltage, name)
                ET.SubElement(base_voltage, f"{{{CIM_NS}}}BaseVoltage.nominalVoltage").text = f"{vn_kv * 1000:.3f}"
                base_voltage_ids[vn_kv] = base_voltage_id

        topo_node_ids: Dict[int, str] = {}
        conn_node_ids: Dict[int, str] = {}
        conn_node_elements: Dict[int, ET.Element] = {}
        for bus_idx, row in _bus_rows().items():
            bus_name = row.get("name") or f"Bus {bus_idx}"
            topo = ET.SubElement(rdf, f"{{{CIM_NS}}}TopologicalNode")
            topo_id = _add_identified_object(topo, bus_name)
            topo_node_ids[bus_idx] = topo_id

            conn = ET.SubElement(rdf, f"{{{CIM_NS}}}ConnectivityNode")
            conn_id = _add_identified_object(conn, bus_name)
            conn_node_ids[bus_idx] = conn_id
            ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.TopologicalNode").set(
                f"{{{RDF_NS}}}resource", f"#{topo_id}"
            )
            ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.ConnectivityNodeContainer").set(
                f"{{{RDF_NS}}}resource", f"#{feeder_id}"
            )

        def _add_terminal(equipment_id: str, bus_idx: int, sequence: int) -> str:
            term = ET.SubElement(rdf, f"{{{CIM_NS}}}Terminal")
            term_id = _add_identified_object(term, f"T_{equipment_id}_{sequence}")
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConductingEquipment").set(
                f"{{{RDF_NS}}}resource", f"#{equipment_id}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConnectivityNode").set(
                f"{{{RDF_NS}}}resource", f"#{conn_node_ids[bus_idx]}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.sequenceNumber").text = str(sequence)
            return term_id

        if config.include_lines and hasattr(net, "line") and not net.line.empty:
            for line_idx, group in net.line.groupby(level=0):
                row = group.iloc[0]
                from_bus = int(row.get("from_bus"))
                to_bus = int(row.get("to_bus"))
                name = row.get("name") or f"Line {line_idx}"
                std_name = row.get("std_type")
                length_km = float(row.get("length_km") or 0.0)
                std_data = {}
                if std_name and hasattr(net, "std_types"):
                    std_data = net.std_types.get("sequence", {}).get(std_name, {})

                line = ET.SubElement(rdf, f"{{{CIM_NS}}}ACLineSegment")
                line_id = _add_identified_object(line, name)
                ET.SubElement(line, f"{{{CIM_NS}}}Conductor.length").text = f"{length_km:.6f}"
                ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.r").text = str(std_data.get("r_ohm_per_km", 0.0))
                ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.x").text = str(std_data.get("x_ohm_per_km", 0.0))
                ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.bch").text = str(std_data.get("c_nf_per_km", 0.0))
                base_id = None
                bus_row = _bus_rows().get(from_bus)
                if bus_row is not None:
                    base_id = base_voltage_ids.get(float(bus_row.get("vn_kv", 0.0) or 0.0))
                if base_id:
                    ET.SubElement(line, f"{{{CIM_NS}}}ConductingEquipment.BaseVoltage").set(
                        f"{{{RDF_NS}}}resource", f"#{base_id}"
                    )
                _add_terminal(line_id, from_bus, 1)
                _add_terminal(line_id, to_bus, 2)

        if config.include_loads and hasattr(net, "asymmetric_load") and not net.asymmetric_load.empty:
            for load_idx, group in net.asymmetric_load.groupby(level=0):
                row = group.iloc[0]
                bus = int(row.get("bus"))
                name = row.get("name") or f"Load {load_idx}"
                p_mw = _sum_series(group.get("p_mw", []))
                q_mvar = _sum_series(group.get("q_mvar", []))
                load = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergyConsumer")
                load_id = _add_identified_object(load, name)
                ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.p").text = f"{p_mw * 1e6:.6f}"
                ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.q").text = f"{q_mvar * 1e6:.6f}"
                _add_terminal(load_id, bus, 1)

        if config.include_generators and hasattr(net, "asymmetric_sgen") and not net.asymmetric_sgen.empty:
            for gen_idx, group in net.asymmetric_sgen.groupby(level=0):
                row = group.iloc[0]
                bus = int(row.get("bus"))
                name = row.get("name") or f"Gen {gen_idx}"
                p_mw = _sum_series(group.get("p_mw", []))
                source = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergySource")
                source_id = _add_identified_object(source, name)
                ET.SubElement(source, f"{{{CIM_NS}}}EnergySource.nominalVoltage").text = "0"
                ET.SubElement(source, f"{{{CIM_NS}}}EnergySource.pMax").text = f"{p_mw * 1e6:.6f}"
                _add_terminal(source_id, bus, 1)

        if config.include_switches and hasattr(net, "switch") and not net.switch.empty:
            for sw_idx, group in net.switch.groupby(level=0):
                row = group.iloc[0]
                if str(row.get("et", "")).lower() != "b":
                    continue
                bus_a = int(row.get("bus"))
                bus_b = int(row.get("element"))
                name = row.get("name") or f"Switch {sw_idx}"
                closed = bool(row.get("closed", True))
                sw = ET.SubElement(rdf, f"{{{CIM_NS}}}Switch")
                sw_id = _add_identified_object(sw, name)
                ET.SubElement(sw, f"{{{CIM_NS}}}Switch.normalOpen").text = "false" if closed else "true"
                _add_terminal(sw_id, bus_a, 1)
                _add_terminal(sw_id, bus_b, 2)

        if config.include_transformers and hasattr(net, "trafo1ph") and not net.trafo1ph.empty:
            for tr_idx, group in net.trafo1ph.groupby(level=0):
                buses = sorted({int(b) for b in group.index.get_level_values("bus")})
                if len(buses) < 2:
                    continue
                name = group.get("name", pd.Series([None])).dropna()
                tr_name = name.iloc[0] if not name.empty else f"Trafo {tr_idx}"
                tr = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformer")
                tr_id = _add_identified_object(tr, tr_name)
                sn_mva = _sum_series(group.get("sn_mva", []))
                rated_s_va = sn_mva * 1e6 if sn_mva > 0 else 0.0
                for end_num, bus in enumerate(buses, start=1):
                    end = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformerEnd")
                    end_id = _add_identified_object(end, f"{tr_name}_end_{end_num}")
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.PowerTransformer").set(
                        f"{{{RDF_NS}}}resource", f"#{tr_id}"
                    )
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.endNumber").text = str(end_num)
                    vn_kv = float(_bus_rows().get(bus, {}).get("vn_kv", 0.0) or 0.0)
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedU").text = f"{vn_kv * 1000:.3f}"
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedS").text = f"{rated_s_va:.3f}"
                    term_id = _add_terminal(tr_id, bus, end_num)
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.Terminal").set(
                        f"{{{RDF_NS}}}resource", f"#{term_id}"
                    )

        tree = ET.ElementTree(rdf)
        os.makedirs(os.path.dirname(os.path.abspath(cim_path)), exist_ok=True)
        tree.write(cim_path, encoding="utf-8", xml_declaration=True)

    def export_cgmes(
        self,
        net,
        *,
        config: Optional[CGMESExportConfig] = None,
    ) -> List[str]:
        if config is None:
            config = CGMESExportConfig()

        os.makedirs(config.output_dir, exist_ok=True)
        base_name = config.base_name

        paths: List[str] = []
        profiles = set(profile.upper() for profile in config.profiles)

        if "EQ" in profiles:
            path = os.path.join(config.output_dir, f"{base_name}_EQ.xml")
            self._export_cgmes_profile(net, path, profile="EQ", config=config)
            paths.append(path)

        if "TP" in profiles:
            path = os.path.join(config.output_dir, f"{base_name}_TP.xml")
            self._export_cgmes_profile(net, path, profile="TP", config=config)
            paths.append(path)

        if "SSH" in profiles and config.include_ssh:
            path = os.path.join(config.output_dir, f"{base_name}_SSH.xml")
            self._export_cgmes_profile(net, path, profile="SSH", config=config)
            paths.append(path)

        if "SV" in profiles and config.include_sv:
            path = os.path.join(config.output_dir, f"{base_name}_SV.xml")
            self._export_cgmes_profile(net, path, profile="SV", config=config)
            paths.append(path)

        if "DL" in profiles and config.include_diagram_layout:
            path = os.path.join(config.output_dir, f"{base_name}_DL.xml")
            self._export_cgmes_profile(net, path, profile="DL", config=config)
            paths.append(path)

        return paths

    def _export_cgmes_profile(self, net, cim_path: str, *, profile: str, config: CGMESExportConfig) -> None:
        ET.register_namespace("cim", CIM_NS)
        ET.register_namespace("rdf", RDF_NS)
        rdf = ET.Element(f"{{{RDF_NS}}}RDF")

        profile = profile.upper()

        def _new_id() -> str:
            return f"_{uuid4()}".upper()

        def _add_identified_object(element, name: str, mrid: Optional[str] = None) -> str:
            oid = mrid or _new_id()
            element.set(f"{{{RDF_NS}}}ID", oid)
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.mRID").text = oid
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.name").text = name
            return oid

        def _stable_id(prefix: str, key: str) -> str:
            return f"_{prefix}_{uuid5(NAMESPACE_OID, key).hex.upper()}"

        def _bus_rows() -> Dict[int, pd.Series]:
            bus_rows: Dict[int, pd.Series] = {}
            if not hasattr(net, "bus"):
                return bus_rows
            for bus_idx in sorted(set(idx[0] for idx in net.bus.index)):
                try:
                    row = net.bus.xs(bus_idx, level=0).iloc[0]
                except Exception:
                    row = net.bus.loc[(bus_idx, 0)] if (bus_idx, 0) in net.bus.index else net.bus.loc[bus_idx]
                bus_rows[bus_idx] = row
            return bus_rows

        def _sum_value(value: object) -> float:
            if value is None:
                return 0.0
            if isinstance(value, (list, tuple)):
                return sum(float(v) for v in value)
            try:
                if pd.isna(value):
                    return 0.0
            except Exception:
                pass
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _sum_series(series: Iterable[object]) -> float:
            total = 0.0
            for value in series:
                total += _sum_value(value)
            return total

        def _add_terminal(equipment_id: str, bus_idx: int, sequence: int, conn_node_ids: Dict[int, str]) -> str:
            term = ET.SubElement(rdf, f"{{{CIM_NS}}}Terminal")
            term_id = _add_identified_object(term, f"T_{equipment_id}_{sequence}")
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConductingEquipment").set(
                f"{{{RDF_NS}}}resource", f"#{equipment_id}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConnectivityNode").set(
                f"{{{RDF_NS}}}resource", f"#{conn_node_ids[bus_idx]}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.sequenceNumber").text = str(sequence)
            return term_id

        full_model = ET.SubElement(rdf, f"{{{CIM_NS}}}FullModel")
        _add_identified_object(full_model, f"{config.base_name}_{profile}")
        ET.SubElement(full_model, f"{{{CIM_NS}}}Model.profile").text = profile

        feeder = ET.SubElement(rdf, f"{{{CIM_NS}}}Feeder")
        feeder_id = _add_identified_object(feeder, config.base_name)

        bus_rows = _bus_rows()
        base_voltage_ids: Dict[float, str] = {}
        if profile == "EQ":
            for bus_idx, row in bus_rows.items():
                vn_kv = float(row.get("vn_kv", 0.0) or 0.0)
                if vn_kv <= 0:
                    continue
                if vn_kv not in base_voltage_ids:
                    base_voltage = ET.SubElement(rdf, f"{{{CIM_NS}}}BaseVoltage")
                    name = f"BaseV_{vn_kv:.4f}"
                    base_voltage_id = _add_identified_object(base_voltage, name)
                    ET.SubElement(base_voltage, f"{{{CIM_NS}}}BaseVoltage.nominalVoltage").text = f"{vn_kv * 1000:.3f}"
                    base_voltage_ids[vn_kv] = base_voltage_id

        topo_node_ids: Dict[int, str] = {}
        conn_node_ids: Dict[int, str] = {}
        conn_node_elements: Dict[int, ET.Element] = {}
        if profile in ("EQ", "TP"):
            for bus_idx, row in bus_rows.items():
                bus_name = row.get("name") or f"Bus {bus_idx}"
                conn = ET.SubElement(rdf, f"{{{CIM_NS}}}ConnectivityNode")
                conn_id = _add_identified_object(conn, bus_name)
                conn_node_ids[bus_idx] = conn_id
                conn_node_elements[bus_idx] = conn
                ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.ConnectivityNodeContainer").set(
                    f"{{{RDF_NS}}}resource", f"#{feeder_id}"
                )

        if profile == "TP":
            island = ET.SubElement(rdf, f"{{{CIM_NS}}}TopologicalIsland")
            island_id = _add_identified_object(island, f"{config.base_name}_Island")
            for bus_idx, row in bus_rows.items():
                bus_name = row.get("name") or f"Bus {bus_idx}"
                topo = ET.SubElement(rdf, f"{{{CIM_NS}}}TopologicalNode")
                topo_id = _add_identified_object(topo, bus_name, _stable_id("TN", str(bus_idx)))
                topo_node_ids[bus_idx] = topo_id
                ET.SubElement(topo, f"{{{CIM_NS}}}TopologicalNode.TopologicalIsland").set(
                    f"{{{RDF_NS}}}resource", f"#{island_id}"
                )
                conn = conn_node_elements.get(bus_idx)
                if conn is not None:
                    ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.TopologicalNode").set(
                        f"{{{RDF_NS}}}resource", f"#{topo_id}"
                    )

        if profile == "EQ":
            if config.include_lines and hasattr(net, "line") and not net.line.empty:
                for line_idx, group in net.line.groupby(level=0):
                    row = group.iloc[0]
                    from_bus = int(row.get("from_bus"))
                    to_bus = int(row.get("to_bus"))
                    name = row.get("name") or f"Line {line_idx}"
                    std_name = row.get("std_type")
                    length_km = float(row.get("length_km") or 0.0)
                    std_data = {}
                    if std_name and hasattr(net, "std_types"):
                        std_data = net.std_types.get("sequence", {}).get(std_name, {})

                    line = ET.SubElement(rdf, f"{{{CIM_NS}}}ACLineSegment")
                    line_id = _add_identified_object(line, name)
                    ET.SubElement(line, f"{{{CIM_NS}}}Conductor.length").text = f"{length_km:.6f}"
                    ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.r").text = str(std_data.get("r_ohm_per_km", 0.0))
                    ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.x").text = str(std_data.get("x_ohm_per_km", 0.0))
                    ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.bch").text = str(std_data.get("c_nf_per_km", 0.0))
                    bus_row = bus_rows.get(from_bus)
                    if bus_row is not None:
                        base_id = base_voltage_ids.get(float(bus_row.get("vn_kv", 0.0) or 0.0))
                        if base_id:
                            ET.SubElement(line, f"{{{CIM_NS}}}ConductingEquipment.BaseVoltage").set(
                                f"{{{RDF_NS}}}resource", f"#{base_id}"
                            )
                    _add_terminal(line_id, from_bus, 1, conn_node_ids)
                    _add_terminal(line_id, to_bus, 2, conn_node_ids)

            if config.include_loads and hasattr(net, "asymmetric_load") and not net.asymmetric_load.empty:
                for load_idx, group in net.asymmetric_load.groupby(level=0):
                    row = group.iloc[0]
                    bus = int(row.get("bus"))
                    name = row.get("name") or f"Load {load_idx}"
                    load = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergyConsumer")
                    load_id = _add_identified_object(load, name)
                    _add_terminal(load_id, bus, 1, conn_node_ids)

            if config.include_generators and hasattr(net, "asymmetric_sgen") and not net.asymmetric_sgen.empty:
                for gen_idx, group in net.asymmetric_sgen.groupby(level=0):
                    row = group.iloc[0]
                    bus = int(row.get("bus"))
                    name = row.get("name") or f"Gen {gen_idx}"
                    source = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergySource")
                    source_id = _add_identified_object(source, name)
                    _add_terminal(source_id, bus, 1, conn_node_ids)

            if config.include_switches and hasattr(net, "switch") and not net.switch.empty:
                for sw_idx, group in net.switch.groupby(level=0):
                    row = group.iloc[0]
                    if str(row.get("et", "")).lower() != "b":
                        continue
                    bus_a = int(row.get("bus"))
                    bus_b = int(row.get("element"))
                    name = row.get("name") or f"Switch {sw_idx}"
                    closed = bool(row.get("closed", True))
                    sw = ET.SubElement(rdf, f"{{{CIM_NS}}}Switch")
                    sw_id = _add_identified_object(sw, name)
                    ET.SubElement(sw, f"{{{CIM_NS}}}Switch.normalOpen").text = "false" if closed else "true"
                    _add_terminal(sw_id, bus_a, 1, conn_node_ids)
                    _add_terminal(sw_id, bus_b, 2, conn_node_ids)

            if config.include_transformers and hasattr(net, "trafo1ph") and not net.trafo1ph.empty:
                for tr_idx, group in net.trafo1ph.groupby(level=0):
                    buses = sorted({int(b) for b in group.index.get_level_values("bus")})
                    if len(buses) < 2:
                        continue
                    name = group.get("name", pd.Series([None])).dropna()
                    tr_name = name.iloc[0] if not name.empty else f"Trafo {tr_idx}"
                    tr = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformer")
                    tr_id = _add_identified_object(tr, tr_name)
                    sn_mva = _sum_series(group.get("sn_mva", []))
                    rated_s_va = sn_mva * 1e6 if sn_mva > 0 else 0.0
                    for end_num, bus in enumerate(buses, start=1):
                        end = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformerEnd")
                        end_id = _add_identified_object(end, f"{tr_name}_end_{end_num}")
                        ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.PowerTransformer").set(
                            f"{{{RDF_NS}}}resource", f"#{tr_id}"
                        )
                        ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.endNumber").text = str(end_num)
                        vn_kv = float(bus_rows.get(bus, {}).get("vn_kv", 0.0) or 0.0)
                        ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedU").text = f"{vn_kv * 1000:.3f}"
                        ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedS").text = f"{rated_s_va:.3f}"
                        term_id = _add_terminal(tr_id, bus, end_num, conn_node_ids)
                        ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.Terminal").set(
                            f"{{{RDF_NS}}}resource", f"#{term_id}"
                        )

        if profile == "SSH" and config.include_ssh:
            if hasattr(net, "asymmetric_load") and not net.asymmetric_load.empty:
                for load_idx, group in net.asymmetric_load.groupby(level=0):
                    row = group.iloc[0]
                    name = row.get("name") or f"Load {load_idx}"
                    p_mw = _sum_series(group.get("p_mw", []))
                    q_mvar = _sum_series(group.get("q_mvar", []))
                    load = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergyConsumer")
                    load_id = _add_identified_object(load, name)
                    ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.p").text = f"{p_mw * 1e6:.6f}"
                    ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.q").text = f"{q_mvar * 1e6:.6f}"

            if hasattr(net, "asymmetric_sgen") and not net.asymmetric_sgen.empty:
                for gen_idx, group in net.asymmetric_sgen.groupby(level=0):
                    row = group.iloc[0]
                    name = row.get("name") or f"Gen {gen_idx}"
                    p_mw = _sum_series(group.get("p_mw", []))
                    source = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergySource")
                    source_id = _add_identified_object(source, name)
                    ET.SubElement(source, f"{{{CIM_NS}}}EnergySource.p").text = f"{p_mw * 1e6:.6f}"

        if profile == "SV" and config.include_sv:
            if hasattr(net, "res_bus") and not net.res_bus.empty:
                for bus_idx in sorted(set(idx[0] for idx in net.res_bus.index)):
                    try:
                        vm_pu = float(net.res_bus.loc[(bus_idx, 1), "vm_pu"])
                        va_deg = float(net.res_bus.loc[(bus_idx, 1), "va_degree"])
                    except Exception:
                        vm_pu = 1.0
                        va_deg = 0.0
                    sv = ET.SubElement(rdf, f"{{{CIM_NS}}}SvVoltage")
                    _add_identified_object(sv, f"SvVoltage_{bus_idx}")
                    topo_id = topo_node_ids.get(bus_idx) or _stable_id("TN", str(bus_idx))
                    ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.TopologicalNode").set(
                        f"{{{RDF_NS}}}resource", f"#{topo_id}"
                    )
                    ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.v").text = f"{vm_pu:.6f}"
                    ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.angle").text = f"{va_deg:.6f}"

        if profile == "DL" and config.include_diagram_layout:
            if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
                location = ET.SubElement(rdf, f"{{{CIM_NS}}}Location")
                location_id = _add_identified_object(location, f"{config.base_name}_Location")
                for bus_idx, row in net.bus_geodata.iterrows():
                    position = ET.SubElement(rdf, f"{{{CIM_NS}}}PositionPoint")
                    _add_identified_object(position, f"Pos_{bus_idx}")
                    ET.SubElement(position, f"{{{CIM_NS}}}PositionPoint.Location").set(
                        f"{{{RDF_NS}}}resource", f"#{location_id}"
                    )
                    ET.SubElement(position, f"{{{CIM_NS}}}PositionPoint.xPosition").text = str(row.get("x", 0.0))
                    ET.SubElement(position, f"{{{CIM_NS}}}PositionPoint.yPosition").text = str(row.get("y", 0.0))

        tree = ET.ElementTree(rdf)
        tree.write(cim_path, encoding="utf-8", xml_declaration=True)

    def _pandapower_to_multiconductor(self, pp_net):
        sn_mva = float(getattr(pp_net, "sn_mva", 100) or 100)
        f_hz = float(getattr(pp_net, "f_hz", 50) or 50)
        rho_ohmm = float(getattr(pp_net, "rho_ohmm", 100) or 100)
        net = create_empty_network(sn_mva=sn_mva, f_hz=f_hz, rho_ohmm=rho_ohmm)

        bus_map: Dict[int, int] = {}
        if hasattr(pp_net, "bus"):
            for bus_idx, row in pp_net.bus.iterrows():
                bus_id = create_bus(
                    net,
                    vn_kv=float(row.get("vn_kv", 0.4) or 0.4),
                    name=row.get("name"),
                    num_phases=4,
                    grounded_phases=(0,),
                    index=int(bus_idx),
                )
                bus_map[int(bus_idx)] = bus_id

        if hasattr(pp_net, "bus_geodata") and pp_net.bus_geodata is not None:
            for bus_idx, row in pp_net.bus_geodata.iterrows():
                if int(bus_idx) in bus_map:
                    net.bus_geodata.loc[int(bus_idx), "x"] = float(row.get("x", 0.0))
                    net.bus_geodata.loc[int(bus_idx), "y"] = float(row.get("y", 0.0))

        if hasattr(pp_net, "ext_grid") and not pp_net.ext_grid.empty:
            for _, eg in pp_net.ext_grid.iterrows():
                bus_idx = int(eg.bus)
                if bus_idx not in bus_map:
                    continue
                vn_kv = float(pp_net.bus.at[bus_idx, "vn_kv"]) if hasattr(pp_net, "bus") else 12.47
                sc_mva = float(eg.get("s_sc_max_mva", 0.0) or 0.0)
                xr = float(eg.get("rx_max", 10.0) or 10.0)
                if sc_mva > 0:
                    z = (vn_kv ** 2) / sc_mva
                else:
                    z = 0.001
                r_ohm = z / math.sqrt(1 + xr * xr)
                x_ohm = r_ohm * xr
                create_ext_grid(
                    net,
                    bus=bus_map[bus_idx],
                    from_phase=(1, 2, 3),
                    to_phase=0,
                    vm_pu=float(eg.get("vm_pu", 1.0) or 1.0),
                    va_degree=float(eg.get("va_degree", 0.0) or 0.0),
                    r_ohm=r_ohm,
                    x_ohm=x_ohm,
                    name=eg.get("name"),
                )

        if hasattr(pp_net, "load") and not pp_net.load.empty:
            for _, load in pp_net.load.iterrows():
                bus_idx = int(load.bus)
                if bus_idx not in bus_map:
                    continue
                p_mw = float(load.p_mw or 0.0)
                q_mvar = float(load.q_mvar or 0.0)
                create_asymmetric_load(
                    net,
                    bus=bus_map[bus_idx],
                    from_phase=(1, 2, 3),
                    to_phase=0,
                    p_mw=(p_mw / 3, p_mw / 3, p_mw / 3),
                    q_mvar=(q_mvar / 3, q_mvar / 3, q_mvar / 3),
                    name=load.get("name"),
                )

        if hasattr(pp_net, "sgen") and not pp_net.sgen.empty:
            for _, gen in pp_net.sgen.iterrows():
                bus_idx = int(gen.bus)
                if bus_idx not in bus_map:
                    continue
                p_mw = float(gen.p_mw or 0.0)
                q_mvar = float(gen.get("q_mvar", 0.0) or 0.0)
                create_asymmetric_sgen(
                    net,
                    bus=bus_map[bus_idx],
                    from_phase=(1, 2, 3),
                    to_phase=0,
                    p_mw=(p_mw / 3, p_mw / 3, p_mw / 3),
                    q_mvar=(q_mvar / 3, q_mvar / 3, q_mvar / 3),
                    name=gen.get("name"),
                )

        if hasattr(pp_net, "line") and not pp_net.line.empty:
            for _, line in pp_net.line.iterrows():
                from_bus = int(line.from_bus)
                to_bus = int(line.to_bus)
                if from_bus not in bus_map or to_bus not in bus_map:
                    continue
                r = float(line.get("r_ohm_per_km", 0.1) or 0.1)
                x = float(line.get("x_ohm_per_km", 0.4) or 0.4)
                c_nf = float(line.get("c_nf_per_km", 0.0) or 0.0)
                std_key = (round(r, 6), round(x, 6), round(c_nf, 6))
                std_name = f"cim_seq_{hash(std_key) & 0xfffffff}"
                if not std_type_exists(net, std_name, element="sequence"):
                    create_std_type(
                        net,
                        {
                            "r_ohm_per_km": r,
                            "x_ohm_per_km": x,
                            "r0_ohm_per_km": r * 3,
                            "x0_ohm_per_km": x * 3,
                            "c_nf_per_km": c_nf,
                            "c0_nf_per_km": c_nf * 3,
                            "max_i_ka": float(line.get("max_i_ka", 1.0) or 1.0),
                        },
                        std_name,
                        element="sequence",
                        overwrite=True,
                        check_required=False,
                    )

                create_line(
                    net,
                    std_type=std_name,
                    model_type="sequence",
                    from_bus=bus_map[from_bus],
                    from_phase=(1, 2, 3),
                    to_bus=bus_map[to_bus],
                    to_phase=(1, 2, 3),
                    length_km=float(line.get("length_km", 1.0) or 1.0),
                    name=line.get("name"),
                )

        if hasattr(pp_net, "switch") and not pp_net.switch.empty:
            for _, sw in pp_net.switch.iterrows():
                if str(sw.get("et", "")).lower() != "b":
                    continue
                bus_a = int(sw.bus)
                bus_b = int(sw.element)
                if bus_a not in bus_map or bus_b not in bus_map:
                    continue
                create_switch(
                    net,
                    bus=bus_map[bus_a],
                    phase=(1, 2, 3),
                    element=bus_map[bus_b],
                    et="b",
                    closed=bool(sw.get("closed", True)),
                    name=sw.get("name"),
                )

        if hasattr(pp_net, "trafo") and not pp_net.trafo.empty:
            for _, tr in pp_net.trafo.iterrows():
                hv_bus = int(tr.hv_bus)
                lv_bus = int(tr.lv_bus)
                if hv_bus not in bus_map or lv_bus not in bus_map:
                    continue
                std_key = (
                    round(float(tr.sn_mva or 0.0), 4),
                    round(float(tr.vn_hv_kv or 0.0), 4),
                    round(float(tr.vn_lv_kv or 0.0), 4),
                    round(float(tr.vk_percent or 0.0), 4),
                    round(float(tr.vkr_percent or 0.0), 4),
                    str(tr.get("vector_group", "Yy0")),
                )
                std_name = f"cim_tr_{hash(std_key) & 0xfffffff}"
                if not std_type_exists(net, std_name, element="trafo"):
                    create_std_type(
                        net,
                        {
                            "sn_mva": float(tr.sn_mva or 0.0),
                            "vn_hv_kv": float(tr.vn_hv_kv or 0.0),
                            "vn_lv_kv": float(tr.vn_lv_kv or 0.0),
                            "vk_percent": float(tr.vk_percent or 0.0),
                            "vkr_percent": float(tr.vkr_percent or 0.0),
                            "pfe_kw": float(tr.get("pfe_kw", 0.0) or 0.0),
                            "i0_percent": float(tr.get("i0_percent", 0.0) or 0.0),
                            "shift_degree": float(tr.get("shift_degree", 0.0) or 0.0),
                            "vector_group": tr.get("vector_group", "Yy0"),
                            "tap_side": tr.get("tap_side", "lv"),
                            "tap_neutral": int(tr.get("tap_neutral", 0) or 0),
                            "tap_min": int(tr.get("tap_min", -5) or -5),
                            "tap_max": int(tr.get("tap_max", 5) or 5),
                            "tap_step_degree": float(tr.get("tap_step_degree", 0.0) or 0.0),
                            "tap_step_percent": float(tr.get("tap_step_percent", 1.25) or 1.25),
                        },
                        std_name,
                        element="trafo",
                        overwrite=True,
                        check_required=False,
                    )

                create_transformer_3ph(
                    net,
                    hv_bus=bus_map[hv_bus],
                    lv_bus=bus_map[lv_bus],
                    std_type=std_name,
                    tap_pos=float(tr.get("tap_pos", 0.0) or 0.0),
                    name=tr.get("name") or "Trafo",
                )

        return net


def _normalize_file_list(file_list: List[str]) -> List[str]:
    normalized: List[str] = []
    for path in file_list:
        if not path:
            continue
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.lower().endswith((".xml", ".zip")):
                    normalized.append(os.path.join(path, name))
        elif os.path.isfile(path):
            normalized.append(path)
    return normalized


def _apply_csv_enrichment(net, csv_dir: str) -> None:
    if not os.path.isdir(csv_dir):
        logger.warning("CSV directory not found: %s", csv_dir)
        return

    bus_coords_path = os.path.join(csv_dir, "ieee9500unbal_Buscoords.csv")
    lines_path = os.path.join(csv_dir, "ieee9500unbal_LinesInstanceZ.csv")
    transformers_path = os.path.join(csv_dir, "ieee9500unbal_Transformers.csv")

    if os.path.exists(bus_coords_path):
        _apply_bus_coords(net, bus_coords_path)
    if os.path.exists(lines_path):
        _apply_line_instance(net, lines_path)
    if os.path.exists(transformers_path):
        _apply_transformer_csv(net, transformers_path)


def _apply_bus_coords(net, csv_path: str) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed reading bus coords: %s", exc)
        return

    if df.empty or "Busname" not in df.columns:
        return

    name_to_xy: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get("Busname") or "").strip()
        if not name:
            continue
        try:
            x = float(row.get("X"))
            y = float(row.get("Y"))
        except (TypeError, ValueError):
            continue
        name_to_xy[name] = (x, y)

    if not hasattr(net, "bus"):
        return

    for bus_idx in sorted(set(idx[0] for idx in net.bus.index)):
        try:
            row = net.bus.xs(bus_idx, level=0).iloc[0]
        except Exception:
            row = net.bus.loc[(bus_idx, 0)] if (bus_idx, 0) in net.bus.index else net.bus.loc[bus_idx]
        name = str(row.get("name") or "").strip()
        if name in name_to_xy:
            x, y = name_to_xy[name]
            net.bus_geodata.loc[bus_idx, "x"] = x
            net.bus_geodata.loc[bus_idx, "y"] = y


def _apply_line_instance(net, csv_path: str) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed reading line instance CSV: %s", exc)
        return

    if df.empty or "Name" not in df.columns:
        return

    if not hasattr(net, "line") or net.line.empty:
        return

    name_to_row = {str(row["Name"]).strip(): row for _, row in df.iterrows()}
    for line_idx, group in net.line.groupby(level=0):
        row = group.iloc[0]
        name = str(row.get("name") or "").strip()
        if not name or name not in name_to_row:
            continue
        csv_row = name_to_row[name]
        r1 = float(csv_row.get("R1", 0.0) or 0.0)
        x1 = float(csv_row.get("X1", 0.0) or 0.0)
        c1 = float(csv_row.get("C1[nF]", 0.0) or 0.0)
        r0 = float(csv_row.get("R0", r1) or r1)
        x0 = float(csv_row.get("X0", x1) or x1)
        c0 = float(csv_row.get("C0[nF]", c1) or c1)
        length = float(csv_row.get("Length", 0.0) or 0.0)
        units = str(csv_row.get("Units") or "km").lower()
        if units in ("ft", "feet"):
            length_km = length * 0.0003048
        elif units in ("m", "meter", "meters"):
            length_km = length / 1000.0
        else:
            length_km = length

        std_name = f"csv_line_{hash((name, r1, x1, c1, r0, x0, c0)) & 0xfffffff}"
        if not std_type_exists(net, std_name, element="sequence"):
            create_std_type(
                net,
                {
                    "r_ohm_per_km": r1,
                    "x_ohm_per_km": x1,
                    "r0_ohm_per_km": r0,
                    "x0_ohm_per_km": x0,
                    "c_nf_per_km": c1,
                    "c0_nf_per_km": c0,
                    "max_i_ka": 1.0,
                },
                std_name,
                element="sequence",
                overwrite=True,
                check_required=False,
            )

        net.line.loc[(line_idx, slice(None)), "std_type"] = std_name
        net.line.loc[(line_idx, slice(None)), "model_type"] = "sequence"
        net.line.loc[(line_idx, slice(None)), "length_km"] = length_km


def _apply_transformer_csv(net, csv_path: str) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed reading transformer CSV: %s", exc)
        return

    if df.empty or "Name" not in df.columns:
        return

    if not hasattr(net, "std_types"):
        return

    for _, row in df.iterrows():
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        sn_mva = float(row.get("kVA1", 0.0) or 0.0) / 1000.0
        vn_hv = float(row.get("kV1", 0.0) or 0.0)
        vn_lv = float(row.get("kV2", 0.0) or 0.0)
        vk_percent = float(row.get("%x12", 0.0) or 0.0)
        vkr_percent = float(row.get("%loadloss", 0.0) or 0.0)
        pfe_kw = float(row.get("%noloadloss", 0.0) or 0.0)
        i0_percent = float(row.get("%imag", 0.0) or 0.0)
        conn1 = str(row.get("Conn1") or "w")
        conn2 = str(row.get("Conn2") or "w")
        vector_group = ("D" if conn1.lower().startswith("d") else "Y") + (
            "d" if conn2.lower().startswith("d") else "y"
        ) + "0"

        std_name = f"csv_tr_{hash((name, sn_mva, vn_hv, vn_lv, vk_percent, vkr_percent)) & 0xfffffff}"
        if not std_type_exists(net, std_name, element="trafo"):
            create_std_type(
                net,
                {
                    "sn_mva": sn_mva,
                    "vn_hv_kv": vn_hv,
                    "vn_lv_kv": vn_lv,
                    "vk_percent": vk_percent,
                    "vkr_percent": vkr_percent,
                    "pfe_kw": pfe_kw,
                    "i0_percent": i0_percent,
                    "shift_degree": 0.0,
                    "vector_group": vector_group,
                    "tap_side": "lv",
                    "tap_neutral": 0,
                    "tap_min": -5,
                    "tap_max": 5,
                    "tap_step_degree": 0.0,
                    "tap_step_percent": 1.25,
                },
                std_name,
                element="trafo",
                overwrite=True,
                check_required=False,
            )
