import math
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandapower as pp
import pandapower.networks as pn

import multiconductor as mc
from multiconductor.pycci import cci_powerflow
from multiconductor.pycci.std_types import create_std_type, std_type_exists
try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency for geodata transform
    Transformer = None


def _scale_geodata_positions(
    coords: Dict[int, Tuple[float, float]],
    target_width: int = 900,
    target_height: int = 600,
    margin: int = 60,
) -> Dict[int, Tuple[int, int]]:
    if not coords:
        return {}

    xs = [v[0] for v in coords.values() if math.isfinite(v[0])]
    ys = [v[1] for v in coords.values() if math.isfinite(v[1])]
    if not xs or not ys:
        return {}

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return {}

    scale = min(target_width / width, target_height / height)
    scaled: Dict[int, Tuple[int, int]] = {}
    for bus_idx, (x, y) in coords.items():
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        sx = int((x - min_x) * scale + margin)
        sy = int((max_y - y) * scale + margin)
        scaled[bus_idx] = (sx, sy)
    return scaled


def _project_epsg4431_to_wgs84(coords: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    if not coords or Transformer is None:
        return {}
    try:
        transformer = Transformer.from_crs("EPSG:4431", "EPSG:4326", always_xy=True)
    except Exception:
        return {}

    projected: Dict[int, Tuple[float, float]] = {}
    for bus_idx, (x, y) in coords.items():
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        lon, lat = transformer.transform(x, y)
        if math.isfinite(lat) and math.isfinite(lon):
            projected[bus_idx] = (lat, lon)
    return projected


def _grid_layout_positions(indices: List[int], cols: int = 12, spacing: int = 120, start_x: int = 100, start_y: int = 100) -> Dict[int, Tuple[int, int]]:
    positions: Dict[int, Tuple[int, int]] = {}
    for idx, bus in enumerate(sorted(indices)):
        row = idx // cols
        col = idx % cols
        positions[bus] = (start_x + col * spacing, start_y + row * spacing)
    return positions


def _bus_type(pp_net: pp.pandapowerNet, bus_idx: int) -> str:
    if not pp_net.ext_grid.empty and bus_idx in pp_net.ext_grid.bus.values:
        return "slack"
    if not pp_net.gen.empty and bus_idx in pp_net.gen.bus.values:
        return "pv"
    return "pq"


def _vector_group_from_connection(connection_type: str) -> str:
    mapping = {
        "Yg-Yg": "Yy0",
        "Yg-D": "Yd1",
        "D-Yg": "Dy1",
        "D-D": "Dd0",
    }
    return mapping.get(connection_type, "Yy0")


def _connection_type_from_vector_group(vector_group: str) -> str:
    vg = (vector_group or "").lower()
    if vg.startswith("yd") or "y" in vg and "d" in vg and vg.index("y") < vg.index("d"):
        return "Yg-D"
    if vg.startswith("dy") or "d" in vg and "y" in vg and vg.index("d") < vg.index("y"):
        return "D-Yg"
    if "d" in vg and "y" not in vg:
        return "D-D"
    return "Yg-Yg"


def _ensure_sequence_line_type(net: pp.pandapowerNet, r: float, x: float, c_nf: float, max_i_ka: float = 1.0) -> str:
    r = max(r, 1e-6)
    x = max(x, 1e-6)
    c_nf = max(c_nf, 0.0)
    key = (round(r, 6), round(x, 6), round(c_nf, 6), round(max_i_ka, 6))
    name = f"seq_{hash(key) & 0xfffffff}"
    if not std_type_exists(net, name, element="sequence"):
        data = {
            "r_ohm_per_km": r,
            "x_ohm_per_km": x,
            "r0_ohm_per_km": r * 3,
            "x0_ohm_per_km": x * 3,
            "c_nf_per_km": c_nf,
            "c0_nf_per_km": c_nf * 3,
            "max_i_ka": max_i_ka,
        }
        create_std_type(net, data, name, element="sequence", overwrite=True, check_required=False)
    return name


def _ensure_trafo_type(net: pp.pandapowerNet, sn_mva: float, vn_hv_kv: float, vn_lv_kv: float, vk_percent: float,
                       vkr_percent: float, vector_group: str, tap_pos: float) -> str:
    key = (round(sn_mva, 4), round(vn_hv_kv, 4), round(vn_lv_kv, 4), round(vk_percent, 4), round(vkr_percent, 4), vector_group)
    name = f"trafo_{hash(key) & 0xfffffff}"
    if not std_type_exists(net, name, element="trafo"):
        data = {
            "sn_mva": sn_mva,
            "vn_hv_kv": vn_hv_kv,
            "vn_lv_kv": vn_lv_kv,
            "vk_percent": vk_percent,
            "vkr_percent": vkr_percent,
            "pfe_kw": 0.0,
            "i0_percent": 0.0,
            "shift_degree": 0,
            "tap_side": "lv",
            "tap_neutral": 0,
            "tap_min": -5,
            "tap_max": 5,
            "tap_step_percent": 1.25,
            "tap_step_degree": 0,
            "vector_group": vector_group,
        }
        create_std_type(net, data, name, element="trafo", overwrite=True, check_required=False)
    return name


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _extract_bus_xy(bus_row) -> Optional[Tuple[float, float]]:
    if bus_row is None or not hasattr(bus_row, "get"):
        return None
    candidates = [
        ("x", "y"),
        ("x_ft", "y_ft"),
        ("x_coord", "y_coord"),
        ("x_coord_ft", "y_coord_ft"),
        ("x_usft", "y_usft"),
        ("utm_x", "utm_y"),
        ("x_utm", "y_utm"),
        ("easting", "northing"),
        ("east", "north"),
    ]
    for x_key, y_key in candidates:
        if x_key in bus_row and y_key in bus_row:
            x = _safe_float(bus_row.get(x_key), float("nan"))
            y = _safe_float(bus_row.get(y_key), float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                return (x, y)
    return None


def _pandapower_net_to_ui(pp_net: pp.pandapowerNet) -> Dict[str, List[Dict]]:
    bus_indices = list(pp_net.bus.index)
    positions = _grid_layout_positions(bus_indices)

    elements: List[Dict] = []
    connections: List[Dict] = []
    bus_id_map: Dict[int, str] = {}

    for bus_idx in bus_indices:
        bus_id = str(uuid4())
        bus_id_map[bus_idx] = bus_id
        x, y = positions[bus_idx]
        bus_type = _bus_type(pp_net, bus_idx)
        elements.append({
            "id": bus_id,
            "name": pp_net.bus.at[bus_idx, "name"] if "name" in pp_net.bus.columns else f"Bus {bus_idx}",
            "type": "bus",
            "x": x,
            "y": y,
            "rotation": 0,
            "enabled": True,
            "nominalVoltageKV": _safe_float(pp_net.bus.at[bus_idx, "vn_kv"]),
            "busType": bus_type,
        })

    # External grid(s)
    for eg_idx, eg in pp_net.ext_grid.iterrows():
        bus_id = bus_id_map.get(int(eg.bus))
        if not bus_id:
            continue
        x, y = positions[int(eg.bus)]
        source_id = str(uuid4())
        elements.append({
            "id": source_id,
            "name": eg.get("name") or f"Grid {eg.bus}",
            "type": "external_source",
            "x": x - 60,
            "y": y,
            "rotation": 0,
            "enabled": True,
            "voltageKV": _safe_float(pp_net.bus.at[int(eg.bus), "vn_kv"]),
            "shortCircuitMVA": 5000,
            "xrRatio": 15,
            "phaseAngle": _safe_float(eg.get("va_degree", 0)),
        })
        connections.append({
            "id": str(uuid4()),
            "fromElementId": source_id,
            "toElementId": bus_id,
            "fromPort": "output",
            "toPort": "input",
        })

    # Generators (PV buses)
    for gen_idx, gen in pp_net.gen.iterrows():
        bus_id = bus_id_map.get(int(gen.bus))
        if not bus_id:
            continue
        x, y = positions[int(gen.bus)]
        gen_id = str(uuid4())
        elements.append({
            "id": gen_id,
            "name": gen.get("name") or f"Gen {gen.bus}",
            "type": "generator",
            "x": x + 60,
            "y": y - 30,
            "rotation": 0,
            "enabled": True,
            "ratingMVA": _safe_float(gen.get("sn_mva")) or max(_safe_float(gen.p_mw) * 1.2, 1.0),
            "activePowerMW": _safe_float(gen.p_mw),
            "voltageSetpointPU": _safe_float(gen.get("vm_pu", 1.0), 1.0),
            "minReactivePowerMVAR": _safe_float(gen.get("min_q_mvar", -_safe_float(gen.p_mw) * 0.5)),
            "maxReactivePowerMVAR": _safe_float(gen.get("max_q_mvar", _safe_float(gen.p_mw) * 0.6)),
            "connectedBusId": bus_id,
        })
        connections.append({
            "id": str(uuid4()),
            "fromElementId": gen_id,
            "toElementId": bus_id,
            "fromPort": "output",
            "toPort": "input",
        })

    # Loads
    for load_idx, load in pp_net.load.iterrows():
        bus_id = bus_id_map.get(int(load.bus))
        if not bus_id:
            continue
        x, y = positions[int(load.bus)]
        load_id = str(uuid4())
        elements.append({
            "id": load_id,
            "name": load.get("name") or f"Load {load.bus}",
            "type": "load",
            "x": x + 60,
            "y": y + 30,
            "rotation": 0,
            "enabled": True,
            "activePowerKW": _safe_float(load.p_mw) * 1000,
            "reactivePowerKVAR": _safe_float(load.q_mvar) * 1000,
            "loadModel": "constant_power",
            "unbalanced": False,
            "phaseAPower": 33.33,
            "phaseBPower": 33.33,
            "phaseCPower": 33.34,
            "connectedBusId": bus_id,
        })
        connections.append({
            "id": str(uuid4()),
            "fromElementId": load_id,
            "toElementId": bus_id,
            "fromPort": "output",
            "toPort": "input",
        })

    # Lines
    for line_idx, line in pp_net.line.iterrows():
        from_bus_id = bus_id_map.get(int(line.from_bus))
        to_bus_id = bus_id_map.get(int(line.to_bus))
        if not from_bus_id or not to_bus_id:
            continue
        fx, fy = positions[int(line.from_bus)]
        tx, ty = positions[int(line.to_bus)]
        line_id = str(uuid4())
        c_nf = float(line.get("c_nf_per_km", 0))
        susceptance = 2 * math.pi * float(pp_net.f_hz) * (c_nf * 1e-9)
        elements.append({
            "id": line_id,
            "name": line.get("name") or f"L{line.from_bus}-{line.to_bus}",
            "type": "line",
            "x": (fx + tx) / 2,
            "y": (fy + ty) / 2,
            "rotation": 0,
            "enabled": True,
            "lengthKm": _safe_float(line.length_km, 1.0) if _safe_float(line.length_km, 1.0) > 0 else 1.0,
            "resistanceOhmPerKm": _safe_float(line.r_ohm_per_km),
            "reactanceOhmPerKm": _safe_float(line.x_ohm_per_km),
            "susceptanceSPerKm": _safe_float(susceptance),
            "fromElementId": from_bus_id,
            "toElementId": to_bus_id,
            "fromBusId": from_bus_id,
            "toBusId": to_bus_id,
        })
        connections.append({
            "id": str(uuid4()),
            "fromElementId": from_bus_id,
            "toElementId": line_id,
            "fromPort": "output",
            "toPort": "input",
        })
        connections.append({
            "id": str(uuid4()),
            "fromElementId": line_id,
            "toElementId": to_bus_id,
            "fromPort": "output",
            "toPort": "input",
        })

    # Transformers
    if hasattr(pp_net, "trafo") and not pp_net.trafo.empty:
        for tr_idx, tr in pp_net.trafo.iterrows():
            from_bus_id = bus_id_map.get(int(tr.hv_bus))
            to_bus_id = bus_id_map.get(int(tr.lv_bus))
            if not from_bus_id or not to_bus_id:
                continue
            fx, fy = positions[int(tr.hv_bus)]
            tx, ty = positions[int(tr.lv_bus)]
            trafo_id = str(uuid4())
            xr_ratio = 10.0
            if _safe_float(tr.get("vkr_percent", 0)) > 0 and _safe_float(tr.get("vk_percent", 0)) > 0:
                r = _safe_float(tr.vkr_percent)
                z = _safe_float(tr.vk_percent)
                x = math.sqrt(max(z * z - r * r, 0))
                if r > 0:
                    xr_ratio = x / r
            elements.append({
                "id": trafo_id,
                "name": tr.get("name") or f"T{tr.hv_bus}-{tr.lv_bus}",
                "type": "transformer",
                "x": (fx + tx) / 2,
                "y": (fy + ty) / 2,
                "rotation": 0,
                "enabled": True,
                "ratingMVA": _safe_float(tr.sn_mva),
                "primaryVoltageKV": _safe_float(tr.vn_hv_kv),
                "secondaryVoltageKV": _safe_float(tr.vn_lv_kv),
                "impedancePercent": _safe_float(tr.vk_percent),
                "xrRatio": xr_ratio,
                "tapPosition": _safe_float(tr.get("tap_pos", 0)),
                "connectionType": _connection_type_from_vector_group(tr.get("vector_group", "")),
                "fromBusId": from_bus_id,
                "toBusId": to_bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": from_bus_id,
                "toElementId": trafo_id,
                "fromPort": "output",
                "toPort": "input",
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": trafo_id,
                "toElementId": to_bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    return {"elements": elements, "connections": connections}


def pandapower_ieee118_to_ui() -> Dict[str, List[Dict]]:
    return _pandapower_net_to_ui(pn.case118())


def pandapower_ieee9_to_ui() -> Dict[str, List[Dict]]:
    return _pandapower_net_to_ui(pn.case9())


def pandapower_ieee30_to_ui() -> Dict[str, List[Dict]]:
    return _pandapower_net_to_ui(pn.case30())


def multiconductor_net_to_ui(pp_net: pp.pandapowerNet) -> Dict[str, List[Dict]]:
    bus_indices = sorted(set(idx[0] for idx in pp_net.bus.index))
    bus_positions: Dict[int, Tuple[int, int]] = {}
    bus_geo_coords: Dict[int, Tuple[float, float]] = {}
    bus_geo_latlon: Dict[int, Tuple[float, float]] = {}

    def get_bus_row(bus_idx: int):
        bus_rows = None
        try:
            bus_rows = pp_net.bus.xs(bus_idx, level=0)
        except Exception:
            try:
                bus_rows = pp_net.bus.loc[bus_idx]
            except Exception:
                bus_rows = None
        if bus_rows is not None:
            if getattr(bus_rows, "ndim", 1) > 1 and not bus_rows.empty:
                return bus_rows.iloc[0]
            return bus_rows
        return None

    if hasattr(pp_net, "bus_geodata") and pp_net.bus_geodata is not None and not pp_net.bus_geodata.empty:
        for bus_idx in bus_indices:
            if bus_idx in pp_net.bus_geodata.index:
                x = _safe_float(pp_net.bus_geodata.at[bus_idx, "x"], float("nan"))
                y = _safe_float(pp_net.bus_geodata.at[bus_idx, "y"], float("nan"))
                if math.isfinite(x) and math.isfinite(y):
                    bus_geo_coords[bus_idx] = (x, y)

    if not bus_geo_coords:
        for bus_idx in bus_indices:
            bus_row = get_bus_row(bus_idx)
            xy = _extract_bus_xy(bus_row)
            if xy:
                bus_geo_coords[bus_idx] = xy

    if bus_geo_coords:
        bus_positions = _scale_geodata_positions(bus_geo_coords)
        bus_geo_latlon = _project_epsg4431_to_wgs84(bus_geo_coords)

    if not bus_positions:
        bus_positions = _grid_layout_positions(bus_indices, cols=20, spacing=80, start_x=80, start_y=80)

    elements: List[Dict] = []
    connections: List[Dict] = []
    bus_id_map: Dict[int, str] = {}

    ext_grid_buses = set()
    if hasattr(pp_net, "ext_grid") and not pp_net.ext_grid.empty:
        ext_grid_buses = set(int(b) for b in pp_net.ext_grid.bus.values)

    for bus_idx in bus_indices:
        bus_row = get_bus_row(bus_idx)
        bus_id = str(uuid4())
        bus_id_map[bus_idx] = bus_id
        x, y = bus_positions.get(bus_idx, (100, 100))
        bus_type = "slack" if bus_idx in ext_grid_buses else "pq"
        elements.append({
            "id": bus_id,
            "name": bus_row.get("name") if bus_row is not None and hasattr(bus_row, "get") else f"Bus {bus_idx}",
            "type": "bus",
            "x": x,
            "y": y,
            "rotation": 0,
            "enabled": True,
            "nominalVoltageKV": _safe_float(bus_row.get("vn_kv") if bus_row is not None and hasattr(bus_row, "get") else None, 12.47),
            "busType": bus_type,
        })
        if bus_idx in bus_geo_coords:
            elements[-1]["geoX"] = bus_geo_coords[bus_idx][0]
            elements[-1]["geoY"] = bus_geo_coords[bus_idx][1]
        if bus_idx in bus_geo_latlon:
            elements[-1]["geoLat"] = bus_geo_latlon[bus_idx][0]
            elements[-1]["geoLon"] = bus_geo_latlon[bus_idx][1]

    # External grid(s)
    if hasattr(pp_net, "ext_grid") and not pp_net.ext_grid.empty:
        for _, eg in pp_net.ext_grid.iterrows():
            bus_idx = int(eg.bus)
            bus_id = bus_id_map.get(bus_idx)
            if not bus_id:
                continue
            x, y = bus_positions.get(bus_idx, (100, 100))
            bus_row = None
            try:
                bus_rows = pp_net.bus.xs(bus_idx, level=0)
                bus_row = bus_rows.iloc[0] if getattr(bus_rows, "ndim", 1) > 1 else bus_rows
            except Exception:
                try:
                    bus_row = pp_net.bus.loc[bus_idx]
                except Exception:
                    bus_row = None
            source_id = str(uuid4())
            elements.append({
                "id": source_id,
                "name": eg.get("name") or f"Grid {bus_idx}",
                "type": "external_source",
                "x": x - 60,
                "y": y,
                "rotation": 0,
                "enabled": True,
                "voltageKV": _safe_float(bus_row.get("vn_kv") if bus_row is not None and hasattr(bus_row, "get") else None, 12.47),
                "shortCircuitMVA": 5000,
                "xrRatio": 15,
                "phaseAngle": _safe_float(eg.get("va_degree", 0)),
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": source_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Loads
    if hasattr(pp_net, "load") and not pp_net.load.empty:
        for _, load in pp_net.load.iterrows():
            bus_id = bus_id_map.get(int(load.bus))
            if not bus_id:
                continue
            x, y = bus_positions.get(int(load.bus), (100, 100))
            load_id = str(uuid4())
            elements.append({
                "id": load_id,
                "name": load.get("name") or f"Load {load.bus}",
                "type": "load",
                "x": x + 60,
                "y": y + 30,
                "rotation": 0,
                "enabled": True,
                "activePowerKW": _safe_float(load.p_mw) * 1000,
                "reactivePowerKVAR": _safe_float(load.q_mvar) * 1000,
                "loadModel": "constant_power",
                "unbalanced": False,
                "phaseAPower": 33.33,
                "phaseBPower": 33.33,
                "phaseCPower": 33.34,
                "connectedBusId": bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": load_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Asymmetric Loads
    if hasattr(pp_net, "asymmetric_load") and not pp_net.asymmetric_load.empty:
        load_df = pp_net.asymmetric_load
        group_cols = ["bus"]
        if "name" in load_df.columns:
            group_cols.append("name")
        grouped = load_df.groupby(group_cols)
        for keys, group in grouped:
            if isinstance(keys, tuple):
                bus_idx = int(keys[0])
                name = keys[1] if len(keys) > 1 else None
            else:
                bus_idx = int(keys)
                name = None
            bus_id = bus_id_map.get(bus_idx)
            if not bus_id:
                continue
            p_mw = _safe_float(group.get("p_mw", 0).sum())
            q_mvar = _safe_float(group.get("q_mvar", 0).sum())
            x, y = bus_positions.get(bus_idx, (100, 100))
            load_id = str(uuid4())
            elements.append({
                "id": load_id,
                "name": name or f"Load {bus_idx}",
                "type": "load",
                "x": x + 40,
                "y": y + 24,
                "rotation": 0,
                "enabled": True,
                "activePowerKW": p_mw * 1000,
                "reactivePowerKVAR": q_mvar * 1000,
                "loadModel": "constant_power",
                "unbalanced": False,
                "phaseAPower": 33.33,
                "phaseBPower": 33.33,
                "phaseCPower": 33.34,
                "connectedBusId": bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": load_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Generators
    if hasattr(pp_net, "sgen") and not pp_net.sgen.empty:
        for _, gen in pp_net.sgen.iterrows():
            bus_id = bus_id_map.get(int(gen.bus))
            if not bus_id:
                continue
            x, y = bus_positions.get(int(gen.bus), (100, 100))
            gen_id = str(uuid4())
            elements.append({
                "id": gen_id,
                "name": gen.get("name") or f"Gen {gen.bus}",
                "type": "generator",
                "x": x + 60,
                "y": y - 30,
                "rotation": 0,
                "enabled": True,
                "ratingMVA": _safe_float(gen.get("sn_mva")) or max(_safe_float(gen.p_mw) * 1.2, 1.0),
                "activePowerMW": _safe_float(gen.p_mw),
                "voltageSetpointPU": 1.0,
                "minReactivePowerMVAR": _safe_float(gen.get("min_q_mvar", -_safe_float(gen.p_mw) * 0.5)),
                "maxReactivePowerMVAR": _safe_float(gen.get("max_q_mvar", _safe_float(gen.p_mw) * 0.6)),
                "connectedBusId": bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": gen_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Asymmetric Generators
    if hasattr(pp_net, "asymmetric_gen") and not pp_net.asymmetric_gen.empty:
        gen_df = pp_net.asymmetric_gen
        group_cols = ["bus"]
        if "name" in gen_df.columns:
            group_cols.append("name")
        grouped = gen_df.groupby(group_cols)
        for keys, group in grouped:
            if isinstance(keys, tuple):
                bus_idx = int(keys[0])
                name = keys[1] if len(keys) > 1 else None
            else:
                bus_idx = int(keys)
                name = None
            bus_id = bus_id_map.get(bus_idx)
            if not bus_id:
                continue
            p_mw = _safe_float(group.get("p_mw", 0).sum())
            sn_mva = _safe_float(group.get("sn_mva", 0).sum())
            vm_pu = _safe_float(group.get("vm_pu", 1.0).mean() if "vm_pu" in group else 1.0, 1.0)
            if p_mw == 0 and sn_mva == 0:
                continue
            x, y = bus_positions.get(bus_idx, (100, 100))
            gen_id = str(uuid4())
            elements.append({
                "id": gen_id,
                "name": name or f"Gen {bus_idx}",
                "type": "generator",
                "x": x + 40,
                "y": y - 24,
                "rotation": 0,
                "enabled": True,
                "ratingMVA": sn_mva or max(p_mw * 1.2, 1.0),
                "activePowerMW": p_mw,
                "voltageSetpointPU": vm_pu,
                "minReactivePowerMVAR": -abs(p_mw) * 0.5,
                "maxReactivePowerMVAR": abs(p_mw) * 0.6,
                "connectedBusId": bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": gen_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    if hasattr(pp_net, "asymmetric_sgen") and not pp_net.asymmetric_sgen.empty:
        sgen_df = pp_net.asymmetric_sgen
        group_cols = ["bus"]
        if "name" in sgen_df.columns:
            group_cols.append("name")
        grouped = sgen_df.groupby(group_cols)
        for keys, group in grouped:
            if isinstance(keys, tuple):
                bus_idx = int(keys[0])
                name = keys[1] if len(keys) > 1 else None
            else:
                bus_idx = int(keys)
                name = None
            bus_id = bus_id_map.get(bus_idx)
            if not bus_id:
                continue
            p_mw = _safe_float(group.get("p_mw", 0).sum())
            sn_mva = _safe_float(group.get("sn_mva", 0).sum())
            if p_mw == 0 and sn_mva == 0:
                continue
            x, y = bus_positions.get(bus_idx, (100, 100))
            gen_id = str(uuid4())
            elements.append({
                "id": gen_id,
                "name": name or f"SGen {bus_idx}",
                "type": "generator",
                "x": x + 40,
                "y": y - 24,
                "rotation": 0,
                "enabled": True,
                "ratingMVA": sn_mva or max(p_mw * 1.2, 1.0),
                "activePowerMW": p_mw,
                "voltageSetpointPU": 1.0,
                "minReactivePowerMVAR": -abs(p_mw) * 0.5,
                "maxReactivePowerMVAR": abs(p_mw) * 0.6,
                "connectedBusId": bus_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": gen_id,
                "toElementId": bus_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Switches (bus-bus)
    if hasattr(pp_net, "switch") and not pp_net.switch.empty:
        for switch_idx, group in pp_net.switch.groupby(level=0):
            row = group.iloc[0]
            if str(row.get("et", "")).lower() != "b":
                continue
            bus_a = int(row.get("bus"))
            bus_b = int(row.get("element"))
            bus_a_id = bus_id_map.get(bus_a)
            bus_b_id = bus_id_map.get(bus_b)
            if not bus_a_id or not bus_b_id:
                continue
            fx, fy = bus_positions.get(bus_a, (100, 100))
            tx, ty = bus_positions.get(bus_b, (100, 100))
            switch_id = str(uuid4())
            closed_val = row.get("closed", True)
            if isinstance(closed_val, str):
                is_closed = closed_val.strip().lower() in ("true", "1", "yes", "y", "t")
            else:
                is_closed = bool(closed_val)
            elements.append({
                "id": switch_id,
                "name": row.get("name") or f"SW{bus_a}-{bus_b}",
                "type": "switch",
                "x": (fx + tx) / 2,
                "y": (fy + ty) / 2,
                "rotation": 0,
                "enabled": True,
                "isClosed": is_closed,
                "fromBusId": bus_a_id,
                "toBusId": bus_b_id,
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": bus_a_id,
                "toElementId": switch_id,
                "fromPort": "output",
                "toPort": "input",
            })
            connections.append({
                "id": str(uuid4()),
                "fromElementId": switch_id,
                "toElementId": bus_b_id,
                "fromPort": "output",
                "toPort": "input",
            })

    # Single-phase transformers (trafo1ph)
    if hasattr(pp_net, "trafo1ph") and not pp_net.trafo1ph.empty:
        for trafo_idx, group in pp_net.trafo1ph.groupby(level=0):
            bus_levels = group.index.get_level_values("bus")
            buses = sorted({int(b) for b in bus_levels})
            if len(buses) < 2:
                continue
            bus_v = {}
            for bus_id, sub in group.groupby(level="bus"):
                bus_v[int(bus_id)] = _safe_float(sub.get("vn_kv", 0).mean(), 0.0)
            buses_sorted = sorted(buses, key=lambda b: bus_v.get(b, 0.0), reverse=True)
            hv_bus = buses_sorted[0]
            hv_bus_id = bus_id_map.get(hv_bus)
            if not hv_bus_id:
                continue
            name = None
            if "name" in group.columns:
                name_series = group["name"].dropna()
                if not name_series.empty:
                    name = name_series.iloc[0]
            sn_mva = _safe_float(group.get("sn_mva", 0).sum(), 0.0)
            vk_percent = _safe_float(group.get("vk_percent", 0).mean(), 5.75)
            vkr_percent = _safe_float(group.get("vkr_percent", 0).mean(), 0.0)
            xr_ratio = 10.0
            if vkr_percent > 0 and vk_percent > 0:
                x = math.sqrt(max(vk_percent * vk_percent - vkr_percent * vkr_percent, 0))
                xr_ratio = x / vkr_percent if vkr_percent > 0 else xr_ratio
            tap_pos = _safe_float(group.get("tap_pos", 0).mean(), 0.0)
            in_service_val = group.get("in_service", True)
            if hasattr(in_service_val, "all"):
                in_service_val = in_service_val.all()
            if isinstance(in_service_val, str):
                in_service = in_service_val.strip().lower() in ("true", "1", "yes", "y", "t")
            else:
                in_service = bool(in_service_val)
            for lv_bus in buses_sorted[1:]:
                lv_bus_id = bus_id_map.get(lv_bus)
                if not lv_bus_id:
                    continue
                fx, fy = bus_positions.get(hv_bus, (100, 100))
                tx, ty = bus_positions.get(lv_bus, (100, 100))
                trafo_id = str(uuid4())
                elements.append({
                    "id": trafo_id,
                    "name": name or f"T1ph{trafo_idx}",
                    "type": "transformer",
                    "x": (fx + tx) / 2,
                    "y": (fy + ty) / 2,
                    "rotation": 0,
                    "enabled": in_service,
                    "ratingMVA": sn_mva or 0.1,
                    "primaryVoltageKV": bus_v.get(hv_bus, 0.0),
                    "secondaryVoltageKV": bus_v.get(lv_bus, 0.0),
                    "impedancePercent": vk_percent,
                    "xrRatio": xr_ratio,
                    "tapPosition": tap_pos,
                    "connectionType": "Yg-Yg",
                    "fromBusId": hv_bus_id,
                    "toBusId": lv_bus_id,
                })
                connections.append({
                    "id": str(uuid4()),
                    "fromElementId": hv_bus_id,
                    "toElementId": trafo_id,
                    "fromPort": "output",
                    "toPort": "input",
                })
                connections.append({
                    "id": str(uuid4()),
                    "fromElementId": trafo_id,
                    "toElementId": lv_bus_id,
                    "fromPort": "output",
                    "toPort": "input",
                })

    # Lines
    if hasattr(pp_net, "line") and not pp_net.line.empty:
        std_sequence = {}
        if hasattr(pp_net, "std_types") and isinstance(pp_net.std_types, dict):
            std_sequence = pp_net.std_types.get("sequence", {}) or {}

        for _, line in pp_net.line.iterrows():
            from_bus_id = bus_id_map.get(int(line.from_bus))
            to_bus_id = bus_id_map.get(int(line.to_bus))
            if not from_bus_id or not to_bus_id:
                continue
            fx, fy = bus_positions.get(int(line.from_bus), (100, 100))
            tx, ty = bus_positions.get(int(line.to_bus), (100, 100))
            line_id = str(uuid4())
            std_name = line.get("std_type")
            std = std_sequence.get(std_name, {}) if std_name else {}
            r = _safe_float(std.get("r_ohm_per_km", 0.1), 0.1)
            x = _safe_float(std.get("x_ohm_per_km", 0.4), 0.4)
            c_nf = _safe_float(std.get("c_nf_per_km", 0.0), 0.0)
            susceptance = 2 * math.pi * float(pp_net.f_hz) * (c_nf * 1e-9)
            elements.append({
                "id": line_id,
                "name": line.get("name") or f"L{line.from_bus}-{line.to_bus}",
                "type": "line",
                "x": (fx + tx) / 2,
                "y": (fy + ty) / 2,
                "rotation": 0,
                "enabled": True,
                "lengthKm": _safe_float(line.length_km, 1.0) if _safe_float(line.length_km, 1.0) > 0 else 1.0,
                "resistanceOhmPerKm": r,
                "reactanceOhmPerKm": x,
                "susceptanceSPerKm": _safe_float(susceptance),
                "fromElementId": from_bus_id,
                "toElementId": to_bus_id,
                "fromBusId": from_bus_id,
                "toBusId": to_bus_id,
            })

    return {"elements": elements, "connections": connections}


def build_mc_from_ui(elements: List[Dict], connections: List[Dict]) -> pp.pandapowerNet:
    net = mc.create_empty_network(sn_mva=100)
    element_by_id = {el["id"]: el for el in elements}
    bus_map: Dict[str, int] = {}

    # Create buses
    for el in elements:
        if el.get("type") != "bus":
            continue
        bus_idx = mc.create_bus(
            net,
            vn_kv=float(el.get("nominalVoltageKV", 13.8)),
            num_phases=4,
            grounded_phases=(0,),
            name=el.get("name"),
        )
        bus_map[el["id"]] = bus_idx

    def connected_bus_id(element_id: str) -> Optional[str]:
        element = element_by_id.get(element_id, {})
        if element.get("connectedBusId"):
            return element["connectedBusId"]
        candidates = []
        for conn in connections:
            if conn.get("fromElementId") == element_id:
                candidates.append(conn.get("toElementId"))
            elif conn.get("toElementId") == element_id:
                candidates.append(conn.get("fromElementId"))
        for cid in candidates:
            if cid in bus_map:
                return cid
        return None

    def connected_bus_ids(element_id: str) -> List[str]:
        element = element_by_id.get(element_id, {})
        if element.get("fromBusId") and element.get("toBusId"):
            return [element.get("fromBusId"), element.get("toBusId")]
        candidates = []
        for conn in connections:
            if conn.get("fromElementId") == element_id:
                candidates.append(conn.get("toElementId"))
            elif conn.get("toElementId") == element_id:
                candidates.append(conn.get("fromElementId"))
        return [cid for cid in candidates if cid in bus_map]

    # External sources
    for el in elements:
        if el.get("type") != "external_source":
            continue
        bus_id = connected_bus_id(el["id"])
        if not bus_id:
            continue
        bus_idx = bus_map[bus_id]
        v_kv = float(el.get("voltageKV", 13.8))
        sc_mva = float(el.get("shortCircuitMVA", 100))
        xr = float(el.get("xrRatio", 10))
        z_ohm = (v_kv ** 2) / sc_mva if sc_mva > 0 else 0.001
        r_ohm = z_ohm / math.sqrt(1 + xr * xr) if xr > 0 else z_ohm
        x_ohm = r_ohm * xr
        mc.create_ext_grid(
            net,
            bus=bus_idx,
            from_phase=(1, 2, 3),
            to_phase=0,
            vm_pu=1.0,
            va_degree=float(el.get("phaseAngle", 0)),
            r_ohm=r_ohm,
            x_ohm=x_ohm,
            name=el.get("name"),
        )

    # Loads
    for el in elements:
        if el.get("type") != "load":
            continue
        bus_id = connected_bus_id(el["id"])
        if not bus_id:
            continue
        bus_idx = bus_map[bus_id]
        p_mw_total = float(el.get("activePowerKW", 0)) / 1000
        q_mvar_total = float(el.get("reactivePowerKVAR", 0)) / 1000
        load_model = el.get("loadModel") or "constant_power"
        const_z_percent_p = 0
        const_i_percent_p = 0
        const_z_percent_q = 0
        const_i_percent_q = 0
        if load_model == "constant_current":
            const_i_percent_p = 100
            const_i_percent_q = 100
        elif load_model == "constant_impedance":
            const_z_percent_p = 100
            const_z_percent_q = 100

        unbalanced = bool(el.get("unbalanced"))
        if unbalanced:
            phase_a = float(el.get("phaseAPower", 33.33) or 0)
            phase_b = float(el.get("phaseBPower", 33.33) or 0)
            phase_c = float(el.get("phaseCPower", 33.34) or 0)
            total = phase_a + phase_b + phase_c
            if total <= 0:
                weights = (1 / 3, 1 / 3, 1 / 3)
            else:
                weights = (phase_a / total, phase_b / total, phase_c / total)
            p_mw = tuple(p_mw_total * w for w in weights)
            q_mvar = tuple(q_mvar_total * w for w in weights)
        else:
            p_mw = p_mw_total / 3
            q_mvar = q_mvar_total / 3
        mc.create_asymmetric_load(
            net,
            bus=bus_idx,
            from_phase=(1, 2, 3),
            to_phase=0,
            p_mw=p_mw,
            q_mvar=q_mvar,
            name=el.get("name"),
            type=load_model,
            const_z_percent_p=const_z_percent_p,
            const_i_percent_p=const_i_percent_p,
            const_z_percent_q=const_z_percent_q,
            const_i_percent_q=const_i_percent_q,
        )

    # Generators (PQ)
    for el in elements:
        if el.get("type") != "generator":
            continue
        bus_id = connected_bus_id(el["id"])
        if not bus_id:
            continue
        bus_idx = bus_map[bus_id]
        p_mw_total = float(el.get("activePowerMW", 0))
        sn_mva = float(el.get("ratingMVA", p_mw_total * 1.2)) / 3 if p_mw_total else None
        mc.create_asymmetric_sgen(
            net,
            bus=bus_idx,
            from_phase=(1, 2, 3),
            to_phase=0,
            p_mw=p_mw_total / 3,
            q_mvar=0.0,
            sn_mva=sn_mva,
            name=el.get("name"),
        )

    # Lines and cables
    for el in elements:
        if el.get("type") not in ["line", "cable"]:
            continue
        bus_ids = connected_bus_ids(el["id"])
        if len(bus_ids) < 2:
            continue
        from_bus_id, to_bus_id = bus_ids[0], bus_ids[1]
        from_bus = bus_map[from_bus_id]
        to_bus = bus_map[to_bus_id]
        length_km = float(el.get("lengthKm", 1)) or 1.0
        r_ohm = float(el.get("resistanceOhmPerKm", 0.1))
        x_ohm = float(el.get("reactanceOhmPerKm", 0.4))
        susceptance = float(el.get("susceptanceSPerKm", 0))
        c_nf = (susceptance / (2 * math.pi * net.f_hz)) * 1e9 if susceptance else 0.0
        std_name = _ensure_sequence_line_type(net, r_ohm, x_ohm, c_nf)
        mc.create_line(
            net,
            std_type=std_name,
            model_type="sequence",
            from_bus=from_bus,
            from_phase=(1, 2, 3),
            to_bus=to_bus,
            to_phase=(1, 2, 3),
            length_km=length_km,
            name=el.get("name"),
        )

    # Switches (bus-bus)
    for el in elements:
        if el.get("type") != "switch":
            continue
        bus_ids = connected_bus_ids(el["id"])
        if len(bus_ids) < 2:
            continue
        from_bus_id, to_bus_id = bus_ids[0], bus_ids[1]
        from_bus = bus_map[from_bus_id]
        to_bus = bus_map[to_bus_id]
        is_closed = bool(el.get("isClosed", True))
        mc.create_switch(
            net,
            bus=from_bus,
            phase=(1, 2, 3),
            element=to_bus,
            et="b",
            closed=is_closed,
            name=el.get("name"),
        )

    # Transformers
    for el in elements:
        if el.get("type") != "transformer":
            continue
        bus_ids = connected_bus_ids(el["id"])
        if len(bus_ids) < 2:
            continue
        from_bus_id, to_bus_id = bus_ids[0], bus_ids[1]
        from_bus_el = element_by_id.get(from_bus_id, {})
        to_bus_el = element_by_id.get(to_bus_id, {})
        from_v = float(from_bus_el.get("nominalVoltageKV", 0) or 0)
        to_v = float(to_bus_el.get("nominalVoltageKV", 0) or 0)
        if from_v >= to_v:
            hv_bus_id, lv_bus_id = from_bus_id, to_bus_id
        else:
            hv_bus_id, lv_bus_id = to_bus_id, from_bus_id
        hv_bus = bus_map[hv_bus_id]
        lv_bus = bus_map[lv_bus_id]
        vk_percent = float(el.get("impedancePercent", 5.75))
        xr_ratio = float(el.get("xrRatio", 10))
        vkr_percent = vk_percent / math.sqrt(1 + xr_ratio * xr_ratio) if xr_ratio > 0 else vk_percent
        std_name = _ensure_trafo_type(
            net,
            sn_mva=float(el.get("ratingMVA", 10)),
            vn_hv_kv=float(el.get("primaryVoltageKV", 13.8)),
            vn_lv_kv=float(el.get("secondaryVoltageKV", 0.48)),
            vk_percent=vk_percent,
            vkr_percent=vkr_percent,
            vector_group=_vector_group_from_connection(el.get("connectionType", "Yg-Yg")),
            tap_pos=float(el.get("tapPosition", 0)),
        )
        mc.create_transformer_3ph(
            net,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            std_type=std_name,
            tap_pos=float(el.get("tapPosition", 0)),
            name=el.get("name", "Trafo"),
        )

    return net


def run_mc_loadflow(elements: List[Dict], connections: List[Dict], network_id: str) -> Dict:
    net = build_mc_from_ui(elements, connections)
    cci_powerflow.run_pf(net)

    bus_results = []
    if hasattr(net, "res_bus"):
        bus_indices = sorted(set(idx[0] for idx in net.res_bus.index))
        for bus_idx in bus_indices:
            def _phase_value(phase: int, field: str, fallback: float = 0.0) -> float:
                try:
                    return float(net.res_bus.loc[(bus_idx, phase), field])
                except Exception:
                    return fallback

            vm_a = _phase_value(1, "vm_pu", 1.0)
            vm_b = _phase_value(2, "vm_pu", vm_a)
            vm_c = _phase_value(3, "vm_pu", vm_a)
            va_a = _phase_value(1, "va_degree", 0.0)
            va_b = _phase_value(2, "va_degree", va_a - 120)
            va_c = _phase_value(3, "va_degree", va_a + 120)

            bus_name = None
            try:
                bus_name = net.bus.loc[(bus_idx, 1), "name"]
            except Exception:
                bus_name = f"Bus {bus_idx}"

            bus_results.append({
                "busId": str(bus_idx),
                "busName": bus_name,
                "voltagePhaseA": {"magnitude": vm_a, "angle": va_a},
                "voltagePhaseB": {"magnitude": vm_b, "angle": va_b},
                "voltagePhaseC": {"magnitude": vm_c, "angle": va_c},
            })

    branch_results = []
    if hasattr(net, "res_line") and not net.res_line.empty:
        for line_idx in sorted(set(idx[0] for idx in net.res_line.index)):
            rows = net.res_line.loc[line_idx]
            if hasattr(rows, "columns"):
                rows_df = rows
            else:
                rows_df = rows.to_frame().T
            i_ka = rows_df["i_ka"] if "i_ka" in rows_df.columns else rows_df["i_from_ka"]
            p_from = rows_df["p_from_mw"] if "p_from_mw" in rows_df.columns else 0
            q_from = rows_df["q_from_mvar"] if "q_from_mvar" in rows_df.columns else 0
            losses = rows_df["pl_mw"] if "pl_mw" in rows_df.columns else 0

            i_ka_values = i_ka.values if hasattr(i_ka, "values") else [float(i_ka)]
            phase_currents = [float(v) * 1000 for v in (list(i_ka_values) + [0, 0, 0])[:3]]

            line_name = None
            try:
                line_name = net.line.loc[(line_idx, 0), "name"]
            except Exception:
                line_name = f"Line {line_idx}"

            branch_results.append({
                "branchId": str(line_idx),
                "branchName": line_name,
                "currentPhaseA": phase_currents[0],
                "currentPhaseB": phase_currents[1],
                "currentPhaseC": phase_currents[2],
                "powerFlowMW": float(abs(p_from.sum())) if hasattr(p_from, "sum") else float(abs(p_from)),
                "powerFlowMVAR": float(abs(q_from.sum())) if hasattr(q_from, "sum") else float(abs(q_from)),
                "losses": float(abs(losses.sum())) if hasattr(losses, "sum") else float(abs(losses)),
            })

    if hasattr(net, "res_trafo") and not net.res_trafo.empty:
        for trafo_idx in sorted(set(idx[0] for idx in net.res_trafo.index)):
            rows = net.res_trafo.loc[trafo_idx]
            if hasattr(rows, "columns"):
                rows_df = rows
            else:
                rows_df = rows.to_frame().T
            i_ka = rows_df["i_ka"] if "i_ka" in rows_df.columns else 0
            p_mw = rows_df["p_mw"] if "p_mw" in rows_df.columns else 0
            q_mvar = rows_df["q_mvar"] if "q_mvar" in rows_df.columns else 0
            losses = rows_df["pl_mw"] if "pl_mw" in rows_df.columns else 0

            i_ka_values = i_ka.values if hasattr(i_ka, "values") else [float(i_ka)]
            phase_currents = [float(v) * 1000 for v in (list(i_ka_values) + [0, 0, 0])[:3]]

            branch_results.append({
                "branchId": f"T{trafo_idx}",
                "branchName": f"Trafo {trafo_idx}",
                "currentPhaseA": phase_currents[0],
                "currentPhaseB": phase_currents[1],
                "currentPhaseC": phase_currents[2],
                "powerFlowMW": float(abs(p_mw.sum())) if hasattr(p_mw, "sum") else float(abs(p_mw)),
                "powerFlowMVAR": float(abs(q_mvar.sum())) if hasattr(q_mvar, "sum") else float(abs(q_mvar)),
                "losses": float(abs(losses.sum())) if hasattr(losses, "sum") else float(abs(losses)),
            })

    import datetime

    return {
        "networkId": network_id,
        "converged": bool(getattr(net.model, "solved", True)),
        "iterations": int(getattr(net.model, "iterations", 0)),
        "timestamp": datetime.datetime.now().isoformat(),
        "busResults": bus_results,
        "branchResults": branch_results,
    }
