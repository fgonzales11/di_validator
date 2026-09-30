"""Matplotlib helpers for plotting network geodata."""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from keplergl import KeplerGl


Point = Tuple[float, float]
WEB_MERCATOR_LAT_LIMIT = 85.05112878
WEB_MERCATOR_RADIUS_M = 6378137.0


def _get_table(net: Any, *names: str) -> Any:
    for name in names:
        if hasattr(net, name):
            table = getattr(net, name)
            if table is not None:
                return table
        try:
            table = net[name]
        except Exception:
            continue
        if table is not None:
            return table
    return None


def _first_index_value(index_value: Any) -> Any:
    if isinstance(index_value, tuple) and index_value:
        return index_value[0]
    return index_value


def _finite_pair(x_value: Any, y_value: Any) -> Optional[Point]:
    try:
        x = float(x_value)
        y = float(y_value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(x) and math.isfinite(y):
        return (x, y)
    return None


def _canonical_key(value: Any) -> Any:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(as_float) and as_float.is_integer():
        return int(as_float)
    return value


def _column_lookup(table: Any) -> Dict[str, Any]:
    columns = getattr(table, "columns", [])
    return {str(column).strip().lower(): column for column in columns}


def _parse_string_geometry(value: str) -> Any:
    text = value.strip()
    if not text:
        return None

    if text.upper().startswith("LINESTRING"):
        pairs = re.findall(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
        return [(float(x), float(y)) for x, y in pairs]

    if text.upper().startswith("POINT"):
        match = re.search(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
        if match:
            return [(float(match.group(1)), float(match.group(2)))]

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def _geometry_to_points(value: Any) -> List[Point]:
    if value is None:
        return []

    if isinstance(value, str):
        value = _parse_string_geometry(value)
        if value is None:
            return []

    if isinstance(value, Mapping):
        lower = {str(key).lower(): key for key in value.keys()}
        if "coordinates" in lower:
            coords = value[lower["coordinates"]]
            geom_type = str(value.get(lower.get("type"), "")).lower()
            if geom_type == "point":
                pair = _finite_pair(coords[0], coords[1]) if isinstance(coords, (list, tuple)) and len(coords) >= 2 else None
                return [pair] if pair is not None else []
            value = coords
        else:
            x_key = lower.get("x") or lower.get("longitude") or lower.get("lon") or lower.get("lng")
            y_key = lower.get("y") or lower.get("latitude") or lower.get("lat")
            if x_key is not None and y_key is not None:
                pair = _finite_pair(value[x_key], value[y_key])
                return [pair] if pair is not None else []
            for key_name in ("coords", "geometry", "geo", "points"):
                key = lower.get(key_name)
                if key is not None:
                    return _geometry_to_points(value[key])
            return []

    if not isinstance(value, (list, tuple)):
        return []

    if len(value) >= 2 and not isinstance(value[0], (list, tuple, Mapping)):
        pair = _finite_pair(value[0], value[1])
        return [pair] if pair is not None else []

    points: List[Point] = []
    for item in value:
        item_points = _geometry_to_points(item)
        if item_points:
            points.extend(item_points)
    return points


def _table_has_rows(table: Any) -> bool:
    try:
        return len(table) > 0
    except TypeError:
        return False


def _has_network_protectors(net: Any) -> bool:
    return _table_has_rows(_get_table(net, "network_protector"))


def _extract_bus_points_from_table(table: Any) -> Dict[Any, Point]:
    coords: Dict[Any, Point] = {}

    if isinstance(table, Mapping):
        for bus_idx, value in table.items():
            points = _geometry_to_points(value)
            if points:
                coords[_canonical_key(bus_idx)] = points[0]
        return coords

    if not _table_has_rows(table) or not hasattr(table, "iterrows"):
        return coords

    columns = _column_lookup(table)
    x_col = columns.get("x") or columns.get("longitude") or columns.get("lon") or columns.get("lng")
    y_col = columns.get("y") or columns.get("latitude") or columns.get("lat")
    geometry_col = (
        columns.get("coords")
        or columns.get("geometry")
        or columns.get("geo")
        or columns.get("points")
    )

    for raw_idx, row in table.iterrows():
        bus_idx = _canonical_key(_first_index_value(raw_idx))
        if bus_idx in coords:
            continue

        pair = None
        if x_col is not None and y_col is not None:
            pair = _finite_pair(row[x_col], row[y_col])
        if pair is None and geometry_col is not None:
            points = _geometry_to_points(row[geometry_col])
            pair = points[0] if points else None
        if pair is not None:
            coords[bus_idx] = pair

    return coords


def _extract_bus_points(net: Any) -> Dict[Any, Point]:
    for table_name in ("bus_geo", "bus_geodata"):
        table = _get_table(net, table_name)
        coords = _extract_bus_points_from_table(table)
        if coords:
            return coords

    bus_table = _get_table(net, "bus", "buses")
    return _extract_bus_points_from_table(bus_table)


def _extract_line_points_from_table(table: Any) -> Dict[Any, List[Point]]:
    lines: Dict[Any, List[Point]] = {}

    if isinstance(table, Mapping):
        for line_idx, value in table.items():
            points = _geometry_to_points(value)
            if len(points) >= 2:
                lines[_canonical_key(line_idx)] = points
        return lines

    if not _table_has_rows(table) or not hasattr(table, "iterrows"):
        return lines

    columns = _column_lookup(table)
    geometry_col = (
        columns.get("coords")
        or columns.get("geometry")
        or columns.get("geo")
        or columns.get("points")
    )
    if geometry_col is None:
        return lines

    for raw_idx, row in table.iterrows():
        line_idx = _canonical_key(_first_index_value(raw_idx))
        if line_idx in lines:
            continue
        points = _geometry_to_points(row[geometry_col])
        if len(points) >= 2:
            lines[line_idx] = points

    return lines


def _extract_explicit_line_points(net: Any) -> Dict[Any, List[Point]]:
    for table_name in ("line_geo", "line_geodata", "branch_geo", "branch_geodata"):
        table = _get_table(net, table_name)
        lines = _extract_line_points_from_table(table)
        if lines:
            return lines
    return {}


def _iter_edge_rows(table: Any) -> Iterable[Tuple[Any, Any, Any]]:
    if not _table_has_rows(table) or not hasattr(table, "iterrows"):
        return

    columns = _column_lookup(table)
    from_col = columns.get("from_bus") or columns.get("from") or columns.get("source")
    to_col = columns.get("to_bus") or columns.get("to") or columns.get("target")
    if from_col is None or to_col is None:
        return

    seen = set()
    for raw_idx, row in table.iterrows():
        edge_idx = _canonical_key(_first_index_value(raw_idx))
        if edge_idx in seen:
            continue
        seen.add(edge_idx)
        from_bus = _canonical_key(row[from_col])
        to_bus = _canonical_key(row[to_col])
        yield edge_idx, from_bus, to_bus


def _infer_line_points(net: Any, bus_points: Mapping[Any, Point]) -> Dict[Any, List[Point]]:
    inferred: Dict[Any, List[Point]] = {}
    for table_name in ("line", "branch", "branches"):
        table = _get_table(net, table_name)
        for edge_idx, from_bus, to_bus in _iter_edge_rows(table) or ():
            start = bus_points.get(from_bus)
            end = bus_points.get(to_bus)
            if start is not None and end is not None:
                inferred[edge_idx] = [start, end]
        if inferred:
            return inferred
    return inferred


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _extract_bus_voltage_kv(net: Any) -> Dict[Any, float]:
    bus_table = _get_table(net, "bus", "buses")
    voltages: Dict[Any, float] = {}

    if isinstance(bus_table, Mapping):
        for bus_idx, value in bus_table.items():
            voltage = None
            if isinstance(value, Mapping):
                lower = {str(key).strip().lower(): key for key in value.keys()}
                for name in ("vn_kv", "voltage_kv", "nominal_voltage_kv", "v_nom_kv", "base_kv", "kv"):
                    key = lower.get(name)
                    if key is not None:
                        voltage = _finite_float(value[key])
                        break
            else:
                voltage = _finite_float(value)
            if voltage is not None:
                voltages[_canonical_key(bus_idx)] = voltage
        return voltages

    if not _table_has_rows(bus_table) or not hasattr(bus_table, "iterrows"):
        return voltages

    columns = _column_lookup(bus_table)
    voltage_col = None
    for name in ("vn_kv", "voltage_kv", "nominal_voltage_kv", "v_nom_kv", "base_kv", "kv"):
        voltage_col = columns.get(name)
        if voltage_col is not None:
            break
    if voltage_col is None:
        return voltages

    for raw_idx, row in bus_table.iterrows():
        bus_idx = _canonical_key(_first_index_value(raw_idx))
        if bus_idx in voltages:
            continue
        voltage = _finite_float(row[voltage_col])
        if voltage is not None:
            voltages[bus_idx] = voltage

    return voltages


def _voltage_class(from_voltage: Optional[float], to_voltage: Optional[float]) -> Any:
    if from_voltage is None:
        return to_voltage
    if to_voltage is None:
        return from_voltage
    if math.isclose(from_voltage, to_voltage, rel_tol=1e-6, abs_tol=1e-9):
        return from_voltage
    return tuple(sorted((from_voltage, to_voltage)))


def _extract_line_voltage_classes(net: Any, bus_voltages: Mapping[Any, float]) -> Dict[Any, Any]:
    line_voltages: Dict[Any, Any] = {}
    for table_name in ("line", "branch", "branches"):
        table = _get_table(net, table_name)
        if not _table_has_rows(table) or not hasattr(table, "iterrows"):
            continue

        columns = _column_lookup(table)
        from_col = columns.get("from_bus") or columns.get("from") or columns.get("source")
        to_col = columns.get("to_bus") or columns.get("to") or columns.get("target")
        voltage_col = None
        for name in ("vn_kv", "voltage_kv", "nominal_voltage_kv", "v_nom_kv", "base_kv", "kv"):
            voltage_col = columns.get(name)
            if voltage_col is not None:
                break

        for raw_idx, row in table.iterrows():
            line_idx = _canonical_key(_first_index_value(raw_idx))
            if line_idx in line_voltages:
                continue

            voltage = _finite_float(row[voltage_col]) if voltage_col is not None else None
            if voltage is None and from_col is not None and to_col is not None:
                from_bus = _canonical_key(row[from_col])
                to_bus = _canonical_key(row[to_col])
                voltage = _voltage_class(bus_voltages.get(from_bus), bus_voltages.get(to_bus))
            if voltage is not None:
                line_voltages[line_idx] = voltage

        if line_voltages:
            return line_voltages

    return line_voltages


def _voltage_sort_key(voltage: Any) -> Tuple[float, str]:
    if isinstance(voltage, tuple):
        numeric = max((_finite_float(item) or 0.0) for item in voltage)
        return numeric, str(voltage)
    numeric = _finite_float(voltage)
    return (numeric if numeric is not None else float("inf"), str(voltage))


def _format_voltage_label(voltage: Any) -> str:
    def format_single(value: Any) -> str:
        numeric = _finite_float(value)
        if numeric is None:
            return str(value)
        return f"{numeric:.2f}"

    if isinstance(voltage, tuple):
        return "/".join(format_single(value) for value in voltage) + " kV"
    return f"{format_single(voltage)} kV"


def _line_voltage_color_map(
    voltages: Sequence[Any],
    *,
    cmap_name: str,
    fallback_color: Any,
    explicit_colors: Optional[Mapping[Any, Any]],
) -> Dict[Any, Any]:
    unique_voltages = sorted(set(voltages), key=_voltage_sort_key)
    if not unique_voltages:
        return {}

    if len(unique_voltages) == 1:
        color_lookup = {unique_voltages[0]: fallback_color}
    else:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
        color_lookup = {
            voltage: cmap(0.15 + 0.75 * index / (len(unique_voltages) - 1))
            for index, voltage in enumerate(unique_voltages)
        }

    if explicit_colors:
        for voltage in unique_voltages:
            label = _format_voltage_label(voltage)
            if voltage in explicit_colors:
                color_lookup[voltage] = explicit_colors[voltage]
            elif label in explicit_colors:
                color_lookup[voltage] = explicit_colors[label]

    return color_lookup


def _extract_trafo1ph_points(
    net: Any,
    bus_points: Mapping[Any, Point],
) -> Tuple[Dict[Any, List[Point]], Dict[Any, Point]]:
    """Infer transformer terminal spans and marker positions from ``trafo1ph``."""
    table = _get_table(net, "trafo1ph")
    if not _table_has_rows(table) or not hasattr(table, "iterrows"):
        return {}, {}

    terminal_points: Dict[Any, List[Point]] = {}
    marker_points: Dict[Any, Point] = {}

    if hasattr(table.index, "names") and "bus" in table.index.names:
        element_level = table.index.names.index("index") if "index" in table.index.names else 0
        bus_level = table.index.names.index("bus")
        grouped_bus_ids: Dict[Any, List[Any]] = {}
        for raw_idx in table.index:
            if isinstance(raw_idx, tuple):
                trafo_idx = _canonical_key(raw_idx[element_level])
                bus_idx = _canonical_key(raw_idx[bus_level])
            else:
                continue
            grouped_bus_ids.setdefault(trafo_idx, [])
            if bus_idx not in grouped_bus_ids[trafo_idx]:
                grouped_bus_ids[trafo_idx].append(bus_idx)
    else:
        columns = _column_lookup(table)
        bus_cols = [
            column
            for key, column in columns.items()
            if key in {"bus", "hv_bus", "lv_bus", "from_bus", "to_bus"}
        ]
        grouped_bus_ids = {}
        for raw_idx, row in table.iterrows():
            trafo_idx = _canonical_key(_first_index_value(raw_idx))
            grouped_bus_ids.setdefault(trafo_idx, [])
            for column in bus_cols:
                bus_idx = _canonical_key(row[column])
                if bus_idx not in grouped_bus_ids[trafo_idx]:
                    grouped_bus_ids[trafo_idx].append(bus_idx)

    for trafo_idx, bus_ids in grouped_bus_ids.items():
        points = []
        for bus_idx in bus_ids:
            point = bus_points.get(bus_idx)
            if point is not None and point not in points:
                points.append(point)

        if not points:
            continue

        marker_points[trafo_idx] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        if len(points) >= 2:
            terminal_points[trafo_idx] = points

    return terminal_points, marker_points


def _trafo_terminal_segments(trafo_points: Mapping[Any, List[Point]]) -> List[List[Point]]:
    segments: List[List[Point]] = []
    for points in trafo_points.values():
        if len(points) == 2:
            segments.append(points)
            continue

        centroid = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for point in points:
            segments.append([centroid, point])
    return segments


def _looks_geographic(points: Sequence[Point]) -> bool:
    if not points:
        return False
    sample = points[: min(len(points), 500)]
    return all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in sample)


def _network_name(net: Any) -> str:
    name = getattr(net, "name", None)
    if name:
        return str(name)
    try:
        name = net["name"]
    except Exception:
        name = None
    return str(name) if name else "Network"


def _load_network_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _iter_networks(networks: Any, pattern: str = "*.pkl") -> Iterable[Tuple[str, Any]]:
    if isinstance(networks, (str, Path)):
        path = Path(networks)
        if path.is_dir():
            for item in sorted(path.glob(pattern)):
                yield item.stem, _load_network_pickle(item)
            return
        if path.is_file():
            yield path.stem, _load_network_pickle(path)
            return
        raise FileNotFoundError(f"No network file or directory found at {path}")

    if _get_table(networks, "bus", "buses") is not None:
        yield _network_name(networks), networks
        return

    if isinstance(networks, Mapping):
        for label, net in networks.items():
            yield str(label), net
        return

    for index, net in enumerate(networks, start=1):
        label = _network_name(net)
        if label == "Network":
            label = f"Network {index}"
        yield label, net


def _lonlat_to_web_mercator(point: Point) -> Point:
    lon, lat = point
    lat = max(min(float(lat), WEB_MERCATOR_LAT_LIMIT), -WEB_MERCATOR_LAT_LIMIT)
    x = WEB_MERCATOR_RADIUS_M * math.radians(float(lon))
    y = WEB_MERCATOR_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def _web_mercator_to_lonlat(x: float, y: float) -> Point:
    lon = math.degrees(float(x) / WEB_MERCATOR_RADIUS_M)
    lat = math.degrees(2.0 * math.atan(math.exp(float(y) / WEB_MERCATOR_RADIUS_M)) - math.pi / 2.0)
    return lon, lat


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(min(float(lat), WEB_MERCATOR_LAT_LIMIT), -WEB_MERCATOR_LAT_LIMIT)
    n = 2 ** zoom
    x_tile = int((float(lon) + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x_tile)), max(0, min(n - 1, y_tile))


def _tile_bounds_web_mercator(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2 ** zoom
    west = x_tile / n * 360.0 - 180.0
    east = (x_tile + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y_tile + 1) / n))))
    west_x, south_y = _lonlat_to_web_mercator((west, south))
    east_x, north_y = _lonlat_to_web_mercator((east, north))
    return west_x, south_y, east_x, north_y


def _basemap_template(provider: Optional[str], dark_mode: bool) -> str:
    if provider and "{z}" in provider and "{x}" in provider and "{y}" in provider:
        return provider

    provider_key = (provider or ("cartodb_dark" if dark_mode else "cartodb_light")).lower()
    templates = {
        "cartodb_dark": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "cartodb_dark_matter": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "dark": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "cartodb_light": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "light": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "openstreetmap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    }
    if provider_key not in templates:
        valid = ", ".join(sorted(templates))
        raise ValueError(f"Unknown basemap provider {provider!r}. Use one of: {valid}, or pass a tile URL template.")
    return templates[provider_key]


def _tile_range_for_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
) -> Tuple[int, int, int, int]:
    x_min, y_max = _lonlat_to_tile(west, south, zoom)
    x_max, y_min = _lonlat_to_tile(east, north, zoom)
    return min(x_min, x_max), max(x_min, x_max), min(y_min, y_max), max(y_min, y_max)


def _choose_basemap_zoom(
    west: float,
    south: float,
    east: float,
    north: float,
    max_tiles: int,
) -> int:
    for zoom in range(18, -1, -1):
        x_min, x_max, y_min, y_max = _tile_range_for_bounds(west, south, east, north, zoom)
        tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
        if tile_count <= max_tiles:
            return zoom
    return 0


def _add_tile_basemap(
    ax: Any,
    lonlat_points: Sequence[Point],
    *,
    dark_mode: bool,
    provider: Optional[str],
    zoom: Optional[int],
    max_tiles: int,
    alpha: float,
    timeout: float,
) -> bool:
    if not lonlat_points:
        return False

    from PIL import Image

    lon_values = [point[0] for point in lonlat_points]
    lat_values = [point[1] for point in lonlat_points]
    west, east = min(lon_values), max(lon_values)
    south, north = min(lat_values), max(lat_values)
    lon_pad = max((east - west) * 0.04, 0.005)
    lat_pad = max((north - south) * 0.04, 0.005)
    west = max(west - lon_pad, -180.0)
    east = min(east + lon_pad, 180.0)
    south = max(south - lat_pad, -WEB_MERCATOR_LAT_LIMIT)
    north = min(north + lat_pad, WEB_MERCATOR_LAT_LIMIT)

    chosen_zoom = zoom if zoom is not None else _choose_basemap_zoom(west, south, east, north, max_tiles)
    x_min, x_max, y_min, y_max = _tile_range_for_bounds(west, south, east, north, chosen_zoom)
    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count > max_tiles:
        raise ValueError(
            f"Basemap zoom {chosen_zoom} needs {tile_count} tiles; "
            f"increase max_tiles or choose a lower basemap_zoom."
        )

    template = _basemap_template(provider, dark_mode)
    tile_size = 256
    mosaic = Image.new(
        "RGB",
        ((x_max - x_min + 1) * tile_size, (y_max - y_min + 1) * tile_size),
        (30, 30, 30) if dark_mode else (240, 240, 240),
    )

    for x_tile in range(x_min, x_max + 1):
        for y_tile in range(y_min, y_max + 1):
            url = template.format(z=chosen_zoom, x=x_tile, y=y_tile)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                tile = Image.open(io.BytesIO(response.read())).convert("RGB")
            mosaic.paste(tile, ((x_tile - x_min) * tile_size, (y_tile - y_min) * tile_size))

    west_x, _, _, north_y = _tile_bounds_web_mercator(x_min, y_min, chosen_zoom)
    _, south_y, east_x, _ = _tile_bounds_web_mercator(x_max, y_max, chosen_zoom)
    ax.imshow(
        mosaic,
        extent=(west_x, east_x, south_y, north_y),
        origin="upper",
        alpha=alpha,
        zorder=0,
    )
    return True


def plot_network_geodata(
    net: Any,
    ax: Any = None,
    *,
    title: Optional[str] = None,
    show_buses: bool = True,
    show_lines: bool = True,
    show_trafo1ph: bool = True,
    infer_lines_from_buses: bool = True,
    bus_size: float = 8.0,
    bus_color: str = "#d62728",
    bus_alpha: float = 0.8,
    line_color: str = "#1f77b4",
    line_width: float = 0.8,
    line_alpha: float = 0.75,
    color_lines_by_voltage: bool = True,
    line_voltage_cmap: str = "plasma",
    line_voltage_colors: Optional[Mapping[Any, Any]] = None,
    trafo_color: str = "#ff7f0e",
    trafo_line_width: float = 1.2,
    trafo_line_alpha: float = 0.9,
    trafo_marker_size: float = 36.0,
    trafo_marker: str = "s",
    trafo_alpha: float = 0.95,
    annotate_buses: bool = False,
    max_annotations: int = 200,
    equal_aspect: bool = True,
    grid: bool = True,
    dark_mode: bool = False,
) -> Tuple[Any, Any]:
    """Plot a pandapower or LFE-style network from bus/line geodata.

    The helper accepts both pandapower names (``bus_geodata`` and
    ``line_geodata``) and converted-network names (``bus_geo``, ``line_geo``,
    or ``branch_geo``). If only bus geodata exists, line segments are inferred
    from ``line.from_bus`` / ``line.to_bus`` when possible. Multiconductor
    ``trafo1ph`` elements are inferred from their indexed bus terminals and
    drawn as transformer spans plus centroid markers.

    When ``color_lines_by_voltage`` is ``True``, line segments are grouped by
    direct line voltage columns when available, otherwise by the nominal
    voltage of their connected buses.

    When ``dark_mode`` is ``True``, the figure and axes use a dark background
    with light text, and the default bus, line, and transformer colors are
    brightened for contrast.

    Returns
    -------
    tuple
        ``(fig, ax)`` from matplotlib.
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if dark_mode:
        bg_color = "#1e1e1e"
        fg_color = "#e0e0e0"
        if bus_color == "#d62728":
            bus_color = "#ff6b6b"
        if line_color == "#1f77b4":
            line_color = "#4da3ff"
        if trafo_color == "#ff7f0e":
            trafo_color = "#ffb86b"
    else:
        bg_color = None
        fg_color = None

    bus_points = _extract_bus_points(net)
    line_points = _extract_explicit_line_points(net)
    if infer_lines_from_buses and bus_points:
        inferred = _infer_line_points(net, bus_points)
        for line_idx, points in inferred.items():
            line_points.setdefault(line_idx, points)
    bus_voltages = _extract_bus_voltage_kv(net) if show_lines and color_lines_by_voltage else {}
    line_voltages = (
        _extract_line_voltage_classes(net, bus_voltages)
        if show_lines and color_lines_by_voltage
        else {}
    )
    trafo_points, trafo_markers = (
        _extract_trafo1ph_points(net, bus_points) if show_trafo1ph and bus_points else ({}, {})
    )
    trafo_segments = _trafo_terminal_segments(trafo_points) if show_trafo1ph else []

    if not bus_points and not line_points and not trafo_markers:
        raise ValueError("No usable bus, line, or trafo1ph geodata found on the network.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    line_items = list(line_points.items()) if show_lines else []
    line_segments = [points for _line_idx, points in line_items]
    line_legend_items: List[Tuple[Any, int, Any]] = []
    unknown_voltage_count = 0
    if line_items:
        voltage_segments: Dict[Any, List[List[Point]]] = {}
        unknown_voltage_segments: List[List[Point]] = []
        if color_lines_by_voltage and line_voltages:
            for line_idx, points in line_items:
                voltage = line_voltages.get(line_idx)
                if voltage is None:
                    unknown_voltage_segments.append(points)
                else:
                    voltage_segments.setdefault(voltage, []).append(points)

        if voltage_segments:
            voltage_colors = _line_voltage_color_map(
                list(voltage_segments),
                cmap_name=line_voltage_cmap,
                fallback_color=line_color,
                explicit_colors=line_voltage_colors,
            )
            for voltage in sorted(voltage_segments, key=_voltage_sort_key):
                segments = voltage_segments[voltage]
                color = voltage_colors[voltage]
                collection = LineCollection(
                    segments,
                    colors=color,
                    linewidths=line_width,
                    alpha=line_alpha,
                    zorder=1,
                )
                ax.add_collection(collection)
                line_legend_items.append((voltage, len(segments), color))

            if unknown_voltage_segments:
                collection = LineCollection(
                    unknown_voltage_segments,
                    colors=line_color,
                    linewidths=line_width,
                    alpha=line_alpha,
                    zorder=1,
                )
                ax.add_collection(collection)
                unknown_voltage_count = len(unknown_voltage_segments)
        else:
            collection = LineCollection(
                line_segments,
                colors=line_color,
                linewidths=line_width,
                alpha=line_alpha,
                zorder=1,
            )
            ax.add_collection(collection)

    if trafo_segments:
        trafo_collection = LineCollection(
            trafo_segments,
            colors=trafo_color,
            linewidths=trafo_line_width,
            alpha=trafo_line_alpha,
            zorder=3,
        )
        ax.add_collection(trafo_collection)

    if show_buses and bus_points:
        xs = [point[0] for point in bus_points.values()]
        ys = [point[1] for point in bus_points.values()]
        ax.scatter(
            xs,
            ys,
            s=bus_size,
            c=bus_color,
            alpha=bus_alpha,
            edgecolors="none",
            label=f"buses ({len(bus_points)})",
            zorder=2,
        )

        if annotate_buses:
            for count, (bus_idx, (x, y)) in enumerate(bus_points.items()):
                if count >= max_annotations:
                    break
                ax.annotate(str(bus_idx), (x, y), fontsize=6, alpha=0.75)

    if line_legend_items:
        for voltage, segment_count, color in line_legend_items:
            ax.plot(
                [],
                [],
                color=color,
                linewidth=line_width,
                label=f"{_format_voltage_label(voltage)} lines ({segment_count})",
            )
        if unknown_voltage_count:
            ax.plot(
                [],
                [],
                color=line_color,
                linewidth=line_width,
                label=f"unknown voltage lines ({unknown_voltage_count})",
            )
    elif line_segments:
        # Invisible handle so the legend reports the line count.
        ax.plot([], [], color=line_color, linewidth=line_width, label=f"lines ({len(line_segments)})")

    if show_trafo1ph and trafo_markers:
        xs = [point[0] for point in trafo_markers.values()]
        ys = [point[1] for point in trafo_markers.values()]
        trafo_label = "transformer w/ network protector" if _has_network_protectors(net) else "trafo1ph"
        ax.scatter(
            xs,
            ys,
            s=trafo_marker_size,
            c=trafo_color,
            marker=trafo_marker,
            alpha=trafo_alpha,
            edgecolors="none",
            label=f"{trafo_label} ({len(trafo_markers)})",
            zorder=4,
        )

    all_points = list(bus_points.values())
    for segment in line_segments:
        all_points.extend(segment)
    for segment in trafo_segments:
        all_points.extend(segment)
    all_points.extend(trafo_markers.values())

    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        x_pad = x_span * 0.03 if x_span else 1.0
        y_pad = y_span * 0.03 if y_span else 1.0
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if _looks_geographic(all_points):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_title(title or f"{_network_name(net)} geodata")
    if grid:
        ax.grid(True, linewidth=0.4, alpha=0.3)

    if dark_mode:
        ax.title.set_color(fg_color)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.tick_params(colors=fg_color)
        for spine in ax.spines.values():
            spine.set_color(fg_color)
        ax.grid(color=fg_color)

    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc="best")
        if dark_mode:
            frame = legend.get_frame()
            frame.set_facecolor(bg_color)
            frame.set_edgecolor(fg_color)
            for text in legend.get_texts():
                text.set_color(fg_color)
    fig.tight_layout()
    return fig, ax


def plot_meshed_networks(
    networks: Any,
    ax: Any = None,
    *,
    pattern: str = "*.pkl",
    title: Optional[str] = None,
    show_buses: bool = True,
    show_lines: bool = True,
    infer_lines_from_buses: bool = True,
    bus_size: float = 2.0,
    bus_alpha: float = 0.65,
    line_width: float = 0.7,
    line_alpha: float = 0.82,
    color_map: str = "turbo",
    equal_aspect: bool = True,
    grid: bool = True,
    dark_mode: bool = True,
    max_legend_items: int = 30,
    basemap: bool = False,
    basemap_provider: Optional[str] = None,
    basemap_zoom: Optional[int] = None,
    basemap_max_tiles: int = 64,
    basemap_alpha: float = 1.0,
    basemap_timeout: float = 10.0,
) -> Tuple[Any, Any]:
    """Plot multiple meshed networks with distinct colors.

    Parameters
    ----------
    networks
        Directory containing network pickle files, a single pickle path, a
        mapping of label -> network, or an iterable of loaded networks.
    pattern
        Glob pattern used when ``networks`` is a directory.
    basemap
        When ``True`` and the network geodata is lon/lat, draw a web tile
        basemap under the networks. This fetches tiles at render time.
    basemap_provider
        Built-in provider key (``"dark"``, ``"light"``, ``"osm"``) or a tile
        URL template containing ``{z}``, ``{x}``, and ``{y}``. Defaults to a
        CARTO dark/light map based on ``dark_mode``.

    Returns
    -------
    tuple
        ``(fig, ax)`` from matplotlib.
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    network_items = list(_iter_networks(networks, pattern=pattern))
    if not network_items:
        raise ValueError("No networks found to plot.")

    if dark_mode:
        bg_color = "#1e1e1e"
        fg_color = "#e0e0e0"
    else:
        bg_color = None
        fg_color = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.figure

    if dark_mode:
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

    cmap = plt.get_cmap(color_map, max(len(network_items), 2))
    network_geodata = []
    all_source_points: List[Point] = []

    for index, (label, net) in enumerate(network_items):
        bus_points = _extract_bus_points(net)
        line_points = _extract_explicit_line_points(net)
        if infer_lines_from_buses and bus_points:
            inferred = _infer_line_points(net, bus_points)
            for line_idx, points in inferred.items():
                line_points.setdefault(line_idx, points)

        if not bus_points and not line_points:
            continue

        line_segments = list(line_points.values()) if show_lines else []
        network_geodata.append((index, label, bus_points, line_segments))
        if show_buses:
            all_source_points.extend(bus_points.values())
        for segment in line_segments:
            all_source_points.extend(segment)

    if not all_source_points:
        raise ValueError("No usable bus or line geodata found on the networks.")

    use_basemap = bool(basemap and _looks_geographic(all_source_points))
    if basemap and not use_basemap:
        raise ValueError("Basemap plotting requires geographic lon/lat geodata.")

    if use_basemap:
        _add_tile_basemap(
            ax,
            all_source_points,
            dark_mode=dark_mode,
            provider=basemap_provider,
            zoom=basemap_zoom,
            max_tiles=basemap_max_tiles,
            alpha=basemap_alpha,
            timeout=basemap_timeout,
        )

    def transform_points(points: Sequence[Point]) -> List[Point]:
        if use_basemap:
            return [_lonlat_to_web_mercator(point) for point in points]
        return list(points)

    all_plot_points: List[Point] = []
    legend_handles = []

    for index, label, bus_points, line_segments in network_geodata:
        color = cmap(index)
        plot_line_segments = [transform_points(segment) for segment in line_segments]
        if line_segments:
            ax.add_collection(
                LineCollection(
                    plot_line_segments,
                    colors=[color],
                    linewidths=line_width,
                    alpha=line_alpha,
                    zorder=2,
                )
            )
            for segment in plot_line_segments:
                all_plot_points.extend(segment)

        if show_buses and bus_points:
            plot_bus_points = transform_points(list(bus_points.values()))
            xs = [point[0] for point in plot_bus_points]
            ys = [point[1] for point in plot_bus_points]
            ax.scatter(
                xs,
                ys,
                s=bus_size,
                c=[color],
                alpha=bus_alpha,
                edgecolors="none",
                zorder=3,
            )
            all_plot_points.extend(plot_bus_points)

        if index < max_legend_items:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=max(line_width * 2.5, 1.5),
                    label=str(label),
                )
            )

    xs = [point[0] for point in all_plot_points]
    ys = [point[1] for point in all_plot_points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = x_span * 0.03 if x_span else 1.0
    y_pad = y_span * 0.03 if y_span else 1.0
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if use_basemap:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{_web_mercator_to_lonlat(value, 0.0)[0]:.3f}")
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{_web_mercator_to_lonlat(0.0, value)[1]:.3f}")
        )
    elif _looks_geographic(all_source_points):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_title(title or f"{len(network_items)} meshed networks")
    if grid:
        ax.grid(True, linewidth=0.4, alpha=0.3)

    if dark_mode:
        ax.title.set_color(fg_color)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.tick_params(colors=fg_color)
        for spine in ax.spines.values():
            spine.set_color(fg_color)
        ax.grid(color=fg_color)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            title="Networks",
        )
        if dark_mode:
            frame = legend.get_frame()
            frame.set_facecolor(bg_color)
            frame.set_edgecolor(fg_color)
            legend.get_title().set_color(fg_color)
            for text in legend.get_texts():
                text.set_color(fg_color)

    fig.tight_layout()
    return fig, ax
    
    
def get_bus_element_multimap(net):
    """Build bus_index -> set of element_types (a bus can belong to multiple element types)."""
    from collections import defaultdict
    bus_elems = defaultdict(set)
    element_types = ['ext_grid_sequence', 'trafo', 'trafo3w', 'trafo1ph', 'asymmetric_shunt', 'asymmetric_load', 'asymmetric_sgen', 'switch']
    for elem in element_types:
        if elem in net and hasattr(net[elem], 'columns') and len(net[elem]) > 0:
            df = net[elem]
            # trafo1ph stores hv_bus in index level 'bus'
            if elem == 'trafo1ph' and 'bus' in df.index.names:
                for bus_idx in set(df.index.get_level_values('bus')):
                    bus_elems[bus_idx].add(elem)
            else:
                bus_cols = [c for c in df.columns if c == 'bus' or c in ('hv_bus', 'lv_bus', 'mv_bus')]
                for col in bus_cols:
                    for bus_idx in df[col]:
                        bus_elems[bus_idx].add(elem)
    return bus_elems

# Color palette: RGB per element type
ELEMENT_RGB = {
    'bus':               [255, 0, 0],      # red
    'ext_grid_sequence': [255, 255, 0],    # yellow
    'switch':            [0, 150, 255],    # blue
    'asymmetric_shunt':  [255, 0, 255],    # magenta
    'asymmetric_load':   [0, 255, 150],    # green-cyan
    'asymmetric_sgen':   [255, 128, 200],  # pink
    'trafo':             [255, 165, 0],    # orange
    'trafo3w':           [255, 165, 0],    # orange
    'trafo1ph':          [255, 165, 0],    # orange
}

RES_BUS_COLS = ['vm_pu', 'va_degree', 'p_mw', 'q_mvar', 'imbalance_percent']
RES_LINE_COLS = ['i_from_ka', 'i_to_ka', 'i_ka', 'p_from_mw', 'p_to_mw',
                 'q_from_mvar', 'q_to_mvar', 'pl_mw', 'ql_mvar', 'loading_percent']

# Line color per nominal voltage (kV). 33 kV keeps the existing green.
VOLTAGE_LINE_RGB = {
    33: [0, 255, 0],     # green (existing)
    12: [255, 165, 0],   # orange
}
# Fallback colors for any other voltage levels found in the network.
EXTRA_LINE_RGB = [
    [0, 200, 255],    # cyan
    [255, 0, 255],    # magenta
    [255, 255, 0],    # yellow
    [200, 120, 255],  # purple
]


def line_voltage_color(level):
    """Return the RGB color for a line at nominal voltage ``level`` (kV).

    33 kV (and the 34.4 kV source side) keep the existing green; 12 kV gets a
    distinct color. Returns None for unrecognized levels so the caller can pick
    a fallback color.
    """
    if level is None:
        return [0, 255, 0]
    rounded = round(level)
    if rounded in VOLTAGE_LINE_RGB:
        return VOLTAGE_LINE_RGB[rounded]
    if 30 <= level <= 36:        # 33 / 34.4 kV source side -> green
        return [0, 255, 0]
    if 10 <= level <= 14:        # 12 kV -> orange
        return [255, 165, 0]
    return None


def build_kepler_map(net, title="Network"):
    bus_elems = get_bus_element_multimap(net)

    # Bus geodata has MultiIndex (bus_number, phase)
    bus_gdf = net.bus_geodata.copy()

    # Join res_bus fields (same MultiIndex)
    res_bus_cols = [c for c in RES_BUS_COLS if c in net.res_bus.columns]
    if len(res_bus_cols) > 0 and len(net.res_bus) > 0:
        bus_gdf = bus_gdf.join(net.res_bus[res_bus_cols], how='left')

    # Also add bus name for tooltip
    if 'name' in net.bus.columns:
        bus_gdf = bus_gdf.join(net.bus[['name']].rename(columns={'name': 'bus_name'}), how='left')

    # Columns to keep for each bus subset
    keep_cols = ['Longitude', 'Latitude']
    if 'bus_name' in bus_gdf.columns:
        keep_cols.append('bus_name')
    keep_cols.extend([c for c in res_bus_cols if c in bus_gdf.columns])

    # Build per-element-type bus sets (a bus can appear in multiple layers)
    from collections import defaultdict
    etype_bus_indices = defaultdict(set)
    for idx in bus_gdf.index:
        bus_num = idx[0]
        etypes = bus_elems.get(bus_num, set())
        if not etypes:
            etype_bus_indices['bus'].add(idx)
        else:
            for et in etypes:
                etype_bus_indices[et].add(idx)

    data_dict = {}
    layers = []
    bus_tooltip_fields = [{'name': c, 'format': None} for c in keep_cols if c not in ('Longitude', 'Latitude')]

    for etype in sorted(etype_bus_indices.keys()):
        indices = sorted(etype_bus_indices[etype])
        subset = bus_gdf.loc[bus_gdf.index.isin(indices), keep_cols].copy()
        if len(subset) == 0:
            continue
        data_name = etype
        data_dict[data_name] = subset
        color = ELEMENT_RGB.get(etype, [255, 0, 0])
        layers.append({
            'id': f'{etype}_layer',
            'type': 'point',
            'config': {
                'dataId': data_name,
                'label': etype,
                'color': color,
                'columns': {'lat': 'Latitude', 'lng': 'Longitude'},
                'isVisible': True,
                'visConfig': {
                    'radius': 5,
                    'opacity': 0.8,
                }
            }
        })
        print(f"  {etype}: {len(subset)} points, color={color}")

    # Build tooltip config: fieldsToShow per dataId (bus layers first)
    fields_to_show = {}
    for etype in etype_bus_indices:
        fields_to_show[etype] = bus_tooltip_fields

    # Build line data with res_line fields
    line_gdf = net.line_geodata.copy()
    # Align index names with res_line for join compatibility
    line_gdf.index.names = net.res_line.index.names
    line_gdf['Geometry'] = line_gdf['Geometry'].apply(
        lambda coords: 'LINESTRING (' + ', '.join(f'{c[0]} {c[1]}' for c in coords) + ')' if len(coords) >= 2 else None
    )
    line_gdf = line_gdf.dropna(subset=['Geometry'])

    # Join res_line fields
    res_line_cols = [c for c in RES_LINE_COLS if c in net.res_line.columns]
    if len(res_line_cols) > 0 and len(net.res_line) > 0:
        line_gdf = line_gdf.join(net.res_line[res_line_cols], how='left')

    # Add line name for tooltip
    if 'name' in net.line.columns:
        line_gdf = line_gdf.join(net.line[['name']].rename(columns={'name': 'line_name'}), how='left')

    # Classify each line by its nominal voltage (from the from_bus' vn_kv).
    def _to_float(value):
        try:
            if value is None or value != value:  # None or NaN
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    bus_vn = {}
    for bidx, brow in net.bus.iterrows():
        bnum = bidx[0] if isinstance(bidx, tuple) else bidx
        if bnum not in bus_vn:
            v = _to_float(brow.get('vn_kv'))
            if v is not None:
                bus_vn[bnum] = v

    line_vn = {}
    for lidx, lrow in net.line.iterrows():
        lid = lidx[0] if isinstance(lidx, tuple) else lidx
        if lid in line_vn:
            continue
        v = bus_vn.get(lrow.get('from_bus'))
        if v is None:
            v = bus_vn.get(lrow.get('to_bus'))
        if v is not None:
            line_vn[lid] = v

    def _line_eid(idx):
        return idx[0] if isinstance(idx, tuple) else idx

    line_gdf['vn_kv'] = [
        round(line_vn[_line_eid(idx)], 1) if _line_eid(idx) in line_vn else None
        for idx in line_gdf.index
    ]

    line_tooltip_fields = []
    if 'line_name' in line_gdf.columns:
        line_tooltip_fields.append({'name': 'line_name', 'format': None})
    line_tooltip_fields.append({'name': 'vn_kv', 'format': None})
    line_tooltip_fields.extend([{'name': c, 'format': None} for c in res_line_cols if c in line_gdf.columns])

    # One geojson layer per voltage level so each level gets its own color.
    levels = sorted({v for v in line_gdf['vn_kv'].tolist() if v is not None}, reverse=True)
    unleveled = line_gdf[line_gdf['vn_kv'].isna()]
    extra_idx = 0
    for level in levels:
        sub = line_gdf[line_gdf['vn_kv'] == level]
        if len(sub) == 0:
            continue
        color = line_voltage_color(level)
        if color is None:
            color = EXTRA_LINE_RGB[extra_idx % len(EXTRA_LINE_RGB)]
            extra_idx += 1
        data_name = f'lines_{level:g}kV'
        data_dict[data_name] = sub
        layers.append({
            'id': f'{data_name}_layer',
            'type': 'geojson',
            'config': {
                'dataId': data_name,
                'label': f'{level:g} kV lines',
                'color': color,
                'columns': {'geojson': 'Geometry'},
                'isVisible': True,
                'visConfig': {
                    'strokeColor': color,
                    'thickness': 2,
                    'opacity': 0.8
                }
            }
        })
        fields_to_show[data_name] = line_tooltip_fields
        print(f"  lines {level:g} kV: {len(sub)} segments, color={color}")

    # Lines whose voltage could not be determined keep the original green.
    if len(unleveled) > 0:
        data_dict['lines'] = unleveled
        layers.append({
            'id': 'lines_layer',
            'type': 'geojson',
            'config': {
                'dataId': 'lines',
                'label': 'lines (unknown kV)',
                'color': [0, 255, 0],
                'columns': {'geojson': 'Geometry'},
                'isVisible': True,
                'visConfig': {
                    'strokeColor': [0, 255, 0],
                    'thickness': 2,
                    'opacity': 0.8
                }
            }
        })
        fields_to_show['lines'] = line_tooltip_fields

    cfg = {
        'version': 'v1',
        'config': {
            'visState': {'layers': layers},
            'mapState': {
                'latitude': float(net.bus_geodata['Latitude'].mean()),
                'longitude': float(net.bus_geodata['Longitude'].mean()),
                'zoom': 13
            },
            'interactionConfig': {
                'tooltip': {
                    'enabled': True,
                    'fieldsToShow': fields_to_show
                }
            }
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmap = KeplerGl(height=800, data=data_dict, config=cfg)
    return kmap


def _infer_transformer_customer_counts(net, ami_csv):
    """Count AMI meters by the transformer bus serving their modeled load."""
    load_table = _get_table(net, "asymmetric_load", "load")
    if load_table is None or "name" not in load_table.columns or "bus" not in load_table.columns:
        return {}

    meter_to_buses = {}
    for _, row in load_table.iterrows():
        meter_name = str(row["name"]).strip().strip("'")
        meter_to_buses.setdefault(meter_name.upper(), set()).add(_canonical_key(row["bus"]))

    counts = {}
    with Path(ami_csv).open(newline="") as stream:
        for meter_name in next(csv.reader(stream), [])[1:]:
            for bus_index in meter_to_buses.get(meter_name.strip().strip("'").upper(), ()):
                counts[bus_index] = counts.get(bus_index, 0) + 1
    return counts


def build_outage_kepler_map(
    net,
    ami_csv,
    title="Network outages",
    deenergized_lines=(),
    deenergized_transformers=(),
):
    """Build a Kepler map with conductor, transformer, and AMI customer status.

    ``deenergized_lines`` accepts endpoint tokens or complete line names. Transformer
    entries may be transformer bus indices or load/meter names from the AMI file.
    """
    customer_counts = _infer_transformer_customer_counts(net, ami_csv)
    transformer_buses = set()
    load_table = _get_table(net, "asymmetric_load", "load")
    if load_table is not None and "name" in load_table.columns and "bus" in load_table.columns:
        requested = {str(value).strip().strip("'").upper() for value in deenergized_transformers}
        for _, row in load_table.iterrows():
            if str(row["name"]).strip().strip("'").upper() in requested:
                transformer_buses.add(_canonical_key(row["bus"]))
    transformer_buses.update(
        _canonical_key(value) for value in deenergized_transformers
        if isinstance(value, (int, float)) or str(value).strip().isdigit()
    )

    line_tokens = {str(value).strip().strip("'").upper() for value in deenergized_lines}
    line_names = net.line["name"].astype(str).str.upper() if "name" in net.line.columns else None
    deenergized_line_ids = set()
    if line_names is not None:
        for raw_index, line_name in line_names.items():
            if any(token in line_name for token in line_tokens):
                deenergized_line_ids.add(_canonical_key(_first_index_value(raw_index)))

    bus_elems = get_bus_element_multimap(net)
    bus_gdf = net.bus_geodata.copy()
    bus_gdf["customer_count"] = [customer_counts.get(_canonical_key(index[0]), 0) for index in bus_gdf.index]
    bus_gdf["transformer_status"] = [
        "de-energized" if _canonical_key(index[0]) in transformer_buses else "energized"
        for index in bus_gdf.index
    ]
    keep_cols = ["Longitude", "Latitude", "customer_count", "transformer_status"]
    if "name" in net.bus.columns:
        bus_gdf = bus_gdf.join(net.bus[["name"]].rename(columns={"name": "bus_name"}), how="left")
        keep_cols.append("bus_name")

    data_dict = {}
    layers = []
    fields_to_show = {}
    transformer_types = {"trafo", "trafo3w", "trafo1ph"}
    element_types = sorted({etype for values in bus_elems.values() for etype in values} | {"bus"})
    for etype in element_types:
        indices = [index for index in bus_gdf.index if (bus_elems.get(index[0], set()) or {"bus"}) and etype in (bus_elems.get(index[0], set()) or {"bus"})]
        subset = bus_gdf.loc[bus_gdf.index.isin(indices), keep_cols].copy()
        if subset.empty:
            continue
        stateful = etype in transformer_types
        subsets = [("de-energized", subset[subset["transformer_status"] == "de-energized"]), ("energized", subset[subset["transformer_status"] == "energized"])] if stateful else [("", subset)]
        for state, state_subset in subsets:
            if state_subset.empty:
                continue
            data_name = etype if not state else f"{etype}_{state.replace('-', '_')}"
            data_dict[data_name] = state_subset
            color = ([255, 0, 0] if state == "de-energized" else [0, 200, 0]) if stateful else ELEMENT_RGB.get(etype, [255, 0, 0])
            label_field = "customer_count" if stateful else data_name
            layers.append({"id": f"{data_name}_layer", "type": "point", "config": {"dataId": data_name, "label": label_field, "color": color, "columns": {"lat": "Latitude", "lng": "Longitude"}, "isVisible": True, "visConfig": {"radius": 5, "opacity": 0.9}}})
            fields_to_show[data_name] = [{"name": column, "format": None} for column in keep_cols if column not in ("Longitude", "Latitude")]

    line_gdf = net.line_geodata.copy()
    line_gdf.index.names = net.res_line.index.names
    line_gdf["Geometry"] = line_gdf["Geometry"].apply(lambda coords: "LINESTRING (" + ", ".join(f"{point[0]} {point[1]}" for point in coords) + ")" if len(coords) >= 2 else None)
    line_gdf = line_gdf.dropna(subset=["Geometry"])
    if "name" in net.line.columns:
        line_gdf = line_gdf.join(net.line[["name"]].rename(columns={"name": "line_name"}), how="left")
    line_gdf["conductor_status"] = [
        "de-energized" if _canonical_key(_first_index_value(index)) in deenergized_line_ids else "energized"
        for index in line_gdf.index
    ]
    line_fields = [{"name": column, "format": None} for column in ("line_name", "conductor_status") if column in line_gdf.columns]
    for state, subset in line_gdf.groupby("conductor_status"):
        data_name = f"conductors_{state.replace('-', '_')}"
        data_dict[data_name] = subset
        color = [255, 255, 0] if state == "de-energized" else [255, 255, 255]
        layers.append({"id": f"{data_name}_layer", "type": "geojson", "config": {"dataId": data_name, "label": data_name, "color": color, "columns": {"geojson": "Geometry"}, "isVisible": True, "visConfig": {"strokeColor": color, "thickness": 3, "opacity": 1}}})
        fields_to_show[data_name] = line_fields

    cfg = {"version": "v1", "config": {"visState": {"layers": layers}, "mapState": {"latitude": float(net.bus_geodata["Latitude"].mean()), "longitude": float(net.bus_geodata["Longitude"].mean()), "zoom": 13}, "interactionConfig": {"tooltip": {"enabled": True, "fieldsToShow": fields_to_show}}}}
    return KeplerGl(height=800, data=data_dict, config=cfg)
    
