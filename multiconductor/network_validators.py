from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd


@dataclass
class ValidationResult:
    issues: pd.DataFrame
    recommendations: pd.DataFrame
    summary: pd.DataFrame


def scan_notebooks_for_validation_snippets(
    notebook_roots: Iterable[str | Path],
    keywords: Iterable[str] | None = None,
    max_snippets_per_notebook: int = 25,
) -> pd.DataFrame:
    """
    Scan notebook code cells for validation-related snippets.

    This helps consolidate existing notebook experiments into one place when
    designing repeatable multiconductor validation checks.
    """
    terms = [
        "voltage",
        "radial",
        "phase",
        "tap",
        "vector",
        "transformer",
        "shunt",
        "connectivity",
        "loop",
        "isolated",
        "p_mw",
        "q_mvar",
    ]
    if keywords:
        terms = [str(k).lower() for k in keywords]

    rows: list[dict[str, Any]] = []
    for root in notebook_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for notebook in root_path.rglob("*.ipynb"):
            try:
                payload = json.loads(notebook.read_text(encoding="utf-8"))
            except Exception:
                continue

            cells = payload.get("cells", [])
            hits = 0
            for index, cell in enumerate(cells, start=1):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                source_l = source.lower()
                matched = [term for term in terms if term in source_l]
                if not matched:
                    continue
                rows.append(
                    {
                        "notebook": str(notebook),
                        "cell_number": index,
                        "matched_terms": ", ".join(matched),
                        "snippet": source[:4000],
                    }
                )
                hits += 1
                if hits >= max_snippets_per_notebook:
                    break

    if not rows:
        return pd.DataFrame(columns=["notebook", "cell_number", "matched_terms", "snippet"])
    return pd.DataFrame(rows)


def run_multiconductor_validations(
    net: Any,
    radial_expected: bool = True,
    voltage_limits: tuple[float, float] = (0.9, 1.1),
) -> ValidationResult:
    """Run all multiconductor quality checks and return a unified report."""
    issues: list[dict[str, Any]] = []

    issues.extend(_check_bus_and_element_voltage_consistency(net, voltage_limits))
    issues.extend(_check_load_and_generation_values(net))
    issues.extend(_check_transformers_and_shunts(net))
    issues.extend(_check_topology_and_radiality(net, radial_expected=radial_expected))
    issues.extend(_check_line_connectivity(net))
    issues.extend(_check_phase_connectivity(net))

    issues_df = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "check",
            "element_type",
            "element_index",
            "field",
            "message",
            "suggestion",
        ],
    )

    if issues_df.empty:
        rec_df = pd.DataFrame(columns=["priority", "issue_type", "recommendation"])
        summary_df = pd.DataFrame([{"severity": "info", "count": 1}])
        return ValidationResult(issues=issues_df, recommendations=rec_df, summary=summary_df)

    rec_df = recommend_corrective_actions(issues_df)
    summary_df = issues_df.groupby("severity", dropna=False).size().reset_index(name="count")
    return ValidationResult(issues=issues_df, recommendations=rec_df, summary=summary_df)


def recommend_corrective_actions(issues_df: pd.DataFrame) -> pd.DataFrame:
    """Build prioritized corrective actions from validation issues."""
    if issues_df.empty:
        return pd.DataFrame(columns=["priority", "issue_type", "recommendation"])

    action_map = {
        "voltage_consistency": "Align nominal voltage data (vn_kv) and verify bus voltage bases or source settings.",
        "power_value_sanity": "Correct missing/negative p_mw/q_mvar values and validate load/generation sign conventions.",
        "transformer_shunt_validity": "Fix transformer vector group/tap settings and shunt control thresholds to physically valid values.",
        "topology_radiality": "Resolve isolated islands/open points and remove unintended loops to satisfy expected radial topology.",
        "line_connectivity": "Reconnect line endpoints to valid buses and complete missing conductor/phase endpoint data.",
        "phase_connectivity": "Correct phase assignments and sequence consistency across buses, lines, loads, and transformers.",
    }
    priority_rank = {"critical": 1, "high": 2, "medium": 3, "low": 4, "info": 5}

    grouped = (
        issues_df.groupby(["check", "severity"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=["severity", "count"], key=lambda s: s.map(priority_rank).fillna(99) if s.name == "severity" else s, ascending=[True, False])
    )

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        issue_type = str(row["check"])
        severity = str(row["severity"])
        rows.append(
            {
                "priority": severity,
                "issue_type": issue_type,
                "recommendation": action_map.get(issue_type, "Investigate and resolve reported issues in this category."),
            }
        )
    return pd.DataFrame(rows)


def _check_bus_and_element_voltage_consistency(net: Any, voltage_limits: tuple[float, float]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    min_v, max_v = voltage_limits

    bus = _get_table(net, "bus")
    bus_ids = _unique_bus_ids(bus)
    res_bus = _get_table(net, "res_bus")

    # Check element endpoint voltage consistency (lines should have same vn_kv at both ends)
    for element_name in ("line", "trafo", "trafo3w", "trafo1ph"):
        table = _get_table(net, element_name)
        if table.empty:
            continue

        # For trafo1ph the bus id lives in MultiIndex level "bus", not in columns
        if element_name == "trafo1ph":
            _check_trafo1ph_vn_consistency(table, bus, bus_ids, issues)
            continue

        checked = set()
        for idx, row in table.iterrows():
            # With MultiIndex lines, deduplicate by element-level index
            elem_id = idx[0] if isinstance(idx, tuple) else idx
            if elem_id in checked:
                continue
            checked.add(elem_id)

            if not _is_in_service(row):
                continue
            buses = [b for b in _candidate_bus_values(row) if b in bus_ids]
            if len(buses) < 2:
                continue
            vn_values = [_get_bus_vn_kv(bus, b) for b in buses]
            vn_values = [v for v in vn_values if v and v > 0]
            if len(vn_values) < 2:
                continue
            max_vn = max(vn_values)
            min_vn = min(vn_values)
            if element_name == "line" and max_vn > 1.15 * min_vn:
                issues.append(
                    _issue(
                        "high",
                        "voltage_consistency",
                        "line",
                        elem_id,
                        "from_bus/to_bus",
                        f"Line endpoints have mismatched nominal voltages ({min_vn:.4f} vs {max_vn:.4f} kV).",
                        "Ensure both line endpoints belong to the same nominal voltage level or model as a transformer.",
                    )
                )

    return issues


def _check_trafo1ph_vn_consistency(trafo1ph: pd.DataFrame, bus: pd.DataFrame, bus_ids: set, issues: list) -> None:
    """Check trafo1ph vn_kv consistency — buses are in MultiIndex level, not columns."""
    if "vn_kv" not in trafo1ph.columns:
        return
    if not isinstance(trafo1ph.index, pd.MultiIndex) or "bus" not in trafo1ph.index.names:
        return

    trafo_idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
    bus_level = trafo1ph.index.names.index("bus")

    for trafo_id in trafo1ph.index.get_level_values(trafo_idx_level).unique():
        try:
            sub = trafo1ph.xs(trafo_id, level=trafo_idx_level)
        except (KeyError, TypeError):
            continue
        trafo_buses = sub.index.get_level_values(bus_level - 1 if trafo_idx_level == 0 else bus_level).unique().tolist()
        if len(trafo_buses) < 2:
            continue
        vn_vals = []
        for tb in trafo_buses:
            try:
                vn = sub.xs(tb, level=bus_level - 1 if trafo_idx_level == 0 else bus_level)["vn_kv"]
                v = _safe_float(vn.iloc[0] if hasattr(vn, "iloc") else vn)
                if v and v > 0:
                    vn_vals.append(v)
            except Exception:
                continue
        # Trafo sides are expected to have different voltages — nothing to flag here.


def _check_load_and_generation_values(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    checks = [
        ("load", "p_mw", "q_mvar", True),
        ("asymmetric_load", "p_mw", "q_mvar", True),
        ("gen", "p_mw", "q_mvar", False),
        ("sgen", "p_mw", "q_mvar", False),
        ("asymmetric_sgen", "p_mw", "q_mvar", False),
    ]

    for table_name, p_col, q_col, is_load in checks:
        table = _get_table(net, table_name)
        if table.empty:
            continue

        # Deduplicate by element-level index for MultiIndex tables
        checked = set()
        for idx, row in table.iterrows():
            elem_id = idx[0] if isinstance(idx, tuple) else idx
            if elem_id in checked:
                continue
            checked.add(elem_id)

            if not _is_in_service(row):
                continue
            p = _safe_float(row.get(p_col)) if p_col in table.columns else None
            q = _safe_float(row.get(q_col)) if q_col in table.columns else None

            if p is None:
                issues.append(_issue("medium", "power_value_sanity", table_name, elem_id, p_col, "Active power value is missing.", "Populate p_mw using source data or estimation."))
            elif is_load and p < 0:
                issues.append(_issue("high", "power_value_sanity", table_name, elem_id, p_col, f"Load p_mw is negative ({p}).", "Flip sign convention or reclassify as generation/storage."))
            elif not is_load and p < 0:
                issues.append(_issue("medium", "power_value_sanity", table_name, elem_id, p_col, f"Generation p_mw is negative ({p}).", "Confirm if this element is intended to absorb power."))

            if q is None:
                issues.append(_issue("low", "power_value_sanity", table_name, elem_id, q_col, "Reactive power value is missing.", "Populate q_mvar or define power factor to derive it."))

            # Check for zero power — skip for asymmetric tables pre-allocation (very common)
            if p is not None and q is not None and abs(p) < 1e-12 and abs(q) < 1e-12:
                # Only flag symmetric load/gen with zero power; asymmetric tables are
                # typically zero-initialized before load allocation — suppress the noise.
                if table_name not in ("asymmetric_load", "asymmetric_sgen"):
                    issues.append(_issue("info", "power_value_sanity", table_name, elem_id, "p_mw/q_mvar", "Both p_mw and q_mvar are zero.", "If not intentional, update to realistic loading/generation values."))

    return issues


def _check_transformers_and_shunts(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    valid_vector_groups = {
        "yy0", "ynyn0", "yd1", "yd5", "yd11", "dy1", "dy5", "dy11", "dd0", "dzn0", "ynzn0", "ynz1"
    }
    for trafo_name in ("trafo", "trafo3w", "trafo1ph"):
        table = _get_table(net, trafo_name)
        if table.empty:
            continue

        checked = set()
        for idx, row in table.iterrows():
            elem_id = idx[0] if isinstance(idx, tuple) else idx
            if elem_id in checked:
                continue
            checked.add(elem_id)

            if not _is_in_service(row):
                continue

            # vector_group check — trafo1ph typically has no vector_group column
            if "vector_group" in table.columns:
                vg = str(row.get("vector_group", "")).strip().lower()
                if vg and vg not in valid_vector_groups:
                    issues.append(
                        _issue(
                            "medium",
                            "transformer_shunt_validity",
                            trafo_name,
                            elem_id,
                            "vector_group",
                            f"Unrecognized transformer vector group '{row.get('vector_group')}'.",
                            "Use a valid IEC vector group (e.g., Dyn5, Yy0, Dd0) consistent with equipment design.",
                        )
                    )

            tap_pos = _safe_float(row.get("tap_pos"))
            tap_min = _safe_float(row.get("tap_min"))
            tap_max = _safe_float(row.get("tap_max"))
            if tap_pos is not None and tap_min is not None and tap_max is not None and not (tap_min <= tap_pos <= tap_max):
                issues.append(
                    _issue(
                        "high",
                        "transformer_shunt_validity",
                        trafo_name,
                        elem_id,
                        "tap_pos",
                        f"Tap position {tap_pos} is outside bounds [{tap_min}, {tap_max}].",
                        "Clamp tap_pos to allowed range or fix tap limits.",
                    )
                )

    for shunt_name in ("shunt", "asymmetric_shunt"):
        table = _get_table(net, shunt_name)
        if table.empty:
            continue
        checked_s = set()
        for idx, row in table.iterrows():
            elem_id = idx[0] if isinstance(idx, tuple) else idx
            if elem_id in checked_s:
                continue
            checked_s.add(elem_id)
            if not _is_in_service(row):
                continue

            v_on = _safe_float(row.get("v_threshold_on"))
            v_off = _safe_float(row.get("v_threshold_off"))
            if v_on is not None and v_off is not None and v_on >= v_off:
                issues.append(
                    _issue(
                        "medium",
                        "transformer_shunt_validity",
                        shunt_name,
                        elem_id,
                        "v_threshold_on/v_threshold_off",
                        f"Shunt threshold ordering is invalid (on={v_on}, off={v_off}).",
                        "Set v_threshold_on lower than v_threshold_off to avoid oscillatory control.",
                    )
                )

            q = _safe_float(row.get("q_mvar"))
            if q is None:
                continue
            if abs(q) < 1e-12:
                issues.append(
                    _issue(
                        "low",
                        "transformer_shunt_validity",
                        shunt_name,
                        elem_id,
                        "q_mvar",
                        "Shunt reactive power is zero.",
                        "Confirm the shunt is intended to be passive or set realistic q_mvar.",
                    )
                )

    return issues


def _check_topology_and_radiality(net: Any, radial_expected: bool = True) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = _get_table(net, "bus")
    if bus.empty:
        return issues

    bus_ids = _unique_bus_ids(bus)

    graph = nx.Graph()
    graph.add_nodes_from(bus_ids)

    def _add_edge_if_valid(u: Any, v: Any, name: str, idx: Any) -> None:
        if u not in bus_ids or v not in bus_ids:
            issues.append(_issue("high", "topology_radiality", name, idx, "bus", "Endpoint bus does not exist in net.bus.", "Fix connectivity node mapping before running topology checks."))
            return
        graph.add_edge(u, v, element=name, index=idx)

    # Lines — deduplicate across MultiIndex circuits
    checked_lines = set()
    for idx, row in _get_table(net, "line").iterrows():
        elem_id = idx[0] if isinstance(idx, tuple) else idx
        if elem_id in checked_lines:
            continue
        checked_lines.add(elem_id)
        if not _is_in_service(row):
            continue
        _add_edge_if_valid(row.get("from_bus"), row.get("to_bus"), "line", elem_id)

    # Pandapower-style trafos
    checked_trafos = set()
    for idx, row in _get_table(net, "trafo").iterrows():
        elem_id = idx[0] if isinstance(idx, tuple) else idx
        if elem_id in checked_trafos:
            continue
        checked_trafos.add(elem_id)
        if not _is_in_service(row):
            continue
        _add_edge_if_valid(row.get("hv_bus"), row.get("lv_bus"), "trafo", elem_id)

    # Multiconductor trafo1ph — bus is in MultiIndex level, not columns
    trafo1ph = _get_table(net, "trafo1ph")
    if not trafo1ph.empty and isinstance(trafo1ph.index, pd.MultiIndex):
        idx_level = trafo1ph.index.names.index("index") if "index" in trafo1ph.index.names else 0
        bus_level_name = "bus" if "bus" in trafo1ph.index.names else None
        if bus_level_name is not None:
            for tid in trafo1ph.index.get_level_values(idx_level).unique():
                try:
                    sub = trafo1ph.xs(tid, level=idx_level)
                    bus_level_pos = sub.index.names.index(bus_level_name) if bus_level_name in sub.index.names else 0
                    trafo_buses = list(dict.fromkeys(sub.index.get_level_values(bus_level_pos).tolist()))
                except Exception:
                    continue
                if len(trafo_buses) >= 2:
                    _add_edge_if_valid(trafo_buses[0], trafo_buses[1], "trafo1ph", tid)

    # Switches
    checked_sw = set()
    for idx, row in _get_table(net, "switch").iterrows():
        elem_id = idx[0] if isinstance(idx, tuple) else idx
        if elem_id in checked_sw:
            continue
        checked_sw.add(elem_id)
        if bool(row.get("closed", True)) and row.get("et") == "b":
            _add_edge_if_valid(row.get("bus"), row.get("element"), "switch", elem_id)
        if not bool(row.get("closed", True)):
            issues.append(_issue("info", "topology_radiality", "switch", elem_id, "closed", "Open point detected.", "Verify this open switch is intentional network configuration."))

    components = list(nx.connected_components(graph))
    if len(components) > 1:
        slack_buses = set(_get_table(net, "ext_grid").get("bus", pd.Series(dtype=object)).tolist())
        # Also check ext_grid_sequence for multiconductor slack buses
        egs = _get_table(net, "ext_grid_sequence")
        if not egs.empty and "bus" in egs.columns:
            slack_buses.update(egs["bus"].tolist())

        for comp in components:
            if slack_buses and comp.intersection(slack_buses):
                continue
            comp_list = list(comp)
            issues.append(
                _issue(
                    "high",
                    "topology_radiality",
                    "network",
                    ",".join(map(str, comp_list[:10])),
                    "connectivity",
                    f"Isolated component with {len(comp)} bus(es) detected.",
                    "Reconnect islanded section to source or mark buses out of service.",
                )
            )

    if radial_expected:
        cycles = nx.cycle_basis(graph)
        for cycle in cycles:
            issues.append(
                _issue(
                    "high",
                    "topology_radiality",
                    "network",
                    ",".join(map(str, cycle)),
                    "loop",
                    "Unintended loop detected in a radial-expected network.",
                    "Open one switching point or correct duplicate branch connectivity.",
                )
            )

    return issues


def _check_line_connectivity(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus = _get_table(net, "bus")
    bus_ids = _unique_bus_ids(bus)
    line = _get_table(net, "line")
    if line.empty:
        return issues

    checked = set()
    for idx, row in line.iterrows():
        elem_id = idx[0] if isinstance(idx, tuple) else idx
        if elem_id in checked:
            continue
        checked.add(elem_id)

        if not _is_in_service(row):
            continue
        from_bus = row.get("from_bus")
        to_bus = row.get("to_bus")

        if from_bus not in bus_ids:
            issues.append(_issue("high", "line_connectivity", "line", elem_id, "from_bus", f"from_bus {from_bus} is missing in net.bus.", "Map line from_bus to a valid bus index."))
        if to_bus not in bus_ids:
            issues.append(_issue("high", "line_connectivity", "line", elem_id, "to_bus", f"to_bus {to_bus} is missing in net.bus.", "Map line to_bus to a valid bus index."))

        if from_bus == to_bus and from_bus in bus_ids:
            issues.append(_issue("high", "line_connectivity", "line", elem_id, "from_bus/to_bus", "Line is connected to the same bus on both ends.", "Correct endpoint mapping or remove invalid self-loop line."))

        # Collect all circuit phases for this line element to check consistency
        line_circuits = _get_element_circuits(line, elem_id)
        if line_circuits is not None and len(line_circuits) > 0:
            from_phases_set = set()
            to_phases_set = set()
            has_from = "from_phase" in line_circuits.columns
            has_to = "to_phase" in line_circuits.columns
            if has_from:
                from_phases_set = set(line_circuits["from_phase"].dropna().tolist())
            if has_to:
                to_phases_set = set(line_circuits["to_phase"].dropna().tolist())
            if has_from and has_to and from_phases_set and to_phases_set and from_phases_set != to_phases_set:
                issues.append(
                    _issue(
                        "medium",
                        "line_connectivity",
                        "line",
                        elem_id,
                        "from_phase/to_phase",
                        f"Phase sets differ between ends (from={sorted(from_phases_set)}, to={sorted(to_phases_set)}).",
                        "Align phase conductors at both line terminals or split line by phase configuration.",
                    )
                )
            if not has_from and not has_to:
                issues.append(_issue("medium", "line_connectivity", "line", elem_id, "phase", "Missing phase data may indicate orphaned conductors.", "Populate from_phase/to_phase fields for each line segment."))

    return issues


def _check_phase_connectivity(net: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    bus = _get_table(net, "bus")
    bus_phase_map: dict[Any, set[int]] = {}

    # Build bus phase map — for MultiIndex (index, phase), extract phases from level 1
    if not bus.empty:
        if isinstance(bus.index, pd.MultiIndex) and len(bus.index.names) >= 2:
            phase_level = 1  # (index, phase)
            idx_level = 0
            for bus_id in bus.index.get_level_values(idx_level).unique():
                phases = set(bus.index.get_level_values(phase_level)[bus.index.get_level_values(idx_level) == bus_id].tolist())
                if phases:
                    bus_phase_map[bus_id] = phases
        else:
            # Flat index — try column-based phase info
            for idx, row in bus.iterrows():
                bus_phases = _extract_phase_set(row, ("phases", "phase", "num_phases"))
                if bus_phases:
                    bus_phase_map[idx] = {_phase_str_to_int(p) for p in bus_phases}

    for table_name, bus_col, phase_col in (
        ("line", "from_bus", "from_phase"),
        ("asymmetric_load", "bus", "from_phase"),
        ("asymmetric_sgen", "bus", "from_phase"),
    ):
        table = _get_table(net, table_name)
        if table.empty:
            continue
        if phase_col not in table.columns:
            continue

        checked = set()
        for idx, row in table.iterrows():
            elem_id = idx[0] if isinstance(idx, tuple) else idx
            if elem_id in checked:
                continue
            checked.add(elem_id)

            if not _is_in_service(row):
                continue

            # Collect all phases for this element across circuits
            element_phases = _get_element_phase_set(table, elem_id, phase_col)
            if not element_phases:
                issues.append(_issue("medium", "phase_connectivity", table_name, elem_id, "phase", "Missing phase assignment.", "Assign explicit phases based on source model and connectivity."))
                continue

            bus_idx = row.get(bus_col)
            bus_phases = bus_phase_map.get(bus_idx)
            if bus_phases and not element_phases.issubset(bus_phases):
                issues.append(
                    _issue(
                        "high",
                        "phase_connectivity",
                        table_name,
                        elem_id,
                        "phase",
                        f"Element phases {sorted(element_phases)} are not compatible with bus phases {sorted(bus_phases)}.",
                        "Correct element phase modeling or update bus phase metadata.",
                    )
                )

    return issues


def _issue(
    severity: str,
    check: str,
    element_type: str,
    element_index: Any,
    field: str,
    message: str,
    suggestion: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "element_type": element_type,
        "element_index": element_index,
        "field": field,
        "message": message,
        "suggestion": suggestion,
    }


def _get_table(net: Any, name: str) -> pd.DataFrame:
    table = getattr(net, name, None)
    if isinstance(table, pd.DataFrame):
        return table
    return pd.DataFrame()


def _unique_bus_ids(bus: pd.DataFrame) -> set:
    """Extract unique scalar bus identifiers from a bus DataFrame.

    Handles both flat index (pandapower) and MultiIndex (index, phase) schemas.
    """
    if bus.empty:
        return set()
    if isinstance(bus.index, pd.MultiIndex):
        return set(bus.index.get_level_values(0).unique().tolist())
    return set(bus.index.tolist())


def _aggregate_bus_metric(res_bus: pd.DataFrame, col: str) -> dict:
    """Return {bus_id: mean_value} aggregating across phases for MultiIndex res_bus."""
    if res_bus.empty or col not in res_bus.columns:
        return {}
    series = pd.to_numeric(res_bus[col], errors="coerce")
    if isinstance(res_bus.index, pd.MultiIndex):
        grouped = series.groupby(level=0).mean()
        return {bid: (_safe_float(v) if pd.notna(v) else None) for bid, v in grouped.items()}
    return {idx: _safe_float(v) for idx, v in series.items()}


def _get_element_circuits(table: pd.DataFrame, elem_id: Any) -> pd.DataFrame | None:
    """Get all circuit rows for a given element from a MultiIndex table."""
    if not isinstance(table.index, pd.MultiIndex):
        try:
            return table.loc[[elem_id]]
        except KeyError:
            return None
    try:
        return table.xs(elem_id, level=0)
    except (KeyError, TypeError):
        return None


def _get_element_phase_set(table: pd.DataFrame, elem_id: Any, phase_col: str) -> set[int]:
    """Collect all phase values for an element across its circuit rows."""
    circuits = _get_element_circuits(table, elem_id)
    if circuits is None or circuits.empty or phase_col not in circuits.columns:
        return set()
    values = circuits[phase_col].dropna().tolist()
    result = set()
    for v in values:
        try:
            result.add(int(v))
        except (ValueError, TypeError):
            continue
    return result


def _phase_str_to_int(phase_str: str) -> int:
    """Convert phase letter to integer: A→1, B→2, C→3, N→0."""
    mapping = {"A": 1, "B": 2, "C": 3, "N": 0}
    return mapping.get(str(phase_str).upper(), -1)


def _bus_exists(bus: pd.DataFrame, bus_id: Any) -> bool:
    return bus_id in _unique_bus_ids(bus)


def _get_bus_vn_kv(bus: pd.DataFrame, bus_id: Any) -> float | None:
    if bus.empty or "vn_kv" not in bus.columns:
        return None

    if isinstance(bus.index, pd.MultiIndex):
        try:
            sub = bus.xs(bus_id, level=0)
            vn = sub["vn_kv"]
            return _safe_float(vn.iloc[0] if hasattr(vn, "iloc") else vn)
        except (KeyError, TypeError, IndexError):
            return None

    try:
        return _safe_float(bus.at[bus_id, "vn_kv"])
    except (KeyError, TypeError):
        return None


def _is_in_service(row: pd.Series) -> bool:
    if "in_service" not in row.index:
        return True
    return bool(row.get("in_service", True))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_bus_values(row: pd.Series) -> list[Any]:
    keys = (
        "from_bus",
        "to_bus",
        "hv_bus",
        "lv_bus",
        "mv_bus",
        "bus",
        "element",
    )
    out: list[Any] = []
    for key in keys:
        if key in row.index:
            out.append(row.get(key))
    return out


def _extract_phase_set(row: pd.Series, columns: Iterable[str]) -> set[str]:
    phases: set[str] = set()
    for col in columns:
        if col not in row.index:
            continue
        raw = row.get(col)
        phases.update(_parse_phases(raw))
    return phases


def _parse_phases(raw: Any) -> set[str]:
    if raw is None:
        return set()

    if isinstance(raw, (int, float)) and not pd.isna(raw):
        raw = str(int(raw))

    if isinstance(raw, (list, tuple, set)):
        out: set[str] = set()
        for item in raw:
            out.update(_parse_phases(item))
        return out

    text = str(raw).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return set()

    # Normalize common representations: 123, A.B.C, [1,2,3], etc.
    for token in ["[", "]", "(", ")", "{", "}", "'", '"', ",", ".", "/", "-"]:
        text = text.replace(token, " ")
    parts = [p for p in text.split() if p]

    mapping = {
        "1": "A",
        "2": "B",
        "3": "C",
        "0": "N",
        "A": "A",
        "B": "B",
        "C": "C",
        "N": "N",
    }
    phases: set[str] = set()
    for part in parts:
        if part in mapping:
            phases.add(mapping[part])
            continue
        # Handle condensed forms like 1230 or ABCN.
        for char in part:
            if char in mapping:
                phases.add(mapping[char])
    return phases


def _has_non_standard_phase_order(row: pd.Series, phase_cols: Iterable[str]) -> bool:
    seq_values: list[str] = []
    for col in phase_cols:
        if col not in row.index:
            continue
        raw = row.get(col)
        if raw is None:
            continue
        txt = str(raw).upper().replace(" ", "")
        if txt and txt not in {"NONE", "NAN"}:
            seq_values.append(txt)
    if not seq_values:
        return False

    canonical = {"123", "12", "13", "23", "ABC", "AB", "AC", "BC", "A", "B", "C"}
    for value in seq_values:
        normalized = value.replace(".", "").replace(",", "").replace("/", "")
        if normalized in canonical:
            continue
        # Ignore neutral markers in sequence checks.
        normalized = normalized.replace("0", "").replace("N", "")
        if normalized and normalized not in canonical:
            return True
    return False

def _check_slack(net):
    import networkx as nx
    import pandas as pd

    def _bus_id(bus):
        if isinstance(bus, tuple):
            return int(bus[0])
        return int(bus)

    def _build_bus_graph(net):
        G = nx.Graph()
        if hasattr(net, "bus") and net.bus is not None and len(net.bus):
            if isinstance(net.bus.index, pd.MultiIndex):
                buses = sorted(set(net.bus.index.get_level_values(0).tolist()))
            else:
                buses = net.bus.index.tolist()
            for b in buses:
                G.add_node(_bus_id(b))
        if hasattr(net, "line") and net.line is not None and len(net.line):
            for _, row in net.line.iterrows():
                if "in_service" in row and not bool(row["in_service"]):
                    continue
                fb = _bus_id(row.get("from_bus"))
                tb = _bus_id(row.get("to_bus"))
                if fb != tb:
                    G.add_edge(fb, tb)
        if hasattr(net, "trafo1ph") and net.trafo1ph is not None and len(net.trafo1ph):
            cols = set(net.trafo1ph.columns)
            if {"hv_bus", "lv_bus"}.issubset(cols):
                for _, row in net.trafo1ph.iterrows():
                    if "in_service" in row and not bool(row["in_service"]):
                        continue
                    hb = _bus_id(row.get("hv_bus"))
                    lb = _bus_id(row.get("lv_bus"))
                    if hb != lb:
                        G.add_edge(hb, lb)
            elif "bus" in cols:
                for tid in net.trafo1ph.index.get_level_values(0).unique():
                    view = net.trafo1ph.loc[tid]
                    bus_levels = view.index.get_level_values("bus") if "bus" in view.index.names else view.index.get_level_values(0)
                    uniq = list(dict.fromkeys([_bus_id(b) for b in bus_levels]))
                    if len(uniq) >= 2 and uniq[0] != uniq[1]:
                        G.add_edge(uniq[0], uniq[1])
        return G

    G = _build_bus_graph(net)
    if len(G.nodes) == 0:
        print("No buses found to build connectivity graph")
    else:
        comps = list(nx.connected_components(G))
        slack_component = None
        for comp in comps:
            if slack_bus in comp:
                slack_component = comp
                break
        print("Connected components:", len(comps))
        if slack_component is None:
            print("Slack bus not found in any component; leaving net unchanged")
        else:
            print("Slack component size:", len(slack_component))
            disabled_bus = 0
            disabled_line = 0
            disabled_trafo = 0
            if isinstance(net.bus.index, pd.MultiIndex):
                bus_mask = ~net.bus.index.get_level_values(0).isin(slack_component)
                disabled_bus = int(bus_mask.sum())
                net.bus.loc[~bus_mask, "in_service"] = True
                net.bus.loc[bus_mask, "in_service"] = False
            else:
                bus_mask = ~net.bus.index.isin(slack_component)
                disabled_bus = int(bus_mask.sum())
                net.bus.loc[~bus_mask, "in_service"] = True
                net.bus.loc[bus_mask, "in_service"] = False
            if hasattr(net, "line") and net.line is not None and len(net.line):
                line_mask = net.line["from_bus"].isin(slack_component) & net.line["to_bus"].isin(slack_component)
                disabled_line = int((~line_mask).sum())
                net.line.loc[~line_mask, "in_service"] = False
            if hasattr(net, "trafo1ph") and net.trafo1ph is not None and len(net.trafo1ph):
                cols = set(net.trafo1ph.columns)
                if {"hv_bus", "lv_bus"}.issubset(cols):
                    trafo_mask = net.trafo1ph["hv_bus"].isin(slack_component) & net.trafo1ph["lv_bus"].isin(slack_component)
                    disabled_trafo = int((~trafo_mask).sum())
                    net.trafo1ph.loc[~trafo_mask, "in_service"] = False
            print("Disabled elements outside slack component:")
            print(f"  buses: {disabled_bus}")
            print(f"  lines: {disabled_line}")
            print(f"  trafos: {disabled_trafo}")

_nan_counts = []
def check_NaNs(net):
    def _bus_id(bus):
        if isinstance(bus, tuple):
            return int(bus[0])
        return int(bus)

    def _build_bus_graph(net):
        G = nx.Graph()
        if hasattr(net, "bus") and net.bus is not None and len(net.bus):
            if isinstance(net.bus.index, pd.MultiIndex):
                buses = sorted(set(net.bus.index.get_level_values(0).tolist()))
            else:
                buses = net.bus.index.tolist()
            for b in buses:
                G.add_node(_bus_id(b))
        if hasattr(net, "line") and net.line is not None and len(net.line):
            for _, row in net.line.iterrows():
                if "in_service" in row and not bool(row.get("in_service", True)):
                    continue
                fb = _bus_id(row.get("from_bus"))
                tb = _bus_id(row.get("to_bus"))
                if fb != tb:
                    G.add_edge(fb, tb)
        if hasattr(net, "trafo1ph") and net.trafo1ph is not None and len(net.trafo1ph):
            cols = set(net.trafo1ph.columns)
            if {"hv_bus", "lv_bus"}.issubset(cols):
                for _, row in net.trafo1ph.iterrows():
                    if "in_service" in row and not bool(row.get("in_service", True)):
                        continue
                    hb = _bus_id(row.get("hv_bus"))
                    lb = _bus_id(row.get("lv_bus"))
                    if hb != lb:
                        G.add_edge(hb, lb)
            elif "bus" in cols:
                for tid in net.trafo1ph.index.get_level_values(0).unique():
                    view = net.trafo1ph.loc[tid]
                    bus_levels = view.index.get_level_values("bus") if "bus" in view.index.names else view.index.get_level_values(0)
                    uniq = list(dict.fromkeys([_bus_id(b) for b in bus_levels]))
                    if len(uniq) >= 2 and uniq[0] != uniq[1]:
                        G.add_edge(uniq[0], uniq[1])
        return G

    def _count_in_service(net, key):
        if not hasattr(net, key):
            return (0, 0)
        tbl = getattr(net, key)
        if tbl is None or len(tbl) == 0:
            return (0, 0)
        if "in_service" in tbl.columns:
            return (int(tbl["in_service"].sum()), int(len(tbl)))
        return (int(len(tbl)), int(len(tbl)))

    print("=== Diagnostics ===")
    print("ext_grid rows:", len(net.ext_grid) if hasattr(net, "ext_grid") else "N/A")
    print("ext_grid_sequence rows:", len(net.ext_grid_sequence) if hasattr(net, "ext_grid_sequence") else "N/A")
    print("Slack bus:", slack_bus)

    for key in ["bus", "line", "trafo1ph", "switch", "load", "sgen", "shunt", "asymmetric_load", "asymmetric_sgen", "asymmetric_shunt"]:
        on, total = _count_in_service(net, key)
        if total:
            print(f"{key}: in_service {on}/{total}")

    if hasattr(net, "bus") and len(net.bus):
        vn = net.bus["vn_kv"] if "vn_kv" in net.bus.columns else None
        if vn is not None:
            print(f"bus vn_kv min/max: {float(vn.min()):.6g} / {float(vn.max()):.6g}")

    if hasattr(net, "asymmetric_load") and len(net.asymmetric_load):
        if "p_mw" in net.asymmetric_load.columns:
            print("asymmetric_load p_mw sum:", float(net.asymmetric_load["p_mw"].sum()))
        if "q_mvar" in net.asymmetric_load.columns:
            print("asymmetric_load q_mvar sum:", float(net.asymmetric_load["q_mvar"].sum()))

    if hasattr(net, "asymmetric_sgen") and len(net.asymmetric_sgen):
        if "p_mw" in net.asymmetric_sgen.columns:
            print("asymmetric_sgen p_mw sum:", float(net.asymmetric_sgen["p_mw"].sum()))
        if "q_mvar" in net.asymmetric_sgen.columns:
            print("asymmetric_sgen q_mvar sum:", float(net.asymmetric_sgen["q_mvar"].sum()))

    G = _build_bus_graph(net)
    if len(G.nodes):
        comps = list(nx.connected_components(G))
        sizes = sorted([len(c) for c in comps], reverse=True)
        print("Connected components:", len(comps), "largest sizes:", sizes[:5])
        if slack_bus is not None:
            in_slack = any(slack_bus in c for c in comps)
            print("Slack bus in component:", in_slack)

    def _nan_counts(net, key, cols):
        if not hasattr(net, key):
            return
        tbl = getattr(net, key)
        if tbl is None or len(tbl) == 0:
            return
        for c in cols:
            if c in tbl.columns:
                nans = int(pd.isna(tbl[c]).sum())
                if nans:
                    print(f"{key}.{c} NaNs: {nans}")

    _nan_counts(net, "line", ["length_km", "from_bus", "to_bus"])
    _nan_counts(net, "trafo1ph", ["hv_bus", "lv_bus", "sn_mva"])
    _nan_counts(net, "bus", ["vn_kv"])
    _nan_counts(net, "ext_grid", ["bus", "from_phase", "to_phase", "vm_pu"])
    _nan_counts(net, "ext_grid_sequence", ["bus", "from_phase", "to_phase", "vm_pu"])


def _find_island(net):
    import numpy as np
    import pandas as pd
    from multiconductor.pycci.model import _initialize_model, get_bus_terminal

    # Build model and use find_islands output stored on the model
    _initialize_model(net, debug_level=0)
    t_lookup = net.model.terminal_to_y_lookup
    valid = t_lookup[t_lookup >= 0]
    unique, counts = np.unique(valid, return_counts=True)
    order = np.argsort(counts)[::-1]
    print("Y-islands:", len(unique))
    print("Largest island sizes (terminals):", counts[order][:10].tolist())

    slack_islands = set()
    if hasattr(net.model, "terminal_is_slack"):
        slack_terms = np.where(net.model.terminal_is_slack)[0]
        slack_terms = slack_terms[slack_terms < len(t_lookup)]
        slack_islands = set(int(t_lookup[t]) for t in slack_terms if t_lookup[t] >= 0)
    print("Slack islands:", sorted(slack_islands))

    bus_y = []
    for b, p in net.bus.index:
        term = get_bus_terminal(net.model, b, p)
        y_idx = int(t_lookup[term]) if term < len(t_lookup) else -1
        bus_y.append(y_idx)
    bus_y = np.array(bus_y)
    bus_unique, bus_counts = np.unique(bus_y[bus_y >= 0], return_counts=True)
    bus_order = np.argsort(bus_counts)[::-1]
    print("Largest island sizes (buses):", bus_counts[bus_order][:10].tolist())
    if slack_islands:
        slack_bus_count = int(np.isin(bus_y, list(slack_islands)).sum())
        print("Buses in slack island(s):", slack_bus_count, "of", len(bus_y))

def _find_breakpoints(net):
    import numpy as np
    import pandas as pd
    import networkx as nx
    from multiconductor.pycci.model import _initialize_model, get_bus_terminal

    # Trace connectivity at bus/phase level and flag likely breakpoints
    _initialize_model(net, debug_level=0)
    t_lookup = net.model.terminal_to_y_lookup

    if isinstance(net.bus.index, pd.MultiIndex):
        bus_phase_iter = [(int(b), int(p)) for b, p in net.bus.index]
        bus_ids = sorted(set(b for b, _ in bus_phase_iter))
    else:
        bus_phase_iter = [(int(b), 1) for b in net.bus.index]
        bus_ids = list(net.bus.index.astype(int))

    bus_phase_island = {}
    for b, p in bus_phase_iter:
        term = get_bus_terminal(net.model, b, p)
        if term < len(t_lookup):
            y_idx = int(t_lookup[term])
        else:
            y_idx = -1
        bus_phase_island[(b, p)] = y_idx

    # Per-bus island and orphan phase summary
    bus_island_sets = {}
    bus_orphan_counts = {}
    for b in bus_ids:
        ys = [bus_phase_island[(b, p)] for bb, p in bus_phase_iter if bb == b]
        bus_island_sets[b] = set(y for y in ys if y >= 0)
        bus_orphan_counts[b] = sum(1 for y in ys if y < 0)

    multi_island_buses = [b for b, s in bus_island_sets.items() if len(s) > 1]
    orphan_phase_buses = [b for b, c in bus_orphan_counts.items() if c > 0]

    print("=== Connectivity Trace ===")
    print("Buses:", len(bus_ids))
    print("Buses with phases in multiple Y-islands:", len(multi_island_buses))
    print("Buses with orphan phases (no Y-island):", len(orphan_phase_buses))

    if multi_island_buses:
        sample = multi_island_buses[:20]
        print("Sample multi-island buses:", sample)
    if orphan_phase_buses:
        sample = orphan_phase_buses[:20]
        print("Sample orphan-phase buses:", sample)

    # Bus-level graph to detect isolated components
    G = nx.Graph()
    G.add_nodes_from(bus_ids)
    if hasattr(net, "line") and net.line is not None and len(net.line):
        for _, row in net.line.iterrows():
            if "in_service" in row and not bool(row.get("in_service", True)):
                continue
            fb = int(row.get("from_bus"))
            tb = int(row.get("to_bus"))
            if fb != tb:
                G.add_edge(fb, tb)
    if hasattr(net, "trafo1ph") and net.trafo1ph is not None and len(net.trafo1ph):
        cols = set(net.trafo1ph.columns)
        if {"hv_bus", "lv_bus"}.issubset(cols):
            for _, row in net.trafo1ph.iterrows():
                if "in_service" in row and not bool(row.get("in_service", True)):
                    continue
                hb = int(row.get("hv_bus"))
                lb = int(row.get("lv_bus"))
                if hb != lb:
                    G.add_edge(hb, lb)

    deg0 = [n for n, d in G.degree() if d == 0]
    print("Isolated buses in bus-graph (degree 0):", len(deg0))
    if deg0:
        print("Sample isolated buses:", deg0[:20])

    comps = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in comps], reverse=True)
    print("Bus-graph components:", len(comps), "largest sizes:", sizes[:5])
    if "slack_bus" in globals():
        in_slack = any(slack_bus in c for c in comps)
        slack_size = next((len(c) for c in comps if slack_bus in c), 0)
        print("Slack bus in bus-graph component:", in_slack, "size:", slack_size)

    # Missing bus references in elements
    bus_set = set(bus_ids)
    missing = {"line": 0, "trafo1ph": 0, "switch": 0}
    if hasattr(net, "line") and net.line is not None and len(net.line):
        for _, row in net.line.iterrows():
            fb = int(row.get("from_bus"))
            tb = int(row.get("to_bus"))
            if fb not in bus_set or tb not in bus_set:
                missing["line"] += 1
    if hasattr(net, "trafo1ph") and net.trafo1ph is not None and len(net.trafo1ph):
        cols = set(net.trafo1ph.columns)
        if {"hv_bus", "lv_bus"}.issubset(cols):
            for _, row in net.trafo1ph.iterrows():
                hb = int(row.get("hv_bus"))
                lb = int(row.get("lv_bus"))
                if hb not in bus_set or lb not in bus_set:
                    missing["trafo1ph"] += 1
    if hasattr(net, "switch") and net.switch is not None and len(net.switch):
        if "bus" in net.switch.columns:
            for _, row in net.switch.iterrows():
                b = int(row.get("bus"))
                if b not in bus_set:
                    missing["switch"] += 1
    print("Missing bus references:", missing)    

