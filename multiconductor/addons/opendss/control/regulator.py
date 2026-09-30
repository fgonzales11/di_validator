"""
OpenDSS RegControl generation from multiconductor tap-changer controllers.

Translates ``LoadTapChangerControl`` and ``LineDropControlExtended``
(stored in ``net.controller``) to OpenDSS ``New RegControl`` DSS commands.

Usage::

    from opendss.control.regulator import write_regcontrols
    write_regcontrols(net, dss_lines)
"""
import math
from typing import List


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _bus_name(net, bus_idx: int) -> str:
    try:
        row = net.bus.xs(bus_idx, level=0).iloc[0]
        raw = row.get("name") if hasattr(row, "get") else getattr(row, "name", None)
        if raw and isinstance(raw, str):
            return raw.replace(" ", "_").replace(".", "_")
    except Exception:
        pass
    return f"bus_{bus_idx}"


def _bus_vn_kv(net, bus_idx: int) -> float:
    try:
        row = net.bus.xs(bus_idx, level=0).iloc[0]
        return float(row["vn_kv"])
    except Exception:
        return 0.4


def _get_trafo_name(net, trafo_top_level_index, circ_idx=0):
    """Return the OpenDSS transformer element name matching ``_write_transformers`` output."""
    if not hasattr(net, "trafo1ph") or net.trafo1ph.empty:
        return None
    try:
        trafo_grp = net.trafo1ph.loc[trafo_top_level_index]
    except KeyError:
        return None

    bus_levels = trafo_grp.index.get_level_values("bus")
    unique_buses = list(dict.fromkeys(int(b) for b in bus_levels))
    if len(unique_buses) < 2:
        return None

    bus_vn = {b: _bus_vn_kv(net, b) for b in unique_buses}
    buses_sorted = sorted(unique_buses, key=lambda b: bus_vn[b], reverse=True)
    hv_bus = buses_sorted[0]
    hv_rows = trafo_grp.xs(hv_bus, level="bus")

    try:
        hv_row = hv_rows.iloc[circ_idx]
    except IndexError:
        hv_row = hv_rows.iloc[0]

    raw_name = None
    try:
        raw_name = hv_row.get("name") if hasattr(hv_row, "get") else getattr(hv_row, "name", None)
        raw_name = None if str(raw_name) == "None" else str(raw_name)
    except Exception:
        pass
    if raw_name:
        base = f"{raw_name}_{trafo_top_level_index}_{circ_idx}"
    else:
        base = f"T_{trafo_top_level_index}_{circ_idx}"
    return base.replace(" ", "_").replace(".", "_")


_LTC_TYPES = ("LoadTapChangerControl", "LineDropControl",
              "LineDropControlExtended")


def write_regcontrols(net, lines: List[str], type_filter=None,
                      section_header: str = "! Regulator Controls") -> None:
    """Append OpenDSS ``New RegControl`` commands to *lines*.

    Iterates over ``net.controller`` looking for multiconductor
    ``LoadTapChangerControl``, ``LineDropControl``, and
    ``LineDropControlExtended`` objects and emits the corresponding
    OpenDSS ``RegControl`` elements.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network with a ``controller`` DataFrame.
    lines : list[str]
        Accumulator list of DSS script lines.
    type_filter : tuple[str], optional
        Restrict emission to controllers whose class name is in this
        tuple. Defaults to all LTC/LDC types.
    section_header : str, optional
        Comment line to emit before the first RegControl entry.
    """
    if not hasattr(net, "controller"):
        return
    try:
        if net.controller is None or net.controller.empty:
            return
    except Exception:
        return

    accepted = tuple(type_filter) if type_filter else _LTC_TYPES
    has_content = False

    for _ctrl_idx, ctrl_row in net.controller.iterrows():
        ctrl = ctrl_row["object"]
        ctrl_type = type(ctrl).__name__

        if ctrl_type not in accepted:
            continue
        if not ctrl_row.get("in_service", True):
            continue

        tidx = ctrl.trafo_top_level_index
        try:
            trafo_grp = net.trafo1ph.loc[tidx]
        except (KeyError, AttributeError):
            continue

        bus_levels = trafo_grp.index.get_level_values("bus")
        unique_buses = list(dict.fromkeys(int(b) for b in bus_levels))
        if len(unique_buses) < 2:
            continue

        bus_vn = {b: _bus_vn_kv(net, b) for b in unique_buses}
        buses_sorted = sorted(unique_buses, key=lambda b: bus_vn[b], reverse=True)
        hv_bus = buses_sorted[0]
        lv_bus = buses_sorted[1]

        tap_side = getattr(ctrl, "side", "lv")
        controlled_bus = lv_bus if tap_side == "lv" else hv_bus
        vn_kv = bus_vn[controlled_bus]

        hv_rows = trafo_grp.xs(hv_bus, level="bus")
        lv_rows = trafo_grp.xs(lv_bus, level="bus")
        ncircuits = len(hv_rows)

        for circ_idx in range(ncircuits):
            trafo_name = _get_trafo_name(net, tidx, circ_idx)
            if trafo_name is None:
                continue

            reg_name = f"Reg_{trafo_name}"

            # Voltage setpoints ------------------------------------------------
            vm_lower = getattr(ctrl, "vm_lower_pu", 0.975)
            vm_upper = getattr(ctrl, "vm_upper_pu", 1.025)
            if hasattr(vm_lower, "__iter__"):
                vm_lower = float(min(vm_lower))
                vm_upper = float(max(vm_upper))
            else:
                vm_lower = float(vm_lower)
                vm_upper = float(vm_upper)

            # OpenDSS vreg / band are on a 120 V secondary base
            vreg_120 = (vm_lower + vm_upper) / 2.0 * 120.0
            band_120 = (vm_upper - vm_lower) * 120.0

            # PT ratio: Vln (volts) / 120
            vln_v = vn_kv / math.sqrt(3) * 1000.0
            ptratio = vln_v / 120.0

            # CT primary: kVA / Vln(kV) as a reasonable default
            try:
                side_rows = lv_rows if tap_side == "lv" else hv_rows
                sn_kva = _safe_float(side_rows.iloc[circ_idx]["sn_mva"], 0.1) * 1000.0
                ct_kv = vn_kv / math.sqrt(3)
                ctprim = sn_kva / ct_kv if ct_kv > 0 else 700.0
            except Exception:
                ctprim = 700.0

            if not has_content:
                lines.append(section_header)
                has_content = True

            cmd = (
                f"New RegControl.{reg_name} transformer={trafo_name} winding=2 "
                f"vreg={vreg_120:.4g} band={band_120:.4g} "
                f"ptratio={ptratio:.6g} ctprim={ctprim:.6g} eventlog=Yes"
            )

            # Line-drop compensation R and X
            if ctrl_type in ("LineDropControl", "LineDropControlExtended"):
                r_ldc = _safe_float(getattr(ctrl, "r_ldc_v", getattr(ctrl, "R_comp", 0.0)))
                x_ldc = _safe_float(getattr(ctrl, "x_ldc_v", getattr(ctrl, "X_comp", 0.0)))
                cmd += f" R={r_ldc:.6g} X={x_ldc:.6g}"

            lines.append(cmd)

    if has_content:
        lines.append("")


# ---------------------------------------------------------------------------
# RegControl for %REGULATOR%-named transformers
# ---------------------------------------------------------------------------

_REGULATOR_NAME_MARKER = "%REGULATOR%"


def _find_ltc_controller_for_trafo(net, tidx):
    """Return the first in-service LTC/LDC controller bound to *tidx*, or ``None``."""
    if not hasattr(net, "controller"):
        return None
    try:
        if net.controller is None or net.controller.empty:
            return None
    except Exception:
        return None
    for _idx, row in net.controller.iterrows():
        if not row.get("in_service", True):
            continue
        ctrl = row["object"]
        if type(ctrl).__name__ not in _LTC_TYPES:
            continue
        if getattr(ctrl, "trafo_top_level_index", None) == tidx:
            return ctrl
    return None


def write_regulator_regcontrols(net, lines: List[str]) -> None:
    """Append ``New RegControl`` commands for every ``trafo1ph`` marked as a
    voltage regulator (``name == '%REGULATOR%'``).

    Reads LTC tap parameters directly from the ``trafo1ph`` table
    (``tap_pos``, ``tap_min``, ``tap_max``, ``tap_neutral``,
    ``tap_step_percent``) and, if a matching ``LoadTapChangerControl``
    or ``LineDropControl[Extended]`` controller is found in
    ``net.controller``, uses its voltage setpoints (``vm_lower_pu``,
    ``vm_upper_pu``) and optional line-drop compensation (``r_ldc_v``,
    ``x_ldc_v``).

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network.
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    if not hasattr(net, "trafo1ph") or net.trafo1ph.empty:
        return

    trafo_indices = net.trafo1ph.index.get_level_values(0).unique()
    has_content = False

    for tidx in trafo_indices:
        trafo_grp = net.trafo1ph.loc[tidx]

        # Determine whether any winding row carries the %REGULATOR% marker
        is_regulator = False
        flat = trafo_grp.reset_index() if hasattr(trafo_grp.index, "names") else trafo_grp
        for _, r in flat.iterrows():
            raw = r.get("name") if hasattr(r, "get") else getattr(r, "name", None)
            if raw is not None and str(raw).strip() == _REGULATOR_NAME_MARKER:
                is_regulator = True
                break
        if not is_regulator:
            continue

        bus_levels = trafo_grp.index.get_level_values("bus")
        unique_buses = list(dict.fromkeys(int(b) for b in bus_levels))
        if len(unique_buses) < 2:
            continue

        bus_vn = {b: _bus_vn_kv(net, b) for b in unique_buses}
        buses_sorted = sorted(unique_buses, key=lambda b: bus_vn[b], reverse=True)
        hv_bus = buses_sorted[0]
        lv_bus = buses_sorted[1]

        hv_rows = trafo_grp.xs(hv_bus, level="bus")
        lv_rows = trafo_grp.xs(lv_bus, level="bus")
        ncircuits = len(hv_rows)

        # Find a matching LTC/LDC controller (may be None)
        ctrl = _find_ltc_controller_for_trafo(net, tidx)
        ctrl_type = type(ctrl).__name__ if ctrl is not None else None

        for circ_idx in range(ncircuits):
            trafo_name = _get_trafo_name(net, tidx, circ_idx)
            if trafo_name is None:
                continue

            reg_name = f"Reg_{trafo_name}"

            # ---- Tap parameters from the LV-side winding row -----------------
            try:
                lv_row = lv_rows.iloc[circ_idx]
            except IndexError:
                lv_row = lv_rows.iloc[0]

            def _col(row, col, default):
                return row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)

            tap_pos      = _safe_float(_col(lv_row, "tap_pos",          0.0),   0.0)
            tap_min      = _safe_float(_col(lv_row, "tap_min",        -16.0), -16.0)
            tap_max      = _safe_float(_col(lv_row, "tap_max",         16.0),  16.0)
            tap_step_pct = _safe_float(_col(lv_row, "tap_step_percent", 0.625), 0.625)

            tap_ratio    = 1.0 + tap_pos * tap_step_pct / 100.0
            maxtap_ratio = 1.0 + tap_max * tap_step_pct / 100.0
            mintap_ratio = 1.0 + tap_min * tap_step_pct / 100.0
            tapincrement = tap_step_pct / 100.0
            numtaps      = int(round(tap_max - tap_min))

            # ---- Voltage setpoints: prefer controller, fall back to defaults -
            if ctrl is not None:
                vm_lower = getattr(ctrl, "vm_lower_pu", 0.975)
                vm_upper = getattr(ctrl, "vm_upper_pu", 1.025)
                if hasattr(vm_lower, "__iter__"):
                    vm_lower = float(min(vm_lower))
                    vm_upper = float(max(vm_upper))
                else:
                    vm_lower = float(vm_lower)
                    vm_upper = float(vm_upper)
            else:
                vm_lower = 0.975
                vm_upper = 1.025

            vreg_120 = (vm_lower + vm_upper) / 2.0 * 120.0
            band_120 = (vm_upper - vm_lower) * 120.0

            vn_kv_lv = bus_vn[lv_bus]
            vln_v    = vn_kv_lv / math.sqrt(3) * 1000.0
            ptratio  = vln_v / 120.0

            try:
                sn_kva = _safe_float(lv_rows.iloc[circ_idx]["sn_mva"], 0.1) * 1000.0
                ct_kv  = vn_kv_lv / math.sqrt(3)
                ctprim = sn_kva / ct_kv if ct_kv > 0 else 700.0
            except Exception:
                ctprim = 700.0

            if not has_content:
                lines.append("! Regulator Controls (%REGULATOR% transformers)")
                has_content = True

            cmd = (
                f"New RegControl.{reg_name} transformer={trafo_name} winding=2 "
                f"vreg={vreg_120:.4g} band={band_120:.4g} "
                f"ptratio={ptratio:.6g} ctprim={ctprim:.6g} "
                f"tap={tap_ratio:.6g} maxtap={maxtap_ratio:.6g} mintap={mintap_ratio:.6g} "
                f"numtaps={numtaps} tapincrement={tapincrement:.6g} "
                f"eventlog=Yes"
            )

            # Line-drop compensation from LDC controllers
            if ctrl_type in ("LineDropControl", "LineDropControlExtended"):
                r_ldc = _safe_float(getattr(ctrl, "r_ldc_v", getattr(ctrl, "R_comp", 0.0)))
                x_ldc = _safe_float(getattr(ctrl, "x_ldc_v", getattr(ctrl, "X_comp", 0.0)))
                cmd += f" R={r_ldc:.6g} X={x_ldc:.6g}"

            lines.append(cmd)

    if has_content:
        lines.append("")
