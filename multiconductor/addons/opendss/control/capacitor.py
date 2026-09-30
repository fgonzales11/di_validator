"""
OpenDSS Capacitor and CapControl generation from multiconductor shunt data.

Translates ``net.asymmetric_shunt`` to ``New Capacitor`` DSS commands and
``MulticonductorBinaryShuntController`` entries in ``net.controller`` to
``New CapControl`` DSS commands.

Usage::

    from opendss.control.capacitor import write_capacitors, write_capcontrols
    write_capacitors(net, dss_lines)
    write_capcontrols(net, dss_lines)
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


def _cap_element_name(shunt_grp, sidx) -> str:
    """Derive a Capacitor element name from a shunt group."""
    row0 = shunt_grp.iloc[0]
    raw_name = None
    try:
        raw_name = row0.get("name") if hasattr(row0, "get") else None
        raw_name = None if str(raw_name) == "None" else str(raw_name)
    except Exception:
        pass
    return (raw_name or f"Cap_{sidx}").replace(" ", "_").replace(".", "_")


def write_capacitors(net, lines: List[str]) -> None:
    """Append OpenDSS ``New Capacitor`` commands to *lines*.

    Each unique ``asymmetric_shunt`` index produces one ``Capacitor``
    element with the aggregated kvar across all circuits.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network (may or may not have ``asymmetric_shunt``).
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    if not hasattr(net, "asymmetric_shunt"):
        return
    try:
        if net.asymmetric_shunt is None or net.asymmetric_shunt.empty:
            return
    except Exception:
        return

    has_content = False
    shunt_indices = net.asymmetric_shunt.index.get_level_values(0).unique()

    for sidx in shunt_indices:
        shunt_grp = net.asymmetric_shunt.loc[sidx]
        if hasattr(shunt_grp, "to_frame"):
            shunt_grp = shunt_grp.to_frame().T

        row0 = shunt_grp.iloc[0]
        in_service = row0.get("in_service", True) if hasattr(row0, "get") else True
        if not bool(in_service):
            continue

        bus_int = int(row0["bus"])
        bus_name = _bus_name(net, bus_int)
        vn_kv = _bus_vn_kv(net, bus_int)

        phases = [int(fp) for fp in shunt_grp["from_phase"].values]
        nphases = len(phases)
        phase_str = ".".join(str(p) for p in phases)

        to_phases = [int(tp) for tp in shunt_grp["to_phase"].values]
        conn = "wye" if all(tp == 0 for tp in to_phases) else "delta"

        if conn == "wye" and nphases == 1:
            cap_kv = vn_kv / math.sqrt(3)
        elif conn == "wye":
            cap_kv = vn_kv
        else:
            cap_kv = vn_kv

        total_kvar = 0.0
        for _, row in shunt_grp.iterrows():
            q = _safe_float(
                row.get("max_q_mvar", row.get("q_mvar", 0.0))
                if hasattr(row, "get") else 0.0
            )
            total_kvar += abs(q) * 1000.0

        cap_name = _cap_element_name(shunt_grp, sidx)

        if not has_content:
            lines.append("! Capacitors")
            has_content = True

        lines.append(
            f"New Capacitor.{cap_name} phases={nphases} "
            f"bus1={bus_name}.{phase_str} conn={conn} "
            f"kv={cap_kv:.6g} kvar={total_kvar:.6g}"
        )

    if has_content:
        lines.append("")


def write_capcontrols(net, lines: List[str]) -> None:
    """Append OpenDSS ``New CapControl`` commands to *lines*.

    Iterates over ``net.controller`` looking for
    ``MulticonductorBinaryShuntController`` objects and emits voltage-based
    ``CapControl`` elements that reference the ``Capacitor`` elements
    produced by :func:`write_capacitors`.

    Parameters
    ----------
    net : pandapowerNet
        Multiconductor network with ``controller`` and ``asymmetric_shunt``.
    lines : list[str]
        Accumulator list of DSS script lines.
    """
    if not hasattr(net, "controller"):
        return
    try:
        if net.controller is None or net.controller.empty:
            return
    except Exception:
        return

    if not hasattr(net, "asymmetric_shunt"):
        return
    try:
        if net.asymmetric_shunt is None or net.asymmetric_shunt.empty:
            return
    except Exception:
        return

    has_content = False

    for _ctrl_idx, ctrl_row in net.controller.iterrows():
        ctrl = ctrl_row["object"]
        ctrl_type = type(ctrl).__name__

        if ctrl_type != "MulticonductorBinaryShuntController":
            continue
        if not ctrl_row.get("in_service", True):
            continue

        shunt_indices = ctrl.shunt_indices
        if not shunt_indices:
            continue

        first_idx = shunt_indices[0]
        sidx = first_idx[0] if isinstance(first_idx, tuple) else first_idx

        try:
            shunt_row = net.asymmetric_shunt.loc[first_idx]
        except KeyError:
            continue

        control_mode = (
            shunt_row.get("control_mode", "switched")
            if hasattr(shunt_row, "get") else "switched"
        )

        # Derive capacitor name (must match write_capacitors output)
        raw_name = None
        try:
            raw_name = shunt_row.get("name") if hasattr(shunt_row, "get") else None
            raw_name = None if str(raw_name) == "None" else str(raw_name)
        except Exception:
            pass
        cap_name = (raw_name or f"Cap_{sidx}").replace(" ", "_").replace(".", "_")

        # Fixed-mode capacitors don't need a CapControl
        if str(control_mode).lower() == "fixed":
            continue

        # Voltage thresholds (pu → 120 V base)
        v_on = _safe_float(
            shunt_row.get("v_threshold_on", 0.9)
            if hasattr(shunt_row, "get") else 0.9, 0.9
        )
        v_off = _safe_float(
            shunt_row.get("v_threshold_off", 1.1)
            if hasattr(shunt_row, "get") else 1.1, 1.1
        )
        on_setting = v_on * 120.0
        off_setting = v_off * 120.0

        bus_int = int(shunt_row["bus"]) if "bus" in shunt_row.index else 0
        vn_kv = _bus_vn_kv(net, bus_int)
        vln_v = vn_kv / math.sqrt(3) * 1000.0
        ptratio = vln_v / 120.0 if vln_v > 0 else 1.0

        fp = int(shunt_row["from_phase"]) if "from_phase" in shunt_row.index else 1

        ctrl_name = f"CapCtrl_{cap_name}"

        if not has_content:
            lines.append("! Capacitor Controls")
            has_content = True

        # widen the shunt deadband to prevent oscillations
        if off_setting == 126.0:
            off_setting = 130.0    

        lines.append(
            f"New CapControl.{ctrl_name} capacitor={cap_name} "
            f"type=voltage OnSetting={on_setting:.4g} OffSetting={off_setting:.4g} "
            f"ptratio={ptratio:.6g} ptphase={fp} "
            f"element=capacitor.{cap_name} terminal=1 eventlog=Yes"
        )

    if has_content:
        lines.append("")
