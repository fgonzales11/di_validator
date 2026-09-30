"""OpenDSS StorageController generation for battery-backed asymmetric sgens.

This module treats ``asymmetric_sgen`` rows whose ``name`` contains
``BATTERY`` as storage devices and emits one ``StorageController`` per
storage element.
"""
from typing import List


def is_battery_name(raw_name) -> bool:
    """Return True when *raw_name* identifies a battery-backed DER."""
    if raw_name is None:
        return False
    try:
        return "BATTERY" in str(raw_name).upper()
    except Exception:
        return False


def storage_element_name(net, lidx, circ) -> str:
    """Return the DSS element name used for a battery-backed asymmetric sgen."""
    try:
        multi = len(net.asymmetric_sgen.loc[lidx]) > 1
    except Exception:
        multi = False

    try:
        raw = net.asymmetric_sgen.loc[(lidx, circ), "name"]
        raw_name = str(raw) if raw is not None and str(raw) != "None" else None
    except Exception:
        raw_name = None

    if raw_name:
        base = raw_name if not multi else f"{raw_name}_{circ}"
    else:
        try:
            bus_int = int(net.asymmetric_sgen.loc[(lidx, circ), "bus"])
        except Exception:
            bus_int = lidx
        base = f"Storage_{bus_int}_{lidx}_{circ}"

    return base.replace(" ", "_").replace(".", "_")


def write_storage_controllers(net, lines: List[str]) -> None:
    """Append OpenDSS ``New StorageController`` commands to *lines*.

    Because the source network only exposes battery devices as
    ``asymmetric_sgen`` rows, the controller targets are inferred from the
    current storage dispatch and use a one-device fleet per controller.
    """
    if not hasattr(net, "asymmetric_sgen"):
        return
    try:
        if net.asymmetric_sgen is None or net.asymmetric_sgen.empty:
            return
    except Exception:
        return

    has_content = False
    used_names = set()

    for (lidx, circ), p_mw, in_service in zip(
        net.asymmetric_sgen.index.values,
        net.asymmetric_sgen["p_mw"].values,
        net.asymmetric_sgen["in_service"].values,
    ):
        if not bool(in_service):
            continue

        try:
            raw = net.asymmetric_sgen.loc[(lidx, circ), "name"]
            raw_name = str(raw) if raw is not None and str(raw) != "None" else None
        except Exception:
            raw_name = None

        if not is_battery_name(raw_name):
            continue

        storage_name = storage_element_name(net, lidx, circ)
        ctrl_name = f"StorageCtrl_{storage_name}"
        if ctrl_name in used_names:
            suffix = 2
            candidate = f"{ctrl_name}_{suffix}"
            while candidate in used_names:
                suffix += 1
                candidate = f"{ctrl_name}_{suffix}"
            ctrl_name = candidate
        used_names.add(ctrl_name)

        kw_target = max(abs(float(p_mw)) * 1000.0, 1e-3)
        kw_band = max(kw_target * 0.05, 0.1)

        if not has_content:
            lines.append("! Storage Controllers")
            has_content = True

        lines.append(
            f"New StorageController.{ctrl_name} "
            f"element=Storage.{storage_name} terminal=1 monphase=MAX "
            f"elementlist=[{storage_name}] modedischarge=PeakShave "
            f"kwtarget={kw_target:.6g} kwband={kw_band:.6g} eventlog=Yes"
        )

    if has_content:
        lines.append("")