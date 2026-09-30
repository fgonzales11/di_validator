"""
Converter from multiconductor pandapowerNet to OpenDSS script format.

Usage::

    from opendss.tools.dss_converter import mc_net_to_opendss
    dss_script = mc_net_to_opendss(net)
    dss_script = mc_net_to_opendss(net, filename="my_network.dss")
"""
import math
from typing import Optional


def _write_optional_controls(net, lines) -> None:
    """Write controller elements when the optional control package is present.

    The benchmark conversion path deliberately uses ``add_controls=False`` and
    only needs static network elements.  Keep these imports behind the explicit
    opt-in so a source checkout without ``opendss.control`` can still provide
    the OpenDSS benchmark engine.
    """
    from opendss.control.shunt_controller import write_shunt_controllers
    from opendss.control.load_tap_changer_control import (
        write_load_tap_changer_controls,
    )
    from opendss.control.line_drop_control import write_line_drop_controls
    from opendss.control.regulator import write_regulator_regcontrols
    from opendss.control.volt_var_control import write_volt_var_controls
    from opendss.control.storage_controller import write_storage_controllers

    write_shunt_controllers(net, lines)
    write_load_tap_changer_controls(net, lines)
    write_line_drop_controls(net, lines)
    write_volt_var_controls(net, lines)
    write_storage_controllers(net, lines)
    write_regulator_regcontrols(net, lines)


def mc_net_to_opendss(
    net,
    filename: Optional[str] = None,
    add_controls: bool = False,
    force_sgens_as_generators: bool = False,
) -> str:
    """Convert a multiconductor pandapowerNet to an OpenDSS script string.

    Parameters
    ----------
    net : pandapowerNet
        A multiconductor network as returned by
        :func:`multiconductor.file_io.create_empty_network` and populated with
        buses, lines, loads, generators, and transformers.
    filename : str, optional
        If provided the DSS script is written to this file in addition to
        being returned as a string.
    add_controls : bool, optional
        If True, control elements (e.g., capacitors, regulators) are added to the DSS script.
    force_sgens_as_generators : bool, optional
        If True, emit static generators as ``Generator`` elements even when
        they have no controller. Certified DOE uses this to preserve exact DER
        identity for element-level perturbation and replay.

    Returns
    -------
    str
        Complete OpenDSS script that describes the network.
    """
    lines = []
    lines.append("! OpenDSS script generated from multiconductor network")
    lines.append("Clear")
    lines.append("")

    _write_circuit(net, lines)
    _write_linecodes(net, lines)
    _write_lines(net, lines)
    _write_switches(net, lines)
    _write_loads(net, lines)
    _write_sgens(net, lines, force_generators=force_sgens_as_generators)
    _write_transformers(net, lines)
    write_capacitors(net, lines)
    if add_controls:
        _write_optional_controls(net, lines)
    _write_footer(net, lines)

    script = "\n".join(lines)
    if filename is not None:
        with open(filename, "w") as fh:
            fh.write(script)
    return script





def _bus_name(net, bus_idx: int) -> str:
    """Return a valid OpenDSS bus name for *bus_idx*."""
    try:
        row = net.bus.xs(bus_idx, level=0).iloc[0]
        raw = row.get("name") if hasattr(row, "get") else getattr(row, "name", None)
        if raw and isinstance(raw, str):
            return raw.replace(" ", "_").replace(".", "_")
    except Exception:
        pass
    return f"bus_{bus_idx}"


def _bus_vn_kv(net, bus_idx: int) -> float:
    """Return the nominal line-to-line voltage in kV for *bus_idx*."""
    try:
        row = net.bus.xs(bus_idx, level=0).iloc[0]
        return float(row["vn_kv"])
    except Exception:
        return 0.4


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _capacitor_element_name(shunt_grp, shunt_idx) -> str:
    """Return the stable OpenDSS name for an asymmetric shunt group."""
    row0 = shunt_grp.iloc[0]
    raw_name = None
    try:
        raw_name = row0.get("name") if hasattr(row0, "get") else None
        raw_name = None if str(raw_name) == "None" else str(raw_name)
    except Exception:
        pass
    return (raw_name or f"Cap_{shunt_idx}").replace(" ", "_").replace(".", "_")


def write_capacitors(net, lines) -> None:
    """Append static OpenDSS capacitor elements from asymmetric shunts.

    Static capacitor conversion is part of the base network model and must not
    depend on the optional controller emitters.
    """
    shunts = getattr(net, "asymmetric_shunt", None)
    try:
        if shunts is None or shunts.empty:
            return
    except Exception:
        return

    has_content = False
    shunt_indices = shunts.index.get_level_values(0).unique()
    for shunt_idx in shunt_indices:
        shunt_grp = shunts.loc[shunt_idx]
        if hasattr(shunt_grp, "to_frame"):
            shunt_grp = shunt_grp.to_frame().T

        row0 = shunt_grp.iloc[0]
        in_service = row0.get("in_service", True) if hasattr(row0, "get") else True
        if not bool(in_service):
            continue

        bus_idx = int(row0["bus"])
        bus_name = _bus_name(net, bus_idx)
        vn_kv = _bus_vn_kv(net, bus_idx)
        phases = [int(phase) for phase in shunt_grp["from_phase"].values]
        to_phases = [int(phase) for phase in shunt_grp["to_phase"].values]
        connection = "wye" if all(phase == 0 for phase in to_phases) else "delta"
        capacitor_kv = (
            vn_kv / math.sqrt(3)
            if connection == "wye" and len(phases) == 1
            else vn_kv
        )

        total_kvar = 0.0
        for _, row in shunt_grp.iterrows():
            reactive_mvar = _safe_float(
                row.get("max_q_mvar", row.get("q_mvar", 0.0))
                if hasattr(row, "get")
                else 0.0
            )
            total_kvar += abs(reactive_mvar) * 1000.0

        if not has_content:
            lines.append("! Capacitors")
            has_content = True

        phase_string = ".".join(str(phase) for phase in phases)
        capacitor_name = _capacitor_element_name(shunt_grp, shunt_idx)
        lines.append(
            f"New Capacitor.{capacitor_name} phases={len(phases)} "
            f"bus1={bus_name}.{phase_string} conn={connection} "
            f"kv={capacitor_kv:.6g} kvar={total_kvar:.6g}"
        )

    if has_content:
        lines.append("")


def _lower_triangular(matrix) -> str:
    """Format a square numpy array as an OpenDSS lower-triangular matrix string.

    The OpenDSS convention is ``[m11 | m21 m22 | m31 m32 m33]``.
    """
    import numpy as np
    mat = np.asarray(matrix, dtype=float)
    n = mat.shape[0]
    parts = []
    for i in range(n):
        row_parts = [f"{mat[i, j]:.8g}" for j in range(i + 1)]
        parts.append(" ".join(row_parts))
    return "[ " + " | ".join(parts) + " ]"


def _matrix_std_type_size(std_type_data) -> int:
    """Infer matrix LineCode dimension from ampacity or row-valued matrix keys."""
    max_i = std_type_data.get("max_i_ka")
    try:
        if max_i is not None and not isinstance(max_i, (str, bytes)):
            n = len(max_i)
            if n:
                return n
    except TypeError:
        pass

    n = 0
    suffixes = ("_ohm_per_km", "_us_per_km")
    for key in std_type_data:
        for suffix in suffixes:
            if key.endswith(suffix):
                prefix = key[:-len(suffix)]
                try:
                    n = max(n, int(prefix.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    pass
    return n


def _normal_amps(std_type_data) -> Optional[float]:
    """Return the conservative positive ampacity encoded by a line type."""
    raw = std_type_data.get("max_i_ka")
    if raw is None:
        return None
    try:
        if isinstance(raw, (str, bytes)):
            values = [float(raw)]
        else:
            values = [float(value) for value in raw]
    except TypeError:
        try:
            values = [float(raw)]
        except (TypeError, ValueError):
            return None
    except ValueError:
        return None
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    return min(positive) * 1000.0 if positive else None


def _dss_conductor_phase(phase: int) -> int:
    """Map internal neutral phase 0 to OpenDSS line/switch conductor node 4."""
    phase = int(phase)
    return 4 if phase == 0 else phase


def _dss_conductor_phase_str(phases) -> str:
    return ".".join(str(_dss_conductor_phase(p)) for p in phases)


def _write_circuit(net, lines):
    """Write the ``New Circuit`` (VSource) command for the slack bus."""
    freq = _safe_float(getattr(net, "f_hz", 50), 50)
    name = (
        (getattr(net, "name", "") or "mc_network")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
        or "mc_network"
    )

    use_seq = hasattr(net, "ext_grid_sequence") and not net.ext_grid_sequence.empty
    use_eg = hasattr(net, "ext_grid") and not net.ext_grid.empty

    if use_seq:
        ext_idx0 = net.ext_grid_sequence.index.get_level_values(0).unique()[0]
        eg_grp = net.ext_grid_sequence.loc[ext_idx0]
        # Positive-sequence row (sequence label = 1)
        if 1 in eg_grp.index:
            pos_row = eg_grp.loc[1]
        else:
            pos_row = eg_grp.iloc[0]
        bus_idx = int(pos_row["bus"])
        vm_pu = _safe_float(pos_row["vm_pu"], 1.0)
        va_degree = _safe_float(pos_row["va_degree"], 0.0)
        r1 = _safe_float(pos_row["r_ohm"], 0.0)
        x1 = _safe_float(pos_row["x_ohm"], 0.0)
        zero_row = eg_grp.loc[0] if 0 in eg_grp.index else pos_row
        r0 = _safe_float(zero_row["r_ohm"], r1)
        x0 = _safe_float(zero_row["x_ohm"], x1)
        phases = sorted(int(p) for p in eg_grp["from_phase"].values if int(p) > 0)
    elif use_eg:
        ext_idx0 = net.ext_grid.index.get_level_values(0).unique()[0]
        eg_grp = net.ext_grid.loc[ext_idx0]
        row0 = eg_grp.iloc[0] if hasattr(eg_grp, "iloc") else eg_grp
        bus_idx = int(row0["bus"])
        vm_pu = _safe_float(row0["vm_pu"], 1.0)
        va_degree = _safe_float(row0["va_degree"], 0.0)
        r1 = _safe_float(row0["r_ohm"], 0.0)
        x1 = _safe_float(row0["x_ohm"], 0.0)
        r0, x0 = r1, x1
        phases = sorted(int(p) for p in eg_grp["from_phase"].values if int(p) > 0)
    else:
        bus_idx = int(net.bus.index.get_level_values(0)[0])
        vm_pu, va_degree = 1.0, 0.0
        r1 = x1 = r0 = x0 = 0.0
        phases = [1, 2, 3]

    nphases = len(phases)
    vn_kv = _bus_vn_kv(net, bus_idx)
    basekv_ll = vn_kv
    bus_name_str = _bus_name(net, bus_idx)
    phase_str = ".".join(str(p) for p in phases)

    # Compute short-circuit MVA from source impedance
    vln_v = vn_kv * 1000.0
    z1 = math.sqrt(r1 ** 2 + x1 ** 2)
    z0 = math.sqrt(r0 ** 2 + x0 ** 2)
    if z1 > 1e-9:
        mvasc3 = (vln_v ** 2 / z1) * 3.0 / 1e6
        z_lll = z1
        z_lg = (2.0 * z1 + z0) / 3.0 if z0 > 1e-9 else z1
        mvasc1 = (3.0 * vln_v ** 2 / (z_lg * 3.0)) / 1e6
    else:
        mvasc3 = 1e12
        mvasc1 = 1e12

    lines.append(
        f"New Circuit.{name} basekv={basekv_ll} pu={vm_pu:.6f} angle={va_degree:.2f} "
        f"frequency={freq} phases={nphases} Mvasc3={mvasc3:.6g} Mvasc1={mvasc1:.6g}"
    )
    lines.append(f"~ bus1={bus_name_str}.{phase_str}")
    lines.append("")


def _write_linecodes(net, lines):
    """Write ``New LineCode`` commands for every std_type used by lines."""
    if not hasattr(net, "line") or net.line.empty:
        return

    freq = _safe_float(getattr(net, "f_hz", 50), 50)
    omega = 2.0 * math.pi * freq

    seen = set()
    has_content = False
    for (_, _), model_type, std_type in zip(
        net.line.index.values,
        net.line["model_type"].values,
        net.line["std_type"].values,
    ):
        if std_type is None or (model_type, std_type) in seen:
            continue
        seen.add((model_type, std_type))

        if model_type == "sequence":
            t = net.std_types.get("sequence", {}).get(std_type)
            if t is None:
                continue
            r1 = _safe_float(t.get("r_ohm_per_km", 0.0))
            x1 = _safe_float(t.get("x_ohm_per_km", 0.0))
            r0 = _safe_float(t.get("r0_ohm_per_km", r1))
            x0 = _safe_float(t.get("x0_ohm_per_km", x1))
            c1_nf = _safe_float(t.get("c_nf_per_km", 0.0))
            c0_nf = _safe_float(t.get("c0_nf_per_km", c1_nf))
            normal_amps = _normal_amps(t)
            ampacity = f" NormAmps={normal_amps:.8g}" if normal_amps is not None else ""
            code_name = std_type.replace(" ", "_")
            if not has_content:
                lines.append("! Line codes")
                has_content = True
            lines.append(
                f"New LineCode.{code_name} nphases=3 units=km "
                f"r1={r1:.8g} x1={x1:.8g} r0={r0:.8g} x0={x0:.8g} "
                f"c1={c1_nf:.8g} c0={c0_nf:.8g}{ampacity}"
            )

        elif model_type == "matrix":
            import numpy as np
            t = net.std_types.get("matrix", {}).get(std_type)
            if t is None:
                continue
            # Reconstruct the n×n impedance and admittance matrices
            n = _matrix_std_type_size(t)
            if n == 0:
                continue
            R = np.zeros((n, n))
            X = np.zeros((n, n))
            B = np.zeros((n, n))  # susceptance in µS/km
            for i in range(1, n + 1):
                r_row = t.get(f"r_{i}_ohm_per_km")
                x_row = t.get(f"x_{i}_ohm_per_km")
                b_row = t.get(f"b_{i}_us_per_km")
                if r_row is not None:
                    r_arr = np.asarray(r_row, dtype=float)
                    R[i - 1, : len(r_arr)] = r_arr
                if x_row is not None:
                    x_arr = np.asarray(x_row, dtype=float)
                    X[i - 1, : len(x_arr)] = x_arr
                if b_row is not None:
                    b_arr = np.asarray(b_row, dtype=float)
                    B[i - 1, : len(b_arr)] = b_arr
            # Convert susceptance (µS/km) to capacitance (nF/km): C = B / omega
            C = B * 1e-6 / omega * 1e9  # nF/km
            normal_amps = _normal_amps(t)
            ampacity = f" NormAmps={normal_amps:.8g}" if normal_amps is not None else ""
            code_name = std_type.replace(" ", "_")
            if not has_content:
                lines.append("! Line codes")
                has_content = True
            lines.append(
                f"New LineCode.{code_name} nphases={n} units=km "
                f"Rmatrix={_lower_triangular(R)} "
                f"Xmatrix={_lower_triangular(X)} "
                f"Cmatrix={_lower_triangular(C)}{ampacity}"
            )

    if has_content:
        lines.append("")


def _write_lines(net, lines):
    """Write ``New Line`` commands for every line in the network."""
    if not hasattr(net, "line") or net.line.empty:
        return

    has_content = False
    prev_lidx = None
    for (lidx, _), model_type, std_type, from_bus, from_phase, to_bus, to_phase, length, in_service in zip(
        net.line.index.values,
        net.line["model_type"].values,
        net.line["std_type"].values,
        net.line["from_bus"].values,
        net.line["from_phase"].values,
        net.line["to_bus"].values,
        net.line["to_phase"].values,
        net.line["length_km"].values,
        net.line["in_service"].values,
    ):
        if lidx == prev_lidx:
            # Already processed this line index in the loop above
            continue
        prev_lidx = lidx

        # Gather all circuits for this line index
        line_grp = net.line.loc[lidx]
        if not bool(in_service):
            continue

        from_bus_int = int(from_bus)
        to_bus_int = int(to_bus)
        length_km = _safe_float(length, 1.0)

        # Collect phase pairs for this line group
        from_phases = [int(fp) for fp in line_grp["from_phase"].values]
        to_phases = [int(tp) for tp in line_grp["to_phase"].values]

        nphases = len(from_phases)
        from_bus_name = _bus_name(net, from_bus_int)
        to_bus_name = _bus_name(net, to_bus_int)

        from_phase_str = _dss_conductor_phase_str(from_phases)
        to_phase_str = _dss_conductor_phase_str(to_phases)

        # Determine line name
        raw_name = None
        try:
            raw_name = line_grp.iloc[0].get("name") if hasattr(line_grp.iloc[0], "get") else None
        except Exception:
            pass
        line_name = (raw_name or f"Line_{from_bus_int}_{to_bus_int}_{lidx}").replace(" ", "_").replace(".", "_")

        if not has_content:
            lines.append("! Lines")
            has_content = True

        if std_type is not None:
            code_name = std_type.replace(" ", "_")
            lines.append(
                f"New Line.{line_name} phases={nphases} "
                f"Bus1={from_bus_name}.{from_phase_str} "
                f"Bus2={to_bus_name}.{to_phase_str} "
                f"LineCode={code_name} Length={length_km:.6g} units=km"
            )
        else:
            # No std_type: use a generic small impedance placeholder
            lines.append(
                f"New Line.{line_name} phases={nphases} "
                f"Bus1={from_bus_name}.{from_phase_str} "
                f"Bus2={to_bus_name}.{to_phase_str} "
                f"r1=0.01 x1=0.01 Length={length_km:.6g} units=km"
            )

    if has_content:
        lines.append("")


def _write_switches(net, lines):
    """Write bus-bus switches as short OpenDSS ``Line`` elements."""
    if not hasattr(net, "switch") or net.switch.empty:
        return

    has_content = False
    switch_indices = net.switch.index.get_level_values(0).unique()

    for sidx in switch_indices:
        sw_grp = net.switch.loc[sidx]
        if hasattr(sw_grp, "to_frame"):
            sw_grp = sw_grp.to_frame().T

        row0 = sw_grp.iloc[0]

        if str(row0.get("et", "")).lower() != "b":
            continue

        try:
            bus1_idx = int(row0["bus"])
            bus2_idx = int(row0["element"])
        except Exception:
            continue

        phases = sorted(
            {int(p) for p in sw_grp["phase"].values if int(p) in {0, 1, 2, 3}},
            key=_dss_conductor_phase,
        )
        if not phases:
            continue

        nphases = len(phases)
        phase_str = _dss_conductor_phase_str(phases)

        bus1_name = _bus_name(net, bus1_idx)
        bus2_name = _bus_name(net, bus2_idx)

        raw_name = None
        try:
            raw_name = row0.get("name") if hasattr(row0, "get") else None
            raw_name = None if str(raw_name) == "None" else str(raw_name)
        except Exception:
            pass
        switch_name = (raw_name or f"SW_{sidx}").replace(" ", "_").replace(".", "_")

        r_ohm_raw = _safe_float(row0.get("r_ohm", 1e-3) if hasattr(row0, "get") else 1e-3, 1e-3)
        r_ohm = max(r_ohm_raw, 1e-3)
        closed_val = row0.get("closed", True) if hasattr(row0, "get") else True
        closed = str(closed_val).strip().lower() in {"true", "1", "yes", "y"}

        if not has_content:
            lines.append("! Switches")
            has_content = True

        cmd = (
            f"New Line.{switch_name} phases={nphases} "
            f"Bus1={bus1_name}.{phase_str} "
            f"Bus2={bus2_name}.{phase_str} "
            f"switch=y r1={r_ohm:.8g} x1=0 r0={r_ohm:.8g} x0=0 "
            f"Length=1 units=none"
        )
        if not closed:
            cmd += " enabled=false"
        lines.append(cmd)

    if has_content:
        lines.append("")


def _write_loads(net, lines):
    """Write ``New Load`` commands for every asymmetric load."""
    if not hasattr(net, "asymmetric_load") or net.asymmetric_load.empty:
        return

    used_names = set()

    # Pre-compute which load indices have more than one circuit
    _multi_circ_loads = {
        lidx
        for lidx in net.asymmetric_load.index.get_level_values(0).unique()
        if len(net.asymmetric_load.loc[lidx]) > 1
    }

    has_content = False
    for (lidx, circ), bus_val, from_phase, to_phase, p_mw, q_mvar, in_service in zip(
        net.asymmetric_load.index.values,
        net.asymmetric_load["bus"].values,
        net.asymmetric_load["from_phase"].values,
        net.asymmetric_load["to_phase"].values,
        net.asymmetric_load["p_mw"].values,
        net.asymmetric_load["q_mvar"].values,
        net.asymmetric_load["in_service"].values,
    ):
        if not bool(in_service):
            continue

        bus_int = int(bus_val)
        fp = int(from_phase)
        tp = int(to_phase)
        p_kw = _safe_float(p_mw) * 1000.0
        q_kvar = _safe_float(q_mvar) * 1000.0
        vn_kv = _bus_vn_kv(net, bus_int)
        # Bus vn_kv is line-to-line. A phase-to-neutral load uses vn/sqrt(3),
        # while a phase-to-phase load uses the line-to-line value directly.
        vln_kv = vn_kv / math.sqrt(3)
        if fp != 0 and tp != 0:
            vln_kv *= math.sqrt(3)

        bus_name_str = _bus_name(net, bus_int)
        # Connection: fp.tp (e.g. 1.0 for phase A to neutral)
        conn_str = f"{fp}.{tp}"

        # Determine load element name; append circuit suffix for multi-circuit loads
        try:
            raw = net.asymmetric_load.loc[(lidx, circ), "name"]
            raw_name = str(raw) if raw and str(raw) != "None" else None
        except Exception:
            raw_name = None
        if raw_name:
            base = raw_name if lidx not in _multi_circ_loads else f"{raw_name}_{circ}"
        else:
            base = f"Load_{bus_int}_{lidx}_{circ}"
        load_name = base.replace(" ", "_").replace(".", "_")
        if load_name in used_names:
            n = 2
            candidate = f"{load_name}_{n}"
            while candidate in used_names:
                n += 1
                candidate = f"{load_name}_{n}"
            load_name = candidate
        used_names.add(load_name)

        if not has_content:
            lines.append("! Loads")
            has_content = True

        lines.append(
            f"New Load.{load_name} Bus1={bus_name_str}.{conn_str} phases=1 "
            f"kV={vln_kv:.6g} kW={p_kw:.6g} kvar={q_kvar:.6g} model=1 Vminpu=0.5 Vmaxpu=1.5"
        )

    if has_content:
        lines.append("")

def _is_pv_sgen_name(raw_name) -> bool:
    """Return True if the asymmetric_sgen name marks it as a photovoltaic DER."""
    if raw_name is None:
        return False
    try:
        return "PHOTOVOLTAIC" in str(raw_name).upper()
    except Exception:
        return False


def is_battery_name(raw_name) -> bool:
    """Return True if an asymmetric sgen name marks a battery-backed DER."""
    if raw_name is None:
        return False
    try:
        return "BATTERY" in str(raw_name).upper()
    except Exception:
        return False


def storage_element_name(net, element_idx, circuit_idx) -> str:
    """Return the stable OpenDSS element name for a battery-backed DER."""
    try:
        has_multiple_circuits = len(net.asymmetric_sgen.loc[element_idx]) > 1
    except Exception:
        has_multiple_circuits = False

    try:
        raw_name = net.asymmetric_sgen.loc[(element_idx, circuit_idx), "name"]
        raw_name = (
            str(raw_name)
            if raw_name is not None and str(raw_name) != "None"
            else None
        )
    except Exception:
        raw_name = None

    if raw_name:
        base = (
            f"{raw_name}_{circuit_idx}"
            if has_multiple_circuits
            else raw_name
        )
    else:
        try:
            bus_idx = int(
                net.asymmetric_sgen.loc[(element_idx, circuit_idx), "bus"]
            )
        except Exception:
            bus_idx = element_idx
        base = f"Storage_{bus_idx}_{element_idx}_{circuit_idx}"
    return base.replace(" ", "_").replace(".", "_")


def _write_sgens(net, lines, force_generators: bool = False):
    """Write OpenDSS DER commands for every asymmetric static generator.

    Generators whose ``name`` contains ``PHOTOVOLTAIC`` are emitted as
    ``PVSystem`` elements (so they can be controlled by ``InvControl``);
    everything else is emitted as a ``Generator`` element.
    """
    if not hasattr(net, "asymmetric_sgen") or net.asymmetric_sgen.empty:
        return

    used_names = set()

    # Pre-compute which sgen indices have more than one circuit
    _multi_circ_sgens = {
        lidx
        for lidx in net.asymmetric_sgen.index.get_level_values(0).unique()
        if len(net.asymmetric_sgen.loc[lidx]) > 1
    }

    has_gen_header = False
    has_pv_header = False
    has_storage_header = False
    for (lidx, circ), bus_val, from_phase, to_phase, p_mw, q_mvar, in_service in zip(
        net.asymmetric_sgen.index.values,
        net.asymmetric_sgen["bus"].values,
        net.asymmetric_sgen["from_phase"].values,
        net.asymmetric_sgen["to_phase"].values,
        net.asymmetric_sgen["p_mw"].values,
        net.asymmetric_sgen["q_mvar"].values,
        net.asymmetric_sgen["in_service"].values,
    ):
        if not bool(in_service):
            continue

        bus_int = int(bus_val)
        fp = int(from_phase)
        tp = int(to_phase)
        p_kw = _safe_float(p_mw) * 1000.0
        q_kvar = _safe_float(q_mvar) * 1000.0
        vn_kv = _bus_vn_kv(net, bus_int)
        # Bus vn_kv is line-to-line; phase-to-neutral DERs use vn/sqrt(3).
        vln_kv = vn_kv / math.sqrt(3)
        if fp != 0 and tp != 0:
            vln_kv *= math.sqrt(3)

        bus_name_str = _bus_name(net, bus_int)
        conn_str = f"{fp}.{tp}"

        try:
            raw = net.asymmetric_sgen.loc[(lidx, circ), "name"]
            raw_name = str(raw) if raw and str(raw) != "None" else None
        except Exception:
            raw_name = None
        is_storage = (not force_generators) and is_battery_name(raw_name)
        is_pv = (not force_generators) and _is_pv_sgen_name(raw_name)

        if is_storage:
            base = storage_element_name(net, lidx, circ)
        elif raw_name:
            base = raw_name if lidx not in _multi_circ_sgens else f"{raw_name}_{circ}"
        else:
            prefix = "PV" if is_pv else "Gen"
            base = f"{prefix}_{bus_int}_{lidx}_{circ}"
        elt_name = base.replace(" ", "_").replace(".", "_")
        if elt_name in used_names:
            n = 2
            candidate = f"{elt_name}_{n}"
            while candidate in used_names:
                n += 1
                candidate = f"{elt_name}_{n}"
            elt_name = candidate
        used_names.add(elt_name)

        if is_storage:
            try:
                sn_mva = _safe_float(
                    net.asymmetric_sgen.loc[(lidx, circ), "sn_mva"], 0.0
                )
            except Exception:
                sn_mva = 0.0
            kva = sn_mva * 1000.0 if sn_mva > 0 else max(abs(p_kw), 1e-3)
            kwh_rated = max(kva * 4.0, 1.0)
            if p_kw >= 0:
                state = "DISCHARGING"
                kw_mag = max(p_kw, 1e-3)
            else:
                state = "CHARGING"
                kw_mag = max(-p_kw, 1e-3)

            if not has_storage_header:
                lines.append("! Storage")
                has_storage_header = True

            lines.append(
                f"New Storage.{elt_name} Bus1={bus_name_str}.{conn_str} phases=1 "
                f"kV={vln_kv:.6g} kWrated={kva:.6g} kWhrated={kwh_rated:.6g} Vminpu=0.5 Vmaxpu=1.5 "
                f"%stored=100 dispmode=DEFAULT state={state} debugtrace=No "
                f"kw={kw_mag:.6g} kvar={q_kvar:.6g}"
            )
        elif is_pv:
            # PVSystem rating: prefer sn_mva; fall back to |P| if missing
            try:
                sn_mva = _safe_float(
                    net.asymmetric_sgen.loc[(lidx, circ), "sn_mva"], 0.0
                )
            except Exception:
                sn_mva = 0.0
            kva = sn_mva * 1000.0 if sn_mva > 0 else max(abs(p_kw), 1e-3)
            pmpp_kw = max(p_kw, 0.0) if p_kw > 0 else kva

            if not has_pv_header:
                lines.append("! PVSystems")
                has_pv_header = True

            lines.append(
                f"New PVSystem.{elt_name} Bus1={bus_name_str}.{conn_str} phases=1 "
                f"kV={vln_kv:.6g} kVA={kva:.6g} Pmpp={pmpp_kw:.6g} "
                f"irradiance=1 pf=1 Vminpu=0.5 Vmaxpu=1.5 MODEL=1 debugtrace=No "
                f"kvar={q_kvar:.6g}"
            )
        else:
            if not has_gen_header:
                lines.append("! Generators")
                has_gen_header = True

            lines.append(
                f"New Generator.{elt_name} Bus1={bus_name_str}.{conn_str} phases=1 "
                f"kV={vln_kv:.6g} kW={p_kw:.6g} kvar={q_kvar:.6g} model=1 Vminpu=0.5 Vmaxpu=1.5 "
            )

    if has_gen_header or has_pv_header or has_storage_header:
        lines.append("")


def _write_transformers(net, lines):
    """Write each transformer circuit without dropping secondary windings.

    Most circuits contain two rows. SMART-DS centre-tapped service
    transformers contain a primary plus two oppositely wound secondary rows.
    Each row is a star-equivalent winding impedance contribution, so the
    OpenDSS pair reactances are sums of the corresponding winding reactances.
    """
    if not hasattr(net, "trafo1ph") or net.trafo1ph.empty:
        return

    has_content = False
    table = net.trafo1ph.reset_index()
    circuit_column = "circuit" if "circuit" in table.columns else None

    for tidx, trafo_rows in table.groupby("index", sort=False):
        circuit_groups = (
            trafo_rows.groupby(circuit_column, sort=False)
            if circuit_column
            else [(0, trafo_rows)]
        )
        for circuit, circuit_rows in circuit_groups:
            if len(circuit_rows) < 2:
                continue

            # Keep source order for equal-voltage windings. The second half of
            # a centre tap uses reversed terminals (0.2), which carries its
            # polarity into the OpenDSS bus specification.
            windings = list(circuit_rows.to_dict("records"))
            windings.sort(
                key=lambda row: _safe_float(
                    row.get("vn_kv"), _bus_vn_kv(net, int(row["bus"]))
                ),
                reverse=True,
            )

            raw_name = windings[0].get("name")
            raw_name = None if raw_name is None or str(raw_name) == "None" else str(raw_name)
            base = f"{raw_name}_{tidx}_{circuit}" if raw_name else f"T_{tidx}_{circuit}"
            name = base.replace(" ", "_").replace(".", "_")

            ratings_kva = [
                max(_safe_float(row.get("sn_mva"), 0.0001) * 1000.0, 0.1)
                for row in windings
            ]
            r_pct = [max(_safe_float(row.get("vkr_percent"), 0.0), 0.0) for row in windings]
            z_pct = [max(_safe_float(row.get("vk_percent"), 0.001), 0.001) for row in windings]
            x_pct = [
                math.sqrt(max(z_value * z_value - r_value * r_value, 0.0))
                for z_value, r_value in zip(z_pct, r_pct)
            ]
            pfe_kw = sum(max(_safe_float(row.get("pfe_kw"), 0.0), 0.0) for row in windings)
            i0_pct = max(
                (_safe_float(row.get("i0_percent"), 0.0) for row in windings),
                default=0.0,
            )
            noloadloss_pct = pfe_kw / ratings_kva[0] * 100.0 if ratings_kva[0] else 0.0
            imag_pct = math.sqrt(max(i0_pct * i0_pct - noloadloss_pct * noloadloss_pct, 0.0))

            pair_terms = []
            for pair_name, (left, right) in zip(
                ("xhl", "xht", "xlt"), ((0, 1), (0, 2), (1, 2))
            ):
                if right < len(windings):
                    pair_terms.append(f"{pair_name}={x_pct[left] + x_pct[right]:.6g}")
            resistance_values = " ".join(f"{value:.6g}" for value in r_pct)

            if not has_content:
                lines.append("! Transformers")
                has_content = True
            lines.append(
                f"New Transformer.{name} phases=1 windings={len(windings)} "
                f"{' '.join(pair_terms)} %Rs=[{resistance_values}] "
                f"%imag={imag_pct:.6g} %noloadloss={noloadloss_pct:.6g}"
            )
            for winding_number, (row, rating_kva) in enumerate(
                zip(windings, ratings_kva), start=1
            ):
                bus = int(row["bus"])
                from_phase = int(row["from_phase"])
                to_phase = int(row["to_phase"])
                vn_kv = _safe_float(row.get("vn_kv"), _bus_vn_kv(net, bus))
                connection = "wye" if 0 in (from_phase, to_phase) else "delta"
                lines.append(
                    f"~ wdg={winding_number} "
                    f"bus={_bus_name(net, bus)}.{from_phase}.{to_phase} "
                    f"kV={vn_kv:.6g} kVA={rating_kva:.6g} conn={connection}"
                )

    if has_content:
        lines.append("")


def _write_footer(net, lines):
    """Write voltage base commands. The consumer is responsible for Solve."""
    bus_voltages = set()
    for bus_idx in net.bus.index.get_level_values(0).unique():
        vn = _bus_vn_kv(net, int(bus_idx))
        if vn > 0:
            bus_voltages.add(round(vn, 6))
    vbases = " ".join(str(v) for v in sorted(bus_voltages, reverse=True))
    lines.append(f"Set voltagebases=[{vbases}]")
    lines.append("CalcVoltageBases")
    lines.append("Solve")
