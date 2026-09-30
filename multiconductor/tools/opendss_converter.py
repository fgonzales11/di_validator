"""
Converter from multiconductor pandapowerNet to OpenDSS script format.

Usage::

    from multiconductor.opendss_converter import mc_net_to_opendss
    dss_script = mc_net_to_opendss(net)
    dss_script = mc_net_to_opendss(net, filename="my_network.dss")
"""
import math
from typing import Optional

from opendss.control.capacitor import write_capacitors, write_capcontrols
from opendss.control.regulator import write_regcontrols


def mc_net_to_opendss(net, filename: Optional[str] = None) -> str:
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
    _write_sgens(net, lines)
    _write_transformers(net, lines)
    write_capacitors(net, lines)
    write_regcontrols(net, lines)
    write_capcontrols(net, lines)
    _write_footer(net, lines)

    script = "\n".join(lines)
    if filename is not None:
        with open(filename, "w") as fh:
            fh.write(script)
    return script


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Return the nominal voltage in kV for *bus_idx*."""
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


def _write_circuit(net, lines):
    """Write the ``New Circuit`` (VSource) command for the slack bus."""
    freq = _safe_float(getattr(net, "f_hz", 50), 50)
    name = (getattr(net, "name", "") or "mc_network").replace(" ", "_") or "mc_network"

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
    bus_name_str = _bus_name(net, bus_idx)
    phase_str = ".".join(str(p) for p in phases)

    # Compute short-circuit MVA from source impedance
    vln_v = vn_kv / math.sqrt(3) * 1000.0
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
        f"New Circuit.{name} basekv={vn_kv} pu={vm_pu:.6f} angle={va_degree:.2f} "
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
            code_name = std_type.replace(" ", "_")
            if not has_content:
                lines.append("! Line codes")
                has_content = True
            lines.append(
                f"New LineCode.{code_name} nphases=3 units=km "
                f"r1={r1:.8g} x1={x1:.8g} r0={r0:.8g} x0={x0:.8g} "
                f"c1={c1_nf:.8g} c0={c0_nf:.8g}"
            )

        elif model_type == "matrix":
            import numpy as np
            t = net.std_types.get("matrix", {}).get(std_type)
            if t is None:
                continue
            # Reconstruct the n×n impedance and admittance matrices
            n = len(t.get("max_i_ka", []))
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
            code_name = std_type.replace(" ", "_")
            if not has_content:
                lines.append("! Line codes")
                has_content = True
            lines.append(
                f"New LineCode.{code_name} nphases={n} units=km "
                f"Rmatrix={_lower_triangular(R)} "
                f"Xmatrix={_lower_triangular(X)} "
                f"Cmatrix={_lower_triangular(C)}"
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

        from_phase_str = ".".join(str(p) for p in from_phases)
        to_phase_str = ".".join(str(p) for p in to_phases)

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

        phases = sorted({int(p) for p in sw_grp["phase"].values if int(p) > 0})
        if not phases:
            continue

        nphases = len(phases)
        phase_str = ".".join(str(p) for p in phases)

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
        x_ohm = 1e-3
        closed_val = row0.get("closed", True) if hasattr(row0, "get") else True
        closed = str(closed_val).strip().lower() in {"true", "1", "yes", "y"}

        if not has_content:
            lines.append("! Switches")
            has_content = True

        cmd = (
            f"New Line.{switch_name} phases={nphases} "
            f"Bus1={bus1_name}.{phase_str} "
            f"Bus2={bus2_name}.{phase_str} "
            f"switch=y r1={r_ohm:.8g} x1={x_ohm:.8g} r0={r_ohm:.8g} x0={x_ohm:.8g} "
            f"Length=0.001 units=km"
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
        # Phase-to-neutral voltage for single-phase load
        vln_kv = vn_kv
        if fp == 0 or tp == 0:
            vln_kv /= math.sqrt(3)

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
            f"kV={vln_kv:.6g} kW={p_kw:.6g} kvar={q_kvar:.6g} model=1"
        )

    if has_content:
        lines.append("")


def _write_sgens(net, lines):
    """Write ``New Generator`` commands for every asymmetric static generator."""
    if not hasattr(net, "asymmetric_sgen") or net.asymmetric_sgen.empty:
        return

    used_names = set()

    # Pre-compute which sgen indices have more than one circuit
    _multi_circ_sgens = {
        lidx
        for lidx in net.asymmetric_sgen.index.get_level_values(0).unique()
        if len(net.asymmetric_sgen.loc[lidx]) > 1
    }

    has_content = False
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
        vln_kv = vn_kv
        if fp == 0 or tp == 0:
            vln_kv /= math.sqrt(3)

        bus_name_str = _bus_name(net, bus_int)
        conn_str = f"{fp}.{tp}"

        try:
            raw = net.asymmetric_sgen.loc[(lidx, circ), "name"]
            raw_name = str(raw) if raw and str(raw) != "None" else None
        except Exception:
            raw_name = None
        if raw_name:
            base = raw_name if lidx not in _multi_circ_sgens else f"{raw_name}_{circ}"
        else:
            base = f"Gen_{bus_int}_{lidx}_{circ}"
        gen_name = base.replace(" ", "_").replace(".", "_")
        if gen_name in used_names:
            n = 2
            candidate = f"{gen_name}_{n}"
            while candidate in used_names:
                n += 1
                candidate = f"{gen_name}_{n}"
            gen_name = candidate
        used_names.add(gen_name)

        if not has_content:
            lines.append("! Generators")
            has_content = True

        lines.append(
            f"New Generator.{gen_name} Bus1={bus_name_str}.{conn_str} phases=1 "
            f"kV={vln_kv:.6g} kW={p_kw:.6g} kvar={q_kvar:.6g} model=1"
        )

    if has_content:
        lines.append("")


def _write_transformers(net, lines):
    """Write ``New Transformer`` commands for every single-phase transformer bank."""
    if not hasattr(net, "trafo1ph") or net.trafo1ph.empty:
        return

    has_content = False
    trafo_indices = net.trafo1ph.index.get_level_values(0).unique()

    for tidx in trafo_indices:
        trafo_grp = net.trafo1ph.loc[tidx]
        # Each trafo_idx has rows grouped by (bus, circuit)
        # Collect the two buses
        bus_levels = trafo_grp.index.get_level_values("bus")
        unique_buses = list(dict.fromkeys(int(b) for b in bus_levels))
        if len(unique_buses) < 2:
            continue

        # Determine HV and LV bus by nominal voltage
        bus_vn = {b: _bus_vn_kv(net, b) for b in unique_buses}
        buses_sorted = sorted(unique_buses, key=lambda b: bus_vn[b], reverse=True)
        hv_bus = buses_sorted[0]
        lv_bus = buses_sorted[1]

        hv_rows = trafo_grp.xs(hv_bus, level="bus")
        lv_rows = trafo_grp.xs(lv_bus, level="bus")

        ncircuits = len(hv_rows)
        for circ_idx in range(ncircuits):
            try:
                hv_row = hv_rows.iloc[circ_idx]
                lv_row = lv_rows.iloc[circ_idx]
            except IndexError:
                continue

            hv_fp = int(hv_row["from_phase"])
            hv_tp = int(hv_row["to_phase"])
            lv_fp = int(lv_row["from_phase"])
            lv_tp = int(lv_row["to_phase"])

            vn_hv = _safe_float(hv_row["vn_kv"], bus_vn[hv_bus])
            vn_lv = _safe_float(lv_row["vn_kv"], bus_vn[lv_bus])
            sn_kva = _safe_float(hv_row["sn_mva"], 0.1) * 1000.0
            vk_pct = _safe_float(hv_row["vk_percent"], 5.0)
            vkr_pct = _safe_float(hv_row["vkr_percent"], 0.5)
            i0_pct = _safe_float(hv_row.get("i0_percent", 0.0) if hasattr(hv_row, "get") else getattr(hv_row, "i0_percent", 0.0), 0.0)
            pfe_kw = _safe_float(hv_row.get("pfe_kw", 0.0) if hasattr(hv_row, "get") else getattr(hv_row, "pfe_kw", 0.0), 0.0)

            # Connection type: if to_phase == 0 → wye; otherwise → delta
            hv_conn = "wye" if hv_tp == 0 else "delta"
            lv_conn = "wye" if lv_tp == 0 else "delta"
            hv_conn_str = f"{hv_fp}.{hv_tp}"
            lv_conn_str = f"{lv_fp}.{lv_tp}"

            hv_bus_name = _bus_name(net, hv_bus)
            lv_bus_name = _bus_name(net, lv_bus)

            raw_name = None
            try:
                raw_name = hv_row.get("name") if hasattr(hv_row, "get") else getattr(hv_row, "name", None)
                raw_name = None if str(raw_name) == "None" else str(raw_name)
            except Exception:
                pass
            if raw_name:
                # Include tidx and circ_idx to guarantee unique element names
                base = f"{raw_name}_{tidx}_{circ_idx}"
            else:
                base = f"T_{tidx}_{circ_idx}"
            trafo_name = base.replace(" ", "_").replace(".", "_")

            # No-load loss as percentage of rated load
            noloadloss_pct = (pfe_kw / sn_kva * 100.0) if sn_kva > 0 else 0.0

            if not has_content:
                lines.append("! Transformers")
                has_content = True

            xhl = math.sqrt((2 * vk_pct)**2 - (2 * vkr_pct)**2)
            loadloss_pct = 2 * vkr_pct
            noloadloss_pct = (pfe_kw / sn_kva * 100.0) if sn_kva > 0 else 0.0
            imag_pct = math.sqrt(max(i0_pct**2 - noloadloss_pct**2, 0))

            lines.append(
                f"New Transformer.{trafo_name} phases=1 windings=2 "
                f"xhl={xhl:.6g} %loadloss={loadloss_pct:.6g} "
                f"%imag={imag_pct:.6g} %noloadloss={noloadloss_pct:.6g}"
            )
            lines.append(f"~ wdg=1 bus={hv_bus_name}.{hv_conn_str} kV={vn_hv:.6g} kVA={sn_kva:.6g} conn={hv_conn}")
            lines.append(f"~ wdg=2 bus={lv_bus_name}.{lv_conn_str} kV={vn_lv:.6g} kVA={sn_kva:.6g} conn={lv_conn}")

    if has_content:
        lines.append("")


def _write_footer(net, lines):
    """Write voltage base and solve commands."""
    # Collect unique nominal voltages from buses
    bus_voltages = set()
    for bus_idx in net.bus.index.get_level_values(0).unique():
        vn = _bus_vn_kv(net, int(bus_idx))
        if vn > 0:
            bus_voltages.add(round(vn, 6))
    #TODO remove
    vbases = " ".join(str(v) for v in sorted(bus_voltages, reverse=True))
    lines.append(f"Set voltagebases=[{vbases}]")
    lines.append("CalcVoltageBases")
    lines.append("Solve")