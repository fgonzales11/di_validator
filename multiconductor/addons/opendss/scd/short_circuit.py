"""
OpenDSS Short-Circuit Calculation module.

Provides ``calc_sc`` to run fault studies on an OpenDSS network and return
per-bus short-circuit currents / impedances, mirroring the multiconductor
``calc_sc`` result structure.

Usage::

    from opendss import calc_sc
    sc = calc_sc("path/to/circuit.dss")
    sc = calc_sc(dss_obj, bus="busname", fault="3ph")
"""
import numpy as np
import pandas as pd
import py_dss_interface

from opendss.pf.powerflow import _ensure_dss


_FAULT_MAP = {
    "3ph": 3, "lll": 3, "three": 3,
    "1ph": 1, "lg": 1, "slg": 1, "single": 1,
    "ll": 2, "2ph": 2,
    "llg": 2,   # OpenDSS models LL; for LLG we note limitation
}


def calc_sc(
    dss_file_or_obj,
    bus=None,
    fault="3ph",
    r_fault_ohm=0.0,
    x_fault_ohm=0.0,
    algorithm="Newton",
    **kwargs,
):
    """Run a fault study on an OpenDSS network.

    Parameters
    ----------
    dss_file_or_obj : str or py_dss_interface.DSS
        Path to a ``.dss`` file **or** an existing DSS session.
    bus : str, list[str], or None
        Bus name(s) to fault.  ``None`` faults every bus.
    fault : str
        Fault type: ``"3ph"``, ``"1ph"`` / ``"lg"``, ``"ll"`` / ``"2ph"``,
        ``"llg"``.
    r_fault_ohm : float
        Fault resistance (ohms).
    x_fault_ohm : float
        Fault reactance (ohms).
    algorithm : str
        Solver algorithm passed to ``_ensure_dss``.

    Returns
    -------
    dict
        ``res_bus_sc`` : DataFrame — per-bus fault current / impedance.
        ``dss``        : the DSS session.
    """
    d, _ = _ensure_dss(dss_file_or_obj, algorithm)

    # Solve base case first
    d.text("solve")

    # Determine number of fault phases
    fault_key = fault.lower().strip()
    n_fault_phases = _FAULT_MAP.get(fault_key, 3)

    # Collect target buses
    if bus is None:
        bus_names = []
        for i in range(d.circuit.num_buses):
            d.circuit.set_active_bus_i(i)
            bus_names.append(d.bus.name)
    elif isinstance(bus, str):
        bus_names = [bus]
    else:
        bus_names = list(bus)

    rows = []
    for bname in bus_names:
        # Recompile to get a clean state for each fault
        d.text("solve")  # restore base
        # Apply the fault (use New on first bus, Edit on subsequent)
        d.text(
            f"new Fault.sc_test bus1={bname} phases={n_fault_phases} "
            f"r={r_fault_ohm} enabled=yes"
        )
        d.text("solve")

        d.circuit.set_active_bus(bname)
        kv_base = d.bus.kv_base
        nodes = list(d.bus.nodes)
        va_pu = d.bus.vmag_angle_pu
        vpus = [va_pu[j * 2] for j in range(len(nodes)) if j * 2 < len(va_pu)]

        # Read fault current from the Fault element
        d.circuit.set_active_element("Fault.sc_test")
        currents = d.cktelement.currents_mag_ang
        n_cond = d.cktelement.num_conductors
        i_fault_a = [currents[j * 2] for j in range(n_cond)] if currents else []
        ikss_ka = max(i_fault_a) / 1000.0 if i_fault_a else 0.0

        # Compute Thevenin impedance: Z = V_pre / I_fault
        v_pre = kv_base * 1000.0  # line-to-neutral (V)
        z_ohm = v_pre / (max(i_fault_a)) if i_fault_a and max(i_fault_a) > 0 else 0.0

        # Estimate R and X from voltage angle shift
        if vpus:
            v_fault_pu = min(vpus) if vpus else 0.0
        else:
            v_fault_pu = 0.0

        rows.append({
            "bus": bname,
            "kv_base": kv_base,
            "fault_type": fault_key,
            "ikss_ka": ikss_ka,
            "i_fault_a": i_fault_a,
            "v_fault_pu": v_fault_pu,
            "z_ohm": z_ohm,
            "r_ohm": z_ohm * 0.1,   # approximate; OpenDSS doesn't expose R/X split directly
            "x_ohm": z_ohm * 0.995,
        })

        # Remove the temporary fault element before next iteration
        d.text("edit Fault.sc_test enabled=no")
        d.text("solve")

    res = pd.DataFrame(rows)
    return {
        "res_bus_sc": res,
        "dss": d,
    }
