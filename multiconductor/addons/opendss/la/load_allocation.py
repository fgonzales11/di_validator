"""
OpenDSS Load Allocation module.

Adjusts individual load kW/kvar values so that the total feeder demand
(measured at the substation head) matches a target measurement, mirroring
the multiconductor ``run_load_allocation`` workflow.

Usage::

    from opendss import run_load_allocation
    result = run_load_allocation("circuit.dss", target_kw=5000)
"""
import numpy as np
import pandas as pd
import py_dss_interface

from opendss.pf.powerflow import _ensure_dss, run_pf


def _get_substation_power(d):
    """Return total (P_kw, Q_kvar) measured at the substation / vsource."""
    d.circuit.set_active_element("Vsource.source")
    powers = d.cktelement.powers
    n = d.cktelement.num_conductors
    p_kw = -sum(powers[j * 2] for j in range(n)) if powers else 0.0
    q_kvar = -sum(powers[j * 2 + 1] for j in range(n)) if powers else 0.0
    return p_kw, q_kvar


def _get_load_names(d):
    """Return a list of all load element names."""
    names = []
    i = d.loads.first()
    while i != 0:
        names.append(d.loads.name)
        i = d.loads.next()
    return names


def _get_load_kw(d, load_name):
    d.circuit.set_active_element(f"Load.{load_name}")
    d.loads.name = load_name
    return d.loads.kw


def _set_load_kw(d, load_name, kw):
    d.loads.name = load_name
    d.loads.kw = kw


def run_load_allocation(
    dss_file_or_obj,
    target_kw=None,
    target_kvar=None,
    tolerance_kw=0.00001,
    max_iter=15,
    adjust_power_factor=True,
    algorithm="Newton",
    verbose=True,
    **kwargs,
):
    """Scale load kW values so feeder head matches a target measurement.

    Parameters
    ----------
    dss_file_or_obj : str or py_dss_interface.DSS
        Path to a ``.dss`` file **or** an existing DSS session.
    target_kw : float or None
        Target active power at the feeder head (kW).  Required.
    target_kvar : float or None
        Target reactive power at the feeder head (kvar).
        When *None*, loads are scaled keeping their original power factor.
    tolerance_kw : float
        Convergence threshold on active power mismatch (kW).
    max_iter : int
        Maximum scaling iterations.
    adjust_power_factor : bool
        If *True* **and** ``target_kvar`` is given, kvar is scaled independently.
    algorithm : str
        Solver algorithm.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    dict
        ``converged``  : bool
        ``iterations`` : int
        ``mismatch_kw`` : float — final P mismatch
        ``scale_factor`` : float — cumulative kW multiplier
        ``res_bus``    : DataFrame — final bus voltages
        ``res_line``   : DataFrame — final line results
        ``dss``        : the DSS session
    """
    if target_kw is None:
        raise ValueError("target_kw is required for load allocation")

    d, _ = _ensure_dss(dss_file_or_obj, algorithm)
    d.text("solve")

    load_names = _get_load_names(d)
    if not load_names:
        raise RuntimeError("No loads found in the circuit")

    # Record original kW and kvar for each load
    orig_kw = {}
    orig_kvar = {}
    for ln in load_names:
        d.loads.name = ln
        orig_kw[ln] = d.loads.kw
        orig_kvar[ln] = d.loads.kvar

    cumulative_scale = 1.0

    for it in range(1, max_iter + 1):
        d.text("solve")
        p_sub, q_sub = _get_substation_power(d)
        mismatch = target_kw - p_sub

        if verbose:
            print(f"  Iter {it}: P_sub={p_sub:.1f} kW, target={target_kw:.1f}, "
                  f"mismatch={mismatch:.1f} kW, scale={cumulative_scale:.6f}")

        if abs(mismatch) <= tolerance_kw:
            break

        # Scale factor for this iteration
        if abs(p_sub) > 1e-6:
            sf = target_kw / p_sub
        else:
            sf = 2.0  # fallback if base load is near zero

        cumulative_scale *= sf

        for ln in load_names:
            new_kw = orig_kw[ln] * cumulative_scale
            _set_load_kw(d, ln, new_kw)
            if adjust_power_factor and target_kvar is not None:
                # Scale kvar proportionally to the overall Q target ratio
                if abs(q_sub) > 1e-6:
                    q_sf = target_kvar / q_sub
                else:
                    q_sf = 1.0
                d.loads.kvar = orig_kvar[ln] * q_sf
    else:
        if verbose:
            print(f"  Load allocation did not converge in {max_iter} iterations")

    # Final solve and extract results
    pf_result = run_pf(d, algorithm=algorithm)

    p_final, _ = _get_substation_power(d)

    return {
        "converged": abs(target_kw - p_final) <= tolerance_kw,
        "iterations": it,
        "mismatch_kw": target_kw - p_final,
        "scale_factor": cumulative_scale,
        "res_bus": pf_result["res_bus"],
        "res_line": pf_result["res_line"],
        "dss": d,
    }
