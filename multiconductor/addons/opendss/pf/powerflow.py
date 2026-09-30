"""
OpenDSS Power Flow module.

Provides ``run_pf`` to compile and solve an OpenDSS network and return
bus / line / transformer results as pandas DataFrames, mirroring the
multiconductor ``run_pf`` result interface.

Usage::

    from opendss import run_pf
    results = run_pf("path/to/circuit.dss")
    results = run_pf(dss_obj)       # reuse an existing py_dss_interface session
"""
import numpy as np
import pandas as pd

try:
    import py_dss_interface
    _DSS_BACKEND = "py_dss_interface"
except ImportError:
    py_dss_interface = None
    _DSS_BACKEND = None

try:
    from opendss._dss_compat import DSSPythonAdapter
    if _DSS_BACKEND is None:
        _DSS_BACKEND = "dss_python"
except ImportError:
    DSSPythonAdapter = None


def _new_dss():
    """Create a new DSS session using whichever backend is available."""
    if py_dss_interface is not None:
        return py_dss_interface.DSS()
    if DSSPythonAdapter is not None:
        return DSSPythonAdapter()
    raise ImportError(
        "No OpenDSS backend available. "
        "Install py_dss_interface (Windows) or DSS-Python (cross-platform)."
    )


def _ensure_dss(dss_file_or_obj, algorithm="Newton"):
    """Return a ready-to-use DSS object.

    Parameters
    ----------
    dss_file_or_obj : str or py_dss_interface.DSS
        Either a ``.dss`` file path (will be compiled) or an already-
        initialised DSS object.
    algorithm : str
        Solver algorithm (``"Newton"`` or ``"NormalSolve"``).

    Returns
    -------
    tuple[py_dss_interface.DSS, bool]
        The DSS handle and whether it was freshly compiled.
    """
    if isinstance(dss_file_or_obj, str):
        d = _new_dss()
        d.text(f"compile {dss_file_or_obj}")
        d.text(f"set algorithm={algorithm}")
        return d, True
    return dss_file_or_obj, False


def _bus_results(d):
    """Extract per-bus voltage results into a DataFrame."""
    rows = []
    for i in range(d.circuit.num_buses):
        d.circuit.set_active_bus_i(i)
        name = d.bus.name
        kv_base = d.bus.kv_base
        nodes = list(d.bus.nodes)
        va_pu = d.bus.vmag_angle_pu        # [mag1, ang1, mag2, ang2, ...]
        vpus = []
        angs = []
        for j in range(len(nodes)):
            if j * 2 < len(va_pu):
                vpus.append(va_pu[j * 2])
                angs.append(va_pu[j * 2 + 1])
        vm_pu = max(vpus) if vpus else 0.0
        va_deg = angs[vpus.index(vm_pu)] if vpus else 0.0
        rows.append({
            "bus": name,
            "kv_base": kv_base,
            "num_phases": len(nodes),
            "vm_pu": vm_pu,
            "va_degree": va_deg,
            "vm_pu_per_phase": vpus,
            "va_degree_per_phase": angs,
        })
    return pd.DataFrame(rows)


def _line_results(d):
    """Extract per-line current / power results into a DataFrame."""
    rows = []
    i_elem = d.lines.first()
    while i_elem != 0:
        name = d.lines.name
        phases = d.lines.phases
        d.circuit.set_active_element(f"Line.{name}")
        currents = d.cktelement.currents_mag_ang  # [I1, A1, I2, A2, ...]
        powers = d.cktelement.powers            # [P1, Q1, P2, Q2, ...]
        n = d.cktelement.num_conductors
        i_from = [currents[j * 2] for j in range(n)] if currents else []
        i_to = [currents[(n + j) * 2] for j in range(n)] if len(currents) > n * 2 else []
        p_from = sum(powers[j * 2] for j in range(n)) if powers else 0.0
        q_from = sum(powers[j * 2 + 1] for j in range(n)) if powers else 0.0
        p_to = sum(powers[(n + j) * 2] for j in range(n)) if len(powers) > n * 2 else 0.0
        q_to = sum(powers[(n + j) * 2 + 1] for j in range(n)) if len(powers) > n * 2 else 0.0
        i_ka = max(i_from) / 1000.0 if i_from else 0.0
        rows.append({
            "line": name,
            "phases": phases,
            "i_from_a": i_from,
            "i_to_a": i_to,
            "i_ka": i_ka,
            "p_from_kw": p_from,
            "q_from_kvar": q_from,
            "p_to_kw": p_to,
            "q_to_kvar": q_to,
        })
        i_elem = d.lines.next()
    return pd.DataFrame(rows)


def _transformer_results(d):
    """Extract per-transformer power / current results into a DataFrame."""
    rows = []
    i_elem = d.transformers.first()
    while i_elem != 0:
        name = d.transformers.name
        d.circuit.set_active_element(f"Transformer.{name}")
        powers = d.cktelement.powers
        currents = d.cktelement.currents_mag_ang
        n = d.cktelement.num_conductors
        p_hv = sum(powers[j * 2] for j in range(n)) if powers else 0.0
        q_hv = sum(powers[j * 2 + 1] for j in range(n)) if powers else 0.0
        p_lv = sum(powers[(n + j) * 2] for j in range(n)) if len(powers) > n * 2 else 0.0
        q_lv = sum(powers[(n + j) * 2 + 1] for j in range(n)) if len(powers) > n * 2 else 0.0
        i_hv = max(currents[j * 2] for j in range(n)) / 1000.0 if currents else 0.0
        rows.append({
            "transformer": name,
            "p_hv_kw": p_hv,
            "q_hv_kvar": q_hv,
            "p_lv_kw": p_lv,
            "q_lv_kvar": q_lv,
            "i_hv_ka": i_hv,
        })
        i_elem = d.transformers.next()
    return pd.DataFrame(rows)


def run_pf(dss_file_or_obj, algorithm="Newton", max_iter=100, tolerance=None, **kwargs):
    """Run a power flow on an OpenDSS network.

    Parameters
    ----------
    dss_file_or_obj : str or py_dss_interface.DSS
        Path to a ``.dss`` file **or** an existing DSS session.
    algorithm : str, optional
        ``"Newton"`` (default) or ``"NormalSolve"``.
    max_iter : int, optional
        Maximum solver iterations (default 100).
    tolerance : float, optional
        Convergence tolerance.  When *None* the OpenDSS default is used.
    **kwargs
        Forwarded as ``set <key>=<value>`` commands before solving.

    Returns
    -------
    dict
        ``converged`` : bool
        ``iterations`` : int
        ``res_bus``   : DataFrame with per-bus voltage results
        ``res_line``  : DataFrame with per-line current/power results
        ``res_trafo`` : DataFrame with per-transformer results
        ``dss``       : the DSS session (can be reused)
    """
    d, freshly_compiled = _ensure_dss(dss_file_or_obj, algorithm)

    if max_iter is not None:
        d.text(f"set maxiterations={max_iter}")
    if tolerance is not None:
        d.text(f"set tolerance={tolerance}")
    for k, v in kwargs.items():
        d.text(f"set {k}={v}")

    d.text("solve")

    return {
        "converged": bool(d.solution.converged),
        "iterations": d.solution.iterations,
        "res_bus": _bus_results(d),
        "res_line": _line_results(d),
        "res_trafo": _transformer_results(d),
        "dss": d,
    }
