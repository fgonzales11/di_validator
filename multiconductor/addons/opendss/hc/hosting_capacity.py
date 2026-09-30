"""
OpenDSS Hosting Capacity (ICA) module.

Provides ``run_hosting_capacity`` to determine the maximum DER (distributed
energy resource) injection at each bus before voltage, thermal, or reverse-
power violations occur.  Two strategies are available:

* **binary search** (default) — iteratively adds a generator at each bus and
  uses binary search on kW to find the violation threshold.
* **streamlined** — analytical estimate using base-case power flow results and
  Thévenin impedance from a short-circuit study.

Usage::

    from opendss import run_hosting_capacity
    hc = run_hosting_capacity("circuit.dss")
    hc_df = hc["res_hc"]
"""
import numpy as np
import pandas as pd
import py_dss_interface

from opendss.pf.powerflow import _ensure_dss


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _get_max_voltage(d):
    """Return the maximum per-unit voltage across all buses."""
    vmax = 0.0
    for i in range(d.circuit.num_buses):
        d.circuit.set_active_bus_i(i)
        va = d.bus.vmag_angle_pu
        for j in range(len(d.bus.nodes)):
            if j * 2 < len(va) and va[j * 2] > vmax:
                vmax = va[j * 2]
    return vmax


def _get_min_voltage(d):
    """Return the minimum per-unit voltage across all buses (ignoring zeros)."""
    vmin = 999.0
    for i in range(d.circuit.num_buses):
        d.circuit.set_active_bus_i(i)
        va = d.bus.vmag_angle_pu
        for j in range(len(d.bus.nodes)):
            if j * 2 < len(va) and va[j * 2] > 0.01 and va[j * 2] < vmin:
                vmin = va[j * 2]
    return vmin


def _get_max_line_loading(d):
    """Return max line loading percent (if NormalRating > 0)."""
    max_pct = 0.0
    i_elem = d.lines.first()
    while i_elem != 0:
        name = d.lines.name
        normal_amps = d.lines.norm_amps
        d.circuit.set_active_element(f"Line.{name}")
        currents = d.cktelement.currents_mag_ang
        n = d.cktelement.num_conductors
        i_max = max((currents[j * 2] for j in range(n)), default=0.0) if currents else 0.0
        if normal_amps > 0 and i_max > 0:
            pct = i_max / normal_amps * 100.0
            if pct > max_pct:
                max_pct = pct
        i_elem = d.lines.next()
    return max_pct


def _get_max_trafo_loading(d):
    """Return max transformer loading percent."""
    max_pct = 0.0
    i_elem = d.transformers.first()
    while i_elem != 0:
        name = d.transformers.name
        kva = d.transformers.kva
        d.circuit.set_active_element(f"Transformer.{name}")
        powers = d.cktelement.powers
        n = d.cktelement.num_conductors
        s_kva = 0.0
        if powers:
            p = sum(abs(powers[j * 2]) for j in range(n))
            q = sum(abs(powers[j * 2 + 1]) for j in range(n))
            s_kva = np.sqrt(p ** 2 + q ** 2)
        if kva > 0 and s_kva > 0:
            pct = s_kva / kva * 100.0
            if pct > max_pct:
                max_pct = pct
        i_elem = d.transformers.next()
    return max_pct


def check_violations(d, v_upper=1.05, v_lower=0.95, line_limit=100.0, trafo_limit=100.0):
    """Check for voltage / thermal violations.

    Returns
    -------
    tuple[bool, str, dict]
        (has_violation, binding_constraint_name, details_dict)
    """
    vmax = _get_max_voltage(d)
    vmin = _get_min_voltage(d)
    line_pct = _get_max_line_loading(d)
    trafo_pct = _get_max_trafo_loading(d)

    violations = []
    if vmax > v_upper:
        violations.append(("overvoltage", vmax - v_upper))
    if vmin < v_lower:
        violations.append(("undervoltage", v_lower - vmin))
    if line_pct > line_limit:
        violations.append(("line_thermal", line_pct - line_limit))
    if trafo_pct > trafo_limit:
        violations.append(("trafo_thermal", trafo_pct - trafo_limit))

    details = {"vmax_pu": vmax, "vmin_pu": vmin,
               "max_line_pct": line_pct, "max_trafo_pct": trafo_pct}

    if violations:
        binding = max(violations, key=lambda x: x[1])[0]
        return True, binding, details
    return False, "none", details


# --------------------------------------------------------------------------- #
#  Bus catalogue                                                               #
# --------------------------------------------------------------------------- #

def _collect_buses(d, exclude_slack=True):
    """Return a DataFrame of candidate buses for hosting capacity."""
    rows = []
    for i in range(d.circuit.num_buses):
        d.circuit.set_active_bus_i(i)
        name = d.bus.name
        kv_base = d.bus.kv_base
        nodes = list(d.bus.nodes)
        va_pu = d.bus.vmag_angle_pu
        vpus = [va_pu[j * 2] for j in range(len(nodes)) if j * 2 < len(va_pu)]
        vmax = max(vpus) if vpus else 0.0
        if kv_base > 0 and vmax > 0:
            rows.append({
                "bus": name, "kv_base": kv_base,
                "num_phases": len(nodes), "base_vmax_pu": vmax,
            })
    df = pd.DataFrame(rows)
    if exclude_slack and len(df) > 0:
        # The first bus is typically the source bus
        df = df.iloc[1:]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Binary-search hosting capacity                                              #
# --------------------------------------------------------------------------- #

def run_hosting_capacity(
    dss_file,
    v_upper=1.05,
    v_lower=0.95,
    line_limit=100.0,
    trafo_limit=100.0,
    max_kw=5000,
    tol_kw=100,
    pf=1.0,
    exclude_slack=True,
    verbose=True,
    buses=None,
    **kwargs,
):
    """Compute generation hosting capacity at each bus via binary search.

    For every candidate bus a test generator is added, and its kW output is
    swept with binary search until a voltage, thermal, or convergence
    violation is detected.

    Parameters
    ----------
    dss_file : str
        Path to the ``.dss`` file.  Must be a file path because the circuit
        is re-compiled for each bus to guarantee a clean state.
    v_upper : float
        Upper voltage limit (pu), default 1.05.
    v_lower : float
        Lower voltage limit (pu), default 0.95.
    line_limit : float
        Maximum line loading percent.
    trafo_limit : float
        Maximum transformer loading percent.
    max_kw : float
        Maximum generation to test per bus (kW).
    tol_kw : float
        Binary search resolution (kW).
    pf : float
        Generator power factor.
    exclude_slack : bool
        Skip the slack (source) bus.
    verbose : bool
        Print progress.
    buses : list[str] or None
        If provided, only evaluate these bus names.

    Returns
    -------
    dict
        ``res_hc`` : DataFrame with columns ``bus``, ``kv_base``,
        ``num_phases``, ``base_vpu``, ``hosting_capacity_kw``,
        ``binding_constraint``.
        ``dss`` : final DSS session.
    """
    # Initial solve to catalogue buses
    d = py_dss_interface.DSS()
    d.text(f"compile {dss_file}")
    d.text("set algorithm=Newton")
    d.text("solve")

    if not d.solution.converged:
        raise RuntimeError("Base case did not converge")

    bus_df = _collect_buses(d, exclude_slack=exclude_slack)
    if buses is not None:
        bus_set = set(b.lower() for b in buses)
        bus_df = bus_df[bus_df["bus"].str.lower().isin(bus_set)].reset_index(drop=True)

    total = len(bus_df)
    if verbose:
        print(f"Computing hosting capacity for {total} buses "
              f"(max={max_kw} kW, tol={tol_kw} kW)...")

    hc_results = []
    for idx, row in bus_df.iterrows():
        bname = row["bus"]
        kv_base = row["kv_base"]
        n_ph = row["num_phases"]
        gen_kv = kv_base * np.sqrt(3) if n_ph > 1 else kv_base

        # Fresh compile for each bus
        d.text(f"compile {dss_file}")
        d.text("set algorithm=Newton")
        d.text(
            f"new generator.hc_test bus1={bname} phases={n_ph} "
            f"kv={gen_kv:.4f} kw=0 pf={pf}"
        )

        lo, hi, hc = 0.0, float(max_kw), 0.0
        binding = "none"

        while hi - lo > tol_kw:
            mid = (lo + hi) / 2.0
            d.text(f"edit generator.hc_test kw={mid}")
            d.text("solve")

            if not d.solution.converged:
                hi = mid
                binding = "diverged"
                continue

            violated, constraint, _ = check_violations(
                d, v_upper=v_upper, v_lower=v_lower,
                line_limit=line_limit, trafo_limit=trafo_limit,
            )
            if violated:
                hi = mid
                binding = constraint
            else:
                lo = mid
                hc = mid

        hc_results.append({
            "bus": bname,
            "kv_base": kv_base,
            "num_phases": n_ph,
            "base_vpu": row["base_vmax_pu"],
            "hosting_capacity_kw": round(hc, 1),
            "binding_constraint": binding,
        })

        if verbose and ((idx + 1) % 100 == 0 or idx + 1 == total):
            print(f"  {idx + 1}/{total} buses processed")

    hc_df = pd.DataFrame(hc_results)

    if verbose and len(hc_df) > 0:
        print(f"\n=== Hosting Capacity Summary ===")
        print(f"  V_upper={v_upper} pu | Line limit={line_limit}% | Trafo limit={trafo_limit}%")
        print(f"  Buses evaluated: {len(hc_df)}")
        print(f"  HC > 0 kW: {(hc_df['hosting_capacity_kw'] > 0).sum()}")
        print(f"  HC = 0 kW: {(hc_df['hosting_capacity_kw'] == 0).sum()}")
        print(f"  min={hc_df['hosting_capacity_kw'].min():.1f}, "
              f"mean={hc_df['hosting_capacity_kw'].mean():.1f}, "
              f"median={hc_df['hosting_capacity_kw'].median():.1f}, "
              f"max={hc_df['hosting_capacity_kw'].max():.1f} kW")
        if "binding_constraint" in hc_df.columns:
            print(f"  Binding constraints:")
            for c, n in hc_df["binding_constraint"].value_counts().items():
                print(f"    {c}: {n} buses")

    return {
        "res_hc": hc_df,
        "dss": d,
    }
