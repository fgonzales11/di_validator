"""
Convergence diagnostics for multiconductor power flow.

Usable from both the backend API and Jupyter notebooks.

    from backend.pf_diagnostics import audit_network, run_pf_verbose, diagnose_failure

    # In a notebook (no package prefix needed if on sys.path):
    from pf_diagnostics import audit_network, run_pf_verbose, diagnose_failure
"""
import logging
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("pf_diagnostics")

_PHASE_LABELS = {1: "A", 2: "B", 3: "C", 0: "N"}
_LOW_V_WARN  = 0.95   # pu — ANSI Range A lower limit
_LOW_V_ERR   = 0.80   # pu — severe under-voltage
_HIGH_V_WARN = 1.05   # pu — ANSI Range A upper limit
_HIGH_V_ERR  = 1.20   # pu — severe over-voltage


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _df_safe(net, attr):
    """Return DataFrame or empty DataFrame."""
    tbl = getattr(net, attr, None)
    if tbl is None or (hasattr(tbl, "empty") and tbl.empty):
        return pd.DataFrame()
    return tbl


def _unique_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if isinstance(df.index, pd.MultiIndex):
        return len(df.index.get_level_values(0).unique())
    return len(df.index.unique())


def _phase_lv0(df: pd.DataFrame) -> pd.Series:
    """Return level-0 of a MultiIndex DataFrame as a Series of ints."""
    if isinstance(df.index, pd.MultiIndex):
        return df.index.get_level_values(0)
    return df.index


def _log_and_print(msg: str, level: str = "info", verbose: bool = True) -> None:
    getattr(logger, level)(msg)
    if verbose:
        print(msg)


def _header(title: str, width: int = 70) -> str:
    return f"\n{'='*width}\n  {title}\n{'='*width}"


def _sub(title: str, width: int = 60) -> str:
    return f"\n  {'─'*width}\n  {title}\n  {'─'*width}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pre-solve network audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_network(net, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Pre-solve network audit.  Checks connectivity, loads, transformers, and
    impedances and returns a findings dict.  Prints findings when verbose=True.

    Call this BEFORE mc.run_pf() to surface issues that cause non-convergence.
    """
    findings: Dict[str, Any] = {
        "warnings": [],
        "errors":   [],
        "info":     [],
    }
    lines: List[str] = []

    def warn(msg):
        findings["warnings"].append(msg)
        lines.append(f"  [WARN]  {msg}")

    def err(msg):
        findings["errors"].append(msg)
        lines.append(f"  [ERROR] {msg}")

    def info(msg):
        findings["info"].append(msg)
        lines.append(f"  [INFO]  {msg}")

    lines.append(_header("PRE-SOLVE NETWORK AUDIT"))

    # ── Element counts ────────────────────────────────────────────────────────
    bus_df    = _df_safe(net, "bus")
    line_df   = _df_safe(net, "line")
    trafo_df  = _df_safe(net, "trafo1ph")
    ct_df     = _df_safe(net, "center_tap_trafo")
    load_df   = _df_safe(net, "asymmetric_load")
    sgen_df   = _df_safe(net, "asymmetric_sgen")
    gen_df    = _df_safe(net, "asymmetric_gen")
    sw_df     = _df_safe(net, "switch")
    eg_df     = _df_safe(net, "ext_grid")
    egs_df    = _df_safe(net, "ext_grid_sequence")
    shunt_df  = _df_safe(net, "asymmetric_shunt")

    n_bus   = _unique_count(bus_df)
    n_line  = _unique_count(line_df)
    n_trafo = _unique_count(trafo_df) + _unique_count(ct_df)
    n_load  = _unique_count(load_df)
    n_sgen  = _unique_count(sgen_df)
    n_gen   = _unique_count(gen_df)
    n_sw    = _unique_count(sw_df)
    n_eg    = _unique_count(eg_df)
    n_shunt = _unique_count(shunt_df)

    findings["n_bus"]   = n_bus
    findings["n_line"]  = n_line
    findings["n_trafo"] = n_trafo
    findings["n_load"]  = n_load

    lines.append(_sub("Element counts"))
    lines.append(f"    buses={n_bus}  lines={n_line}  trafos={n_trafo}  "
                 f"loads={n_load}  sgen={n_sgen}  gen={n_gen}")
    lines.append(f"    ext_grid={n_eg}  switches={n_sw}  shunts={n_shunt}")

    if n_bus == 0:
        err("No buses found — network is empty.")
        _log_and_print("\n".join(lines), "error", verbose)
        return findings

    if n_eg == 0 and not egs_df.empty:
        info("No ext_grid rows; slack reference comes from ext_grid_sequence.")
    elif n_eg == 0:
        err("No external grid (slack bus) found — solver cannot establish a voltage reference.")

    # ── Slack bus ─────────────────────────────────────────────────────────────
    lines.append(_sub("Slack / ext_grid"))
    if not eg_df.empty:
        eg_r = eg_df.reset_index()
        for col in ["bus", "vm_pu", "va_degree"]:
            if col not in eg_r.columns:
                eg_r[col] = np.nan
        for _, row in eg_r.drop_duplicates(
            subset=[c for c in ["bus", "vm_pu"] if c in eg_r.columns]
        ).iterrows():
            vm = _safe_float(row.get("vm_pu", 1.0), 1.0)
            lines.append(f"    bus={row.get('bus', '?')}  vm_pu={vm:.4f}")
            if vm < 0.9 or vm > 1.1:
                warn(f"ext_grid vm_pu={vm:.4f} is outside normal range [0.90, 1.10]")

    # ── Voltage bases ─────────────────────────────────────────────────────────
    lines.append(_sub("Bus voltage bases"))
    if not bus_df.empty and "vn_kv" in bus_df.columns:
        vbases = bus_df["vn_kv"].dropna().unique()
        vbases.sort()
        lines.append(f"    unique vn_kv (kV): {[round(v,4) for v in vbases]}")
        findings["vbases_kv"] = list(vbases)
        if np.any(vbases <= 0):
            err("One or more buses have vn_kv <= 0 — invalid voltage base.")
        extreme = vbases[vbases > 1000]
        if len(extreme):
            warn(f"Buses with vn_kv > 1000 kV detected: {list(extreme)} — check unit (V vs kV?).")
    else:
        warn("Bus table missing 'vn_kv' column.")

    # ── Load summary ──────────────────────────────────────────────────────────
    lines.append(_sub("Load summary"))
    if not load_df.empty:
        p_col = "p_mw" if "p_mw" in load_df.columns else None
        q_col = "q_mvar" if "q_mvar" in load_df.columns else None
        if p_col:
            total_p = _safe_float(load_df[p_col].sum())
            total_q = _safe_float(load_df[q_col].sum()) if q_col else 0.0
            max_p   = _safe_float(load_df[p_col].max())
            lines.append(f"    total load: {total_p:.4f} MW + j{total_q:.4f} MVAR")
            findings["total_load_mw"]   = total_p
            findings["total_load_mvar"] = total_q

            # Per-load extremes
            if max_p > 100:
                warn(f"Maximum single load p_mw={max_p:.2f} MW is very large — check units.")
            bad_p = load_df[load_df[p_col] < 0] if p_col else pd.DataFrame()
            if not bad_p.empty:
                warn(f"{len(bad_p)} load rows have negative p_mw — could be confused with generation.")

            # Power factor check
            if p_col and q_col:
                pq = load_df[[p_col, q_col]].dropna()
                s  = np.sqrt(pq[p_col]**2 + pq[q_col]**2)
                valid = s > 1e-9
                if valid.any():
                    pf = (pq.loc[valid, p_col] / s[valid]).abs()
                    low_pf = pf[pf < 0.5]
                    if not low_pf.empty:
                        warn(f"{len(low_pf)} loads have power factor < 0.50 (very reactive) — "
                             f"min PF={pf.min():.3f}.")
    else:
        info("No asymmetric_load elements.")

    # ── Generation summary ────────────────────────────────────────────────────
    lines.append(_sub("Generation summary"))
    total_gen_mw = 0.0
    total_trafo_mva = 0.0
    if not sgen_df.empty and "p_mw" in sgen_df.columns:
        g = _safe_float(sgen_df["p_mw"].sum())
        total_gen_mw += g
        lines.append(f"    asymmetric_sgen: {g:.4f} MW")
    if not gen_df.empty and "p_mw" in gen_df.columns:
        g = _safe_float(gen_df["p_mw"].sum())
        total_gen_mw += g
        lines.append(f"    asymmetric_gen:  {g:.4f} MW")
    if total_gen_mw > 0:
        lines.append(f"    total generation: {total_gen_mw:.4f} MW")
        findings["total_gen_mw"] = total_gen_mw

    # ── Transformer summary ────────────────────────────────────────────────────
    lines.append(_sub("Transformer summary"))
    for tdf, tname in [(trafo_df, "trafo1ph"), (ct_df, "center_tap_trafo")]:
        if tdf.empty:
            continue
        sn_col = "sn_mva" if "sn_mva" in tdf.columns else None
        vn_col = "vn_kv"  if "vn_kv"  in tdf.columns else None
        n_t = _unique_count(tdf)
        if sn_col:
            total_sn = _safe_float(tdf[sn_col].sum())
            total_trafo_mva += total_sn
            max_sn   = _safe_float(tdf[sn_col].max())
            zero_sn  = (tdf[sn_col] <= 0).sum()
            lines.append(f"    {tname}: count={n_t}  total_sn={total_sn:.4f} MVA  "
                         f"max_sn={max_sn:.4f} MVA")
            if zero_sn:
                warn(f"{tname}: {zero_sn} transformers have sn_mva <= 0.")
        if vn_col:
            bad_vn = (tdf[vn_col] <= 0).sum() if not tdf.empty else 0
            if bad_vn:
                err(f"{tname}: {bad_vn} rows have vn_kv <= 0 — invalid transformer voltage base.")

    findings["total_trafo_mva"] = total_trafo_mva
    if total_trafo_mva > 0 and "total_load_mw" in findings:
        load_pf_assumed = 0.95
        load_apparent   = findings["total_load_mw"] / load_pf_assumed
        if load_apparent > total_trafo_mva * 1.5:
            warn(
                f"Total load apparent power ({load_apparent:.2f} MVA) exceeds 150 % of "
                f"transformer capacity ({total_trafo_mva:.2f} MVA) — likely to cause "
                f"convergence failure."
            )
        elif load_apparent > total_trafo_mva:
            warn(
                f"Total load ({load_apparent:.2f} MVA) exceeds transformer capacity "
                f"({total_trafo_mva:.2f} MVA)."
            )

    # ── Line impedance sanity ─────────────────────────────────────────────────
    lines.append(_sub("Line impedance sanity"))
    if not line_df.empty:
        r_col = "r_ohm_per_km" if "r_ohm_per_km" in line_df.columns else None
        x_col = "x_ohm_per_km" if "x_ohm_per_km" in line_df.columns else None
        l_col = "length_km"     if "length_km"     in line_df.columns else None

        if r_col and x_col and l_col:
            r_tot = (line_df[r_col] * line_df[l_col]).dropna()
            x_tot = (line_df[x_col] * line_df[l_col]).dropna()
            z_mag  = np.sqrt(r_tot**2 + x_tot**2)

            zero_z  = (z_mag < 1e-9).sum()
            tiny_z  = ((z_mag > 0) & (z_mag < 1e-6)).sum()
            large_z = (z_mag > 1e3).sum()

            lines.append(f"    line R*len range: [{r_tot.min():.4e}, {r_tot.max():.4e}] Ω")
            lines.append(f"    line |Z|   range: [{z_mag.min():.4e}, {z_mag.max():.4e}] Ω")

            if zero_z:
                warn(f"{zero_z} lines have zero impedance — may cause singular Y-bus.")
            if tiny_z:
                warn(f"{tiny_z} lines have near-zero impedance (|Z|<1e-6 Ω) — numerical issues likely.")
            if large_z:
                warn(f"{large_z} lines have very large impedance (|Z|>1000 Ω) — "
                     f"check units or disconnected stubs.")

            # Negative resistance
            neg_r = (line_df[r_col] < 0).sum()
            if neg_r:
                err(f"{neg_r} lines have negative r_ohm_per_km — physically invalid.")

            findings["z_min_ohm"] = float(z_mag.min())
            findings["z_max_ohm"] = float(z_mag.max())
        else:
            info("Line table missing r/x/length columns — cannot check impedances.")

        # Zero-length lines
        if l_col:
            zero_len = (line_df[l_col] <= 0).sum()
            if zero_len:
                warn(f"{zero_len} lines have length_km <= 0.")

    # ── Switch / open-loop topology ────────────────────────────────────────────
    lines.append(_sub("Switch / topology"))
    if not sw_df.empty:
        closed_col = "closed" if "closed" in sw_df.columns else None
        if closed_col:
            n_open   = (~sw_df[closed_col].astype(bool)).sum()
            n_closed = sw_df[closed_col].astype(bool).sum()
            lines.append(f"    switches: {n_closed} closed, {n_open} open")
            findings["open_switches"] = int(n_open)
            if n_open > 0:
                info(f"{n_open} open switches — ensure network remains radially connected.")
        else:
            lines.append(f"    {_unique_count(sw_df)} switch rows (closed column not found).")

    # ── NaN / missing value check ─────────────────────────────────────────────
    lines.append(_sub("NaN / missing value check"))
    for tname, tdf in [("line", line_df), ("trafo1ph", trafo_df),
                       ("asymmetric_load", load_df), ("asymmetric_sgen", sgen_df)]:
        if tdf.empty:
            continue
        nan_counts = tdf.isnull().sum()
        bad_cols   = nan_counts[nan_counts > 0]
        if not bad_cols.empty:
            warn(f"{tname} has NaN in columns: "
                 f"{dict(bad_cols[bad_cols > 0].head(8).to_dict())}")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append(_header("AUDIT SUMMARY"))
    lines.append(f"  Errors   : {len(findings['errors'])}")
    lines.append(f"  Warnings : {len(findings['warnings'])}")
    for e in findings["errors"]:
        lines.append(f"    ✗ {e}")
    for w in findings["warnings"]:
        lines.append(f"    ⚠ {w}")
    if not findings["errors"] and not findings["warnings"]:
        lines.append("  ✓ No obvious issues detected — network looks well-formed.")

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.info(output)
    findings["report"] = output
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 2. Post-failure convergence diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_failure(net, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyse a failed (or partial) power flow to surface likely convergence blockers.
    Call this AFTER mc.run_pf() raises or after checking net.converged == False.
    """
    findings: Dict[str, Any] = {"voltage_issues": [], "power_balance": {}}
    lines: List[str] = []

    lines.append(_header("POST-FAILURE CONVERGENCE DIAGNOSTICS"))

    converged = getattr(net, "converged", None)
    model     = getattr(net, "model", None)
    iterations = getattr(model, "iterations", None)
    lines.append(f"  net.converged = {converged}   iterations = {iterations}")

    # ── Internal model state (available even when res_bus is empty) ───────────
    if model is not None:
        lines.append(_sub("Internal model state"))
        y_size     = getattr(model, "y_size",     "?")
        y_isolated = getattr(model, "y_isolated", None)
        y_nonslack = getattr(model, "y_nonslack", None)

        n_isolated = int(len(y_isolated)) if y_isolated is not None else "?"
        n_nonslack = int(len(y_nonslack)) if y_nonslack is not None else "?"
        lines.append(f"    y_size (total terminals) : {y_size}")
        lines.append(f"    y_nonslack (free nodes)  : {n_nonslack}")
        lines.append(f"    y_isolated (NaN, no path to slack): {n_isolated}")
        findings["n_isolated_terminals"] = n_isolated if n_isolated != "?" else 0

        if n_isolated != "?" and n_isolated > 0:
            findings["voltage_issues"].append(
                f"{n_isolated} isolated terminals have no path to the slack bus "
                f"(NaN voltage). These prevent convergence — prune islands first."
            )

        # Residual from last iteration voltages
        if hasattr(model, "E") and hasattr(model, "Y_tot") and y_nonslack is not None:
            try:
                E        = model.E
                Ytot     = model.Y_tot
                y_fix    = model.y_fix
                E_fix    = model.E_fix
                Y1       = Ytot[y_nonslack, :]
                Yll      = Y1[:, y_nonslack]
                Ylg      = Y1[:, y_fix]
                IFIX     = Ylg @ E_fix
                En       = E[y_nonslack]
                residual = Yll @ En + IFIX  # should be ≈ I_corr (correction currents)
                res_norm = float(np.linalg.norm(residual))
                lines.append(f"    ||Y_ll·E_n + Y_lg·E_fix|| (linear residual): {res_norm:.4e}")
                findings["linear_residual_norm"] = res_norm
                if res_norm > 1e3:
                    findings["voltage_issues"].append(
                        f"Very large linear residual ({res_norm:.2e}) → "
                        f"likely near-singular Y-bus or extreme load/impedance mismatch."
                    )
            except Exception as _ex:
                lines.append(f"    Could not compute linear residual: {_ex}")

        # Y-matrix diagonal check
        if hasattr(model, "Yll"):
            try:
                Yll    = model.Yll
                d      = np.abs(Yll.diagonal())
                if len(d):
                    lines.append(f"    Y_ll diagonal: min={d.min():.3e}  max={d.max():.3e}")
                    tiny = int((d < 1e-6).sum())
                    if tiny:
                        findings["voltage_issues"].append(
                            f"{tiny} near-zero Y_ll diagonal entries → singular admittance matrix."
                        )
                        lines.append(f"    ⚠ {tiny} near-zero diagonal entries (< 1e-6)!")
            except Exception:
                pass

        # Flat-start voltage range from model.E
        if hasattr(model, "E"):
            try:
                vmag = np.abs(model.E).flatten()
                vmag_valid = vmag[vmag > 0]
                if len(vmag_valid):
                    lines.append(f"    Last-iteration |E| range: "
                                 f"[{vmag_valid.min():.4f}, {vmag_valid.max():.4f}] pu")
                    n_collapsed = int((vmag_valid < 0.3).sum())
                    if n_collapsed:
                        lines.append(f"    ⚠ {n_collapsed} terminals with |E| < 0.30 pu — voltage collapse!")
                        findings["voltage_issues"].append(
                            f"{n_collapsed} terminal voltages collapsed below 0.30 pu."
                        )
            except Exception:
                pass

    res_bus = _df_safe(net, "res_bus")
    bus_df  = _df_safe(net, "bus")

    if res_bus.empty:
        lines.append("  [INFO] No res_bus results available — solver may not have produced any output.")
        _log_and_print("\n".join(lines), "warning", verbose)
        return findings

    # ── Bus voltage profile ───────────────────────────────────────────────────
    lines.append(_sub("Bus voltage profile"))
    if "vm_pu" in res_bus.columns:
        vm = res_bus["vm_pu"].dropna()
        nan_count = res_bus["vm_pu"].isna().sum()

        lines.append(f"    Total bus-phase rows : {len(res_bus)}")
        lines.append(f"    NaN vm_pu            : {nan_count}")
        lines.append(f"    vm_pu  min={vm.min():.4f}  max={vm.max():.4f}  "
                     f"mean={vm.mean():.4f}  std={vm.std():.4f}")
        findings["vm_min"] = float(vm.min())
        findings["vm_max"] = float(vm.max())
        findings["vm_nan"] = int(nan_count)

        if nan_count > 0:
            findings["voltage_issues"].append(
                f"{nan_count} bus-phase entries have NaN voltage — likely disconnected.")

        # Severe violations
        collapsed = vm[vm < 0.5]
        low       = vm[(vm >= 0.5) & (vm < _LOW_V_ERR)]
        warn_low  = vm[(vm >= _LOW_V_ERR) & (vm < _LOW_V_WARN)]
        high_v    = vm[vm > _HIGH_V_ERR]

        for label, subset, threshold in [
            ("COLLAPSED (<0.50 pu)", collapsed, 0.5),
            ("SEVERE LOW (<0.80 pu)", low,       0.80),
            ("OVER-VOLTAGE (>1.20 pu)", high_v,  1.20),
        ]:
            if not subset.empty:
                lines.append(f"    {label}: {len(subset)} bus-phase entries")
                sample_idx = subset.nsmallest(5).index if threshold < 1.0 else subset.nlargest(5).index
                for idx in sample_idx:
                    bname = _bus_name_from_idx(net, idx)
                    lines.append(f"      idx={idx}  name={bname}  vm_pu={subset[idx]:.4f}")
                findings["voltage_issues"].append(f"{label}: {len(subset)} entries")

        if not warn_low.empty:
            lines.append(f"    LOW VOLTAGE ({_LOW_V_ERR:.0%}–{_LOW_V_WARN:.0%} pu): "
                         f"{len(warn_low)} bus-phase entries")

    # ── Phase unbalance ───────────────────────────────────────────────────────
    lines.append(_sub("Phase unbalance (worst 10 buses)"))
    if "vm_pu" in res_bus.columns and isinstance(res_bus.index, pd.MultiIndex):
        try:
            vm_wide = res_bus["vm_pu"].unstack(level=1)  # buses × phases
            ph_cols = [c for c in vm_wide.columns if c in (1, 2, 3)]
            if len(ph_cols) >= 2:
                vm_abc = vm_wide[ph_cols].dropna(how="all")
                vm_mean = vm_abc.mean(axis=1)
                vm_dev  = vm_abc.subtract(vm_mean, axis=0).abs().max(axis=1)
                unbal_pct = (vm_dev / vm_mean.replace(0, np.nan) * 100).dropna()
                worst = unbal_pct.nlargest(10)
                lines.append(f"    mean unbalance = {unbal_pct.mean():.2f}%  "
                              f"max = {unbal_pct.max():.2f}%")
                for bus_idx, val in worst.items():
                    bname = _bus_name_from_idx(net, (bus_idx, 1))
                    lines.append(f"      bus {bus_idx} ({bname})  unbalance={val:.2f}%")
                findings["max_unbalance_pct"] = float(unbal_pct.max())
                if unbal_pct.max() > 10:
                    findings["voltage_issues"].append(
                        f"Phase unbalance > 10% at {(unbal_pct > 10).sum()} buses.")
        except Exception as ex:
            lines.append(f"    Could not compute unbalance: {ex}")

    # ── Line loading ──────────────────────────────────────────────────────────
    lines.append(_sub("Line loading (top 10 overloaded)"))
    res_line = _df_safe(net, "res_line")
    line_df  = _df_safe(net, "line")
    if not res_line.empty:
        if "i_ka" in res_line.columns:
            i_actual = res_line["i_ka"].dropna()
            lines.append(f"    i_ka range: [{i_actual.min():.4e}, {i_actual.max():.4e}] kA")
            if not line_df.empty and "max_i_ka" in line_df.columns:
                try:
                    loading_pct = (i_actual / line_df["max_i_ka"].reindex(
                        i_actual.index, method=None
                    ) * 100).dropna()
                    overloaded = loading_pct[loading_pct > 100]
                    if not overloaded.empty:
                        lines.append(f"    OVERLOADED lines: {len(overloaded)}")
                        for idx, pct in overloaded.nlargest(10).items():
                            lines.append(f"      idx={idx}  loading={pct:.1f}%")
                        findings["overloaded_lines"] = int(len(overloaded))
                except Exception:
                    pass
            # Flag extreme currents regardless of rating
            extreme_i = i_actual[i_actual > 10]
            if not extreme_i.empty:
                lines.append(f"    Lines with i_ka > 10 kA: {len(extreme_i)} "
                             f"(max={extreme_i.max():.2f} kA) — check impedance scaling.")

    # ── Power balance ─────────────────────────────────────────────────────────
    lines.append(_sub("Power balance check"))
    try:
        res_eg = _df_safe(net, "res_ext_grid")
        load_df = _df_safe(net, "asymmetric_load")
        sgen_df = _df_safe(net, "asymmetric_sgen")

        p_load = _safe_float(load_df["p_mw"].sum()) if "p_mw" in load_df.columns else 0.0
        p_gen  = _safe_float(sgen_df["p_mw"].sum()) if "p_mw" in sgen_df.columns else 0.0

        if not res_eg.empty and "p_mw" in res_eg.columns:
            p_eg = _safe_float(res_eg["p_mw"].sum())
            lines.append(f"    ext_grid injection : {p_eg:.4f} MW")
            lines.append(f"    load consumption   : {p_load:.4f} MW")
            lines.append(f"    sgen generation    : {p_gen:.4f} MW")
            balance = p_eg + p_gen - p_load
            lines.append(f"    apparent imbalance : {balance:.4f} MW")
            findings["power_balance"] = {
                "ext_grid_mw": p_eg,
                "load_mw":     p_load,
                "sgen_mw":     p_gen,
                "imbalance_mw": balance,
            }
        else:
            lines.append(f"    load={p_load:.4f} MW  sgen={p_gen:.4f} MW  "
                         f"(no res_ext_grid available)")
    except Exception as ex:
        lines.append(f"    Power balance check failed: {ex}")

    # ── Likely root causes ────────────────────────────────────────────────────
    lines.append(_sub("Likely convergence blockers"))
    causes = _infer_causes(findings)
    if causes:
        for i, c in enumerate(causes, 1):
            lines.append(f"    {i}. {c}")
    else:
        lines.append("    No obvious single cause identified — try increasing MaxIter or relaxing tolerance.")

    findings["likely_causes"] = causes

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.warning(output)
    findings["report"] = output
    return findings


def _infer_causes(findings: Dict) -> List[str]:
    causes = []
    vm_min = findings.get("vm_min", 1.0)
    vm_nan = findings.get("vm_nan", 0)
    z_min  = findings.get("z_min_ohm", 1.0)

    if vm_nan > 0:
        causes.append(
            f"{vm_nan} NaN voltages → disconnected buses or islands not tied to the slack bus. "
            f"Use trace_upstream_phase_breaks() or contract_closed_switches() to prune islands."
        )
    if vm_min < 0.5:
        causes.append(
            f"Voltage collapse: vm_min={vm_min:.3f} pu. "
            f"Possible load > transformer capacity, or extreme R/X ratio causing divergence. "
            f"Try reducing total load or relaxing tolerance (tol_vmag_pu=1e-2)."
        )
    if findings.get("overloaded_lines", 0):
        causes.append(
            f"{findings['overloaded_lines']} lines overloaded → excessive current → "
            f"voltage drop amplification per iteration."
        )
    if findings.get("max_unbalance_pct", 0) > 15:
        causes.append(
            f"Severe phase unbalance ({findings['max_unbalance_pct']:.1f}%) → neutral/ground currents "
            f"may be destabilising the iteration."
        )
    if any("zero impedance" in e.lower() or "singular" in e.lower()
           for e in findings.get("errors", [])):
        causes.append(
            "Zero-impedance lines detected → singular Y-bus matrix → solver cannot proceed."
        )
    return causes


def _bus_name_from_idx(net, idx) -> str:
    try:
        bus_df = _df_safe(net, "bus")
        if bus_df.empty or "name" not in bus_df.columns:
            return "?"
        if isinstance(idx, tuple):
            bus_idx = idx[0]
        else:
            bus_idx = idx
        if isinstance(bus_df.index, pd.MultiIndex):
            sub = bus_df.xs(bus_idx, level=0)
            if not sub.empty:
                return str(sub["name"].iloc[0])
        else:
            return str(bus_df.loc[bus_idx, "name"])
    except Exception:
        pass
    return "?"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Post-solve result summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize_pf_results(net, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Print a human-readable summary of a completed power flow.
    Call after mc.run_pf() succeeds (net.converged == True).
    """
    findings: Dict[str, Any] = {}
    lines: List[str] = []

    lines.append(_header("POWER FLOW RESULT SUMMARY"))

    iterations = getattr(getattr(net, "model", None), "iterations", None)
    lines.append(f"  converged={getattr(net, 'converged', '?')}   iterations={iterations}")

    res_bus = _df_safe(net, "res_bus")
    if res_bus.empty:
        lines.append("  [INFO] No res_bus — nothing to summarize.")
        _log_and_print("\n".join(lines), "info", verbose)
        return findings

    # ── Voltage profile ───────────────────────────────────────────────────────
    lines.append(_sub("Voltage profile (per phase)"))
    if "vm_pu" in res_bus.columns and isinstance(res_bus.index, pd.MultiIndex):
        try:
            vm_wide = res_bus["vm_pu"].unstack(level=1)
            for ph in sorted(vm_wide.columns):
                col = vm_wide[ph].dropna()
                if col.empty:
                    continue
                label = _PHASE_LABELS.get(ph, str(ph))
                lines.append(f"    Phase {label}: min={col.min():.4f}  max={col.max():.4f}  "
                             f"mean={col.mean():.4f}  std={col.std():.4f} pu")
                findings[f"vm_min_ph{label}"] = float(col.min())
                findings[f"vm_max_ph{label}"] = float(col.max())

            # Worst buses per phase
            lines.append(_sub("Worst (lowest) voltage buses — top 5 per phase"))
            for ph in sorted(vm_wide.columns):
                col = vm_wide[ph].dropna()
                if col.empty:
                    continue
                label = _PHASE_LABELS.get(ph, str(ph))
                worst = col.nsmallest(5)
                for bus_idx, vm in worst.items():
                    bname = _bus_name_from_idx(net, (bus_idx, ph))
                    lines.append(f"    Ph{label}  bus {bus_idx} ({bname})  vm={vm:.4f} pu")
        except Exception as ex:
            lines.append(f"    Could not build voltage profile: {ex}")

    # ── Phase angle profile ───────────────────────────────────────────────────
    if "va_degree" in res_bus.columns and isinstance(res_bus.index, pd.MultiIndex):
        try:
            va_wide = res_bus["va_degree"].unstack(level=1)
            lines.append(_sub("Voltage angle profile (per phase)"))
            for ph in sorted(va_wide.columns):
                col = va_wide[ph].dropna()
                if col.empty:
                    continue
                label = _PHASE_LABELS.get(ph, str(ph))
                lines.append(f"    Phase {label}: min={col.min():.2f}°  max={col.max():.2f}°  "
                             f"range={col.max()-col.min():.2f}°")
        except Exception:
            pass

    # ── Ext grid output ───────────────────────────────────────────────────────
    lines.append(_sub("Ext-grid (source) output"))
    res_eg = _df_safe(net, "res_ext_grid")
    if not res_eg.empty:
        for col in ["p_mw", "q_mvar"]:
            if col in res_eg.columns:
                tot = _safe_float(res_eg[col].sum())
                lines.append(f"    total {col}: {tot:.4f}")

    # ── Line losses ───────────────────────────────────────────────────────────
    lines.append(_sub("System losses"))
    res_line = _df_safe(net, "res_line")
    if not res_line.empty and "pl_mw" in res_line.columns:
        total_loss_p = _safe_float(res_line["pl_mw"].sum())
        total_loss_q = _safe_float(res_line["ql_mvar"].sum()) if "ql_mvar" in res_line.columns else 0.0
        lines.append(f"    line losses: {total_loss_p:.4f} MW + j{total_loss_q:.4f} MVAR")
        findings["losses_mw"] = total_loss_p

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.info(output)
    findings["report"] = output
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verbose run_pf wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_pf_verbose(
    net,
    *,
    tol_vmag_pu: float = 1e-3,
    tol_vang_rad: float = 1e-3,
    MaxIter: int = 100,
    run_control: bool = True,
    verbose: bool = True,
    audit: bool = True,
    reraise: bool = False,
    **kwargs,
) -> bool:
    """
    Wrapper around mc.run_pf() with full pre/post diagnostic logging.

    Returns True if converged, False otherwise.

    Key differences from mc.run_pf():
      - reraise=False by default: the LoadflowNotConverged exception is caught
        so you always get the diagnostic summary, even on failure.
      - When the solver fails, net.res_bus is force-populated from net.model.E
        (the last-iteration voltages) so diagnose_failure() has data to work with.
      - Set reraise=True to re-raise the exception after printing diagnostics
        (matches legacy behaviour if you need it in a try/except chain).

    Usage (notebook):
        from pf_diagnostics import run_pf_verbose
        converged = run_pf_verbose(mc_net, tol_vmag_pu=1e-3, MaxIter=50)
        # → always prints audit + diagnostics; never raises by default

    Drop-in for mc.run_pf with diagnostics but same raise behaviour:
        run_pf_verbose(net, tol_vmag_pu=1e-3, MaxIter=50, reraise=True)
    """
    try:
        import multiconductor as mc
    except ImportError as exc:
        raise ImportError("multiconductor package not found") from exc

    if audit:
        audit_network(net, verbose=verbose)

    if verbose:
        print(f"\n[pf_diagnostics] Calling mc.run_pf("
              f"tol_vmag_pu={tol_vmag_pu}, tol_vang_rad={tol_vang_rad}, "
              f"MaxIter={MaxIter}, run_control={run_control})")

    t0 = time.perf_counter()
    converged = False
    exc_info  = None
    try:
        mc.run_pf(
            net,
            tol_vmag_pu=tol_vmag_pu,
            tol_vang_rad=tol_vang_rad,
            MaxIter=MaxIter,
            run_control=run_control,
            **kwargs,
        )
        converged = bool(getattr(net, "converged", True))
    except Exception as exc:
        exc_info = exc
        converged = False
        if verbose:
            print(f"[pf_diagnostics] mc.run_pf raised: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0
    iterations = getattr(getattr(net, "model", None), "iterations", "?")

    if verbose:
        status = "CONVERGED" if converged else "DID NOT CONVERGE"
        print(f"[pf_diagnostics] {status}  iterations={iterations}  elapsed={elapsed:.2f}s")

    if not converged:
        # Force-populate net.res_bus from net.model.E (last-iteration voltages).
        # The solver sets model.E before raising, but never calls _bus_results_pf.
        # Without this step diagnose_failure() would have no voltage data.
        _force_populate_res_bus(net, verbose=verbose)
        diagnose_failure(net, verbose=verbose)
    else:
        summarize_pf_results(net, verbose=verbose)

    if exc_info and reraise:
        raise exc_info

    return converged


def _force_populate_res_bus(net, *, verbose: bool = True) -> None:
    """
    Populate net.res_bus from net.model.E when the solver raised before
    _bus_results_pf() could run.  Safe to call even if results already exist.
    """
    model = getattr(net, "model", None)
    if model is None or not hasattr(model, "E"):
        return
    if hasattr(net, "res_bus") and not _df_safe(net, "res_bus").empty:
        return  # already populated by the solver

    try:
        from multiconductor.pycci.pf_results import _bus_results_pf
        _bus_results_pf(net)
        if verbose:
            vm = _df_safe(net, "res_bus").get("vm_pu", pd.Series(dtype=float)).dropna()
            print(f"[pf_diagnostics] Extracted last-iteration voltages from net.model.E "
                  f"({len(vm)} bus-phase rows, vm_pu range "
                  f"[{vm.min():.4f}, {vm.max():.4f}])")
    except Exception as ex:
        if verbose:
            print(f"[pf_diagnostics] Could not extract voltages from model state: {ex}")
        # Fallback: build a minimal res_bus directly from model.E
        try:
            _force_populate_res_bus_fallback(net)
        except Exception:
            pass


def _force_populate_res_bus_fallback(net) -> None:
    """Minimal fallback: build res_bus from model.E + terminal_to_y_lookup."""
    model = net.model
    E = model.E
    bus_df = _df_safe(net, "bus")
    if bus_df.empty or not isinstance(bus_df.index, pd.MultiIndex):
        return

    phase_vals = pd.to_numeric(bus_df.index.get_level_values(1), errors="coerce")
    res_idx = bus_df.index[phase_vals != 0]

    ypos = model.terminal_to_y_lookup[[b * 4 + p for b, p in res_idx]]
    r = E[ypos].flatten()
    r[ypos == -1] = np.nan

    net["res_bus"] = pd.DataFrame(
        {"vm_pu": np.abs(r), "va_degree": np.angle(r) * 180.0 / np.pi},
        index=res_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FBSW per-iteration diagnostic hook
# ─────────────────────────────────────────────────────────────────────────────

def fbsw_iteration_report(
    iteration: int,
    bfs_order,
    V_pu: dict,
    V_prev: dict,
    I_branch: dict,
    loads_by_bus: dict,
    n_phases: int,
    I_BASE: float,
    *,
    verbose: bool = True,
    top_n: int = 3,
) -> float:
    """
    Called after each FBSW iteration.  Returns max |ΔV|.
    Prints per-phase breakdown and worst-offender buses when verbose=True.
    """
    PHASE_LBLS = ["A", "B", "C", "N"] if n_phases == 4 else [f"Ph{i}" for i in range(n_phases)]

    # Per-bus, per-phase |ΔV|
    dv_all = {}
    for bus in bfs_order[1:]:
        dv_all[bus] = np.abs(V_pu[bus] - V_prev[bus])

    max_dv      = max(dv_all[b].max() for b in dv_all)
    ph_max_dv   = np.array([max(dv_all[b][p] for b in dv_all) for p in range(n_phases)])

    # Worst bus per phase
    worst_bus_ph = []
    for p in range(n_phases):
        worst_b = max(dv_all.keys(), key=lambda b: dv_all[b][p])
        worst_bus_ph.append((worst_b, dv_all[worst_b][p]))

    # Minimum voltage across all non-root buses
    v_mags = {bus: np.abs(V_pu[bus]) for bus in bfs_order[1:]}
    min_v_phase = np.array([
        min(v_mags[b][p] for b in v_mags) for p in range(n_phases)
    ])

    # Source current (root's children, first branch)
    root = bfs_order[0]
    root_children = [b for b in bfs_order[1:] if I_branch.get(b) is not None]
    src_I_pu = I_branch[root_children[0]] if root_children else np.zeros(n_phases)
    src_I_A  = np.abs(src_I_pu) * I_BASE

    lines = [f"  Iter {iteration:3d}  │  max|ΔV|={max_dv:.4e} pu"]
    if verbose:
        # Per-phase breakdown
        ph_str = "  ".join(
            f"{PHASE_LBLS[p]}:{ph_max_dv[p]:.3e}" for p in range(n_phases)
        )
        lines.append(f"           │  per-phase ΔV: {ph_str}")

        # Worst buses
        for p in range(n_phases):
            wb, wdv = worst_bus_ph[p]
            lines.append(f"           │  worst Ph{PHASE_LBLS[p]}: bus={wb}  |ΔV|={wdv:.4e} pu")

        # Min voltages
        mv_str = "  ".join(
            f"{PHASE_LBLS[p]}:{min_v_phase[p]:.4f}" for p in range(n_phases)
        )
        lines.append(f"           │  min|V| (pu): {mv_str}")

        # Source currents
        si_str = "  ".join(
            f"{PHASE_LBLS[p]}:{src_I_A[p]:.1f}A" for p in range(n_phases)
        )
        lines.append(f"           │  source I:    {si_str}")

        # Near-zero voltage warning
        collapse_phases = [PHASE_LBLS[p] for p in range(n_phases) if min_v_phase[p] < 0.5]
        if collapse_phases:
            lines.append(f"           │  ⚠ VOLTAGE COLLAPSE risk on phases {collapse_phases}!")

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.debug(output)
    return max_dv


def fbsw_pre_solve_report(
    buses_df,
    branches_df,
    loads_by_bus: dict,
    n_phases: int,
    V_BASE_LN: float,
    S_BASE: float,
    Z_BASE: float,
    I_BASE: float,
    MAX_ITER: int,
    TOL: float,
    *,
    verbose: bool = True,
) -> None:
    """Print a detailed pre-solve diagnostic for the FBSW solver."""
    PHASE_LBLS = ["A", "B", "C", "N"] if n_phases == 4 else [f"Ph{i}" for i in range(n_phases)]
    lines = []
    lines.append(_header(f"{n_phases}-Phase FBSW Load Flow Solver"))
    lines.append(f"  Network : {len(buses_df)} buses,  {len(branches_df)} branches")
    lines.append(f"  Vbase   : {V_BASE_LN:.2f} V (LN)  |  {V_BASE_LN*np.sqrt(3)/1e3:.3f} kV (LL)")
    lines.append(f"  Sbase   : {S_BASE/1e6:.2f} MVA  |  Zbase : {Z_BASE:.4f} Ω  |  Ibase : {I_BASE:.2f} A")
    lines.append(f"  Tol     : {TOL:.1e} pu  |  MaxIter : {MAX_ITER}")

    # Load summary
    lines.append(_sub("Load summary (pu)"))
    total_S = np.zeros(n_phases, dtype=complex)
    for bus, S in loads_by_bus.items():
        total_S += S[:n_phases]
        s_mag = np.abs(S[:n_phases])
        if s_mag.max() > 5.0:
            lines.append(f"    ⚠ LARGE LOAD at {bus}: |S|={s_mag} pu — may cause divergence.")
    lines.append(f"  Total  : P={total_S.real * S_BASE / 1e6:.4f} MW  "
                 f"Q={total_S.imag * S_BASE / 1e6:.4f} MVAR (summed across phases)")

    # Branch impedance summary
    lines.append(_sub("Branch impedance summary (pu)"))
    z_mags = []
    for _, row in branches_df.iterrows():
        Z = row["Z_pu"]
        z_mags.append(np.abs(Z).max())
        diag = np.abs(np.diag(Z))
        if diag.min() < 1e-9:
            lines.append(f"    ⚠ Near-zero impedance on branch {row['from_bus']}→{row['to_bus']}")
        cond = np.linalg.cond(Z) if Z.shape[0] > 1 else 1.0
        if cond > 1e6:
            lines.append(f"    ⚠ Ill-conditioned Z matrix on {row['from_bus']}→{row['to_bus']} "
                         f"(cond={cond:.2e})")
    if z_mags:
        lines.append(f"  |Z_pu| range: [{min(z_mags):.4e}, {max(z_mags):.4e}]")

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.info(output)


def fbsw_post_solve_report(
    V_pu: dict,
    I_branch: dict,
    branch_map: dict,
    bfs_order,
    conv_history: list,
    n_phases: int,
    I_BASE: float,
    S_BASE: float,
    converged: bool,
    *,
    verbose: bool = True,
) -> None:
    """Print convergence history analysis and voltage/current summary after FBSW."""
    PHASE_LBLS = ["A", "B", "C", "N"] if n_phases == 4 else [f"Ph{i}" for i in range(n_phases)]
    lines = []

    lines.append(_sub("Convergence history"))
    lines.append(f"  Converged: {converged}  after {len(conv_history)} iterations")
    if conv_history:
        lines.append(f"  Initial  |ΔV|: {conv_history[0]:.4e} pu")
        lines.append(f"  Final    |ΔV|: {conv_history[-1]:.4e} pu")
        if len(conv_history) >= 3:
            rates = [conv_history[i+1] / max(conv_history[i], 1e-16)
                     for i in range(len(conv_history) - 1)]
            lines.append(f"  Conv. rate (ΔV_n+1/ΔV_n): "
                         f"min={min(rates):.3f}  max={max(rates):.3f}  "
                         f"last={rates[-1]:.3f}")
            if rates[-1] >= 1.0:
                lines.append(f"  ⚠ Diverging! Last rate={rates[-1]:.3f} ≥ 1.0")
            elif rates[-1] >= 0.9:
                lines.append(f"  ⚠ Slow convergence — rate={rates[-1]:.3f}. "
                             f"Consider increasing MaxIter or loosening tolerance.")

    if not converged:
        lines.append(_sub("Divergence analysis"))
        worst_buses = sorted(
            bfs_order[1:],
            key=lambda b: np.abs(V_pu[b]).max(),
            reverse=False,
        )[:5]
        lines.append("  Lowest voltage buses (potential collapse):")
        for bus in worst_buses:
            mags = [f"{PHASE_LBLS[p]}:{abs(V_pu[bus][p]):.4f}" for p in range(n_phases)]
            lines.append(f"    {bus}: " + "  ".join(mags) + " pu")

        # Highest current branches
        if I_branch:
            branch_i_max = {
                b: np.abs(I_branch[b]).max() * I_BASE
                for b in bfs_order[1:] if I_branch.get(b) is not None
            }
            hot_branches = sorted(branch_i_max, key=branch_i_max.get, reverse=True)[:5]
            lines.append("  Highest current branches:")
            for b in hot_branches:
                fb = branch_map[b]["from_bus"]
                lines.append(f"    {fb}→{b}: |I|_max={branch_i_max[b]:.2f} A")

    # Voltage violations
    lines.append(_sub("Voltage violation summary"))
    below_95 = below_90 = above_105 = 0
    for bus in bfs_order[1:]:
        for p in range(min(3, n_phases)):
            vm = abs(V_pu[bus][p])
            if vm < 0.90:
                below_90 += 1
            elif vm < 0.95:
                below_95 += 1
            elif vm > 1.05:
                above_105 += 1
    lines.append(f"  |V| < 0.90 pu : {below_90} bus-phase(s)")
    lines.append(f"  |V| < 0.95 pu : {below_95} bus-phase(s)")
    lines.append(f"  |V| > 1.05 pu : {above_105} bus-phase(s)")

    output = "\n".join(lines)
    if verbose:
        print(output)
    logger.info(output)
