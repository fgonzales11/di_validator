#!/usr/bin/env python3
import copy
import math
import os
import pickle
import sys

import numpy as np
import pandapower as pp
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_PKL = os.path.join(PROJECT_ROOT, "public/networks_demo/barcelona_geo.pkl")


def _convert_buses(mc_net, pp_net):
    bus_index_map = {}
    for bus_idx in mc_net.bus.index.get_level_values(0).unique():
        rows = mc_net.bus.loc[bus_idx]
        vn_kv = float(rows["vn_kv"].iloc[0])
        raw_name = rows["name"].iloc[0]
        name = str(raw_name) if raw_name is not None else ""
        pp_bus = pp.create_bus(pp_net, vn_kv=vn_kv, name=name)
        bus_index_map[int(bus_idx)] = pp_bus
    return bus_index_map


def _add_ext_grid(mc_net, pp_net, bus_index_map):
    if hasattr(mc_net, "ext_grid_sequence") and len(mc_net.ext_grid_sequence) > 0:
        eg = mc_net.ext_grid_sequence.reset_index()
        pos = eg[eg["sequence"] == 1]
        if len(pos) == 0:
            pos = eg.iloc[[0]]
        row = pos.iloc[0]
        pp.create_ext_grid(
            pp_net,
            bus=bus_index_map[int(row["bus"])],
            vm_pu=float(row["vm_pu"]),
            va_degree=float(row["va_degree"]),
        )
    elif len(mc_net.ext_grid) > 0:
        eg = mc_net.ext_grid.reset_index()
        ph1 = eg[eg["from_phase"] == 1]
        if len(ph1) == 0:
            ph1 = eg.iloc[[0]]
        row = ph1.iloc[0]
        pp.create_ext_grid(
            pp_net,
            bus=bus_index_map[int(row["bus"])],
            vm_pu=float(row["vm_pu"]),
            va_degree=float(row["va_degree"]),
        )
    else:
        raise ValueError(
            "mc_net has no external grid — cannot define a slack bus"
            " (both ext_grid_sequence and ext_grid are empty)"
        )


def _get_pos_seq_params(std_types, std_type_name, model_type, f_hz):
    if model_type == "sequence":
        p = std_types["sequence"][std_type_name]
        return (
            float(p["r1_ohm_per_km"]),
            float(p["x1_ohm_per_km"]),
            float(p.get("c1_nf_per_km", 0.0)),
            float(p.get("max_i_ka", 1.0)),
        )
    elif model_type == "matrix":
        p = std_types["matrix"][std_type_name]

        def _diag(key, idx, fallback=0.0):
            v = p.get(key)
            if v is None:
                return fallback
            try:
                return float(v[idx])
            except (TypeError, IndexError):
                return float(v) if v is not None else fallback

        r1 = float(np.mean([_diag("r_1_ohm_per_km", 0), _diag("r_2_ohm_per_km", 1), _diag("r_3_ohm_per_km", 2)]))
        x1 = float(np.mean([_diag("x_1_ohm_per_km", 0), _diag("x_2_ohm_per_km", 1), _diag("x_3_ohm_per_km", 2)]))
        b1 = float(np.mean([_diag("b_1_us_per_km", 0), _diag("b_2_us_per_km", 1), _diag("b_3_us_per_km", 2)]))
        c1 = max(0.0, b1 * 1000.0 / (2.0 * math.pi * f_hz))
        raw = p.get("max_i_ka", 1.0)
        max_i_ka = float(raw[0]) if hasattr(raw, "__getitem__") else float(raw)
        return r1, x1, c1, max_i_ka
    else:
        print(f"Warning: unknown model_type '{model_type}' for '{std_type_name}' — using fallback r=0.1, x=0.3")
        return 0.1, 0.3, 0.0, 1.0


def _add_lines(mc_net, pp_net, bus_index_map):
    if len(mc_net.line) == 0:
        return
    lines = mc_net.line.reset_index()
    for line_idx, grp in lines.groupby("index"):
        row = grp.iloc[0]
        r1, x1, c1, max_i_ka = _get_pos_seq_params(
            mc_net.std_types, row["std_type"], row["model_type"], mc_net.f_hz
        )
        pp.create_line_from_parameters(
            pp_net,
            from_bus=bus_index_map[int(row["from_bus"])],
            to_bus=bus_index_map[int(row["to_bus"])],
            length_km=float(row["length_km"]),
            r_ohm_per_km=r1,
            x_ohm_per_km=x1,
            c_nf_per_km=c1,
            max_i_ka=max_i_ka,
            name=str(line_idx),
        )


def _add_switches(mc_net, pp_net, bus_index_map):
    if not hasattr(mc_net, "switch") or len(mc_net.switch) == 0:
        return
    switches = mc_net.switch.reset_index()
    for switch_idx, grp in switches.groupby("index"):
        row = grp.iloc[0]
        if str(row.get("et", "")).strip().lower() != "b":
            continue

        from_bus = int(row["bus"])
        to_bus = int(row["element"])
        if from_bus not in bus_index_map or to_bus not in bus_index_map:
            continue

        kwargs = {}
        switch_type = row.get("type", None)
        if pd.notna(switch_type):
            kwargs["type"] = str(switch_type)

        r_ohm = row.get("r_ohm", None)
        if pd.notna(r_ohm):
            kwargs["z_ohm"] = float(r_ohm)

        switch_name = row.get("name", None)
        pp.create_switch(
            pp_net,
            bus=bus_index_map[from_bus],
            element=bus_index_map[to_bus],
            et="b",
            closed=bool(row.get("closed", True)),
            name=str(switch_name) if pd.notna(switch_name) else str(switch_idx),
            **kwargs,
        )


def _add_loads(mc_net, pp_net, bus_index_map):
    if len(mc_net.asymmetric_load) == 0:
        return
    loads = mc_net.asymmetric_load.reset_index()
    loads = loads[loads["from_phase"].isin([1, 2, 3])]
    for load_idx, grp in loads.groupby("index"):
        row = grp.iloc[0]
        pp.create_load(
            pp_net,
            bus=bus_index_map[int(row["bus"])],
            p_mw=float(grp["p_mw"].sum()),
            q_mvar=float(grp["q_mvar"].sum()),
            name=str(load_idx),
        )


def _add_sgens(mc_net, pp_net, bus_index_map):
    if len(mc_net.asymmetric_sgen) == 0:
        return
    sgens = mc_net.asymmetric_sgen.reset_index()
    sgens = sgens[sgens["from_phase"].isin([1, 2, 3])]
    for sgen_idx, grp in sgens.groupby("index"):
        row = grp.iloc[0]
        pp.create_sgen(
            pp_net,
            bus=bus_index_map[int(row["bus"])],
            p_mw=float(grp["p_mw"].sum()),
            q_mvar=float(grp["q_mvar"].sum()),
            name=str(sgen_idx),
        )


def _add_shunts(mc_net, pp_net, bus_index_map):
    if not hasattr(mc_net, "asymmetric_shunt") or len(mc_net.asymmetric_shunt) == 0:
        return
    shunts = mc_net.asymmetric_shunt.reset_index()
    shunts = shunts[shunts["from_phase"].isin([1, 2, 3])]
    for shunt_idx, grp in shunts.groupby("index"):
        row = grp.iloc[0]
        kwargs = {}
        vn_kv = row.get("vn_kv", None)
        if vn_kv is not None and not (isinstance(vn_kv, float) and np.isnan(vn_kv)):
            kwargs["vn_kv"] = float(vn_kv)
        pp.create_shunt(
            pp_net,
            bus=bus_index_map[int(row["bus"])],
            p_mw=float(grp["p_mw"].sum()),
            q_mvar=float(grp["q_mvar"].sum()),
            name=str(shunt_idx),
            **kwargs,
        )


def _add_transformers(mc_net, pp_net, bus_index_map):
    if not hasattr(mc_net, "trafo1ph") or len(mc_net.trafo1ph) == 0:
        return
    trafos = mc_net.trafo1ph.reset_index()
    for trafo_idx, grp in trafos.groupby("index"):
        buses = grp["bus"].unique()
        if len(buses) != 2:
            continue
        vn0 = float(mc_net.bus.loc[int(buses[0])]["vn_kv"].iloc[0])
        vn1 = float(mc_net.bus.loc[int(buses[1])]["vn_kv"].iloc[0])
        if vn0 >= vn1:
            hv_mc, lv_mc, vn_hv, vn_lv = int(buses[0]), int(buses[1]), vn0, vn1
        else:
            hv_mc, lv_mc, vn_hv, vn_lv = int(buses[1]), int(buses[0]), vn1, vn0
        row = grp.iloc[0]
        n_circuits = int(grp["circuit"].nunique())
        n_rows = len(grp)
        sn_mva = float(row["sn_mva"]) * n_circuits
        vk_percent = float(row["vk_percent"]) * 2
        vkr_percent = float(row["vkr_percent"]) * 2
        pfe_kw = float(row["pfe_kw"]) * n_rows  # n_rows = 2 × n_circuits: pfe_kw stored per winding-end per circuit
        i0_percent = float(row["i0_percent"]) * 2
        pp.create_transformer_from_parameters(
            pp_net,
            hv_bus=bus_index_map[hv_mc],
            lv_bus=bus_index_map[lv_mc],
            sn_mva=sn_mva,
            vn_hv_kv=vn_hv,
            vn_lv_kv=vn_lv,
            vk_percent=max(vk_percent, 0.001),
            vkr_percent=vkr_percent,
            pfe_kw=pfe_kw,
            i0_percent=max(i0_percent, 1e-6),
            name=str(trafo_idx),
        )


def mc_to_balanced_pp(mc_net):
    pp_net = pp.create_empty_network(f_hz=mc_net.f_hz, sn_mva=mc_net.sn_mva)
    bus_index_map = _convert_buses(mc_net, pp_net)
    _add_ext_grid(mc_net, pp_net, bus_index_map)
    _add_lines(mc_net, pp_net, bus_index_map)
    _add_switches(mc_net, pp_net, bus_index_map)
    _add_loads(mc_net, pp_net, bus_index_map)
    _add_sgens(mc_net, pp_net, bus_index_map)
    _add_shunts(mc_net, pp_net, bus_index_map)
    _add_transformers(mc_net, pp_net, bus_index_map)
    return pp_net


def run_balanced_loadflow(pkl_path=DEFAULT_PKL):
    mc_net = pickle.load(open(pkl_path, "rb"))
    pp_net = mc_to_balanced_pp(mc_net)
    try:
        pp.runpp(pp_net)
        converged = True
    except pp.powerflow.LoadflowNotConverged:
        converged = False
    return converged, pp_net


def run_pp_loadflow(mc_net):
    pp_net = mc_to_balanced_pp(mc_net)
    try:
        pp.runpp(pp_net)
        converged = True
    except pp.powerflow.LoadflowNotConverged:
        converged = False
    return converged, pp_net

def _normalize_ts_frame(ts_frame):
    if ts_frame is None:
        return pd.DataFrame(columns=["REPORTED_DTTM", "MEASURE_VALUE"])
    if not isinstance(ts_frame, pd.DataFrame):
        raise TypeError("ts_frame must be a pandas DataFrame")
    if ts_frame.empty:
        return pd.DataFrame(columns=["REPORTED_DTTM", "MEASURE_VALUE"])

    if "REPORTED_DTTM" not in ts_frame.columns:
        raise KeyError("ts_frame must include REPORTED_DTTM")

    value_col = None
    for candidate in ("MEASURE_VALUE", "MEASURED_VALUE"):
        if candidate in ts_frame.columns:
            value_col = candidate
            break
    if value_col is None:
        raise KeyError("ts_frame must include MEASURE_VALUE or MEASURED_VALUE")

    normalized = ts_frame[["REPORTED_DTTM", value_col]].copy()
    normalized["REPORTED_DTTM"] = pd.to_datetime(normalized["REPORTED_DTTM"], errors="coerce")
    normalized[value_col] = pd.to_numeric(normalized[value_col], errors="coerce").fillna(0.0)
    normalized = normalized.dropna(subset=["REPORTED_DTTM"])
    normalized = normalized.sort_values("REPORTED_DTTM").reset_index(drop=True)
    return normalized.rename(columns={value_col: "MEASURE_VALUE"})


def _apply_total_load_target(pp_net, base_p, base_q, target_p_mw):
    if len(pp_net.load) == 0:
        return

    base_total = float(base_p.sum())
    target = float(target_p_mw)

    if abs(base_total) < 1e-12:
        pp_net.load.loc[:, "p_mw"] = 0.0
        pp_net.load.loc[:, "q_mvar"] = 0.0
        return

    factor = target / base_total
    pp_net.load.loc[:, "p_mw"] = base_p.values * factor
    pp_net.load.loc[:, "q_mvar"] = base_q.values * factor


def run_balanced_time_series(mc_net, ts_frame=None, load_round_decimals=3, cache_by_unique_values=True):
    """
    Run balanced power flow over time-series load values.

    Parameters
    ----------
    mc_net : object
        Multiconductor network object.
    ts_frame : pd.DataFrame, optional
        Time-series DataFrame. If None, uses mc_net.ts.
    load_round_decimals : int
        Rounding precision for cache keys.
    cache_by_unique_values : bool
        If True, solves each rounded load target once and reuses results.

    Returns
    -------
    pd.DataFrame
        Long-form bus results with columns:
        [REPORTED_DTTM, hr, load_target_mw, converged, bus, vm_pu, va_degree, p_mw, q_mvar]
    """
    source_ts = ts_frame if ts_frame is not None else getattr(mc_net, "ts", None)
    profile = _normalize_ts_frame(source_ts)
    if profile.empty:
        return pd.DataFrame(
            columns=[
                "REPORTED_DTTM",
                "hr",
                "load_target_mw",
                "converged",
                "bus",
                "vm_pu",
                "va_degree",
                "p_mw",
                "q_mvar",
            ]
        )

    base_pp = mc_to_balanced_pp(mc_net)
    base_p = base_pp.load["p_mw"].astype(float).copy() if len(base_pp.load) else pd.Series(dtype=float)
    base_q = base_pp.load["q_mvar"].astype(float).copy() if len(base_pp.load) else pd.Series(dtype=float)

    cache = {}
    rows = []

    for hr, row in enumerate(profile.itertuples(index=False), start=1):
        ts = pd.Timestamp(row.REPORTED_DTTM)
        load_target = float(row.MEASURE_VALUE)
        cache_key = round(load_target, int(load_round_decimals))

        if cache_by_unique_values and cache_key in cache:
            converged, solved_res = cache[cache_key]
            res = solved_res.copy()
        else:
            pp_net = copy.deepcopy(base_pp)
            _apply_total_load_target(pp_net, base_p, base_q, load_target)
            try:
                pp.runpp(pp_net, init="auto")
                converged = True
            except pp.powerflow.LoadflowNotConverged:
                converged = False

            if converged and len(pp_net.res_bus) > 0:
                res = pp_net.res_bus.reset_index().rename(columns={"index": "bus"})
            else:
                res = pd.DataFrame(columns=["bus", "vm_pu", "va_degree", "p_mw", "q_mvar"])

            for required in ("bus", "vm_pu", "va_degree", "p_mw", "q_mvar"):
                if required not in res.columns:
                    res[required] = np.nan

            if cache_by_unique_values:
                cache[cache_key] = (converged, res.copy())

        if res.empty:
            rows.append(
                {
                    "REPORTED_DTTM": ts,
                    "hr": hr,
                    "load_target_mw": load_target,
                    "converged": converged,
                    "bus": np.nan,
                    "vm_pu": np.nan,
                    "va_degree": np.nan,
                    "p_mw": np.nan,
                    "q_mvar": np.nan,
                }
            )
            continue

        for res_row in res.itertuples(index=False):
            rows.append(
                {
                    "REPORTED_DTTM": ts,
                    "hr": hr,
                    "load_target_mw": load_target,
                    "converged": converged,
                    "bus": getattr(res_row, "bus", np.nan),
                    "vm_pu": getattr(res_row, "vm_pu", np.nan),
                    "va_degree": getattr(res_row, "va_degree", np.nan),
                    "p_mw": getattr(res_row, "p_mw", np.nan),
                    "q_mvar": getattr(res_row, "q_mvar", np.nan),
                }
            )

    return pd.DataFrame(rows)


# if __name__ == "__main__":
#     pkl_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
#     converged, pp_net = run_balanced_loadflow(pkl_path)
#     print(f"Converged: {converged}")
#     if converged:
#         print(pp_net.res_bus[["vm_pu", "va_degree"]].describe())
