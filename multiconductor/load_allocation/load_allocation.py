import numpy as np
import pandas as pd
import networkx as nx
from multiconductor import run_pf

from contextlib import contextmanager

def _disable_table(net, table_name, row_mask=None):
    """
     deactivate  elements in net.<table_name>
    - If column 'in_service' exists set to False
    - Else if 'p_mw'/'q_mvar' exist set to 0
    Returns a reactivate dict
    """
    if not hasattr(net, table_name):
        return None

    df = getattr(net, table_name)
    if df is None or len(df) == 0:
        return None

    if row_mask is None:
        idx = df.index
    else:
        idx = df.index[row_mask]

    if len(idx) == 0:
        return None

    reactivate = {"table": table_name, "idx": idx, "data": {}}

    if "in_service" in df.columns:
        reactivate["data"]["in_service"] = df.loc[idx, "in_service"].copy()
        df.loc[idx, "in_service"] = False
        return reactivate

    # fallback to zeroing active and reactive power if no in_service exists
    for col in ("p_mw", "q_mvar"):
        if col in df.columns:
            reactivate["data"][col] = df.loc[idx, col].copy()
            df.loc[idx, col] = 0.0

    return reactivate if reactivate["data"] else None


def _reactivate_disabled(net, reactivate_list):
    if not reactivate_list:
        return
    for r in reactivate_list:
        if r is None:
            continue
        df = getattr(net, r["table"])
        for col, series in r["data"].items():
            df.loc[r["idx"], col] = series


@contextmanager
def _apply_ignore_options(net, ignore_generators=False, ignore_fixed_capacitors=False, ignore_controlled_capacitors=False):
    """
    Temporarily disable generators or shunts or capacitors (or combination) the PF + simulated meter
    represent a network without those injections

    - Generators:  net.asymmetric_sgen, net.asymmetric_gen
    - Capacitors/Shunts: net.asymmetric_shunt
      fixed vs controlled  using control_mode column
    """
    reactivate = []

    try:
        if ignore_generators:
            reactivate += [_disable_table(net, "asymmetric_gen"), _disable_table(net, "asymmetric_sgen")]

        # shunts "fixed" vs "controlled" depends on 'control_mode' if present.
        # If no control_mode info exists, ignore_fixed_capacitors will disable all shunts
        if ignore_fixed_capacitors or ignore_controlled_capacitors:
            for tname in ("asymmetric_shunt"):
                if not hasattr(net, tname):
                    continue
                df = getattr(net, tname)
                if df is None or len(df) == 0:
                    continue

                if "control_mode" in df.columns:
                    if ignore_fixed_capacitors:
                        reactivate.append(_disable_table(net, tname, row_mask=(df["control_mode"].fillna(False)).values))
                    if ignore_controlled_capacitors:
                        reactivate.append(_disable_table(net, tname, row_mask=(df["control_mode"].fillna(False)).values))
                else:
                    if ignore_fixed_capacitors:
                        reactivate.append(_disable_table(net, tname))

        yield

    finally:
        _reactivate_disabled(net, reactivate)


def get_load_phases(row):
    """
    consider (from_phase, to_phase):
    if  from_phase = 1,2,3
        to_phase   = 0, then  single-phase load on 'from_phase'
    If to_phase != 0, 3ph load [from_phase, to_phase]
    returns the phases that a load belongs to
    """
    from_ph = int(row.from_phase)
    to_ph = int(row.to_phase)

    if to_ph == 0:
        return [from_ph]

    return list((from_ph, to_ph))

def measurement_to_pq_target(net, meas_row):
    """
    Convert a measurement (P/Q/S/PF/I) AMP-PF (current magnitude and power factor), kW-kvar (active and reactive power), or kW-kVA (active and apparent power)
    to pandapower compliant  P [MW] and Q [MVAr]
    """
    mtype = meas_row.measurement_type.lower()
    value = float(meas_row["value"])
    side = str(meas_row.get("side", "from")).lower()

    # power factor, mode ("lagging"/"leading")
    pf = float(meas_row.get("pf", np.nan)) if "pf" in meas_row.index else np.nan
    mode = str(meas_row.get("pf_mode", "")).lower() if "pf_mode" in meas_row.index else ""

    # PF to angle
    def pf_to_angle(pf_val):
        # pf in [-100,100], default 95 if NaN
        if np.isnan(pf_val):
            pf_val = 95.0
        pf_abs = abs(pf_val) / 100.0
        pf_abs = max(min(pf_abs, 1.0), 0.0)
        return np.arccos(pf_abs)

    # sign of Q from lagging/leading and PF sign
    def pf_to_q_sign(pf_val, mode_str):
        sign_p = 1.0 if pf_val >= 0 else -1.0
        if "leading" in mode_str:
            # leading: Q < 0 (capacitive) for positive P
            return -1.0 * sign_p
        elif "lagging" in mode_str:
            # lagging: Q > 0 (inductive) for positive P
            return 1.0 * sign_p
        # default assume lagging
        return 1.0 * sign_p

    if mtype == "p":
        # normalize so "to" is treated as positive import
        if side == "to":
            value = abs(value)
        return value, None

    if mtype == "q":
        return None, value

    # Apparent power , PF
    if mtype == "s":
        phi = pf_to_angle(pf)
        S = value  # MVA
        pf_sign = 1.0 if pf >= 0 else -1.0
        P = pf_sign * S * np.cos(phi)
        q_sign = pf_to_q_sign(pf, mode)
        Q = q_sign * S * np.sin(phi)
        return P, Q

    # Current , PF
    if mtype == "i":
        etype = meas_row.element_type.lower()
        element = int(meas_row.element)
        if etype == "line":
            line_row = net.line.loc[(element, 0)]
            bus = line_row.from_bus if meas_row.side == "from" else line_row.to_bus
        elif etype == "ext_grid":
            eg_row = net.ext_grid_sequence.loc[(element, 0)]
            bus = eg_row.bus
        else:
            raise NotImplementedError

        vn_kv = float(net.bus.at[(bus, 0), "vn_kv"])
        U_phase_kV = vn_kv / np.sqrt(3)
        I_kA = value

        # Single-phase apparent power
        S = U_phase_kV * I_kA
        phi = pf_to_angle(pf)
        pf_sign = 1.0 if pf >= 0 else -1.0
        P = pf_sign * S * np.cos(phi)
        q_sign = pf_to_q_sign(pf, mode)
        Q = q_sign * S * np.sin(phi)
        return P, Q

    raise NotImplementedError(f"measurement_type '{mtype}' not supported yet.")

def get_trafo_hv_lv_buses(net, trafo_idx: int):

    trafo_index = net.trafo1ph.xs(trafo_idx, level="index")
    buses = list(map(int, trafo_index.index.get_level_values("bus").unique()))
    b_a, b_b = buses
    vn_a = float(net.bus.at[(b_a, 0), "vn_kv"])
    vn_b = float(net.bus.at[(b_b, 0), "vn_kv"])

    hv_bus, lv_bus = (b_a, b_b) if vn_a >= vn_b else (b_b, b_a)
    return hv_bus, lv_bus


#  Graph creation
def build_measurement_graph(net):
    """
    Build a directed phase MultiGraph with nodes (bus, phase)

    Edges:
      - lines (per phase)
      - switches (per phase, if present)

    """
    mg = nx.MultiDiGraph()

    for (idx, circuit), line in net.line.iterrows():
        for phase in range(line.from_phase, line.to_phase + 1):
            mg.add_edge((line.from_bus, phase),(line.to_bus, phase),
                key=("line", (idx, circuit)),
                weight=line.length_km)

    if "switch" in net and len(net.switch):
        for (idx, circuit), switch in net.switch.iterrows():
            mg.add_edge((switch.bus, switch.phase),(switch.element, switch.phase),
                key=("switch", (idx, circuit)),
                weight=0.01)

    if hasattr(net, "trafo1ph") and len(net.trafo1ph):
        for tidx, trafo in net.trafo1ph.groupby(level=0):
            hv_bus, lv_bus = get_trafo_hv_lv_buses(net, int(tidx))
            phases = sorted({int(p) for p in trafo["from_phase"].unique() if pd.notna(p)})
            for ph in phases:
                mg.add_edge((hv_bus, ph),(lv_bus, ph), key=("trafo1ph", (int(tidx), ph)), weight=0.01)
                #mg.add_edge((b1, ph), (b0, ph), key=("trafo1ph", (int(tidx), ph, "rev")), weight=0.01)

    return mg

# find all loads that are under the measurement
def get_downstream_load_indices(net, mg, meas_idx):
    meas = net.measurement.loc[meas_idx]

    if isinstance(meas, pd.DataFrame):
        meas = meas.iloc[0]

    meas_bus = get_measurement_downstream_bus(net, meas)

    # reachable (bus, phase) from all phases of meas_bus
    meas_phases = {ph for (bus, ph) in mg.nodes if bus == meas_bus}
    reachable = set()
    for ph in meas_phases:
        _, paths = nx.single_source_dijkstra(mg, (meas_bus, ph))
        reachable.update(paths.keys())

    # loads whose (bus,phase) are reachable
    loads = []
    for lidx, lrow in net.asymmetric_load.iterrows():
        for ph in get_load_phases(lrow):
            if (int(lrow.bus), int(ph)) in reachable:
                loads.append(lidx)
                break

    return loads

def get_measurement_bus(net, meas_row):
    elem_type = str(meas_row["element_type"]).lower()
    side = meas_row.get("side", None)
    if elem_type == "line":
        line_idx = int(meas_row["element"])
        line_row = net.line.loc[(line_idx, 0)]
        return int(line_row.from_bus) if side == "from" else int(line_row.to_bus)

    if elem_type == "ext_grid":
        eg_row = net.ext_grid_sequence.loc[(int(meas_row["element"]), 0)]
        return int(eg_row.bus)
    if elem_type == "trafo1ph":
        trafo_idx = int(meas_row["element"])
        hv_bus, lv_bus = get_trafo_hv_lv_buses(net, trafo_idx)
        return hv_bus if side == "hv" else lv_bus
    raise NotImplementedError(elem_type)

def get_measurement_downstream_bus(net, meas_row):
    etype = str(meas_row["element_type"]).lower()
    side = str(meas_row.get("side", "from")).lower()

    if etype == "line":
        line_idx = int(meas_row["element"])
        line_row = net.line.loc[(line_idx, 0)]
        # downstream region is the opp end of the measured side
        return int(line_row.to_bus) if side == "from" else int(line_row.from_bus)

    if etype == "ext_grid":
        eg_row = net.ext_grid_sequence.loc[(int(meas_row["element"]), 0)]
        return int(eg_row.bus)

    if etype == "trafo1ph":
        trafo_idx = int(meas_row["element"])
        hv_bus, lv_bus = get_trafo_hv_lv_buses(net, trafo_idx)
        # for radial net, downstream of the trafo is the LV bus
        return lv_bus

    raise NotImplementedError(etype)

def reachable_nodes_from_bus(mg, start_bus):
    """
    Return all downstream (bus, phase) nodes reachable from any phase at any start bus
    """
    start_nodes = [n for n in mg.nodes if n[0] == int(start_bus)]
    reached = set(start_nodes)
    for n0 in start_nodes:
        reached.update(nx.descendants(mg, n0))
    return reached


def filter_loads_under_other_measurement(net, mg, load_indices, meas_idx):

    # loads initially connected to this measurement
    conn_loads = set(load_indices)

    meas = net.measurement.loc[meas_idx]
    if isinstance(meas, pd.DataFrame):
        meas = meas.iloc[0]

    meas_bus = get_measurement_downstream_bus(net, meas)

    # nodes reachable downstream from this measurement
    reachable_nodes = reachable_nodes_from_bus(mg, meas_bus)

    disconn_loads = set()

    for other_idx in net.measurement.index:
        if other_idx == meas_idx:
            continue

        other = net.measurement.loc[other_idx]
        if isinstance(other, pd.DataFrame):
            other = other.iloc[0]

        other_bus = get_measurement_downstream_bus(net, other)

        other_start_nodes = [n for n in mg.nodes if n[0] == int(other_bus)]
        if not any(n in reachable_nodes for n in other_start_nodes):
            continue

        # phase-aware downstream nodes under other downstream measurement
        other_reachable_nodes = reachable_nodes_from_bus(mg, other_bus)

        # load on those downstream feeder head must be removed
        for lidx in conn_loads:
            load_row = net.asymmetric_load.loc[lidx]
            for ph in get_load_phases(load_row):
                if (int(load_row.bus), int(ph)) in other_reachable_nodes:
                    disconn_loads.add(lidx)
                    break

    return list(conn_loads - disconn_loads)


# get rated power of load
#ToDo: sn_mva from nearest connected trafo to be implemented
def get_sn_mva(net, load_indices):
    # def calc_sn_mva(idx):
    row = net.asymmetric_load.loc[load_indices]
    if "sn_mva" in row.index and pd.notna(row.sn_mva):
        return float(row.sn_mva)
    p = float(row.p_mw)
    q = float(row.q_mvar)
    return np.sqrt(p**2 + q**2)

    # if not isinstance(load_indices, (list, tuple)):
    #     return calc_sn_mva(load_indices)

    # # Multiple list of tuples
    # return {idx: calc_sn_mva(idx) for idx in load_indices}

def get_rated_S(net, mg, load_indices, include_trafo=False):
    """
    Return dict {load_idx: rated_S}, where rated power comes from:

    If include_trafo=True:
        1) connected capacity from nearest upstream trafo (OR)
        2) load's own sn_mva if upstream trafo lookup fails
        3) fallback sqrt(p^2 + q^2)

    If include_trafo=False:
        1) load's own sn_mva
        2) fallback sqrt(p^2 + q^2)
    """
    rated = {}

    for idx in load_indices:
        cap = np.nan
        if include_trafo:
            cap = get_connected_capacity_from_trafo(net, mg, idx)
        if pd.notna(cap) and cap > 0:
            rated[idx] = float(cap)
        else:
            # load's own sn_mva or |S|
            rated[idx] = get_sn_mva(net, idx)
    return rated

def get_connected_capacity_from_trafo(net, mg, load_idx):
    """
    Connected capacity weight for this load based on nearest upstream trafo.

    Steps:
      - for each load phase, BFS upstream (reverse graph) from (load_bus, phase)
      - stop at the first encountered trafo1ph edge
      - read trafo sn_mva from net.trafo1ph
      - split equally across load phases (phase-to-phase: 50/50 or 1/3 across 3 phases)
    """
    if mg is None or not hasattr(net, "trafo1ph") or len(net.trafo1ph) == 0:
        return np.nan
    # identify the bus where to load is and phases it's connected to
    row = net.asymmetric_load.loc[load_idx]
    load_bus = int(row.bus)
    phases = get_load_phases(row)
    if not phases:
        return np.nan
    # upstream graph search
    reverse_graph = mg.reverse(copy=False)
    number_of_phases = len(phases)

    def get_trafo_sn_mva(net, trafo_index):
        """
        Return the rated power (sn_mva) of a transformer
        """
        try:
            trafo_rows = net.trafo1ph.xs(trafo_index, level="index")
        except KeyError:
            return np.nan

        return float(trafo_rows["sn_mva"].iloc[0])

    capacities = []

    for phase in phases:
        start_node = (load_bus, int(phase))

        if start_node not in reverse_graph:
            continue

        nodes_to_visit = [start_node]
        visited_nodes = {start_node}
        trafo_capacity = np.nan

        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)

            for _, upstream_node, edge_key, _ in reverse_graph.out_edges(
                    current_node, keys=True, data=True
            ):
                # Look for transformer edge
                if isinstance(edge_key, tuple) and edge_key[0] == "trafo1ph":
                    trafo_index = int(edge_key[1][0])
                    trafo_capacity = get_trafo_sn_mva(net, trafo_index)
                    nodes_to_visit.clear()  # stop BFS
                    break

                if upstream_node not in visited_nodes:
                    visited_nodes.add(upstream_node)
                    nodes_to_visit.append(upstream_node)

            if not nodes_to_visit and pd.notna(trafo_capacity):
                break

        if pd.notna(trafo_capacity):
            capacities.append(trafo_capacity / number_of_phases)
    return float(sum(capacities)) if capacities else np.nan


def connected_capacity_allocation(net, load_indices, rated_S, target_p_mw):
    """
    Distribute a target active power (MW) to loads proportional to rated power.

    Parameters
    ----------
    net : pandapowerNet
    load_indices : list
        Indices of loads to allocate
    rated_S : dict
        {load_idx: rated apparent power}
    target_p_mw : float
        Total active power (MW) to be distributed
    """

    target = float(target_p_mw)

    if target <= 0.0 or not load_indices:
        return

    S_total = sum(rated_S.values())
    if S_total <= 0:
        raise RuntimeError("Sum of rated powers is zero, cannot allocate load power")

    for idx in load_indices:
        net.asymmetric_load.at[idx, "p_mw"] = (rated_S[idx] / S_total) * target


#ToDo: add 'locked' to net.asymmetric_load
def filter_locked_loads(net, load_indices):
    """
    Remove loads whose 'locked' is True
    """
    if "locked" not in net.asymmetric_load.columns:
        return load_indices

    net.asymmetric_load.loc[net.asymmetric_load.locked.isna(), "locked"] = False
    locked = net.asymmetric_load.loc[load_indices, "locked"].astype(bool)

    return [idx for idx, is_locked in zip(load_indices, locked.values) if not is_locked]


def get_overloaded_loads(net, load_overload_factor=1.0):
    """
    Overloaded loads if:
      |S| > sn_mva * load_overload_factor (considered 130% for connected trafo as per spec)
    """
    overloaded = set()
    for lidx, row in net.asymmetric_load.iterrows():
        if "sn_mva" not in row.index or pd.isna(row.sn_mva):
            continue

        p = float(row.p_mw)
        q = float(row.q_mvar)
        S = np.sqrt(p**2 + q**2)

        sn = float(row.sn_mva)
        if sn <= 0:
            continue

        threshold = sn * load_overload_factor
        if S > threshold:
            overloaded.add(lidx)

    return overloaded


def filter_overloaded_loads(net, load_indices):
    """
    Drop loads that currently exceed their own sn_mva rating.
    To be used inside the iterative loop.
    """
    overloaded = get_overloaded_loads(net)
    return [idx for idx in load_indices if idx not in overloaded]


def compute_delta(meas_t, simul):
    """
     determine delta between measurement target and value after load flow
    """
    meas_target = float(meas_t)
    simulated = float(simul)
    if abs(meas_target) < 1e-9:
        return 0.0 if abs(simulated) < 1e-9 else 0.0
    return (meas_target - simulated) / max(abs(meas_target), 1e-9)


def get_simulated_measurement_value(net, meas_row):
    """
    Map a measurement definition to its simulated value from PF results.

      - element_type "line" with measurement_type "p" at "from"/"to" side
      - element_type "ext_grid" with "p" (via res_bus or res_ext_grid_sequence)

    """
    mtype = meas_row.measurement_type.lower()
    etype = meas_row.element_type.lower()
    element = int(meas_row.element)
    side = meas_row.side

    if mtype != "p":
        raise NotImplementedError

    if etype == "line":
        col = f"p_{side}_mw"
        res = net.res_line.loc[(element, slice(None)), col]
        p = float(res.sum())
        if str(side).lower() == "to":
            p = abs(p)
        return p

    if etype == "ext_grid":
        if hasattr(net, "res_ext_grid_sequence") and len(net.res_ext_grid_sequence):
            try:
                res = net.res_ext_grid_sequence.loc[(element, slice(None)), "p_mw"]
                val = float(res.sum())
                if abs(val) > 1e-6:
                    return val
            except KeyError:
                pass
        if hasattr(net, "res_bus") and len(net.res_bus):
            ext_grid_rows = net.ext_grid_sequence.xs(element, level=0)
            bus = int(ext_grid_rows["bus"].iloc[0])
            bus_power = net.res_bus.loc[(bus, slice(None)), "p_mw"]
            phases = bus_power.index.get_level_values(-1)
            return abs(float(bus_power[phases != 0].sum()))

        raise RuntimeError("No ext_grid result available to map measurement")

    if etype == "trafo1ph":
        if not hasattr(net, "res_trafo"):
            raise RuntimeError("No res_trafo available")

        hv_bus, lv_bus = get_trafo_hv_lv_buses(net, element)

        if side == "hv":
            bus_side = hv_bus
        elif side == "lv":
            bus_side = lv_bus
        else:
            raise ValueError("Trafo measurement side must be 'hv' or 'lv'")

        res = net.res_trafo.loc[(element, bus_side, slice(None)), "p_mw"]
        p = float(res.sum())

        # LV-side measurement as positive import
        if side == "lv":
            p = abs(p)

        return p

    raise NotImplementedError(f"Measurement mapping not implemented for type={mtype}, element_type={etype}")

def lock_loads(net, load_indices):
    if "locked" not in net.asymmetric_load.columns:
        net.asymmetric_load["locked"] = False
    net.asymmetric_load.loc[load_indices, "locked"] = True


def _cap_load_powers(net, mg, load_indices, cap_to_load_rating=False,
                     cap_to_transformer_rating=False, load_overload_factor=1.0,
                     trafo_overload_factor=1.3, adjust_power_factor=False):
    """Clamp each load's apparent power to its rating in place.

    Applies the smaller of the load rating cap (sn_mva * load_overload_factor)
    and the transformer rating cap (connected trafo capacity * trafo_overload_factor)
    to the load's current p_mw/q_mvar. Loads already within their cap are left
    untouched. Returns the list of capped load indices.
    """
    if not (cap_to_load_rating or cap_to_transformer_rating):
        return []

    capped = []
    for idx in load_indices:
        row = net.asymmetric_load.loc[idx]
        p_old = float(row.p_mw)
        q_old = float(row.q_mvar)

        S_old = float(np.sqrt(p_old ** 2 + q_old ** 2))
        S_cap = None

        # load cap (1.0 * sn_mva by default)
        if cap_to_load_rating and pd.notna(row.sn_mva):
            load_cap = float(row.sn_mva) * float(load_overload_factor)
            S_cap = load_cap if S_cap is None else min(S_cap, load_cap)

        # trafo cap (typically 1.3 * trafo sn_mva, split across load phases)
        if cap_to_transformer_rating:
            trafo_cap = get_connected_capacity_from_trafo(net, mg, idx)
            if pd.notna(trafo_cap) and float(trafo_cap) > 0:
                trafo_cap = float(trafo_cap) * float(trafo_overload_factor)
                S_cap = trafo_cap if S_cap is None else min(S_cap, trafo_cap)

        if S_cap is not None and S_old > S_cap:
            if not adjust_power_factor:
                # scale P and Q down proportionally
                factor = S_cap / S_old
                p_new = p_old * factor
                q_new = q_old * factor
            else:
                p_mag = min(abs(p_old), S_cap)
                q_mag = float(np.sqrt(max(S_cap ** 2 - p_mag ** 2, 0.0)))
                q_sign = np.sign(q_old) if abs(q_old) > 1e-12 else 1.0
                p_new = np.sign(p_old) * p_mag
                q_new = q_sign * q_mag

            net.asymmetric_load.at[idx, "p_mw"] = p_new
            net.asymmetric_load.at[idx, "q_mvar"] = q_new
            capped.append(idx)

    return capped


def _run_load_allocation_for_measurement(net, mg, meas_idx, tolerance=0.5, max_iter=8, verbose=True, adjust_after_load_flow=False,
                                         cap_to_load_rating=False, cap_to_transformer_rating=False, ignore_generators=True, ignore_fixed_capacitors=False, ignore_controlled_capacitors=False,
                                         load_overload_factor=1.0, trafo_overload_factor=1.3, warn_overloads=True,
                                         unlock_all_locked_loads=None, adjust_power_factor=False,
                                         disable_downstream_meters=False,  max_tolerance_for_relaxation=None):

    meas = net.measurement.loc[meas_idx]
    # MultiIndex first idx
    if isinstance(meas, pd.DataFrame):
        meas = meas.iloc[0]
    P_target, Q_target = measurement_to_pq_target(net, meas)

    # When no factor is provided, fall back to the default 1.3 (130%).
    if trafo_overload_factor is None:
        trafo_overload_factor = 1.3

    meas_bus = get_measurement_downstream_bus(net, meas)
    reachable_nodes = reachable_nodes_from_bus(mg, meas_bus)

    downstream_meas = []
    for idx in net.measurement.index:
        if idx == meas_idx:
            continue
        other = net.measurement.loc[idx]
        if str(other.element_type).lower() == "ext_grid":
            continue
        if isinstance(other, pd.DataFrame):
            other = other.iloc[0]
        other_bus = get_measurement_downstream_bus(net, other)
        other_nodes = [n for n in mg.nodes if n[0] == int(other_bus)]
        if any(n in reachable_nodes for n in other_nodes):
            downstream_meas.append(idx)

    meas_elem_type = str(meas.element_type).lower()
    meas_side = str(meas.get("side", "")).lower()

    if meas_elem_type == "trafo1ph":
        downstream_meas_for_residual = []
    else:
        downstream_meas_for_residual = downstream_meas

    downstream_sum = 0.0
    for idx in downstream_meas_for_residual:
        other = net.measurement.loc[idx]
        if isinstance(other, pd.DataFrame):
            other = other.iloc[0]
        p_other, _ = measurement_to_pq_target(net, other)
        downstream_sum += float(p_other)

    target = P_target - downstream_sum

    if verbose and downstream_meas_for_residual:
        print(
            f"measurement {meas_idx}: base={float(P_target):.4f} MW, "
            f"downstream={downstream_sum:.4f} MW, residual={target:.4f} MW"
        )


    # filters loads under other measurements
    load_indices = get_downstream_load_indices(net, mg, meas_idx)
    #load_indices = filter_loads_under_other_measurement(net, mg, load_indices, meas_idx)
    #meas_elem_type = str(meas.element_type).lower()
    #meas_side = str(meas.get("side", "")).lower()

    if not (meas_elem_type == "trafo1ph" and meas_side == "lv"):
        # for everything except trafo LV
        load_indices = filter_loads_under_other_measurement(net, mg, load_indices, meas_idx)
    if not load_indices:
        if verbose:
            print(f"measurement {meas_idx}: no downstream loads to adjust")
        return

    known_sum_p = 0.0
    if "locked" in net.asymmetric_load.columns and len(load_indices):
        locked_mask = net.asymmetric_load.loc[load_indices, "locked"].fillna(False)
        locked_loads = [idx for idx, is_locked in zip(load_indices, locked_mask.values) if is_locked]
        if locked_loads:
            known_sum_p = float(net.asymmetric_load.loc[locked_loads, "p_mw"].abs().sum())

    target = float(target) - known_sum_p

    if verbose and known_sum_p > 0:
        print(
            f"Known locked loads ΣP={known_sum_p:.4f} MW")

    # filter locked loads
    if not unlock_all_locked_loads:
        load_indices = filter_locked_loads(net, load_indices)

    # rated power and initial guess for load allocation considering loads
    # trafo included
    rated_S = get_rated_S(net, mg, load_indices)

    connected_capacity_allocation(net, load_indices, rated_S, target)

    # Enforce rating caps on the initial allocation so loads are limited even
    # when the measurement target is met before the iterative loop runs (or when
    # adjust_after_load_flow=False).
    capped = _cap_load_powers(
        net, mg, load_indices,
        cap_to_load_rating=cap_to_load_rating,
        cap_to_transformer_rating=cap_to_transformer_rating,
        load_overload_factor=load_overload_factor,
        trafo_overload_factor=trafo_overload_factor,
        adjust_power_factor=adjust_power_factor,
    )
    if verbose and capped:
        print(f"capped {len(capped)} loads to their rating after initial allocation")

    # warn-only overload checks after initial allocation
    if warn_overloads and not (cap_to_load_rating or cap_to_transformer_rating):
        overloaded = get_overloaded_loads(net, load_overload_factor=load_overload_factor)
        if overloaded and verbose:
            print(f"[WARN] loads exceed rating (factor={load_overload_factor}): {sorted(overloaded)}")
        if cap_to_transformer_rating is False and hasattr(net, "trafo1ph"):
            pass

    if not adjust_after_load_flow:
        if verbose:
            print("adjust_after_load_flow=False, skipping iterative PF adjustment")
        return

    for it in range(max_iter):
        with _apply_ignore_options(net,
                    ignore_generators=ignore_generators,
                    ignore_fixed_capacitors=ignore_fixed_capacitors,
                    ignore_controlled_capacitors=ignore_controlled_capacitors):

            run_pf(net)
            simulated_total = get_simulated_measurement_value(net, meas)
            delta = compute_delta(P_target, simulated_total)

            if verbose:
                print(
                    f"Iter {it}: measurement_target={P_target:.4f} MW, "
                    f"sim={simulated_total:.4f} MW, delta={delta * 100:.3f}%"
                )

            if abs(delta * 100) <= tolerance:
                if verbose:
                    print(f"Converged within ±{tolerance}% after {it} iterations")
                # lock_loads(net, load_indices)
                return

            scale = max(0.0, 1.0 + delta)

            for idx in load_indices:
                row = net.asymmetric_load.loc[idx]
                p_old = float(row.p_mw)
                q_old = float(row.q_mvar)

                p_new = p_old * scale
                q_new = q_old * scale # TODO: there needs to be 2 different scaling factors for P and Q based on the Q_target variable above

                net.asymmetric_load.at[idx, "p_mw"] = p_new
                net.asymmetric_load.at[idx, "q_mvar"] = q_new

            # Re-enforce rating caps after scaling so loads never exceed their rating.
            _cap_load_powers(
                net, mg, load_indices,
                cap_to_load_rating=cap_to_load_rating,
                cap_to_transformer_rating=cap_to_transformer_rating,
                load_overload_factor=load_overload_factor,
                trafo_overload_factor=trafo_overload_factor,
                adjust_power_factor=adjust_power_factor,
            )

    if verbose:
        print(f"load allocation for measurement {meas_idx} did not converge in Iter {it}")


def run_load_allocation(net, mg, adjust_after_load_flow=True, tolerance=0.5, ignore_fixed_capacitors=False,
                        ignore_controlled_capacitors=False, ignore_generators=False, unlock_all_locked_loads=None,
                        disable_downstream_meters=False, convert_by_phase_to_total_demands=False, max_tolerance_for_relaxation=None,
                        cap_to_transformer_rating=False, cap_to_load_rating=False, trafo_overload_factor=None,
                        load_overload_factor=1.0, adjust_power_factor=True, measurement_indices=None, max_iter=8, verbose=True):
    """
    This function loops over all selected measurements and calls the _run_load_allocation_for_measurement() for each one
    """

    if measurement_indices is None:
        measurement_indices = list(net.measurement.index)

    for meas_idx in measurement_indices:
        if verbose:
            print(f"\n run_load_allocation for measurement {meas_idx}")

        _run_load_allocation_for_measurement(net, mg, meas_idx, tolerance=tolerance, max_iter=max_iter,
                                             verbose=verbose, adjust_after_load_flow=adjust_after_load_flow, cap_to_load_rating=cap_to_load_rating,
                                             unlock_all_locked_loads=unlock_all_locked_loads, adjust_power_factor=adjust_power_factor, ignore_generators=ignore_generators,
                                             ignore_fixed_capacitors=ignore_fixed_capacitors, ignore_controlled_capacitors=ignore_controlled_capacitors,
                                             cap_to_transformer_rating=cap_to_transformer_rating, trafo_overload_factor=trafo_overload_factor, 
                                             load_overload_factor=load_overload_factor)


