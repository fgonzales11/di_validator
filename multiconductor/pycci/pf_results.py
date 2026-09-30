import math
import numpy as np
import pandas as pd
from scipy.sparse import block_diag
from numba import njit

from multiconductor.pycci.make_Y_elements import get_c_type

def _bus_results_pf(net):
    model = net["model"]
    sbase = net.sn_mva * 1e6
    E = model.E

    ypos = model.terminal_to_y_lookup[[b * 4 + p for b, p in net.bus.index]]
    r = E[ypos]
    r[ypos == -1] = np.nan

    res_bus = net["res_bus"] = pd.DataFrame(index=net.bus.index, columns=["vm_pu", "va_degree", "p_mw", "q_mvar", "imbalance_percent"])

    res_bus["vm_pu"] = np.abs(r)
    res_bus["va_degree"] = np.angle(r) * 180 / np.pi

    # Add "imbalance percent" to bus results: calculate from vm_pu phase 1-3 and cast back to multiindex
    vm_wide = net.res_bus.vm_pu.loc[pd.IndexSlice[:, [1, 2, 3]]].unstack().values
    phase_count = (vm_wide > 0).sum(axis=1)
    V_avg = vm_wide.mean(1)
    V_avg[V_avg == 0] = 1
    max_dev = np.abs(vm_wide - V_avg[:, None]).max(1)
    imbalance = np.where(phase_count == 3, (max_dev / V_avg) * 100, np.nan)
    bus_counts = net.res_bus.index.get_level_values(0).value_counts().sort_index()
    imbalance_expl = pd.Series(np.repeat(imbalance, bus_counts.values), index=net.res_bus.index,
                                name='imbalance_percent')
    net.res_bus['imbalance_percent'] = imbalance_expl

    Ypass = model.Y_tran + model.Y_network + model.Y_ground

    # Evaluate complex power at bus terminals
    Sbus = E * np.conjugate(Ypass @ E) * sbase
    Sbus = Sbus[ypos, 0]
    # Change sign to bus power to fix convention (P>0 = injected)
    res_bus["p_mw"] = -np.real(Sbus) / 1e6
    res_bus["q_mvar"] = -np.imag(Sbus) / 1e6


def _shunt_results_pf(net):
    for what in ["load", "sgen", "shunt"]:
        swhat = f"S_{what}"
        if swhat not in net.model or net.model[swhat] is None:
            continue
        net[f"res_asymmetric_{what}"] = pd.DataFrame(
            index=net[f"asymmetric_{what}"].index,
            columns=["p_mw", "q_mvar"],
        )

        #rtab["p_mw"] = np.real(net.model[swhat]) / (net.sn_mva * 1e6)
        #rtab["q_mvar"] = np.imag(net.model[swhat]) / (net.sn_mva * 1e6)


def _generator_results_pf(net):
    columns = ["p_mw", "q_mvar", "va_degree", "vm_pu"]
    res_gen = pd.DataFrame(np.nan, index=net.asymmetric_gen.index, columns=columns, dtype=np.float64)
    gen_state = getattr(net.model, "gen_state", None)
    if gen_state is None or len(gen_state.get("q_mvar", [])) == 0:
        net["res_asymmetric_gen"] = res_gen
        return

    E = net.model.E
    for idx, y_from, y_to, p_mw, q_mvar, phase_to_phase in zip(
        gen_state["active_pos"],
        gen_state["y_from"],
        gen_state["y_to"],
        gen_state["p_mw"],
        gen_state["q_mvar"],
        gen_state["phase_to_phase"],
    ):
        row_index = gen_state["row_index"][idx]
        terminal_voltage = E[y_from, 0] - E[y_to, 0]
        res_gen.loc[row_index, "p_mw"] = p_mw
        res_gen.loc[row_index, "q_mvar"] = q_mvar
        res_gen.loc[row_index, "vm_pu"] = abs(terminal_voltage) * (1.0 / math.sqrt(3.0) if phase_to_phase else 1.0)
        res_gen.loc[row_index, "va_degree"] = np.angle(terminal_voltage) * 180.0 / np.pi

    net["res_asymmetric_gen"] = res_gen


@njit(cache=True)
def _line_results_pf_numba(indizes, Yprimitives, Em, Ibase, sbase, r):
    rad_to_deg = 57.29577951308232  # 180/pi
    sbase_MW = sbase / 1e6

    k = 0
    indpos = 0
    primpos = 0

    while indpos < len(indizes):
        size = indizes[indpos]
        hs = size // 2

        for i in range(size):
            i_div_hs = i // hs
            i_mod_hs = i % hs
            row = k + i_mod_hs
            col_offset = 2 * i_div_hs

            current = 0j
            for j in range(size):
                y = indizes[indpos + 2 + j]
                current += Yprimitives[i, primpos + j] * Em[y, 0]

            y = indizes[indpos + 2 + i]
            E = Em[y, 0]
            I_ka = current * Ibase[y, 0] * 0.001

            S = E * (current.real - 1j * current.imag) * sbase_MW

            r[row, 0 + col_offset] = math.sqrt(I_ka.real * I_ka.real + I_ka.imag * I_ka.imag)
            r[row, 1 + col_offset] = math.atan2(I_ka.imag, I_ka.real) * rad_to_deg
            r[row, 5 + i_div_hs] = S.real
            r[row, 7 + i_div_hs] = S.imag
            r[row, 11 + i_div_hs] = math.sqrt(E.real * E.real + E.imag * E.imag)
            r[row, 13 + i_div_hs] = math.atan2(E.imag, E.real) * rad_to_deg

            if i >= hs:
                r[row, 4] = max(r[row, 0], r[row, 2])
                r[row, 9] = r[row, 5] + r[row, 6]
                r[row, 10] = r[row, 7] + r[row, 8]

        k += hs
        indpos += 2 + size
        primpos += size


def _line_results_pf(net):
    columns = [
        'i_from_ka', 'ia_from_degree', 'i_to_ka', 'ia_to_degree', 'i_ka', 'p_from_mw', 'p_to_mw',
        'q_from_mvar', 'q_to_mvar', 'pl_mw', 'ql_mvar',
        'vm_from_pu', 'vm_to_pu', 'va_from_degree', 'va_to_degree', 'loading_percent',
    ]
    res_line = pd.DataFrame(np.nan, index=net.line.index, columns=columns, dtype=np.float64)
    sbase = net.sn_mva * 1e6
    Ibase = net.model.Ibase_y.reshape(-1, 1)
    Em = net.model.E
    li = net.model.line_info
    active_rows = [row for rows in li.get("active_rows", []) for row in rows]
    active_row_count = len(active_rows)
    if active_row_count:
        active_results = np.zeros(shape=(active_row_count, len(columns)), dtype=np.float64)
        _line_results_pf_numba(li["indizes"], li["Yprimitives"], Em, Ibase, sbase, active_results)
        active_frame = pd.DataFrame(active_results, index=pd.MultiIndex.from_tuples(active_rows), columns=columns)
        active_frame["loading_percent"] = _line_loading_percent(net, li, active_frame.index, active_frame["i_ka"])
        res_line.loc[active_frame.index, columns] = active_frame[columns]
    net.res_line = res_line


def _line_loading_percent(net, line_info, active_rows, currents_ka):
    loading = pd.Series(np.nan, index=active_rows, dtype=np.float64)
    for line_id, active_circuits in zip(line_info["indices"], line_info.get("active_circuits", [])):
        group = net.line.xs(line_id, level=0, drop_level=False)
        row0 = group.iloc[0]
        conductor_limits = np.asarray(
            get_c_type(net, row0["model_type"], row0["std_type"]).get("Imax", []),
            dtype=np.float64,
        ).reshape(-1)
        for circuit_id in active_circuits:
            row_index = (line_id, circuit_id)
            if circuit_id < len(conductor_limits) and conductor_limits[circuit_id] > 0:
                loading.loc[row_index] = currents_ka.loc[row_index] / conductor_limits[circuit_id] * 100.0
            else:
                loading.loc[row_index] = np.nan
    return loading.to_numpy(dtype=np.float64)


def _line_results_pf_old(net):
    sbase = net.sn_mva * 1e6
    E = net.model.E

    active_YpL = net.model.line_info["Yprimitive"]
    active_line_indices = net.model.line_info["indices"]
    circuit_counts = net.model.line_info["num_circuits"]
    
    if not active_YpL:
        return pd.DataFrame(index=net.line.index)
   
    y_from_to = net.model.line_info["y_from_to"].astype(int)
    Ibase_all = net.model.Ibase_y[y_from_to]
    
    YpL_combined = block_diag(active_YpL, format='csr')
    E_all = E[y_from_to]
    I_all = YpL_combined @ E_all
    S_all = E_all * np.conj(I_all) * sbase

    mask = np.concatenate([np.repeat([True, False], count) for count in circuit_counts])
    
    I_all_kA = I_all * Ibase_all / 1000

    I_all_send_kA = I_all_kA[mask]
    I_all_recv_kA = I_all_kA[~mask]
    
    S_all_send = S_all[mask]
    S_all_recv = S_all[~mask]

    E_all_send = E_all[mask]
    E_all_recv = E_all[~mask]
    
    results = []
    indices = []
    
    #TODO vectorize everything!
    phase_offset = 0
    for i, (ni, n_circuits) in enumerate(zip(active_line_indices, circuit_counts)):
        
        end_offset = phase_offset + n_circuits
        
        I_send_kA = I_all_send_kA[phase_offset:end_offset]
        I_recv_kA = I_all_recv_kA[phase_offset:end_offset]
        S_send = S_all_send[phase_offset:end_offset]
        S_recv = S_all_recv[phase_offset:end_offset]
        E_send = E_all_send[phase_offset:end_offset]
        E_recv = E_all_recv[phase_offset:end_offset]
        
        for circuit_idx in range(n_circuits):
            indices.append((ni, circuit_idx))
            
            i_send = abs(I_send_kA[circuit_idx, 0])
            ia_send = np.angle(I_send_kA[circuit_idx, 0]) * 180 / np.pi
            i_recv = abs(I_recv_kA[circuit_idx, 0])
            ia_recv = np.angle(I_recv_kA[circuit_idx, 0]) * 180 / np.pi

            results.append([
                i_send, ia_send, i_recv, ia_recv, max(i_send, i_recv),
                np.real(S_send[circuit_idx, 0]) / 1e6,
                np.real(S_recv[circuit_idx, 0]) / 1e6,
                np.imag(S_send[circuit_idx, 0]) / 1e6,
                np.imag(S_recv[circuit_idx, 0]) / 1e6,
                np.real(S_send[circuit_idx, 0] + S_recv[circuit_idx, 0]) / 1e6,
                np.imag(S_send[circuit_idx, 0] + S_recv[circuit_idx, 0]) / 1e6,
                abs(E_send[circuit_idx, 0]),
                abs(E_recv[circuit_idx, 0]),
                np.angle(E_send[circuit_idx, 0]) * 180/np.pi,
                np.angle(E_recv[circuit_idx, 0]) * 180/np.pi,
                np.nan
            ])
        
        phase_offset = end_offset
    
    columns = ['i_from_ka', 'ia_from_degree', 'i_to_ka', 'ia_to_degree', 'i_ka', 'p_from_mw', 'p_to_mw', 
               'q_from_mvar', 'q_to_mvar', 'pl_mw', 'ql_mvar',
               'vm_from_pu', 'vm_to_pu', 'va_from_degree', 'va_to_degree', 'loading_percent']
    
    multi_index = pd.MultiIndex.from_tuples(indices, names=['index', 'circuit'])
    res_line = pd.DataFrame(results, index=multi_index, columns=columns)
    
    net["res_line"] = res_line


# @numba.njit(cache=True)
# def f(Ybr, y_send_rec, Em, Ibase_y, buses, pns, is_p2p, r):
#     k = 0
#     E = np.zeros(shape=(4), dtype=np.complex128)
#     for c in range(len(y_send_rec) // 4):
#         for ii in range(4):
#             E[ii] = Em[ii]
#         I = Ybr[4*c: 4*c+4] @ E

#         I_ka = np.abs(I * Ibase_y[y_send_rec[4*c: 4*c + 4]]) / 1e3
#         S = E * np.conjugate(I) * sbase
#         S_loss_total = np.sum(S)
#         S_loss_per_side = S_loss_total / 2

#         for i, (bus, pn, p2p) in enumerate(zip(buses[2*k: 2*k + 2], pns[2*k: 2*k + 2], is_p2p[2*k: 2*k + 2])):
#             S_side = S[2 * i, 0] + S[2 * i + 1, 0]
#             loading_percent = np.abs(S_side) / pn * 100

#             E_from_to = E[2 * i, 0] - E[2 * i + 1, 0]

#             r[k, 0] = np.real(S[2 * i, 0]) / 1e6
#             r[k, 1] = np.imag(S[2 * i, 0]) / 1e6
#             r[k, 2] = I_ka[2 * i, 0]
#             r[k, 3] = np.abs(E_from_to) * (1. / 1.7320508075688772 if p2p else 1)
#             r[k, 4] = np.angle(E_from_to) * 180 / np.pi
#             r[k, 5] = np.real(S_loss_per_side) / 1e6
#             r[k, 6] = np.imag(S_loss_per_side) / 1e6
#             r[k, 7] = loading_percent
#             k += 1


def _trafo_results_pf(net):
    indices = []
    results = []
    sbase = net.sn_mva * 1e6
    Ibase_y = net.model.Ibase_y.reshape(-1, 1)

    for (tidx, cidx), circ in net.model.trafo_circuits.items():
        y_send_rec = circ["y_send_rec"]
        E = net.model.E[y_send_rec]
        branch_current = circ["Ybr"] @ E
        I_ka = abs(branch_current * Ibase_y[y_send_rec]) / 1e3

        S = E * np.conjugate(branch_current) * sbase
        
        S_loss_total = np.sum(S)
        S_loss_per_side = S_loss_total / len(circ["buses"])


        for i, (bus, pn, p2p) in enumerate(zip(circ["buses"], circ["pns"], circ["is_p2p"])):
            indices.append((tidx, bus, cidx))

            S_from = S[2 * i, 0]
            S_to = S[2 * i + 1, 0]
            I_from = I_ka[2 * i, 0]
            I_to = I_ka[2 * i + 1, 0]
            max_current = max(I_from, I_to)
            loading_percent = max(abs(S_from), abs(S_to)) / pn * 100 if pn and pn > 0 else np.nan
            S_through = S_from

            E_from_to = E[2 * i, 0] - E[2 * i + 1, 0]

            results.append([
                np.real(S_through) / 1e6,
                np.imag(S_through) / 1e6,
                max_current,
                abs(E_from_to) * (1. / math.sqrt(3.) if p2p else 1),
                np.angle(E_from_to) * 180 / np.pi,
                np.real(S_loss_per_side) / 1e6,
                np.imag(S_loss_per_side) / 1e6,
                loading_percent,
            ])

    columns = ['p_mw', 'q_mvar', 'i_ka', 'vm_pu', 'va_degree', 'pl_mw', 'ql_mvar', 'loading_percent']

    multi_index = pd.MultiIndex.from_tuples(indices, names=['index', 'bus', 'circuit'])
    net.res_trafo = pd.DataFrame(results, index=multi_index, columns=columns).sort_index()
