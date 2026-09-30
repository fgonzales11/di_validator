import numpy as np
import math
from scipy.sparse import csc_matrix, eye, linalg, coo_matrix, diags
from multiconductor.pycci.pf_results import _bus_results_pf, _generator_results_pf, _line_results_pf, _trafo_results_pf, _shunt_results_pf
from multiconductor.studies import InvalidStudyInput
from .model import _initialize_model, find_islands, get_active_ext_grid_table

# TEMP fix
import pandapower.control
import inspect
source = inspect.getsource(pandapower.control.run_control).replace("ctrl_variables = prepare_run_ctrl(net, ctrl_variables)", "ctrl_variables = prepare_run_ctrl(net, ctrl_variables, **kwargs)")
exec(source, pandapower.control.__dict__)


class ppException(Exception):
    """
    General pandapower custom parent exception.
    """
    pass


class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


class ControllerNotConverged(ppException):
    """
    Exception being raised in case a controller does not converge.
    """
    pass


# @njit(cache=True)
# def infclean(I):
#     for k in range(len(I)):
#         if np.isinf(I[k,0]):
#             I[k,0]=0
#     return I

def set_controllers_in_service(net, ctrtype, in_service):
    # ctrtype: ['All', 'LoadTapChangerControl','LineDropControl', VoltVarController', 'BinaryShuntController']
    for i,co in net.controller.object.items():
        if ctrtype=="All" or ctrtype in str(type(co)):
            net.controller.at[i,'in_service']=in_service


def correct_trafo_vnkv(net):
    tt = net.trafo1ph.reset_index()
    tt = tt[tt.to_phase == 0]
    corrected = 0
    for _, t in tt.iterrows():
        if abs(t.vn_kv - net.bus.vn_kv.at[(t.bus, t.to_phase)]) < 0.01:
            b = (t["index"], t.bus, t.circuit)
            net.trafo1ph.loc[b, "vn_kv"]  = net.trafo1ph.loc[b, "vn_kv"] / math.sqrt(3)
            corrected += 1
        elif t.vn_kv / net.bus.vn_kv.at[(t.bus, t.to_phase)] == 0.5:
            b = (t["index"], t.bus, t.circuit)
            net.trafo1ph.loc[b, "vn_kv"]  = net.trafo1ph.loc[b, "vn_kv"] *2 / math.sqrt(3)
            corrected += 1
    return corrected


def run_pf(net, tol_vmag_pu=1e-5, tol_vang_rad=1e-5, MaxIter=100, run_control=False, debug_level=0, **kwargs):
    """
    Run a snapshot power flow calculation in a generic multi-conductor grid using the Correction-Current-Injection
    method.

    INPUT:
        **net** - The pandapower-like format network where elements are defined

    OPTIONAL:
    
        **tol_vmag_pu** (float, 1e-5) - Tolerance of iterative process on voltage magnitude in pu. Default is 1e-5

        **tol_vang_rad** (float, 1e-5) - Tolerance of iterative process on voltage angle in rad. Default is 1e-5

        **MaxIter** (int, 100) - Max number of iterations in the iterative process. Default is 100.
        
        **kwargs** - Additional keyword arguments. Effective ones may be: \
            * init_model (True), False - specify if model has to be initialized
            * run_capacitor_control (boolean, True) - if True, capacitor controllers are used. If False, they are deactivated.  
            * run_ltc_control (boolean, True) - if True, LTC controllers are used. If False, they are deactivated.
            * run_ldc_control (boolean, True) - if True, LDC controllers are used. If False, they are deactivated.
            * run_voltvar_control (boolean, True) - if True, volt/var controllers are used. If False, they are deactivated.    
            * NOTE: the run_x_control flags modify the in_service property of controllers. If no flag is given, this property will remain unchanged.
            * If, on the other side, any flag is given, all in_service properties will be changed (eventually to default True for missing flags). 
            * In-service properties will not be re-set after run_pp. This needs to be considered when using run_pp multiple times on the same grid.           
    """
    
    allow_islands = bool(kwargs.pop("allow_islands", False))
    init_model = bool(kwargs.pop("init_model", True))
    warm_start = bool(kwargs.pop("warm_start", False))
    previous_voltage = None
    if warm_start and hasattr(net, "model") and hasattr(net.model, "E"):
        candidate = np.asarray(net.model.E, dtype=complex).reshape(-1)
        if candidate.size:
            previous_voltage = candidate.copy()
    if net.switch.closed.dtype != np.dtype("bool"):
        net.switch.closed = net.switch.closed.astype(bool)
    if run_control and any([flag in kwargs.keys() for flag in ['run_capacitor_control','run_ltc_control','run_ldc_control','run_voltvar_control'] ]):         
        set_controllers_in_service(net, 'BinaryShuntController', kwargs.get('run_capacitor_control', True))
        set_controllers_in_service(net, 'LoadTapChangerControl', kwargs.get('run_ltc_control', True))
        set_controllers_in_service(net, 'LineDropControl', kwargs.get('run_ldc_control', True))
        set_controllers_in_service(net, 'VoltVarController', kwargs.get('run_voltvar_control', True))                    

    correct_trafo_vnkv(net)

    if run_control and net.controller.in_service.any():
        def f(net, **kwargs):
            _initialize_model(net)
            _init_pf(net, initial_voltage=previous_voltage)
            _raise_for_unsupplied_injections(net, allow_islands=allow_islands)
            snap_pf(net, tol_vmag_pu, tol_vang_rad, MaxIter, **kwargs)
            net['_control_steps'] += 1
            net['converged'] = True
        net['_control_steps'] = 0
        pandapower.control.run_control(net, run=f, max_iter=60)
    else:
        has_reusable_model = (
            not init_model
            and hasattr(net, "model")
            and hasattr(net.model, "Y_tot")
            and hasattr(net.model, "y_nonslack")
        )
        if not has_reusable_model:
            _initialize_model(net, debug_level=debug_level)
            _init_pf(net, initial_voltage=previous_voltage)
        else:
            net.model.solved = False
            _apply_initial_voltage(net.model, previous_voltage)
        _raise_for_unsupplied_injections(net, allow_islands=allow_islands)
        snap_pf(net, tol_vmag_pu, tol_vang_rad, MaxIter)


def _raise_for_unsupplied_injections(net, *, allow_islands):
    """Reject active injections on source-disconnected terminals by default."""

    if allow_islands or len(net.model.y_isolated) == 0:
        return
    isolated = set(np.asarray(net.model.y_isolated, dtype=int).tolist())
    affected = []
    # Source-disconnected generation is simply unavailable; disconnected
    # demand is the condition that makes an ordinary power-flow result
    # incomplete and therefore requires explicit study opt-in.
    for table_name in ("asymmetric_load",):
        table = net.get(table_name)
        if table is None or len(table) == 0:
            continue
        for index, row in table.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            p_mw = float(row.get("p_mw", 0.0) or 0.0)
            q_mvar = float(row.get("q_mvar", 0.0) or 0.0)
            if abs(p_mw) + abs(q_mvar) == 0.0:
                continue
            terminal = int(row["bus"]) * 4 + int(row["from_phase"])
            y_index = int(net.model.terminal_to_y_lookup[terminal])
            if y_index < 0 or y_index in isolated:
                affected.append(f"{table_name}:{index!r}")
    if affected:
        net.model.solved = False
        preview = ", ".join(affected[:5])
        suffix = " ..." if len(affected) > 5 else ""
        raise LoadflowNotConverged(
            "Active injections are unsupplied by any slack source: "
            f"{preview}{suffix}. Pass allow_islands=True only for a study "
            "that explicitly records unsupplied demand."
        )


def _apply_initial_voltage(model, initial_voltage):
    """Warm-start non-slack terminals without changing fixed source voltages."""

    if initial_voltage is None:
        return
    candidate = np.asarray(initial_voltage, dtype=complex).reshape(-1)
    if candidate.size != int(model.y_size):
        return
    nonslack = np.asarray(model.y_nonslack, dtype=int)
    usable = np.isfinite(candidate[nonslack])
    if not np.any(usable):
        return
    warm = np.asarray(model.E0, dtype=complex).reshape(-1).copy()
    warm[nonslack[usable]] = candidate[nonslack[usable]]
    model.E0 = warm.reshape(-1, 1)


def _init_pf(net, initial_voltage=None):
    """
        Initialize the power flow results for first calculation.
        Voltage in all terminals is assumed as the corresponding phase at the slack bus.

        INPUT:
            **net** - The pandapower-like format network where elements are defined
            
    """
    net.model.solved = False
    model = net.model

    Y_pass = (model.Y_tran + model.Y_network + model.Y_ground + model.Y_source +
              model.Y_shunt + model.Y_switch)

    # Defining the set-point for the voltage source(s) at the slack bus
    slack_kind, slack = get_active_ext_grid_table(net, require_active=False)
    if slack is None or len(slack) == 0:
        raise LoadflowNotConverged("No active slack source found for power flow.")

    if slack_kind == "ext_grid":
        Vslack = slack['vm_pu'].to_numpy(dtype=np.float64) * np.exp(1j * np.deg2rad(slack['va_degree'].to_numpy(dtype=np.float64)))
    elif slack_kind == "ext_grid_sequence":
        alpha = np.exp(1j * np.pi * 2 / 3)
        T = np.array([[1, 1, 1], [1, alpha ** 2, alpha], [1, alpha, alpha ** 2]])
        Vslack = T @ (
            slack['vm_pu'].to_numpy(dtype=np.float64) * np.exp(1j * np.deg2rad(slack['va_degree'].to_numpy(dtype=np.float64)))
        )
    else:
        raise InvalidStudyInput("Unsupported active slack configuration.")

    y_slack = model.terminal_to_y_lookup[model.terminal_is_slack]
    model.y_fixed_voltage[y_slack] = Vslack.reshape(-1, 1)

    C = Y_pass.copy()
    for tname in ["asymmetric_load", "asymmetric_sgen"]:
        shunt = net[tname]
        buses = shunt["bus"].values
        from_phases = shunt["from_phase"].values
        to_phases = shunt["to_phase"].values
        y_from = net.model.terminal_to_y_lookup[buses * 4 + from_phases]
        y_to = net.model.terminal_to_y_lookup[buses * 4 + to_phases]
        mapped = (y_from >= 0) & (y_to >= 0)
        if not np.any(mapped):
            continue
        y_from = y_from[mapped]
        y_to = y_to[mapped]
        C += coo_matrix((np.ones(2 * len(y_from)), (np.hstack([y_from, y_to]), np.hstack([y_to, y_from]))),
                        shape=(model.y_size, model.y_size)).tocsr()
    y_connected = find_islands(net.model.y_size, y_slack.astype(int), C.indptr, C.indices)
    y_isolated = np.where(y_connected == -1)[0]
    net.model.y_fixed_voltage[y_isolated] = np.nan

    fixed_voltage = net.model.y_fixed_voltage[:, 0]
    y_fix = np.where(np.isfinite(fixed_voltage) & (fixed_voltage != -1))[0]
    y_nonslack = np.where(fixed_voltage == -1)[0]
    E_fix = net.model.y_fixed_voltage[y_fix]

    E0 = np.full((net.model.y_size, 1), np.nan + 0j, dtype=complex)
    E0[y_fix] = E_fix
    if len(y_nonslack) > 0:
        Ytot_0 = Y_pass + eye(net.model.y_size) * 1e-6
        Y_from_nonslack = Ytot_0[y_nonslack, :]
        Yna = Y_from_nonslack[:, y_fix]
        Ynn = Y_from_nonslack[:, y_nonslack]
        rhs = Yna @ E_fix
        Ynn_solver = linalg.splu(csc_matrix(Ynn))
        En = -Ynn_solver.solve(rhs)
        E0[y_nonslack] = En
    
    net.model.E0 = E0
    net.model.E_fix = E_fix
    net.model.Y_tot = Y_pass
    net.model.y_fix = y_fix
    net.model.y_nonslack = y_nonslack
    net.model.y_isolated = y_isolated
    _apply_initial_voltage(net.model, initial_voltage)


def snap_pf(net, tol_vmag_pu, tol_vang_rad, MaxIter,**kwargs):
    _cci_pf(net, tol_vmag_pu, tol_vang_rad, MaxIter)

    _bus_results_pf(net)
    _shunt_results_pf(net)
    _generator_results_pf(net)
    _line_results_pf(net)
    _trafo_results_pf(net)

    return net


def _prepare_pv_generators(net):
    gen = net.asymmetric_gen
    count = len(gen)
    result = {
        "row_index": list(gen.index),
        "active_mask": np.zeros(count, dtype=bool),
        "y_from": np.empty(0, dtype=np.int64),
        "y_to": np.empty(0, dtype=np.int64),
        "p_mw": np.empty(0, dtype=np.float64),
        "vm_target": np.empty(0, dtype=np.float64),
        "q_mvar": np.empty(0, dtype=np.float64),
        "q_min": np.empty(0, dtype=np.float64),
        "q_max": np.empty(0, dtype=np.float64),
        "q_limited": np.empty(0, dtype=bool),
        "bound_mode": np.empty(0, dtype=np.int8),
        "phase_to_phase": np.empty(0, dtype=bool),
        "voltage_history": np.empty(0, dtype=np.float64),
        "q_history": np.empty(0, dtype=np.float64),
    }
    if count == 0:
        return result

    buses = gen["bus"].to_numpy(dtype=np.int64)
    from_phases = gen["from_phase"].to_numpy(dtype=np.int64)
    to_phases = gen["to_phase"].to_numpy(dtype=np.int64)
    in_service = gen["in_service"].to_numpy(dtype=bool)
    scaling = gen["scaling"].to_numpy(dtype=np.float64) if "scaling" in gen.columns else np.ones(count, dtype=np.float64)
    p_mw = gen["p_mw"].to_numpy(dtype=np.float64) * scaling
    vm_target = gen["vm_pu"].to_numpy(dtype=np.float64)
    q_initial = gen["q_mvar"].to_numpy(dtype=np.float64) * scaling if "q_mvar" in gen.columns else np.zeros(count, dtype=np.float64)
    sn_mva = gen["sn_mva"].to_numpy(dtype=np.float64) if "sn_mva" in gen.columns else np.full(count, np.nan)
    explicit_q_min = gen["min_q_mvar"].to_numpy(dtype=np.float64) if "min_q_mvar" in gen.columns else np.full(count, np.nan)
    explicit_q_max = gen["max_q_mvar"].to_numpy(dtype=np.float64) if "max_q_mvar" in gen.columns else np.full(count, np.nan)

    y_from_all = net.model.terminal_to_y_lookup[buses * 4 + from_phases]
    y_to_all = net.model.terminal_to_y_lookup[buses * 4 + to_phases]
    nonisolated = (~np.isin(y_from_all, net.model.y_isolated)) & (~np.isin(y_to_all, net.model.y_isolated))
    mapped = (y_from_all >= 0) & (y_to_all >= 0)
    active_mask = in_service & mapped & nonisolated & np.isfinite(vm_target)
    result["active_mask"] = active_mask
    if not active_mask.any():
        return result

    active_pos = np.flatnonzero(active_mask)
    p_active = p_mw[active_pos]
    sn_active = sn_mva[active_pos]
    q_cap = np.where(
        np.isfinite(sn_active),
        np.sqrt(np.maximum(sn_active ** 2 - p_active ** 2, 0.0)),
        np.inf,
    )
    q_min = np.where(np.isfinite(explicit_q_min[active_pos]), explicit_q_min[active_pos], -q_cap)
    q_max = np.where(np.isfinite(explicit_q_max[active_pos]), explicit_q_max[active_pos], q_cap)
    q_min = np.maximum(q_min, -q_cap)
    q_max = np.minimum(q_max, q_cap)
    if np.any(q_min > q_max):
        raise InvalidStudyInput("asymmetric_gen reactive limits are inconsistent.")

    q_initial = np.clip(q_initial[active_pos], q_min, q_max)

    result.update(
        {
            "active_pos": active_pos,
            "y_from": y_from_all[active_pos].astype(np.int64),
            "y_to": y_to_all[active_pos].astype(np.int64),
            "p_mw": p_active,
            "vm_target": vm_target[active_pos],
            "q_mvar": q_initial,
            "q_min": q_min,
            "q_max": q_max,
            "q_limited": np.zeros(len(active_pos), dtype=bool),
            "bound_mode": np.zeros(len(active_pos), dtype=np.int8),
            "phase_to_phase": (to_phases[active_pos] != 0),
            "voltage_history": np.full(len(active_pos), np.nan, dtype=np.float64),
            "q_history": np.full(len(active_pos), np.nan, dtype=np.float64),
        }
    )
    return result


def _update_pv_generators(gen_state, E, voltage_tol):
    if len(gen_state["q_mvar"]) == 0:
        return 0.0, True

    voltage = E[gen_state["y_from"]] - E[gen_state["y_to"]]
    vm = np.abs(voltage).astype(np.float64).flatten()
    error = gen_state["vm_target"] - vm
    q_old = gen_state["q_mvar"].copy()
    max_q_change = 0.0
    all_satisfied = True

    for i in range(len(q_old)):
        limited_low = q_old[i] <= gen_state["q_min"][i] + 1e-9
        limited_high = q_old[i] >= gen_state["q_max"][i] - 1e-9
        if abs(error[i]) <= voltage_tol:
            gen_state["q_limited"][i] = limited_low or limited_high
            gen_state["bound_mode"][i] = -1 if limited_low else (1 if limited_high else 0)
            continue

        if (limited_high and error[i] > 0.0) or (limited_low and error[i] < 0.0):
            gen_state["q_limited"][i] = True
            gen_state["bound_mode"][i] = 1 if limited_high else -1
            continue

        span = max(gen_state["q_max"][i] - gen_state["q_min"][i], 0.01)
        if np.isfinite(gen_state["voltage_history"][i]) and np.isfinite(gen_state["q_history"][i]):
            dq_hist = q_old[i] - gen_state["q_history"][i]
            dv_hist = vm[i] - gen_state["voltage_history"][i]
            if abs(dq_hist) > 1e-9:
                sensitivity = dv_hist / dq_hist
            else:
                sensitivity = np.nan
        else:
            sensitivity = np.nan

        if not np.isfinite(sensitivity) or sensitivity <= 1e-6:
            sensitivity = 0.5 / span

        q_proposed = q_old[i] + error[i] / sensitivity
        max_step = 0.5 * span
        q_step = np.clip(q_proposed - q_old[i], -max_step, max_step)
        q_new = np.clip(q_old[i] + 0.6 * q_step, gen_state["q_min"][i], gen_state["q_max"][i])
        max_q_change = max(max_q_change, abs(q_new - q_old[i]))
        gen_state["q_mvar"][i] = q_new
        gen_state["q_limited"][i] = q_new <= gen_state["q_min"][i] + 1e-9 or q_new >= gen_state["q_max"][i] - 1e-9
        gen_state["bound_mode"][i] = -1 if q_new <= gen_state["q_min"][i] + 1e-9 else (1 if q_new >= gen_state["q_max"][i] - 1e-9 else 0)
        if abs(error[i]) > voltage_tol and not gen_state["q_limited"][i]:
            all_satisfied = False
        elif abs(error[i]) > voltage_tol and gen_state["q_limited"][i]:
            all_satisfied = all_satisfied and True

    gen_state["voltage_history"] = vm
    gen_state["q_history"] = q_old
    return max_q_change, all_satisfied


def _cci_pf(net,Tol_EM,Tol_EA,MaxIter):
    model = net["model"]
    E0 = model.E0
    E_fix = model.E_fix
    y_fix = model.y_fix
    y_nonslack = net.model.y_nonslack
    Ytot = model.Y_tot
    
    if len(y_nonslack) == 0:
        model.E = E0
        model.solved = True
        model.iterations = 0
        return

    En = E0[y_nonslack.astype(int)]

    Eph = abs(E0) > 0.2
    E0[Eph] = E0[Eph] / abs(E0[Eph])
    E0[y_fix] = E_fix

    Y1 = Ytot[y_nonslack, :]
    Ylg = Y1[:, y_fix]
    Yll = Y1[:, y_nonslack]

    model.Yll = Yll

    d = np.abs(Yll.diagonal())
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise LoadflowNotConverged(
            "Power flow topology contains disconnected or floating non-slack terminals."
        )
    D_inv = 1.0 / np.sqrt(d)
    D_inv_mat = diags(D_inv)
    Yll_scaled = D_inv_mat @ Yll @ D_inv_mat
    try:
        Yll_1 = linalg.splu(csc_matrix(Yll_scaled))
    except RuntimeError as exc:
        raise LoadflowNotConverged(
            "Power flow topology contains disconnected or floating non-slack terminals."
        ) from exc

    it = 0
    solved = False

    sbase = net.sn_mva * 1e6

    def ding(tname):
        shunt = net[tname]
        buses = shunt["bus"].values
        from_phases = shunt["from_phase"].values
        to_phases = shunt["to_phase"].values
        y_from = model.terminal_to_y_lookup[buses * 4 + from_phases]
        y_to = model.terminal_to_y_lookup[buses * 4 + to_phases]
        connected_y = (~np.isin(y_from, net.model.y_isolated)) & (y_from >= 0) & (y_to >= 0)
        absEsh0 = np.abs(E0[y_from] - E0[y_to]).flatten()
        S = (shunt["p_mw"].values + 1j * shunt["q_mvar"].values) * 1e6 * shunt["in_service"]
        kIp = shunt["const_i_percent_p"].values / 100.
        kZp = shunt["const_z_percent_p"].values / 100.
        kPp = 1 - (kIp + kZp)
        kIq = shunt["const_i_percent_q"].values / 100.
        kZq = shunt["const_z_percent_q"].values / 100.
        kPq = 1 - (kIq + kZq)
        S_const_power = kPp * np.real(S) + kPq * 1j*np.imag(S)
        S_const_current = kIp * np.real(S) + kIq * 1j*np.imag(S)
        S_const_impediance = kZp * np.real(S) + kZq * 1j*np.imag(S)
        return (y_from[connected_y], y_to[connected_y], absEsh0[connected_y], S_const_power[connected_y],
                S_const_current[connected_y], S_const_impediance[connected_y])
    
    load_y_from, load_y_to, load_absEsh0, load_S_const_power, load_S_const_current, load_S_const_impediance = ding("asymmetric_load")
    sgen_y_from, sgen_y_to, sgen_absEsh0, sgen_S_const_power, sgen_S_const_current, sgen_S_const_impediance = ding("asymmetric_sgen")
    gen_state = _prepare_pv_generators(net)

    y_from = np.hstack([load_y_from, sgen_y_from])
    y_to = np.hstack([load_y_to, sgen_y_to])
    absEsh0 = np.hstack([load_absEsh0, sgen_absEsh0])
    S_const_power = np.hstack([load_S_const_power, -sgen_S_const_power])
    S_const_current = np.hstack([load_S_const_current, -sgen_S_const_current])
    S_const_impediance = np.hstack([load_S_const_impediance, -sgen_S_const_impediance])

    def sum_currents(y_from, y_to, I_corr_sh):
        Icorr = np.zeros((model.y_size, 1), dtype=np.complex128)
        np.add.at(Icorr[:, 0], y_from, I_corr_sh)
        np.add.at(Icorr[:, 0], y_to, -I_corr_sh)
        return Icorr

    En_old = np.zeros((len(En), 1), dtype=complex)
    E = E0.copy()
    IFIX = Ylg @ E_fix
    pv_q_change = np.inf if len(gen_state["q_mvar"]) > 0 else 0.0
    pv_solved = len(gen_state["q_mvar"]) == 0
    while True:
        solved = np.max(np.abs(np.abs(En) - np.abs(En_old))) <= Tol_EM

        if solved:
            significant = np.abs(En) > 0.01
            if np.any(significant) and np.max(np.abs(np.angle(En[significant]) - np.angle(En_old[significant]))) > Tol_EA:
                 solved = False

        if (solved and pv_solved and pv_q_change <= 1e-6) or it == MaxIter:
            break

        En_old = En.copy()
        E_shunt = E[y_from] - E[y_to]
        E_shunt[E_shunt == 0] = 1

        absEsh = np.abs(E_shunt).flatten()
        S_act = S_const_power + \
                S_const_current * absEsh / absEsh0 + \
                S_const_impediance  * absEsh**2 / absEsh0**2
        I_corr_sh = -np.conj(S_act / sbase / E_shunt.flatten())
        injection_current = sum_currents(y_from, y_to, I_corr_sh)
        if len(gen_state["q_mvar"]) > 0:
            E_gen = E[gen_state["y_from"]] - E[gen_state["y_to"]]
            E_gen[E_gen == 0] = 1
            S_gen = -(gen_state["p_mw"] + 1j * gen_state["q_mvar"]) * 1e6
            I_corr_gen = -np.conj(S_gen / sbase / E_gen.flatten())
            injection_current += sum_currents(
                gen_state["y_from"], gen_state["y_to"], I_corr_gen
            )
        Il = injection_current[y_nonslack]
        Il_scaled = D_inv.reshape(-1, 1) * (Il - IFIX)
        En = D_inv.reshape(-1, 1) * Yll_1.solve(Il_scaled)
        E[y_nonslack] = En
        pv_q_change, pv_solved = _update_pv_generators(gen_state, E, max(Tol_EM * 5, 1e-5))
        it = it + 1

        if solved and pv_solved and pv_q_change <= 1e-6:
            break

    net.model.E = E
    net.model.gen_state = gen_state

#    net.model.S_load = S_act[:len(load_y_from)]
#    net.model.I_load = sum_currents(load_y_from, load_y_to, I_corr_sh[:len(load_y_from)])

#    net.model.S_sgen = S_act[len(load_y_from):]
#    net.model.I_sgen = sum_currents(sgen_y_from, sgen_y_to, I_corr_sh[len(load_y_from):])

    net.model.iterations = it
    net.model.solved = solved and pv_solved and pv_q_change <= 1e-6

    if it == MaxIter and (len(gen_state["q_mvar"]) == 0 or not pv_solved or pv_q_change > 1e-6):
        raise LoadflowNotConverged("Power Flow did not converge after {0} iterations!".format(MaxIter))



def _CorrCurr_PV(net,E,nph_G,nph_SG,CVa,D,kx):
    # Costruction of the correction current array for PV elements
    #Y_index=net.model.Y_index
    Icorr = np.zeros((net.model.y_size, 1), dtype=np.complex128)
    # Eg=np.zeros((nph_G.shape[0],1),dtype=complex)
    # for ng in range(Gen.shape[0]):
    #     N=Gen.iloc[ng]['bus']
    #     ph1=Gen.iloc[ng]['from_phase']
    #     ph2=Gen.iloc[ng]['to_phase']
    #     # Position in the overall system of equations of the terminals
    #     ph1_g=Y_index.loc[(N,ph1)]['yrow'].astype(int)
    #     ph2_g=Y_index.loc[(N,ph2)]['yrow'].astype(int)
    #     # Voltage set-point
    #     vm=Gen.iloc[ng]['vm_pu']
        
    #     # Get the voltage angle at the generator's terminals
    #     Eg[nph_G==ph1_g,0]=vm*(E[ph1_g]-E[ph2_g])/abs(E[ph1_g]-E[ph2_g])
    #     Eg[nph_G==ph2_g,0]=E[ph2_g]
        
    #     # Calculate the active power exchanged according to const admittance        
        
    # Ix=CVa + D@Eg + kx[np.in1d(nph_SG,nph_G)==True]
    
    # for ng in range(Gen.shape[0]):
    #     N=Gen.iloc[ng]['bus']
    #     ph1=Gen.iloc[ng]['from_phase']
    #     ph2=Gen.iloc[ng]['to_phase']
    #     # Position in the overall system of equations of the terminals
    #     ph1_g=Y_index.loc[(N,ph1)]['yrow'].astype(int)
    #     ph2_g=Y_index.loc[(N,ph2)]['yrow'].astype(int)
    #     # Active and reactive power set-points
    #     p=Gen.iloc[ng]['p_mw']/sbase
        
    #     # Calculate active power injection due to const admitt
    #     Ey=Eg[nph_G==ph1_g]-Eg[nph_G==ph2_g]
        
    #     # Calculate the active power variation with respect to P set-point
    #     sx=Ey*np.conjugate(Ix[nph_G==ph1_g])
    #     px=np.real(sx)
    #     deltap=p-(px)
    #     # Calculate current injection for active power adjustment
    #     Ir=deltap/np.conjugate(Ey)
        
        
    #     Icorr[ph1_g]=Icorr[ph1_g]+Ix[nph_G==ph1_g]+Ir
    #     Icorr[ph2_g]=Icorr[ph2_g]-Ix[nph_G==ph1_g]-Ir
        
    return Icorr
