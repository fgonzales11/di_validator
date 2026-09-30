"""
KLU-based power flow solver for multiconductor networks.

This module provides ``run_klu_pf``, which is functionally identical to the
existing ``run_pf`` in ``cci_powerflow.py`` but replaces ALL SciPy sparse
solves with KLU factorization via KLUSolve.dll — both the initial voltage
guess (``_init_pf_klu``) and the CCI iteration (``_cci_pf_klu``).
"""

import numpy as np
from scipy.sparse import csc_matrix, eye, coo_matrix

import multiconductor
from multiconductor.pycci.pf_results import (
    _bus_results_pf,
    _line_results_pf,
    _trafo_results_pf,
    _shunt_results_pf,
)
from multiconductor.pycci.cci_powerflow import set_controllers_in_service
from multiconductor.pycci.model import _initialize_model, find_islands

from .klu_solver import KLUSolver

# pandapower monkey-patch is already applied by cci_powerflow (imported above)
import pandapower.control


def run_klu_pf(
    net,
    tol_vmag_pu=1e-5,
    tol_vang_rad=1e-5,
    MaxIter=100,
    run_control=False,
    debug_level=0,
    **kwargs,
):
    """
    Run a snapshot power flow using the KLU sparse solver (via KLUSolve.dll).

    This function is a drop-in replacement for ``run_pf`` — the only difference
    is that the sparse LU factorisation is performed by KLU instead of SciPy's
    SuperLU.

    INPUT:
        **net** - The pandapower-like format network where elements are defined

    OPTIONAL:
        **tol_vmag_pu** (float, 1e-5) - Voltage magnitude tolerance in pu.
        **tol_vang_rad** (float, 1e-5) - Voltage angle tolerance in rad.
        **MaxIter** (int, 100) - Maximum CCI iterations.
        **run_control** (bool, False) - Enable controller loop.
        **debug_level** (int, 0) - Verbosity.
        **kwargs** - Same controller flags as ``run_pf``.
    """
    if net.switch.closed.dtype != np.dtype("bool"):
        net.switch.closed = net.switch.closed.astype(bool)

    if run_control and any(
        flag in kwargs
        for flag in [
            "run_capacitor_control",
            "run_ltc_control",
            "run_ldc_control",
            "run_voltvar_control",
        ]
    ):
        set_controllers_in_service(
            net, "BinaryShuntController", kwargs.get("run_capacitor_control", True)
        )
        set_controllers_in_service(
            net, "LoadTapChangerControl", kwargs.get("run_ltc_control", True)
        )
        set_controllers_in_service(
            net, "LineDropControl", kwargs.get("run_ldc_control", True)
        )
        set_controllers_in_service(
            net, "VoltVarController", kwargs.get("run_voltvar_control", True)
        )

    if run_control and net.controller.in_service.any():

        def f(net, **kwargs):
            _initialize_model(net)
            _init_pf_klu(net)
            _snap_pf_klu(net, tol_vmag_pu, tol_vang_rad, MaxIter, **kwargs)
            net["_control_steps"] += 1
            net["converged"] = True

        net["_control_steps"] = 0
        pandapower.control.run_control(net, run=f, max_iter=60)
    else:
        _initialize_model(net, debug_level=debug_level)
        _init_pf_klu(net)
        _snap_pf_klu(net, tol_vmag_pu, tol_vang_rad, MaxIter)


def _init_pf_klu(net):
    """
    Initialize the power flow — KLU version.

    Same as ``_init_pf`` in ``cci_powerflow.py`` but uses KLUSolver for the
    initial voltage solve instead of ``scipy.sparse.linalg.splu``.
    """
    net.model.solved = False
    model = net.model

    Y_pass = (model.Y_tran + model.Y_network + model.Y_ground + model.Y_source +
              model.Y_shunt + model.Y_switch)

    # Defining the set-point for the voltage source(s) at the slack bus
    if net.ext_grid.shape[0] > 0:
        slack = net.ext_grid
        Vslack = slack['vm_pu'].values * np.exp(1j * slack['va_degree'].values * np.pi / 180)
    elif net.ext_grid_sequence.shape[0] > 0:
        alpha = np.exp(1j * np.pi * 2 / 3)
        T = np.array([[1, 1, 1], [1, alpha ** 2, alpha], [1, alpha, alpha ** 2]])
        slack = net.ext_grid_sequence
        Vslack = T @ (slack['vm_pu'] * np.exp(1j * slack['va_degree'] * np.pi / 180))

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
        C += coo_matrix((np.ones(2 * len(y_from)), (np.hstack([y_from, y_to]), np.hstack([y_to, y_from]))),
                        shape=(model.y_size, model.y_size)).tocsr()
    y_connected = find_islands(net.model.y_size, y_slack, C.indptr, C.indices)
    y_isolated = np.where(y_connected == -1)[0]
    net.model.y_fixed_voltage[y_isolated] = np.nan

    y_fix = np.where((net.model.y_fixed_voltage != -1))[0]
    y_nonslack = np.where(net.model.y_fixed_voltage == -1)[0]
    E_fix = net.model.y_fixed_voltage[y_fix]

    Ytot_0 = Y_pass + eye(net.model.y_size) * 1e-6
    Y_from_nonslack = Ytot_0[y_nonslack, :]
    Yna = Y_from_nonslack[:, y_fix]
    Ynn = Y_from_nonslack[:, y_nonslack]
    rhs = Yna @ E_fix

    # ── KLU instead of scipy.sparse.linalg.splu ──
    Ynn_solver = KLUSolver(csc_matrix(Ynn))
    En = -Ynn_solver.solve(rhs)

    E0 = np.zeros((net.model.y_size, 1), dtype=complex)
    E0[y_fix] = E_fix
    E0[y_nonslack] = En

    net.model.E0 = E0
    net.model.E_fix = E_fix
    net.model.Y_tot = Y_pass
    net.model.y_fix = y_fix
    net.model.y_nonslack = y_nonslack
    net.model.y_isolated = y_isolated


def _snap_pf_klu(net, tol_vmag_pu, tol_vang_rad, MaxIter, **kwargs):
    """Run the CCI iteration with KLU, then compute results."""
    _cci_pf_klu(net, tol_vmag_pu, tol_vang_rad, MaxIter)

    _bus_results_pf(net)
    _shunt_results_pf(net)
    _line_results_pf(net)
    _trafo_results_pf(net)

    return net


def _cci_pf_klu(net, Tol_EM, Tol_EA, MaxIter):
    """
    Correction-Current-Injection power flow iteration using KLU factorisation.

    Identical algorithm to ``_cci_pf`` in ``cci_powerflow.py`` — the only
    change is ``KLUSolver(csc_matrix(Yll))`` instead of
    ``scipy.sparse.linalg.splu(csc_matrix(Yll))``.
    """
    model = net["model"]
    E0 = model.E0
    E_fix = model.E_fix
    y_fix = model.y_fix
    y_nonslack = net.model.y_nonslack
    Ytot = model.Y_tot

    if len(y_fix) == model.y_size:
        model.E = E0
        return

    En = E0[y_nonslack.astype(int)]

    Eph = abs(E0) > 0.2
    E0[Eph] = E0[Eph] / abs(E0[Eph])

    Y1 = Ytot[y_nonslack, :]
    Ylg = Y1[:, y_fix]
    Yll = Y1[:, y_nonslack]

    model.Yll = Yll
    model.y_nonslack = y_nonslack

    # ── KLU factorisation instead of SciPy splu ──
    Yll_1 = KLUSolver(csc_matrix(Yll))

    it = 0
    solved = False

    sbase = net.sn_mva * 1e6

    def ding(tname):
        shunt = net[tname]
        buses = shunt["bus"].values
        from_phases = shunt["from_phase"].values
        to_phases = shunt["to_phase"].values
        y_from = model.terminal_to_y_lookup[buses * 4 + from_phases]
        connected_y = ~np.isin(y_from, net.model.y_isolated)
        y_to = model.terminal_to_y_lookup[buses * 4 + to_phases]
        absEsh0 = np.abs(E0[y_from] - E0[y_to]).flatten()
        if any(absEsh0 == 0):
            if model["debug_level"] > 0:
                print(f"disconnected {sum(absEsh0 == 0)} {tname}")
            absEsh0[absEsh0 == 0] = 1
        S = (shunt["p_mw"].values + 1j * shunt["q_mvar"].values) * 1e6 * shunt["in_service"]
        kIp = shunt["const_i_percent_p"].values / 100.0
        kZp = shunt["const_z_percent_p"].values / 100.0
        kPp = 1 - (kIp + kZp)
        kIq = shunt["const_i_percent_q"].values / 100.0
        kZq = shunt["const_z_percent_q"].values / 100.0
        kPq = 1 - (kIq + kZq)
        S_const_power = kPp * np.real(S) + kPq * 1j * np.imag(S)
        S_const_current = kIp * np.real(S) + kIq * 1j * np.imag(S)
        S_const_impediance = kZp * np.real(S) + kZq * 1j * np.imag(S)
        return (
            y_from[connected_y],
            y_to[connected_y],
            absEsh0[connected_y],
            S_const_power[connected_y],
            S_const_current[connected_y],
            S_const_impediance[connected_y],
        )

    (
        load_y_from, load_y_to, load_absEsh0,
        load_S_const_power, load_S_const_current, load_S_const_impediance,
    ) = ding("asymmetric_load")
    (
        sgen_y_from, sgen_y_to, sgen_absEsh0,
        sgen_S_const_power, sgen_S_const_current, sgen_S_const_impediance,
    ) = ding("asymmetric_sgen")

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
    while True:
        solved = np.max(np.abs(np.abs(En) - np.abs(En_old))) <= Tol_EM

        if solved:
            significant = np.abs(En) > 0.01
            if np.any(significant) and np.max(
                np.abs(np.angle(En[significant]) - np.angle(En_old[significant]))
            ) > Tol_EA:
                solved = False

        if solved or it == MaxIter:
            break

        En_old = En.copy()
        E_shunt = E[y_from] - E[y_to]
        E_shunt[E_shunt == 0] = 1

        absEsh = np.abs(E_shunt).flatten()
        S_act = (
            S_const_power
            + S_const_current * absEsh / absEsh0
            + S_const_impediance * absEsh**2 / absEsh0**2
        )
        I_corr_sh = -np.conj(S_act / sbase / E_shunt.flatten())
        I = sum_currents(y_from, y_to, I_corr_sh)
        Il = I[y_nonslack]
        En = Yll_1.solve(Il - IFIX)

        E[y_nonslack] = En
        it = it + 1

    net.model.E = E
    net.model.iterations = it
    net.model.solved = solved
