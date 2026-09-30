# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings

import numpy as np
from scipy.sparse.linalg import inv as inv_sparse
from scipy.linalg import inv

BIG_NUMBER = 1e20  # was previously importable from pandapower.pd2ppc_zero; now defined locally
from pandapower.pypower.idx_bus_sc import R_EQUIV, X_EQUIV
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_STATUS
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.auxiliary import _clean_up

try:
    from pandapower.pf.makeYbus_numba import makeYbus
except ImportError:
    from pandapower.pypower.makeYbus import makeYbus


def _calc_rx(net, ppci, bus_idx, sequence):
    if bus_idx is None or len(bus_idx) == 0:
        return
    # Vectorized for multiple bus
    fault = net._options["fault"]
    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppci["bus"][bus_idx, BASE_KV]) / ppci["baseMVA"]
        fault_impedance = (r_fault + x_fault * 1j) / base_r
    else:
        fault_impedance = 0 + 0j
    net._options["fault_impedance"] = fault_impedance

    if net["_options"]["inverse_y"]:
        Zbus = ppci["internal"].get("Zbus")
        if Zbus is None:
            from multiconductor.shortcircuit.impedance import _calc_zbus
            _calc_zbus(net, ppci)
            Zbus = ppci["internal"].get("Zbus")
        if Zbus is None:
            return
        Zbus = np.asarray(Zbus)
        if Zbus.ndim < 2:
            z_equiv = np.zeros(bus_idx.shape[0], dtype=np.complex128) + fault_impedance
            ppci["bus"][bus_idx, R_EQUIV] = z_equiv.real
            ppci["bus"][bus_idx, X_EQUIV] = z_equiv.imag
            return
        if (fault == "LL"):
            if (sequence == 1) or (sequence == 2):
                z_equiv = np.diag(Zbus)[bus_idx] + fault_impedance/2
            else:
                z_equiv = np.diag(Zbus)[bus_idx]
        else:
            z_equiv = np.diag(Zbus)[bus_idx] + fault_impedance
    else: 
        z_equiv = _calc_zbus_diag(net, ppci, bus_idx) + fault_impedance
    ppci["bus"][bus_idx, R_EQUIV] = z_equiv.real
    ppci["bus"][bus_idx, X_EQUIV] = z_equiv.imag


def _calc_ybus(ppci):
    branch = ppci["branch"]
    if branch.size:
        zero_z = (branch[:, BR_R] == 0) & (branch[:, BR_X] == 0) & (branch[:, BR_STATUS] != 0)
        if np.any(zero_z):
            branch = branch.copy()
            branch[zero_z, BR_X] = 1e-9
            ppci["branch"] = branch
    Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    if np.isnan(Ybus.data).any():
        raise ValueError("nan value detected in Ybus matrix - check calculation parameters for nan values")

    nonzero = Ybus.nonzero()
    if len(nonzero[0]) > 0:
        nz_vals = Ybus.data
        if nz_vals.size:
            mask = np.abs(nz_vals) <= (10 / (BIG_NUMBER * ppci["baseMVA"]))
            if mask.any():
                rows = nonzero[0][mask]
                cols = nonzero[1][mask]
                Ybus[rows[rows != cols], cols[rows != cols]] = 0
                Ybus.eliminate_zeros()

    # nonzero = Yf.nonzero()
    # nonzero_mask = np.array(abs(Yf[nonzero]) <= (10 / (BIG_NUMBER * ppci["baseMVA"])))[0]
    # if len(nonzero_mask) > 0:
    #     rows = nonzero[0][nonzero_mask]
    #     cols = nonzero[1][nonzero_mask]
    #     Yf[rows[rows != cols], cols[rows != cols]] = 0
    #     Yf.eliminate_zeros()
    #
    # nonzero = Yt.nonzero()
    # nonzero_mask = np.array(abs(Yt[nonzero]) <= (10 / (BIG_NUMBER * ppci["baseMVA"])))[0]
    # if len(nonzero_mask) > 0:
    #     rows = nonzero[0][nonzero_mask]
    #     cols = nonzero[1][nonzero_mask]
    #     Yt[rows[rows != cols], cols[rows != cols]] = 0
    #     Yt.eliminate_zeros()

    ppci["internal"]["Yf"] = Yf
    ppci["internal"]["Yt"] = Yt
    ppci["internal"]["Ybus"] = Ybus


def _calc_zbus(net, ppci):
    try:
        Ybus = ppci["internal"]["Ybus"]
        if Ybus.shape[0] == 0:
            ppci["internal"]["Zbus"] = np.zeros((0, 0), dtype=np.complex128)
            return
        sparsity = Ybus.nnz / Ybus.shape[0]**2
        if sparsity < 0.002:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ppci["internal"]["Zbus"] = inv_sparse(Ybus).toarray()
        else:
            ppci["internal"]["Zbus"] = inv(Ybus.toarray())
    except Exception as e:
        _clean_up(net, res=False)
        raise (e)


def _calc_zbus_diag(net, ppci, bus_idx=None):
    ybus_fact = ppci["internal"]["ybus_fact"]
    n_ppci_bus = ppci["bus"].shape[0]

    if bus_idx is None:
        diagZ = np.zeros(n_ppci_bus, dtype=np.complex128)
        for i in range(n_ppci_bus):
            b = np.zeros(n_ppci_bus, dtype=np.complex128)
            b[i] = 1 + 0j
            diagZ[i] = ybus_fact(b)[i]
        ppci["internal"]["diagZ"] = diagZ
        return diagZ
    else:
        diagZ = np.zeros(bus_idx.shape[0], dtype=np.complex128)
        for ix, b in enumerate(bus_idx):
            rhs = np.zeros(n_ppci_bus, dtype=np.complex128)
            rhs[b] = 1 + 0j
            diagZ[ix] = ybus_fact(rhs)[b]
        return diagZ
