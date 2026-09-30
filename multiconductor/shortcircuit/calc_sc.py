# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import copy
import dataclasses
import inspect
import warnings
from pandapower.build_bus import _add_load_sc_impedances_ppc
import numpy as np
from scipy.sparse.linalg import factorized
from numbers import Number
from pandapower.auxiliary import (
    _clean_up,
    _add_ppc_options,
    _add_sc_options,
    _add_auxiliary_elements,
)
from pandapower.pd2ppc import _ppc2ppci
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.results import _copy_results_ppci_to_ppc
from multiconductor.shortcircuit.currents import (
    _calc_ikss_to_g,
    _calc_branch_currents_complex,
)
from multiconductor.shortcircuit.impedance import _calc_zbus, _calc_ybus, _calc_rx
from multiconductor.shortcircuit.ppc_conversion import (
    _init_ppc,
    _create_k_updated_ppci,
    _get_is_ppci_bus,
)
from multiconductor.shortcircuit.kappa import _add_kappa_to_ppc
from multiconductor.shortcircuit.results import _extract_results
from multiconductor.shortcircuit.native import (
    FaultSpec,
    ShortCircuitSourceError,
    calc_sc_native,
    fault_spec_from_options,
)
from pandapower.results import init_results
from pandapower.pypower.idx_brch_sc import K_ST
import logging

logger = logging.getLogger(__name__)


def _add_ppc_options_compat(net, **kwargs):
    """
    Call pandapower's _add_ppc_options with only the keyword arguments supported
    by the installed pandapower version.

    Some environments expose an _add_ppc_options signature that includes
    `enforce_p_lims`, while others do not. Short-circuit analysis only needs a
    stable subset, so we filter dynamically instead of binding to one version.
    """
    signature = inspect.signature(_add_ppc_options)
    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return _add_ppc_options(net, **supported_kwargs)


def _ensure_sc_tables(net):
    """Ensure pandapower-required tables exist for short-circuit calculation."""
    import pandas as pd

    try:
        import pandapower as pp
    except Exception:  # pragma: no cover - pandapower is a hard dependency in this repo
        return

    pp_net = pp.create_empty_network()
    for name, value in pp_net.items():
        if not isinstance(value, pd.DataFrame):
            continue
        if name not in net:
            net[name] = value.copy()
            continue
        # add any missing columns with sensible defaults
        missing_cols = [col for col in value.columns if col not in net[name].columns]
        for col in missing_cols:
            dtype = value[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                default = False
            elif dtype.kind in ("i", "u"):
                default = 0
            elif dtype.kind == "f":
                default = np.nan
            else:
                default = None
            net[name][col] = default


def _flatten_mc_index(df):
    import pandas as pd

    if not hasattr(df, "index") or not hasattr(df.index, "get_level_values"):
        return df
    if not isinstance(df.index, pd.MultiIndex):
        return df
    flat = df.copy()
    flat.index = flat.index.get_level_values(0)
    flat = flat[~flat.index.duplicated(keep="first")]
    flat.index.name = None
    return flat


def _to_bool_series(series, default=False):
    import pandas as pd

    if series.dtype == bool:
        return series.fillna(default)

    truthy = {"true", "t", "1", "yes", "y", "on"}
    falsy = {"false", "f", "0", "no", "n", "off"}

    def _coerce(v):
        if pd.isna(v):
            return default
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer, float, np.floating)):
            return bool(v)
        text = str(v).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return default

    return series.map(_coerce).astype(bool)


def _clone_net_for_sc(net):
    import pandas as pd

    net_sc = net.__class__({})
    if hasattr(net, "_allow_invalid_attributes"):
        net_sc._setattr("_allow_invalid_attributes", net._allow_invalid_attributes)

    for key, value in net.items():
        if isinstance(value, pd.DataFrame):
            net_sc[key] = value.copy(deep=True)

    for key in ("name", "f_hz", "sn_mva", "user_pf_options", "std_types"):
        if key in net:
            net_sc[key] = copy.deepcopy(net[key])

    if "_pd2ppc_lookups" not in net_sc:
        net_sc["_pd2ppc_lookups"] = {
            "bus": None,
            "gen": None,
            "branch": None,
            "ext_grid": None,
            "bus_dc": None,
            "branch_dc": None,
            "aux": None,
            "aux_dc": None,
            "merged_bus": None,
        }
    if "_is_elements" not in net_sc:
        net_sc["_is_elements"] = None
    if "_is_elements_final" not in net_sc:
        net_sc["_is_elements_final"] = None

    return net_sc


def _flatten_mc_net_for_sc(net):
    import pandas as pd

    net_sc = _clone_net_for_sc(net)
    net_sc["_mc_sc_flattened"] = True
    _ensure_sc_tables(net_sc)
    if isinstance(net_sc.get("_pd2ppc_lookups"), pd.DataFrame):
        net_sc["_pd2ppc_lookups"] = {
            "bus": None,
            "gen": None,
            "branch": None,
            "ext_grid": None,
            "bus_dc": None,
            "branch_dc": None,
            "aux": None,
            "aux_dc": None,
            "merged_bus": None,
        }
    # Fill line parameters from multiconductor sequence std types for pandapower pd2ppc
    if "line" in net_sc and isinstance(net_sc["line"], pd.DataFrame):
        seq_types = (
            net_sc.std_types.get("sequence") if hasattr(net_sc, "std_types") else None
        )
        if seq_types:
            line = net_sc["line"]
            if "parallel" not in line.columns:
                line["parallel"] = 1
            else:
                line["parallel"] = line["parallel"].replace(0, 1).fillna(1)
            if "df" not in line.columns:
                line["df"] = 1.0
            else:
                line["df"] = line["df"].replace(0, 1.0).fillna(1.0)
            if "g_us_per_km" not in line.columns:
                line["g_us_per_km"] = 0.0
            for col in (
                "r_ohm_per_km",
                "x_ohm_per_km",
                "c_nf_per_km",
                "r0_ohm_per_km",
                "x0_ohm_per_km",
                "c0_nf_per_km",
                "max_i_ka",
            ):
                if col not in line.columns:
                    line[col] = np.nan
            for idx, row in line.iterrows():
                std = row.get("std_type")
                if not std or std not in seq_types:
                    continue
                st = seq_types[std]
                for col in (
                    "r_ohm_per_km",
                    "x_ohm_per_km",
                    "c_nf_per_km",
                    "r0_ohm_per_km",
                    "x0_ohm_per_km",
                    "c0_nf_per_km",
                    "max_i_ka",
                ):
                    if pd.isna(line.at[idx, col]) and col in st:
                        line.at[idx, col] = st[col]
            # Fill remaining NaNs with safe defaults to avoid NaNs in Ybus
            for col in (
                "r_ohm_per_km",
                "x_ohm_per_km",
                "c_nf_per_km",
                "r0_ohm_per_km",
                "x0_ohm_per_km",
                "c0_nf_per_km",
                "g_us_per_km",
            ):
                if col in line.columns:
                    line[col] = line[col].fillna(0.0)
            if "max_i_ka" in line.columns:
                line["max_i_ka"] = line["max_i_ka"].fillna(0.0)
            if "length_km" in line.columns:
                line["length_km"] = line["length_km"].fillna(0.0)
            if "endtemp_degree" not in line.columns:
                line["endtemp_degree"] = 20.0
            else:
                line["endtemp_degree"] = line["endtemp_degree"].fillna(20.0)                    
            if "r_ohm_per_km" in line.columns and "x_ohm_per_km" in line.columns:
                zero_z = (line["r_ohm_per_km"] == 0) & (line["x_ohm_per_km"] == 0)
                if zero_z.any():
                    line.loc[zero_z, "x_ohm_per_km"] = 1e-6
            net_sc["line"] = line
    if "switch" in net_sc and isinstance(net_sc["switch"], pd.DataFrame):
        if "closed" in net_sc["switch"].columns:
            net_sc["switch"]["closed"] = _to_bool_series(
                net_sc["switch"]["closed"], default=True
            )
        if "z_ohm" not in net_sc["switch"].columns:
            if "r_ohm" in net_sc["switch"].columns:
                net_sc["switch"]["z_ohm"] = net_sc["switch"]["r_ohm"].fillna(0.0)
            else:
                net_sc["switch"]["z_ohm"] = 0.0
        else:
            if "r_ohm" in net_sc["switch"].columns:
                net_sc["switch"]["z_ohm"] = (
                    net_sc["switch"]["z_ohm"]
                    .fillna(net_sc["switch"]["r_ohm"])
                    .fillna(0.0)
                )
            else:
                net_sc["switch"]["z_ohm"] = net_sc["switch"]["z_ohm"].fillna(0.0)
    if "ext_grid" in net_sc and isinstance(net_sc["ext_grid"], pd.DataFrame):
        ext_grid = net_sc["ext_grid"]
        ext_grid_columns = list(ext_grid.columns)
        if (
            ext_grid.empty
            and "ext_grid_sequence" in net_sc
            and not net_sc["ext_grid_sequence"].empty
        ):

            def _scalar(val):
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    if len(val) >= 2:
                        return float(val[1])
                    if len(val) == 1:
                        return float(val[0])
                    return np.nan
                return float(val) if val is not None else np.nan

            rows = []
            for _, row in net_sc["ext_grid_sequence"].groupby(level=0):
                r0 = row.iloc[0]
                rows.append(
                    {
                        "name": r0.get("name", None),
                        "bus": r0.get("bus", np.nan),
                        "vm_pu": _scalar(r0.get("vm_pu", 1.0)),
                        "va_degree": _scalar(r0.get("va_degree", 0.0)),
                        "in_service": bool(r0.get("in_service", True)),
                    }
                )
            if rows:
                ext_grid = pd.DataFrame(rows)
                for col in ext_grid_columns:
                    if col not in ext_grid.columns:
                        ext_grid[col] = np.nan
                ext_grid = ext_grid[ext_grid_columns]
                ext_grid.index = pd.RangeIndex(len(ext_grid))
                net_sc["ext_grid"] = ext_grid
        # Ensure SC-required columns exist
        sc_cols = [
            "s_sc_max_mva",
            "rx_max",
            "x0x_max",
            "r0x0_max",
            "s_sc_min_mva",
            "rx_min",
            "x0x_min",
            "r0x0_min",
        ]
        for col in sc_cols:
            if col not in net_sc["ext_grid"].columns:
                net_sc["ext_grid"][col] = np.nan
        missing_strength = [
            col
            for col in ("s_sc_max_mva", "rx_max", "s_sc_min_mva", "rx_min")
            if col in net_sc["ext_grid"].columns
            and net_sc["ext_grid"][col].isna().any()
        ]
        if missing_strength:
            raise ShortCircuitSourceError(
                "balanced short-circuit conversion requires explicit external-grid "
                f"source-strength fields; missing values in {missing_strength}"
            )
    for key, value in list(net_sc.items()):
        if isinstance(value, pd.DataFrame):
            net_sc[key] = _flatten_mc_index(value)
    return net_sc


def _expand_sc_results(net_sc, net):
    import pandas as pd

    def _expand(df, target_index):
        if not isinstance(target_index, pd.MultiIndex):
            return df
        base_index = target_index.get_level_values(0)
        expanded = df.reindex(base_index).copy()
        expanded.index = target_index
        return expanded

    for name in (
        "res_bus_sc",
        "res_line_sc",
        "res_switch_sc",
        "res_trafo_sc",
        "res_trafo3w_sc",
    ):
        if name not in net_sc:
            continue
        df = net_sc[name]
        if name == "res_bus_sc":
            expanded = _expand(df, net.bus.index)
        elif name == "res_line_sc" and "line" in net:
            expanded = _expand(df, net.line.index)
        elif name == "res_switch_sc" and "switch" in net:
            expanded = _expand(df, net.switch.index)
        elif name == "res_trafo_sc" and "trafo" in net:
            expanded = _expand(df, net.trafo.index)
        elif name == "res_trafo3w_sc" and "trafo3w" in net:
            expanded = _expand(df, net.trafo3w.index)
        else:
            expanded = df

        # Fill NaNs with 0.0 for current and power columns representing isolated elements
        for col in expanded.columns:
            if (
                "ikss" in col
                or "skss" in col
                or "ip" in col
                or "ith" in col
                or col.startswith("p_")
                or col.startswith("q_")
            ):
                expanded[col] = expanded[col].fillna(0.0)

        net[name] = expanded


def calc_sc(
    net,
    bus=None,
    fault="3ph",
    case="max",
    lv_tol_percent=10,
    topology="auto",
    ip=False,
    ith=False,
    tk_s=1.0,
    kappa_method="C",
    r_fault_ohm=0.0,
    x_fault_ohm=0.0,
    branch_results=False,
    check_connectivity=True,
    return_all_currents=False,
    inverse_y=True,
    use_pre_fault_voltage=False,
    phases=None,
    fault_spec=None,
):
    """
    Calculates minimal or maximal symmetrical short-circuit currents.
    The calculation is based on the method of the equivalent voltage source
    according to DIN/IEC EN 60909.
    The initial short-circuit alternating current *ikss* is the basis of the short-circuit
    calculation and is therefore always calculated.
    Other short-circuit currents can be calculated from *ikss* with the conversion factors defined
    in DIN/IEC EN 60909.

    The output is stored in the net.res_bus_sc table as a short_circuit current
    for each bus.

    INPUT:
        **net** (pandapowerNet) pandapower Network

        **bus** (int, list, np.array, None) defines if short-circuit calculations should only be calculated for defined bus

        ***fault** (str, 3ph) type of fault

            - "3ph" for three-phase

            - "2ph" for two-phase (phase-to-phase) short-circuits

            - "1ph" for single-phase-to-ground faults

        ***fault** (str, LLL) type of fault

            - "LLL" for three-phase

            - "LL" for two-phase (phase-to-phase) short-circuits

            - "LG" for single-phase-to-ground faults

            - "LLG" for double-phase-to-ground faults

        **case** (str, "max")

            - "max" for maximal current calculation

            - "min" for minimal current calculation

        **lv_tol_percent** (int, 10) voltage tolerance in low voltage grids

            - 6 for 6% voltage tolerance

            - 10 for 10% voltage olerance

        **ip** (bool, False) if True, calculate aperiodic short-circuit current

        **ith** (bool, False) if True, calculate equivalent thermical short-circuit current Ith

        **topology** (str, "auto") define option for meshing (only relevant for ip and ith)

            - "meshed" - it is assumed all buses are supplied over multiple paths

            - "radial" - it is assumed all buses are supplied over exactly one path

            - "auto" - topology check for each bus is performed to see if it is supplied over multiple paths

        **tk_s** (float, 1) failure clearing time in seconds (only relevant for ith)

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

        **branch_results** (bool, False) defines if short-circuit results should also be generated for branches

        **return_all_currents** (bool, False) applies only if branch_results=True, if True short-circuit currents for
        each (branch, bus) tuple is returned otherwise only the max/min is returned

        **inverse_y** (bool, True) defines if complete inverse should be used instead of LU factorization, factorization version is in experiment which should be faster and memory efficienter

        **use_pre_fault_voltage** (bool, False) whether to consider the pre-fault grid state (superposition method, "Type C"). The user must first execute pp.runpp(net) before executing sc.calc_sc in this case

        **phases** (tuple, None) selected multiconductor phase labels. Required
        counts are three for LLL, two for LL/LLG, and one for LG.

        **fault_spec** (FaultSpec, None) explicit native multiconductor fault
        definition. When supplied, its calculation settings take precedence;
        ``bus`` and ``phases`` remain optional selection overrides.


    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
    """
    import pandas as pd

    is_mc_net = isinstance(net.bus.index, pd.MultiIndex)
    if not is_mc_net and "ext_grid_sequence" in net:
        try:
            is_mc_net = (
                isinstance(net.ext_grid_sequence, pd.DataFrame)
                and not net.ext_grid_sequence.empty
            )
        except Exception:
            is_mc_net = False
    if net.get("_mc_sc_flattened", False):
        is_mc_net = False

    if is_mc_net:
        if fault_spec is None:
            native_spec = fault_spec_from_options(
                fault=fault,
                bus=None,
                phases=phases,
                case=case,
                r_fault_ohm=r_fault_ohm,
                x_fault_ohm=x_fault_ohm,
                ip=ip,
                ith=ith,
                tk_s=tk_s,
                use_pre_fault_voltage=use_pre_fault_voltage,
            )
        elif isinstance(fault_spec, FaultSpec):
            native_spec = fault_spec
            if phases is not None:
                native_spec = dataclasses.replace(native_spec, phases=tuple(phases))
        else:
            raise TypeError("fault_spec must be a FaultSpec instance")
        calc_sc_native(
            net,
            native_spec,
            bus=bus,
            lv_tol_percent=lv_tol_percent,
            branch_results=branch_results,
            return_all_currents=return_all_currents,
        )
        return

    _ensure_sc_tables(net)
    if fault in ["3ph", "2ph", "1ph", "2ph-g"]:
        msg = (
            "Short-circuit fault types 3ph, 2ph, 2ph-g and 1ph have been renamed to LLL, LL, LLG and LG, "
            "please use the new naming convention as the old convention will be removed in future pandapower versions."
        )
        warnings.warn(msg, DeprecationWarning)
        mapping = {"3ph": "LLL", "2ph": "LL", "1ph": "LG", "2ph-g": "LLG"}
        fault = mapping[fault]

    if fault not in ["LLL", "LL", "LG", "LLG"]:
        raise NotImplementedError(
            "Only LLL, LL, LLG and LG short-circuit faults implemented"
        )

    if len(net.gen) and (ip or ith):
        logger.warning(
            "aperiodic, thermal short-circuit currents are only implemented for "
            "faults far from generators!"
        )

    if case not in ["max", "min"]:
        raise ValueError(
            'case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current'
        )

    if topology not in ["meshed", "radial", "auto"]:
        raise ValueError('specify network structure as "meshed", "radial" or "auto"')

    # not neccesarry anymore as the issues werer fixed and testes
    """if branch_results:
        logger.warning("Branch results are in beta mode and might not always be reliable, "
                       "especially for transformers")"""

    if use_pre_fault_voltage:
        init_vm_pu = init_va_degree = "results"
        trafo_model = net._options[
            "trafo_model"
        ]  # trafo model for SC must match the trafo model for PF calculation
        if not isinstance(bus, Number) and len(net.sgen.query("in_service")) > 0:
            raise NotImplementedError(
                "Short-circuit with Type C method and sgen is only implemented for a single bus"
            )
    else:
        init_vm_pu = init_va_degree = "flat"
        trafo_model = "pi"

    # Convert bus to numpy array
    if bus is None:
        bus = net.bus.index.values
    else:
        bus = np.array([bus]).ravel()

    kappa = ith or ip
    net["_options"] = {}
    _add_ppc_options_compat(
        net,
        calculate_voltage_angles=False,
        trafo_model=trafo_model,
        check_connectivity=check_connectivity,
        mode="sc",
        switch_rx_ratio=2,
        init_vm_pu=init_vm_pu,
        init_va_degree=init_va_degree,
        enforce_p_lims=False,
        enforce_q_lims=False,
        recycle=None,
    )
    _add_sc_options(
        net,
        fault=fault,
        case=case,
        lv_tol_percent=lv_tol_percent,
        tk_s=tk_s,
        topology=topology,
        r_fault_ohm=r_fault_ohm,
        x_fault_ohm=x_fault_ohm,
        kappa=kappa,
        ip=ip,
        ith=ith,
        branch_results=branch_results,
        kappa_method=kappa_method,
        return_all_currents=return_all_currents,
        inverse_y=inverse_y,
        use_pre_fault_voltage=use_pre_fault_voltage,
    )
    net._options["fault_impedance"] = 0 + 0j
    init_results(net, "sc")

    # if fault == ("LLL"):
    #     _calc_sc(net, bus)
    if fault in ("LLL", "LG", "LLG", "LL"):
        _calc_sc_to_g(net, bus)
    else:
        raise ValueError("Invalid fault %s" % fault)


# def _calc_current(net, ppci_orig, bus):
#     # Select required ppci bus
#     ppci_bus = _get_is_ppci_bus(net, bus)

#     # update ppci
#     non_ps_gen_ppci_bus, non_ps_gen_ppci, ps_gen_bus_ppci_dict =\
#         _create_k_updated_ppci(net, ppci_orig, ppci_bus=ppci_bus)

#     # For each ps_gen_bus one unique ppci is required
#     ps_gen_ppci_bus = list(ps_gen_bus_ppci_dict.keys())

#     for calc_bus in ps_gen_ppci_bus+[non_ps_gen_ppci_bus]:
#         if isinstance(calc_bus, np.ndarray):
#             # Use ppci for general bus
#             this_ppci, this_ppci_bus = non_ps_gen_ppci, calc_bus
#         else:
#             # Use specific ps_gen_bus ppci
#             this_ppci, this_ppci_bus = ps_gen_bus_ppci_dict[calc_bus], np.array([calc_bus])

#         _calc_ybus(this_ppci)
#         if net["_options"]["inverse_y"]:
#             _calc_zbus(net, this_ppci)
#         else:
#             # Factorization Ybus once
#             # scipy.sparse.linalg.factorized converts the input matrix to csc from csr and raises a warning
#             # todo: create Ybus in CSC format instead of CSR format if known that inverse_y is False?
#             this_ppci["internal"]["ybus_fact"] = factorized(this_ppci["internal"]["Ybus"].tocsc())

#         _calc_rx(net, this_ppci, this_ppci_bus)
#         _calc_ikss(net, this_ppci, this_ppci_bus)
#         _add_kappa_to_ppc(net, this_ppci)
#         if net["_options"]["ip"]:
#             _calc_ip(net, this_ppci)
#         if net["_options"]["ith"]:
#             _calc_ith(net, this_ppci)

#         if net._options["branch_results"]:
#             # if net._options["fault"] == "LLL":
#             _calc_branch_currents_complex(net, this_ppci_bus, None, this_ppci, None, 1)
#             # else:
#             #     _calc_branch_currents(net, this_ppci, this_ppci_bus)

#         _copy_result_to_ppci_orig(ppci_orig, this_ppci, this_ppci_bus,
#                                   calc_options=net._options)


# def _calc_sc(net, bus):
#     ppc, ppci = _init_ppc(net)
#     if net._options.get("use_pre_fault_voltage", False):
#         _add_load_sc_impedances_ppc(net, ppc)  # add SC impedances for loads
#         ppci = _ppc2ppci(ppc, net)

#     _calc_current(net, ppci, bus)

#     ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
#     _extract_results(net, ppc_0=None, ppc_1=ppc, ppc_2=None, bus=bus)
#     _clean_up(net)

#     if "ybus_fact" in ppci["internal"]:
#         # Delete factorization object
#         ppci["internal"].pop("ybus_fact")


def _calc_sc_to_g(net, bus):
    """
    calculation method for phase to ground short-circuit currents
    """
    _add_auxiliary_elements(net)
    # pos. seq bus impedance
    ppc_1, ppci_1 = _init_ppc(net)
    # Create k updated ppci_1
    ppci_bus = _get_is_ppci_bus(net, bus)
    _, ppci_1, _ = _create_k_updated_ppci(net, ppci_1, ppci_bus=ppci_bus)
    _calc_ybus(ppci_1)

    ppc_2, ppci_2 = _init_ppc(net, sequence=2)
    # Create k updated ppci_2
    _, ppci_2, _ = _create_k_updated_ppci(net, ppci_2, ppci_bus=ppci_bus)
    _calc_ybus(ppci_2)

    # input for negative sequence is same as for positive sequence
    # ppc_2 = copy.deepcopy(ppc_1)
    # ppci_2 = copy.deepcopy(ppci_1)

    # placing this here allows saving the calculation of Ybus if not type C
    if net._options.get("use_pre_fault_voltage", False):
        _add_load_sc_impedances_ppc(net, ppc_1)  # add SC impedances for sgens and loads
        ppci_1 = _ppc2ppci(ppc_1, net)
        _, ppci_1, _ = _create_k_updated_ppci(net, ppci_1, ppci_bus=ppci_bus)
        _calc_ybus(ppci_1)

        _add_load_sc_impedances_ppc(
            net, ppc_2, relevant_elements=("load",)
        )  # add SC impedances for loads
        ppci_2 = _ppc2ppci(ppc_2, net)
        _calc_ybus(ppci_2)

    # zero seq bus impedance
    ppc_0, ppci_0 = _pd2ppc_zero(net, ppc_1["branch"][:, K_ST])
    _calc_ybus(ppci_0)

    if net["_options"]["inverse_y"]:
        _calc_zbus(net, ppci_0)
        _calc_zbus(net, ppci_1)
        _calc_zbus(net, ppci_2)
    else:
        # Factorization Ybus once
        ppci_0["internal"]["ybus_fact"] = factorized(ppci_0["internal"]["Ybus"].tocsc())
        ppci_1["internal"]["ybus_fact"] = factorized(ppci_1["internal"]["Ybus"].tocsc())
        ppci_2["internal"]["ybus_fact"] = factorized(ppci_2["internal"]["Ybus"].tocsc())

    _calc_rx(net, ppci_1, ppci_bus, 1)
    _add_kappa_to_ppc(net, ppci_1)  # todo add kappa only to ppci_1?

    _calc_rx(net, ppci_0, ppci_bus, 0)
    _calc_rx(net, ppci_2, ppci_bus, 2)

    _calc_ikss_to_g(net, ppci_0, ppci_1, ppci_2, ppci_bus)
    if net._options["branch_results"]:
        _calc_branch_currents_complex(net, ppci_bus, ppci_0, ppci_1, ppci_2, 0)
        _calc_branch_currents_complex(net, ppci_bus, ppci_0, ppci_1, ppci_2, 1)
        _calc_branch_currents_complex(net, ppci_bus, ppci_0, ppci_1, ppci_2, 2)

    ppc_0 = _copy_results_ppci_to_ppc(ppci_0, ppc_0, "sc")
    ppc_1 = _copy_results_ppci_to_ppc(ppci_1, ppc_1, "sc")
    ppc_2 = _copy_results_ppci_to_ppc(ppci_2, ppc_2, "sc")
    _extract_results(net, ppc_0, ppc_1, ppc_2, bus=bus)
    _clean_up(net)
