"""
Newton-Raphson Iteration Scrubber - Data cleanup functions for power flow convergence.

This module provides scrubbing functions to fix common data quality issues that
prevent Newton-Raphson power flow convergence.

Usage:
    from sce.nr_iteration_scrubber import (
        fix_zero_sn_mva,
        normalize_in_service_columns,
        coerce_boolean_columns,
        update_std_type_impedances,
        fix_ext_grid_impedances,
        fix_asymmetric_load_sn_mva,
        fix_negative_gen_power,
        fix_shunt_p_mw,
        fix_switch_r_ohm,
        fix_load_p_mw,
        fix_transformer_configuration,
        attach_auto_fix,
        apply_all_scrubbers
    )
    
    # Apply all scrubbers
    apply_all_scrubbers(net)

Copyright 2026, iTron.
Authors: Frank M Gonzales, Ajith Joseph
"""

import pandas as pd
import numpy as np


# Boolean token sets for parsing
TRUE_TOKENS = {"true", "t", "1", "yes", "y", "on"}
FALSE_TOKENS = {"false", "f", "0", "no", "n", "off", "", "none", "nan", "null"}

# Line std types that need impedance fixes
TARGET_STD_TYPES = {
    "None_4",
    "None_3",
    "None_2",
    "None_1",
    "UNK_UNK_2PH_12KV_2",
    "UNK_UNK_3PH_12KV_3",
    "None_3_N",
    "UNK_UNK_1PH_12KV_1",
    "__3PH_12KV_3",
}


def attach_auto_fix(net, auto_fix: bool = True):
    """
    Attach auto_fix boolean variable to network object.
    
    Parameters:
        net: pandapower/multiconductor network object
        auto_fix: Default value for auto_fix flag (default True)
    
    Returns:
        The network object with auto_fix attribute set
    """
    net.auto_fix = auto_fix
    return net


def _to_bool(v):
    """Convert value to boolean."""
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    if s in TRUE_TOKENS:
        return True
    if s in FALSE_TOKENS:
        return False
    return False


def _to_bool_from_string(value):
    """Convert string value to boolean with extended token support."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in TRUE_TOKENS:
        return True
    if text in FALSE_TOKENS:
        return False
    return False


def coerce_boolean_columns(net):
    """
    Convert string boolean values to Python bool type in all element tables.
    
    Targets columns: 'in_service', 'closed'
    
    Parameters:
        net: pandapower/multiconductor network object
    """
    for _, table in net.items():
        if isinstance(table, pd.DataFrame):
            for col in ["in_service", "closed"]:
                if col in table.columns and table[col].dtype == object:
                    table[col] = table[col].map(_to_bool).astype(bool)


def normalize_in_service_columns(net, net_name: str = "net") -> pd.DataFrame:
    """
    Normalize in_service columns across all element tables with detailed reporting.
    
    Parameters:
        net: pandapower/multiconductor network object
        net_name: Name for logging (default "net")
    
    Returns:
        pd.DataFrame: Summary of normalization with before/after dtype info
    """
    updates = []
    for table_name, table in net.items():
        if not isinstance(table, pd.DataFrame) or "in_service" not in table.columns:
            continue

        before_dtype = str(table["in_service"].dtype)
        before_true = int((table["in_service"].astype(str).str.strip().str.lower().isin(TRUE_TOKENS)).sum())
        before_false_like = int((table["in_service"].astype(str).str.strip().str.lower().isin(FALSE_TOKENS)).sum())

        table["in_service"] = table["in_service"].map(_to_bool_from_string).astype(bool)

        updates.append({
            "table": table_name,
            "rows": len(table),
            "dtype_before": before_dtype,
            "dtype_after": str(table["in_service"].dtype),
            "true_like_before": before_true,
            "false_like_before": before_false_like,
        })

    return pd.DataFrame(updates).sort_values("table").reset_index(drop=True) if updates else pd.DataFrame()


def fix_zero_sn_mva(net, default_sn_mva: float = 0.075):
    """
    Fix zero or null transformer nameplate ratings (sn_mva).
    
    Parameters:
        net: pandapower/multiconductor network object
        default_sn_mva: Default rating in MVA (default 0.075 = 75 kVA)
    """
    for tname in ["trafo", "trafo1ph"]:
        if tname in net and isinstance(net[tname], pd.DataFrame) and len(net[tname]) > 0 and "sn_mva" in net[tname].columns:
            sn = pd.to_numeric(net[tname]["sn_mva"], errors="coerce")
            mask = sn.fillna(0).eq(0)
            if mask.any():
                net[tname].loc[mask, "sn_mva"] = default_sn_mva


def fix_asymmetric_load_sn_mva(net, default_sn_mva: float = 0.075) -> int:
    """
    Fix asymmetric_load rows with sn_mva = 0 or None.
    
    Parameters:
        net: pandapower/multiconductor network object
        default_sn_mva: Default rating in MVA (default 0.075 = 75 kVA)
    
    Returns:
        int: Number of rows fixed
    """
    if "asymmetric_load" not in net or not isinstance(net["asymmetric_load"], pd.DataFrame):
        return 0
    
    table = net["asymmetric_load"]
    if len(table) == 0 or "sn_mva" not in table.columns:
        return 0
    
    sn = pd.to_numeric(table["sn_mva"], errors="coerce")
    mask = sn.fillna(0).eq(0)
    
    if mask.any():
        net["asymmetric_load"].loc[mask, "sn_mva"] = default_sn_mva
        return int(mask.sum())
    
    return 0


def fix_ext_grid_impedances(net, default_impedance: float = 1e-9):
    """
    Fix r_ohm and x_ohm in ext_grid_sequence, setting to default if 0 or null.
    
    Parameters:
        net: pandapower/multiconductor network object
        default_impedance: Default impedance value (default 1e-9)
    """
    if hasattr(net, 'ext_grid_sequence') and isinstance(net.ext_grid_sequence, pd.DataFrame):
        for col in ['r_ohm', 'x_ohm']:
            if col in net.ext_grid_sequence.columns:
                values = pd.to_numeric(net.ext_grid_sequence[col], errors='coerce')
                mask = values.isna() | (values == 0)
                if mask.any():
                    net.ext_grid_sequence.loc[mask, col] = default_impedance


def update_std_type_impedances(net, default_impedance: float = 0.0001):
    """
    Update line std_type impedances for unknown conductor types.
    
    Parameters:
        net: pandapower/multiconductor network object
        default_impedance: Default impedance value (default 0.0001)
    """
    line_std = net.get("line_std_types", {})
    if not isinstance(line_std, dict):
        line_std = {}
        net["line_std_types"] = line_std

    def conductor_count(std_name: str):
        if std_name.endswith("_4"):
            return 4
        if std_name.endswith("_3_N"):
            return 3
        if std_name.endswith("_3"):
            return 3
        if std_name.endswith("_2"):
            return 2
        if std_name.endswith("_1"):
            return 1
        return None

    def make_matrix_values(n: int, value: float):
        rows = {}
        for prefix in ["r", "x", "b", "g"]:
            for i in range(1, 5):
                key = f"{prefix}_{i}_ohm_per_km" if prefix in {"r", "x"} else f"{prefix}_{i}_us_per_km"
                if i <= n:
                    row = [0.0] * n
                    if prefix in {"r", "x"}:
                        row[i - 1] = value
                    rows[key] = row
                else:
                    rows[key] = None
        rows["max_i_ka"] = [1.0] * n
        return rows

    present_target_std_types = set()
    if hasattr(net, "line") and isinstance(net.line, pd.DataFrame) and "std_type" in net.line.columns:
        present_target_std_types = set(net.line["std_type"].astype(str).unique()).intersection(TARGET_STD_TYPES)

    for std_name in present_target_std_types:
        std = line_std.get(std_name)
        if not isinstance(std, dict):
            n = conductor_count(std_name)
            if n is None:
                continue
            line_std[std_name] = make_matrix_values(n, default_impedance)
            continue

        for key, value in list(std.items()):
            key_l = str(key).lower()
            if not (key_l.startswith("r_") or key_l.startswith("x_")):
                continue

            if isinstance(value, list):
                std[key] = [default_impedance for _ in value]
            elif value is None:
                continue
            else:
                std[key] = default_impedance


def fix_negative_gen_power(net) -> int:
    """
    Set p_mw and q_mvar to positive if negative for asymmetric_gen and asymmetric_sgen.
    Generators should inject positive power.
    
    Parameters:
        net: pandapower/multiconductor network object
    
    Returns:
        int: Total number of values fixed
    """
    fixed_count = 0
    
    for table_name in ["asymmetric_gen", "asymmetric_sgen"]:
        if table_name not in net or not isinstance(net[table_name], pd.DataFrame):
            continue
        
        table = net[table_name]
        if len(table) == 0:
            continue
        
        for col in ["p_mw", "q_mvar"]:
            if col not in table.columns:
                continue
            
            # Handle list columns (per-phase values)
            for idx, value in table[col].items():
                if isinstance(value, list):
                    new_value = [abs(v) if isinstance(v, (int, float)) and v < 0 else v for v in value]
                    if new_value != value:
                        net[table_name].at[idx, col] = new_value
                        fixed_count += 1
                elif isinstance(value, (int, float)) and value < 0:
                    net[table_name].at[idx, col] = abs(value)
                    fixed_count += 1
    
    return fixed_count


def fix_shunt_p_mw(net) -> int:
    """
    Fix shunt p_mw values > 0 by setting them to 0.
    Shunt elements should not have positive active power (consuming).
    
    Parameters:
        net: pandapower/multiconductor network object
    
    Returns:
        int: Number of rows fixed
    """
    fixed_count = 0
    
    for table_name in ["shunt", "asymmetric_shunt"]:
        if table_name not in net or not isinstance(net[table_name], pd.DataFrame):
            continue
        
        table = net[table_name]
        if len(table) == 0 or "p_mw" not in table.columns:
            continue
        
        # Handle list columns (per-phase values)
        for idx, value in table["p_mw"].items():
            if isinstance(value, list):
                new_value = [0 if isinstance(v, (int, float)) and v > 0 else v for v in value]
                if new_value != value:
                    net[table_name].at[idx, "p_mw"] = new_value
                    fixed_count += 1
            else:
                p_mw = pd.to_numeric(value, errors="coerce")
                if not pd.isna(p_mw) and p_mw > 0:
                    net[table_name].at[idx, "p_mw"] = 0
                    fixed_count += 1
    
    return fixed_count


def fix_switch_r_ohm(net) -> int:
    """
    Set switch r_ohm to zero if value >= 1.
    High switch resistance can cause convergence issues.
    
    Parameters:
        net: pandapower/multiconductor network object
    
    Returns:
        int: Number of rows fixed
    """
    if "switch" not in net or not isinstance(net["switch"], pd.DataFrame):
        return 0
    
    table = net["switch"]
    if len(table) == 0 or "r_ohm" not in table.columns:
        return 0
    
    r_ohm = pd.to_numeric(table["r_ohm"], errors="coerce")
    mask = r_ohm >= 1
    
    if mask.any():
        net["switch"].loc[mask, "r_ohm"] = 0
        return int(mask.sum())
    
    return 0


def fix_load_p_mw(net, percent_of_sn_mva: float = 0.25) -> int:
    """
    Set p_mw to percentage of sn_mva for loads that have zero or blank p_mw.
    
    Parameters:
        net: pandapower/multiconductor network object
        percent_of_sn_mva: Percentage of sn_mva to use (default 0.25 = 25%)
    
    Returns:
        int: Number of rows fixed
    """
    fixed_count = 0
    
    for table_name in ["asymmetric_load", "load"]:
        if table_name not in net or not isinstance(net[table_name], pd.DataFrame):
            continue
        
        table = net[table_name]
        if len(table) == 0:
            continue
        
        if "p_mw" not in table.columns or "sn_mva" not in table.columns:
            continue
        
        for idx in table.index:
            p_mw = table.at[idx, "p_mw"]
            sn_mva = table.at[idx, "sn_mva"]
            
            # Handle list columns (per-phase values)
            if isinstance(p_mw, list):
                sn_val = pd.to_numeric(sn_mva, errors="coerce") if not isinstance(sn_mva, list) else None
                if sn_val and not pd.isna(sn_val) and sn_val > 0:
                    default_p = sn_val * percent_of_sn_mva
                    new_value = []
                    modified = False
                    for v in p_mw:
                        num_v = pd.to_numeric(v, errors="coerce") if not isinstance(v, (int, float)) else v
                        if pd.isna(num_v) or num_v == 0:
                            new_value.append(default_p)
                            modified = True
                        else:
                            new_value.append(v)
                    if modified:
                        net[table_name].at[idx, "p_mw"] = new_value
                        fixed_count += 1
            else:
                p_val = pd.to_numeric(p_mw, errors="coerce")
                sn_val = pd.to_numeric(sn_mva, errors="coerce")
                
                if (pd.isna(p_val) or p_val == 0) and not pd.isna(sn_val) and sn_val > 0:
                    net[table_name].at[idx, "p_mw"] = sn_val * percent_of_sn_mva
                    fixed_count += 1
    
    return fixed_count


def fix_transformer_configuration(net, default_sn_mva: float = 0.075,
                                   default_vk_percent: float = 2.0) -> int:
    """
    Fix invalid transformer configurations in trafo1ph.

    Corrections applied:
      1. **vn_kv**: If vn_kv does not match bus voltage for either Y or D,
         recompute as V_LL/√3 (Y) based on from_phase/to_phase connection.
      2. **sn_mva**: Set to ``default_sn_mva`` if zero or null.
      3. **vk_percent**: Set to ``default_vk_percent`` if zero or null.
      4. **from_phase / to_phase**: Flag out-of-range values (no auto-fix;
         phase assignment is installation-specific).

    Parameters
    ----------
    net : pandapower/multiconductor network object
    default_sn_mva : float
        Default nameplate rating in MVA (default 0.075 = 75 kVA).
    default_vk_percent : float
        Default short-circuit voltage in percent (default 2.0).

    Returns
    -------
    int
        Number of fields corrected.
    """
    import math

    if not hasattr(net, 'trafo1ph') or not isinstance(net.trafo1ph, pd.DataFrame):
        return 0

    trafo = net.trafo1ph
    if len(trafo) == 0:
        return 0

    fixed_count = 0

    # Build bus voltage lookup
    bus_vn_kv = {}
    if hasattr(net, 'bus') and isinstance(net.bus, pd.DataFrame) and 'vn_kv' in net.bus.columns:
        for bus_idx in net.bus.index:
            bus_vn_kv[bus_idx] = net.bus.at[bus_idx, 'vn_kv']

    for idx in trafo.index:
        if isinstance(idx, tuple):
            tidx, bus, circuit = idx
        else:
            tidx = idx
            bus = None
            circuit = None

        # ── Fix vn_kv ──
        if 'vn_kv' in trafo.columns and bus is not None and bus in bus_vn_kv:
            vn_kv = trafo.at[idx, 'vn_kv']
            bus_v = bus_vn_kv[bus]

            try:
                vn_kv_f = float(vn_kv)
            except (TypeError, ValueError):
                vn_kv_f = -1

            expected_y = round(bus_v / math.sqrt(3), 4)
            expected_d = round(bus_v, 4)
            actual = round(vn_kv_f, 2) if vn_kv_f > 0 else -1

            if actual != round(expected_y, 2) and actual != round(expected_d, 2):
                # Determine connection from to_phase
                to_phase = trafo.at[idx, 'to_phase'] if 'to_phase' in trafo.columns else None
                try:
                    tp = int(to_phase)
                except (TypeError, ValueError):
                    tp = 0  # default to Y

                if tp == 0:
                    # Y connection → vn_kv = V_LL / √3
                    trafo.at[idx, 'vn_kv'] = expected_y
                else:
                    # D/P2P connection → vn_kv = V_LL
                    trafo.at[idx, 'vn_kv'] = expected_d
                fixed_count += 1

        # ── Fix sn_mva ──
        if 'sn_mva' in trafo.columns:
            sn = trafo.at[idx, 'sn_mva']
            try:
                sn_f = float(sn)
            except (TypeError, ValueError):
                sn_f = 0

            if sn_f <= 0 or pd.isna(sn_f):
                trafo.at[idx, 'sn_mva'] = default_sn_mva
                fixed_count += 1

        # ── Fix vk_percent ──
        if 'vk_percent' in trafo.columns:
            vk = trafo.at[idx, 'vk_percent']
            try:
                vk_f = float(vk)
            except (TypeError, ValueError):
                vk_f = 0

            if vk_f <= 0 or pd.isna(vk_f):
                trafo.at[idx, 'vk_percent'] = default_vk_percent
                fixed_count += 1

    return fixed_count


def fix_voltage_regulators(net) -> int:
    """
    Fix voltage regulator transformers whose LV-side vn_kv was incorrectly
    set to a distribution-secondary voltage instead of the correct winding
    voltage for the bus.

    Voltage regulators are identified by having 'REG' or ':LR' in the
    trafo1ph ``name`` field.  For each regulator, every side's ``vn_kv``
    is recomputed from the bus ``vn_kv`` and the winding's connection type
    (Y  -> V_LL / sqrt3,  D/P2P -> V_LL).

    Returns
    -------
    int
        Number of entries corrected.
    """
    import math

    if not hasattr(net, 'trafo1ph') or not isinstance(net.trafo1ph, pd.DataFrame):
        return 0

    trafo = net.trafo1ph
    if len(trafo) == 0:
        return 0

    # Identify regulator entries by name
    reg_mask = trafo['name'].str.upper().str.contains('REG|LR_|:LR', na=False)
    if not reg_mask.any():
        return 0

    # Build bus voltage lookup (by bus index level)
    bus_vn_kv = {}
    if hasattr(net, 'bus') and isinstance(net.bus, pd.DataFrame) and 'vn_kv' in net.bus.columns:
        for bus_idx in net.bus.index.get_level_values('index').unique():
            try:
                bus_vn_kv[bus_idx] = net.bus.xs(bus_idx, level='index')['vn_kv'].iloc[0]
            except Exception:
                pass

    fixed_count = 0
    reg_indices = trafo[reg_mask].index.get_level_values('index').unique()

    for ri in reg_indices:
        sub = trafo.xs(ri, level='index')
        buses = sub.index.get_level_values('bus').unique()

        for b in buses:
            side = sub.xs(b, level='bus')
            current_vn = side['vn_kv'].iloc[0]
            to_phase = side['to_phase'].iloc[0]
            bus_v = bus_vn_kv.get(b)
            if bus_v is None:
                continue

            # Expected winding voltage for this bus and connection
            if int(to_phase) == 0:  # Y / phase-to-neutral
                expected_vn = round(bus_v / math.sqrt(3), 4)
            else:  # Delta / phase-to-phase
                expected_vn = round(bus_v, 4)

            if abs(current_vn - expected_vn) > 0.01:
                for circuit in side.index:
                    c = circuit if not isinstance(circuit, tuple) else circuit[0]
                    trafo.at[(ri, b, c), 'vn_kv'] = expected_vn
                    fixed_count += 1

    return fixed_count


def apply_grounding_defaults(net, r_ohm: float = 1e-6, x_ohm: float = 1e-6):
    """
    Apply default grounding values to bus table.
    
    Parameters:
        net: pandapower/multiconductor network object
        r_ohm: Default grounding resistance (default 1e-6)
        x_ohm: Default grounding reactance (default 1e-6)
    """
    if hasattr(net, 'bus') and isinstance(net.bus, pd.DataFrame):
        net.bus['grounding_r_ohm'] = r_ohm
        net.bus['grounding_x_ohm'] = x_ohm


def apply_switch_defaults(net, r_ohm: float = 0):
    """
    Apply default switch resistance values.
    
    Parameters:
        net: pandapower/multiconductor network object
        r_ohm: Default switch resistance (default 0)
    """
    if hasattr(net, 'switch') and isinstance(net.switch, pd.DataFrame):
        net.switch['r_ohm'] = r_ohm


def apply_all_scrubbers(net, net_name: str = "net"):
    """
    Apply all scrubbing functions to a network.
    
    Parameters:
        net: pandapower/multiconductor network object
        net_name: Name for logging (default "net")
    
    Returns:
        dict: Summary of fixes applied
    """
    summary = {}
    
    # Apply grounding defaults
    apply_grounding_defaults(net)
    summary['grounding_defaults'] = True
    
    # Apply switch defaults
    apply_switch_defaults(net)
    summary['switch_defaults'] = True
    
    # Fix transformer ratings
    fix_zero_sn_mva(net)
    summary['fix_zero_sn_mva'] = True
    
    # Fix transformer configurations (vn_kv, phases, sn_mva, vk_percent)
    trafo_fixed = fix_transformer_configuration(net)
    summary['fix_transformer_configuration'] = trafo_fixed

    # Fix voltage regulator winding voltages
    vreg_fixed = fix_voltage_regulators(net)
    summary['fix_voltage_regulators'] = vreg_fixed

    # Fix asymmetric load ratings
    al_fixed = fix_asymmetric_load_sn_mva(net)
    summary['fix_asymmetric_load_sn_mva'] = al_fixed
    
    # Normalize boolean columns
    normalize_in_service_columns(net, net_name)
    summary['normalize_in_service'] = True
    
    # Coerce boolean columns
    coerce_boolean_columns(net)
    summary['coerce_boolean'] = True
    
    # Fix line std type impedances
    update_std_type_impedances(net)
    summary['update_std_type_impedances'] = True
    
    # Fix external grid impedances
    fix_ext_grid_impedances(net)
    summary['fix_ext_grid_impedances'] = True
    
    # Fix negative generator power
    gen_fixed = fix_negative_gen_power(net)
    summary['fix_negative_gen_power'] = gen_fixed
    
    # Fix shunt p_mw
    shunt_fixed = fix_shunt_p_mw(net)
    summary['fix_shunt_p_mw'] = shunt_fixed
    
    # Fix switch r_ohm
    switch_fixed = fix_switch_r_ohm(net)
    summary['fix_switch_r_ohm'] = switch_fixed
    
    # Fix load p_mw
    load_fixed = fix_load_p_mw(net)
    summary['fix_load_p_mw'] = load_fixed
    
    return summary

