"""
Newton Rules Engine - Field mapping and validation for power flow networks.

This module provides field mappings and validation rules for multiconductor
power flow networks. Use `get_pf_rule_mismatches()` to validate a network
against the field mapping rules.

Usage:
    from sce.newton_rules_engine import get_pf_rule_mismatches, new_pf_field_mapping
    
    mismatch_df = get_pf_rule_mismatches(net, new_pf_field_mapping)
    print(f"Found {len(mismatch_df)} rule mismatches")

Copyright 2026, iTron.
Authors: Frank M Gonzales, Ajith Joseph
"""

import re
import math
import numpy as np
import pandas as pd
from typing import TypedDict


class MappedField(TypedDict):
    field: str
    source: str
    required: bool
    pf_required: bool
    default_val: str | int | float | bool | None
    range: str


class FieldMapping(TypedDict):
    create_method: str
    field_mapping: list[MappedField]


new_pf_field_mapping: FieldMapping = {
    "bus": [
        #{"field": "vn_kv", "source": "BUS_VN_KV", "required": True, "pf_required": True, "default_val": 1.03, "range": "0 - 1.05"},
        {"field": "name", "source": "CONNECTIVITY_NODEID", "required": True, "pf_required": False, "default_val": None, "range": ""},
        {"field": "num_phases", "source": "BUS_PHASES", "required": False, "pf_required": False, "default_val": None, "range": ""},
        {"field": "grounded", "source": "BUS_GROUNDED_PHASES", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "grounding_r_ohm", "source": "BUS_GROUNDING_R_OHM", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "grounding_x_ohm", "source": "BUS_GROUNDING_X_OHM", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "in_service", "source": "BUS_IN_SERVICE", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "type", "source": "BUS_TYPE", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "zone", "source": "BUS_ZONE", "required": False, "pf_required": False, "default_val": None, "range": ""}
    ],
    "line": [
        {"field": "std_type", "source": "LINE_TYPE", "required": True, "pf_required": True, "default_val": "overhead", "range": "Any - Lower Case"},
        {"field": "model_type", "source": "LINE_MODEL_TYPE", "required": True, "pf_required": True, "default_val": "matrix", "range": "matrix - Lower Case"},
        {"field": "from_bus", "source": "UPSTREAM_CONNECTIVITYNODEID_INDEX", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "LINE_FROM_PHASE", "required": True, "pf_required": True, "default_val": 123, "range": "[0:3]"},
        {"field": "to_bus", "source": "DOWNSTREAM_CONNECTIVITYNODEID_INDEX", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "to_phase", "source": "LINE_TO_PHASE", "required": True, "pf_required": True, "default_val": 123, "range": "[0:3]"},
        {"field": "length_km", "source": "LINE_LENGTH_KM", "required": True, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": False, "pf_required": False, "default_val": None, "range": ""},
        {"field": "in_service", "source": "LINE_IN_SERVICE", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - it should be in Camel Case"}
    ],
    "switch": [
        {"field": "bus", "source": "UPSTREAM_CONNECTIVITYNODEID", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "phase", "source": "SWITCH_PHASE", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "element", "source": "DOWNSTREAM_CONNECTIVITYNODEID", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "et", "source": "SWITCH_ET", "required": True, "pf_required": True, "default_val": "b", "range": "\"b\" or \"l\""},
        {"field": "closed", "source": "SWITCH_CLOSED", "required": False, "pf_required": True, "default_val": True, "range": "True OR False -  Camel Case"},
        {"field": "type", "source": "SWITCH_TYPE", "required": False, "pf_required": False, "default_val": "LBS", "range": "Any string [It is an Informational Field] - Upper Case"},
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": False, "pf_required": False, "default_val": None, "range": ""}
    ],
    "asymmetric_shunt": [
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": True, "pf_required": False, "default_val": None, "range": ""},
        {"field": "bus", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "SHUNT_FROM_PHASE", "required": True, "pf_required": True, "default_val": 123, "range": "[1:3]"},
        {"field": "to_phase", "source": "SHUNT_TO_PHASE", "required": True, "pf_required": True, "default_val": 0, "range": "[0:3]"},
        {"field": "p_mw", "source": "SHUNT_P_MW", "required": True, "pf_required": True, "default_val": 0, "range": "Any"},
        {"field": "q_mvar", "source": "SHUNT_Q_MVAR", "required": True, "pf_required": True, "default_val": 1e-06, "range": "Any"},
        {"field": "control_mode", "source": "SHUNT_CONTROL_MODE", "required": False, "pf_required": True, "default_val": "switched", "range": "\"none, \"fixed\" or \"switched\" - Lower Case"},
        {"field": "closed", "source": "SHUNT_CLOSED", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - Camel Case"},
        {"field": "v_threshold_on", "source": "SHUNT_V_THR_ON", "required": False, "pf_required": True, "default_val": 0.9833, "range": ">0"},
        {"field": "v_threshold_off", "source": "SHUNT_V_THR_OFF", "required": False, "pf_required": True, "default_val": 1.05, "range": ">0"},
        {"field": "vn_kv", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""}
    ],
    "ext_grid_sequence": [
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": True, "pf_required": False, "default_val": None, "range": ""},
        {"field": "bus", "source": "DOWNSTREAM_CONNECTIVITYNODEID", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "EXTERNALGRID_FROM_PHASE", "required": True, "pf_required": True, "default_val": 123, "range": "123"},
        {"field": "to_phase", "source": "EXTERNALGRID_TO_PHASE", "required": True, "pf_required": True, "default_val": 0, "range": "000"},
        {"field": "vm_pu", "source": "EXTERNALGRID_VM_PU", "required": True, "pf_required": True, "default_val": 1.03, "range": ">0.7 & < 1.3"},
        {"field": "va_degree", "source": "EXTERNALGRID_VA_DEGREE", "required": True, "pf_required": True, "default_val": 0, "range": "[0:360]"},
        {"field": "r_ohm", "source": "EXTERNALGRID_SC_MVA", "required": True, "pf_required": True, "default_val": 100, "range": ">0"},
        {"field": "x_ohm", "source": "EXTERNALGRID_RX", "required": True, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "in_service", "source": "EXTERNALGRID_IN_SERVICE", "required": True, "pf_required": True, "default_val": True, "range": "True OR False - it should be in Camel Case"}
    ],
    "trafo3ph": [
        {"field": "hv_bus", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "lv_bus", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "std_type", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "tap_pos", "source": "", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "name", "source": "", "required": False, "pf_required": False, "default_val": None, "range": ""}
    ],
    "trafo1ph": [
        {"field": "buses", "source": "UPSTREAM_CONNECTIVITYNODEID, DOWNSTREAM_CONNECTIVITYNODEID", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "XFRMR_HV_FROM_PHASE, XFRMR_LV_FROM_PHASE", "required": True, "pf_required": True, "default_val": None, "range": "[1:3]; len=PHASE_COUNT; MUST MATCH xfrmr_lookup_table per CONFIG"},
        {"field": "to_phase", "source": "XFRMR_HV_TO_PHASE, XFRMR_LV_TO_PHASE", "required": True, "pf_required": True, "default_val": None, "range": "[0:3]; IF CONNECTION=Y|P2P THEN 0 ELIF CONNECTION=D THEN delta rotation; MUST MATCH xfrmr_lookup_table per CONFIG"},
        {"field": "vn_kv", "source": "XFRMR_VN_HV_KV, XFRMR_VN_LV_KV", "required": True, "pf_required": True, "default_val": None, "range": ">0; IF CONNECTION=Y THEN BUS_VN_KV/sqrt(3) ELIF CONNECTION=P2P|D THEN BUS_VN_KV; MUST MATCH xfrmr_lookup_table"},
        {"field": "sn_mva", "source": "XFRMR_SN_MVA", "required": True, "pf_required": True, "default_val": 0.075, "range": ">0"},
        {"field": "vk_percent", "source": "XFRMR_VK_PERCENT", "required": True, "pf_required": True, "default_val": 1, "range": ">0"},
        {"field": "vkr_percent", "source": "XFRMR_VKR_PERCENT", "required": True, "pf_required": True, "default_val": 0.707106781, "range": ">=0"},
        {"field": "pfe_kw", "source": "XFRMR_PFE_KW", "required": True, "pf_required": True, "default_val": 0, "range": ">=0"},
        {"field": "i0_percent", "source": "XFRMR_I0_PERCENT", "required": True, "pf_required": True, "default_val": 1e-06, "range": ">=0"},
        {"field": "tap_neutral", "source": "XFRMR_TAP_NEUTRAL", "required": False, "pf_required": True, "default_val": 0, "range": "between tap_min and tap_max"},
        {"field": "tap_min", "source": "XFRMR_TAP_MIN", "required": False, "pf_required": True, "default_val": -16, "range": "any float"},
        {"field": "tap_max", "source": "XFRMR_TAP_MAX", "required": False, "pf_required": True, "default_val": 16, "range": ">= tap_min"},
        {"field": "tap_pos", "source": "XFRMR_TAP_POS", "required": False, "pf_required": True, "default_val": 0, "range": "between tap_min and tap_max"},
        {"field": "tap_step_percent", "source": "XFRMR_TAP_STEP_PERCENT", "required": False, "pf_required": True, "default_val": 0.625, "range": ">0"},
        {"field": "in_service", "source": "XFRMR_IN_SERVICE", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - Camel Case"},
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": False, "pf_required": False, "default_val": None, "range": ""}
    ],
    "regulator_control": [
        {"field": "trafo_top_level_index", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "mode", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "v_set_secondary_v", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "bandwidth_secondary_v", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "pt_ratio", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "ct_primary_rating_a", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "r_ldc_v", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "x_ldc_v", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""}
    ],
    "tap_control": [
        {"field": "trafo_top_level_index", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "mode", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "vm_lower_pu", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "vm_upper_pu", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "detect_oscillation", "source": "CALCULATED", "required": True, "pf_required": True, "default_val": None, "range": ""}
    ],
    "trafo3w": [
        {"field": "hv_bus", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "mv_bus", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "lv_bus", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "std_type", "source": "", "required": True, "pf_required": True, "default_val": None, "range": ""},
        {"field": "tap_pos", "source": "", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "name", "source": "", "required": False, "pf_required": False, "default_val": None, "range": ""}
    ],
    "line_std_type": [
        {"field": "r_ohm_per_km", "source": "LINE_R_OHM_PER_KM", "required": False, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "x_ohm_per_km", "source": "LINE_X_OHM_PER_KM", "required": False, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "r0_ohm_per_km", "source": "LINE_R0_OHM_PER_KM", "required": False, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "x0_ohm_per_km", "source": "LINE_X0_OHM_PER_KM", "required": False, "pf_required": True, "default_val": 1e-06, "range": ">0"},
        {"field": "c_nf_per_km", "source": "LINE_C_NF_PER_KM", "required": False, "pf_required": True, "default_val": 0, "range": ">=0"},
        {"field": "c0_nf_per_km", "source": "LINE_C0_NF_PER_KM", "required": False, "pf_required": True, "default_val": 0, "range": ">=0"},
        {"field": "max_i_ka", "source": "LINE_MAX_I_KA", "required": False, "pf_required": False, "default_val": 1, "range": ">=0"},
        {"field": "name", "source": "LINE_TYPE", "required": False, "pf_required": False, "default_val": "overhead", "range": "Any - Lower Case"}
    ],
    "asymmetric_load": [
        {"field": "bus", "source": "UPSTREAM_CONNECTIVITYNODEID", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "LOAD_FROM_PHASE", "required": False, "pf_required": True, "default_val": 123, "range": "[0:3]"},
        {"field": "to_phase", "source": "LOAD_TO_PHASE", "required": False, "pf_required": True, "default_val": 0, "range": "[0:3]"},
        {"field": "p_mw", "source": "LOAD_P_MW", "required": False, "pf_required": True, "default_val": "0.000000,0.000000,0.000000", "range": "any"},
        {"field": "q_mvar", "source": "LOAD_Q_MVAR", "required": False, "pf_required": True, "default_val": "0.000000,0.000000,0.000000", "range": "any"},
        {"field": "const_z_percent_p", "source": "LOAD_CONST_Z_PERCENT_P", "required": False, "pf_required": True, "default_val": 0, "range": "[0:100]"},
        {"field": "const_i_percent_p", "source": "LOAD_CONST_I_PERCENT_P", "required": False, "pf_required": True, "default_val": 0, "range": "[0:100]"},
        {"field": "const_z_percent_q", "source": "LOAD_CONST_Z_PERCENT_Q", "required": False, "pf_required": True, "default_val": 0, "range": "[0:100]"},
        {"field": "const_i_percent_q", "source": "LOAD_CONST_I_PERCENT_Q", "required": False, "pf_required": True, "default_val": 0, "range": "[0:100]"},
        {"field": "sn_mva", "source": "LOAD_SN_MVA", "required": False, "pf_required": True, "default_val": 0, "range": ">0"},
        {"field": "scaling", "source": "LOAD_SCALING", "required": False, "pf_required": True, "default_val": 1, "range": ">0"},
        {"field": "in_service", "source": "LOAD_IN_SERVICE", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - Camel Case"},
        {"field": "name", "source": "STRUCTURE_NUM", "required": False, "pf_required": False, "default_val": None, "range": ""},
        {"field": "type", "source": "LOAD_TYPE", "required": False, "pf_required": False, "default_val": "customer type", "range": "Any string - Lower Case"}
    ],
    "asymmetric_gen": [
        {"field": "bus", "source": "UPSTREAM_CONNECTIVITYNODEID", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "GEN_FROM_PHASE", "required": False, "pf_required": True, "default_val": 123, "range": "[0:3]"},
        {"field": "to_phase", "source": "GEN_TO_PHASE", "required": False, "pf_required": True, "default_val": 0, "range": "[0:3]"},
        {"field": "p_mw", "source": "gen_p_mw_list", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "vm_pu", "source": "GEN_VM_PU", "required": False, "pf_required": True, "default_val": 1, "range": ">0"},
        {"field": "sn_mva", "source": "GEN_SN_MVA", "required": False, "pf_required": True, "default_val": "RATED_KW/1000", "range": ">0"},
        {"field": "scaling", "source": "GEN_SCALING", "required": False, "pf_required": True, "default_val": 1, "range": ">0"},
        {"field": "in_service", "source": "GEN_IN_SERVICE", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - Camel Case"},
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": False, "pf_required": False, "default_val": None, "range": ""},
        {"field": "type", "source": "GEN_TYPE", "required": False, "pf_required": False, "default_val": "TECH TYPE", "range": "Any string"}
    ],
    "asymmetric_sgen": [
        {"field": "bus", "source": "UPSTREAM_CONNECTIVITYNODEID", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "from_phase", "source": "SGEN_FROM_PHASE", "required": False, "pf_required": True, "default_val": 123, "range": "[0:3]"},
        {"field": "to_phase", "source": "SGEN_TO_PHAS", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "p_mw", "source": "SGEN_P_MW", "required": False, "pf_required": True, "default_val": "0.000000,0.000000,0.000000", "range": "any float (Make it Absolute)"},
        {"field": "q_mvar", "source": "SGEN_Q_MVAR", "required": False, "pf_required": True, "default_val": "0.000000,0.000000,0.000000", "range": "any float (Make it Absolute)"},
        {"field": "sn_mva", "source": "SGEN_SN_MVA", "required": False, "pf_required": True, "default_val": 0, "range": ">0"},
        {"field": "scaling", "source": "SGEN_SCALING", "required": False, "pf_required": True, "default_val": 1, "range": ">0"},
        {"field": "in_service", "source": "SGEN_IN_SERVICE", "required": False, "pf_required": True, "default_val": True, "range": "True OR False - Camel Case"},
        {"field": "name", "source": "CONDUCTING_EQUIPMENTID", "required": False, "pf_required": False, "default_val": None, "range": ""},
        {"field": "type", "source": "SGEN_TYPE", "required": False, "pf_required": False, "default_val": "TECH TYPE", "range": "Any string"}
    ],
    "sgen_control": [
        {"field": "element", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "qv_curve", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "et", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "mode", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""},
        {"field": "damping_coef", "source": "CALCULATED", "required": False, "pf_required": True, "default_val": None, "range": ""}
    ]
}


def get_pf_rule_mismatches(net, field_mapping: FieldMapping = None) -> pd.DataFrame:
    """
    Validate a network against field mapping rules.
    
    Parameters:
        net: pandapower/multiconductor network object
        field_mapping: Field mapping dictionary (defaults to new_pf_field_mapping)
    
    Returns:
        pd.DataFrame: DataFrame with columns ['element', 'field', 'rule_mismatch']
    """
    if field_mapping is None:
        field_mapping = new_pf_field_mapping
    
    def _is_missing(value):
        if value is None:
            return True
        try:
            return pd.isna(value)
        except Exception:
            return False

    def _is_bool_scalar(value):
        return isinstance(value, (bool, np.bool_))

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_scalar(value):
        if isinstance(value, str):
            v = value.strip()
            if v.lower() in {"true", "false"}:
                return v.lower() == "true"
            n = _to_float(v)
            if n is not None and math.isfinite(n):
                return int(n) if n.is_integer() else n
            return v
        return value

    def _check_default(value, default_val):
        if default_val is None:
            return True
        if _is_missing(value):
            return False
        left = _normalize_scalar(value)
        right = _normalize_scalar(default_val)
        return left == right

    def _check_range(value, rule):
        if _is_missing(value):
            return False

        range_rule = (rule or "").strip()
        if not range_rule:
            return True

        lowered = range_rule.lower()
        if lowered.startswith("any") or "any string" in lowered:
            return True
        if "true or false" in lowered:
            return _is_bool_scalar(value)

        condition_parts = [p.strip() for p in re.split(r"\s*&\s*", range_rule) if p.strip()]
        if len(condition_parts) > 1:
            return all(_check_range(value, part) for part in condition_parts)

        bracket_match = re.fullmatch(r"\[(\-?\d+(?:\.\d+)?)\s*:\s*(\-?\d+(?:\.\d+)?)\]", range_rule)
        if bracket_match:
            lower = float(bracket_match.group(1))
            upper = float(bracket_match.group(2))
            if isinstance(value, str) and value.isdigit():
                return all(lower <= int(ch) <= upper for ch in value)
            number = _to_float(value)
            return number is not None and lower <= number <= upper

        comp_match = re.fullmatch(r"(>=|<=|>|<)\s*(\-?\d+(?:\.\d+)?)", range_rule)
        if comp_match:
            op, raw = comp_match.groups()
            bound = float(raw)
            number = _to_float(value)
            if number is None:
                return False
            if op == ">":
                return number > bound
            if op == ">=":
                return number >= bound
            if op == "<":
                return number < bound
            return number <= bound

        return True

    mismatches = []
    for element, mappings in field_mapping.items():
        element_table = None
        if hasattr(net, element):
            element_table = getattr(net, element)
        elif isinstance(net, dict) and element in net:
            element_table = net[element]

        if not isinstance(element_table, pd.DataFrame) or element_table.empty:
            continue

        for mapped in mappings:
            field = mapped.get("field")
            if not field or field not in element_table.columns:
                continue

            default_val = mapped.get("default_val")
            range_rule = (mapped.get("range", "") or "").strip()
            has_range_rule = bool(range_rule)
            bool_rule = "true or false" in range_rule.lower()

            for row_idx, value in element_table[field].items():
                reasons = []

                if bool_rule and not _is_bool_scalar(value):
                    reasons.append(f"boolean type mismatch (value={value!r}, type={type(value).__name__})")

                if has_range_rule:
                    if not _check_range(value, range_rule):
                        reasons.append(f"range mismatch (value={value!r}, range={range_rule!r})")
                elif default_val is not None:
                    if not _check_default(value, default_val):
                        reasons.append(f"default mismatch (value={value!r}, default={default_val!r})")

                if reasons:
                    mismatches.append({
                        "element": element,
                        "field": field,
                        "rule_mismatch": f"row={row_idx}: " + "; ".join(reasons),
                    })

    return pd.DataFrame(mismatches, columns=["element", "field", "rule_mismatch"])


def get_mismatches_by_element(net, field_mapping: FieldMapping = None) -> dict:
    """
    Get rule mismatches grouped by element type.
    
    Parameters:
        net: pandapower/multiconductor network object
        field_mapping: Field mapping dictionary
    
    Returns:
        dict: Dictionary with element names as keys and mismatch DataFrames as values
    """
    df = get_pf_rule_mismatches(net, field_mapping)
    if df.empty:
        return {}
    return {element: group for element, group in df.groupby('element')}


# ── Valid transformer configurations ─────────────────────────────────────────
# For each primary kV (L-L), the valid winding voltages and phase assignments.
# Y (phase-to-ground): vn_kv = V_LL / √3, to_phase = 0
# D / P2P (phase-to-phase): vn_kv = V_LL, to_phase ≠ 0

_VALID_PRIMARY_KV_LL = [33.0, 25.0, 16.0, 12.0, 4.16, 2.4]
_VALID_HV_WINDING_KV = {}
for _kv in _VALID_PRIMARY_KV_LL:
    _VALID_HV_WINDING_KV[round(_kv / math.sqrt(3), 2)] = ('Y', _kv)   # wye
    _VALID_HV_WINDING_KV[round(_kv, 2)] = ('D/P2P', _kv)              # delta or P2P

_VALID_SECONDARY = {
    # vn_kv → (label, expected_connection)
    0.12:  '120V winding (Y from 208V or 240V L-L)',
    0.14:  '138.6V winding (Y from 240V L-L)',
    0.24:  '240V winding (D from 240V L-L)',
    0.28:  '277V winding (Y from 480V L-L)',
    0.48:  '480V winding (D from 480V L-L)',
}


def check_transformer_configuration(net) -> pd.DataFrame:
    """
    Check all trafo1ph entries for invalid transformer configurations.

    Validates:
      1. **vn_kv** matches a valid winding voltage for the bus voltage
         (Y: V_LL/√3, D: V_LL)
      2. **from_phase** and **to_phase** are consistent with the connection type
         (Y → to_phase=0, D → to_phase≠0)
      3. **sn_mva** is positive and non-zero
      4. **vk_percent** is positive and non-zero

    Parameters
    ----------
    net : pandapower/multiconductor network object

    Returns
    -------
    pd.DataFrame
        Columns: ['trafo_index', 'bus', 'circuit', 'field', 'value', 'issue']
        Empty if all configurations are valid.
    """
    issues = []

    if not hasattr(net, 'trafo1ph') or not isinstance(net.trafo1ph, pd.DataFrame):
        return pd.DataFrame(columns=['trafo_index', 'bus', 'circuit', 'field', 'value', 'issue'])

    trafo = net.trafo1ph
    NUM_PHASES = 4

    # Build bus voltage lookup from net.bus
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

        row = trafo.loc[idx]

        # ── Check vn_kv ──
        vn_kv = row.get('vn_kv')
        if vn_kv is not None:
            try:
                vn_kv_f = float(vn_kv)
            except (TypeError, ValueError):
                issues.append({
                    'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                    'field': 'vn_kv', 'value': vn_kv,
                    'issue': 'vn_kv is not numeric'
                })
                continue

            if vn_kv_f <= 0:
                issues.append({
                    'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                    'field': 'vn_kv', 'value': vn_kv_f,
                    'issue': 'vn_kv must be > 0'
                })

            # Check if this vn_kv is consistent with the bus voltage
            if bus is not None and bus in bus_vn_kv:
                bus_v = bus_vn_kv[bus]
                expected_y = round(bus_v / math.sqrt(3), 2)
                expected_d = round(bus_v, 2)
                actual = round(vn_kv_f, 2)

                if actual != expected_y and actual != expected_d:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'vn_kv', 'value': vn_kv_f,
                        'issue': (f'vn_kv={vn_kv_f} does not match bus vn_kv={bus_v} '
                                  f'(expected Y={expected_y} or D={expected_d})')
                    })

        # ── Check from_phase / to_phase consistency ──
        from_phase = row.get('from_phase')
        to_phase = row.get('to_phase')

        if from_phase is not None and to_phase is not None:
            try:
                fp = int(from_phase)
                tp = int(to_phase)
            except (TypeError, ValueError):
                fp = tp = None

            if fp is not None and tp is not None:
                # Phase-to-neutral (Y): to_phase should be 0
                # Phase-to-phase (D/P2P): to_phase should be non-zero
                if tp == 0 and fp == 0:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'from_phase/to_phase', 'value': f'{fp}/{tp}',
                        'issue': 'from_phase=0 and to_phase=0 is invalid (neutral-to-neutral)'
                    })
                if fp < 0 or fp > 3:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'from_phase', 'value': fp,
                        'issue': f'from_phase={fp} out of range [0-3]'
                    })
                if tp < 0 or tp > 3:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'to_phase', 'value': tp,
                        'issue': f'to_phase={tp} out of range [0-3]'
                    })

        # ── Check sn_mva ──
        sn_mva = row.get('sn_mva')
        if sn_mva is not None:
            try:
                sn = float(sn_mva)
                if sn <= 0:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'sn_mva', 'value': sn,
                        'issue': 'sn_mva must be > 0'
                    })
            except (TypeError, ValueError):
                issues.append({
                    'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                    'field': 'sn_mva', 'value': sn_mva,
                    'issue': 'sn_mva is not numeric'
                })

        # ── Check vk_percent ──
        vk = row.get('vk_percent')
        if vk is not None:
            try:
                vk_f = float(vk)
                if vk_f <= 0:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'vk_percent', 'value': vk_f,
                        'issue': 'vk_percent must be > 0'
                    })
                elif vk_f > 20:
                    issues.append({
                        'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                        'field': 'vk_percent', 'value': vk_f,
                        'issue': 'vk_percent > 20% is unusually high'
                    })
            except (TypeError, ValueError):
                issues.append({
                    'trafo_index': tidx, 'bus': bus, 'circuit': circuit,
                    'field': 'vk_percent', 'value': vk,
                    'issue': 'vk_percent is not numeric'
                })

    return pd.DataFrame(issues, columns=['trafo_index', 'bus', 'circuit', 'field', 'value', 'issue'])
