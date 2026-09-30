import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from pandapower import __version__, __format_version__, openpyxl_INSTALLED
from pandapower import pandapowerNet, soft_dependency_error, io_utils
from multiconductor.pycci.std_types import add_basic_std_types
from numpy import dtype

_multi_index_columns = {"trafo1ph": [0, 1, 2],
                        "line": [0, 1],
                        "switch": [0, 1],
                        "configuration_std_type": [0, 1],
                        "matrix_std_type": [0, 1],
                        "asymmetric_load": [0, 1],                        
                        "asymmetric_sgen": [0, 1],
                        "asymmetric_gen": [0, 1],
                        "ext_grid": [0, 1],
                        "ext_grid_sequence": [0, 1],
                        "bus": [0, 1]}


_SCHEMA_TABLES = {
    "arc_flash_location": [
        "name", "bus", "equipment_id", "access_area", "equipment_type",
        "electrode_configuration", "conductor_gap_mm", "working_distance_mm",
        "enclosure_height_mm", "enclosure_width_mm", "enclosure_depth_mm",
        "protection_device_ids", "site_ppe", "limited_approach_boundary_mm",
        "restricted_approach_boundary_mm", "glove_class", "label_count", "in_service",
    ],
    "arc_flash_scenario": [
        "name", "fault_case", "operating_condition", "description", "actions", "in_service",
    ],
    "arc_flash_source": [
        "name", "element_type", "element", "source_type", "decrement_profile", "in_service",
    ],
    "measurement": [
        "name",
        "measurement_type",
        "element_type",
        "element",
        "value",
        "std_dev",
        "side",
        "phase",
        "circuit",
        "terminal",
        "units",
        "status",
    ],
    "pwl_cost": ["power_type", "element", "et", "phase", "circuit", "points"],
    "poly_cost": [
        "element",
        "et",
        "phase",
        "circuit",
        "cp0_eur",
        "cp1_eur_per_mw",
        "cp2_eur_per_mw2",
        "cq0_eur",
        "cq1_eur_per_mvar",
        "cq2_eur_per_mvar2",
    ],
    "characteristic": ["object"],
    "controller": ["object", "in_service", "order", "level", "initial_run", "recycle"],
    "group": ["name", "element_type", "element", "reference_column"],
    "protection": [
        "name",
        "element_type",
        "element",
        "phase",
        "circuit",
        "side",
        "device_type",
        "curve_id",
        "pickup_current_a",
        "instantaneous_current_a",
        "time_dial",
        "time_grading_s",
        "ct_ratio",
        "opening_time_s",
        "curve_kind",
        "in_service",
        "settings",
        "tripped",
    ],
    "protection_curve": ["curve_id", "curve_type", "points", "metadata"],
    "grid_equivalent": [
        "name",
        "eq_type",
        "representation",
        "boundary_terminals",
        "y_eq",
        "i_offset",
        "base_voltage",
        "validation",
        "provenance",
    ],
}

_RESULT_TABLE_COLUMNS = {
    "res_bus": ["vm_pu", "va_degree", "p_mw", "q_mvar", "imbalance_percent"],
    "res_line": [
        "i_from_ka",
        "ia_from_degree",
        "i_to_ka",
        "ia_to_degree",
        "i_ka",
        "p_from_mw",
        "p_to_mw",
        "q_from_mvar",
        "q_to_mvar",
        "pl_mw",
        "ql_mvar",
        "vm_from_pu",
        "vm_to_pu",
        "va_from_degree",
        "va_to_degree",
        "loading_percent",
    ],
    "res_trafo": ["p_mw", "q_mvar", "pl_mw", "ql_mvar", "i_ka", "vm_pu", "va_degree", "loading_percent"],
    "res_ext_grid": ["p_mw", "q_mvar"],
    "res_ext_grid_sequence": ["p_mw", "q_mvar"],
}

_ELEMENT_TABLE_COLUMNS = {
    "asymmetric_load": ["controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
    "asymmetric_sgen": ["controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
    "asymmetric_gen": ["q_mvar", "controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
    "ext_grid": ["controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
    "ext_grid_sequence": ["controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"],
    "line": ["max_loading_percent"],
    "trafo1ph": ["max_loading_percent"],
}

_NORMALIZED_RESULT_TABLES = {
    "res_arc_flash": [
        "location_id", "scenario_id", "location_name", "scenario_name",
        "equipment_id", "access_area", "bus", "fault_case", "operating_condition",
        "arcing_current_case", "nominal_voltage_kv", "working_distance_mm", "phase_currents_ka",
        "phase_current_spread_percent", "bolted_fault_current_ka",
        "initial_arcing_current_ka", "clearing_time_s", "duration_capped",
        "incident_energy_cal_cm2", "arc_flash_boundary_mm", "controlling_device_ids",
        "segment_count", "segments", "status", "warnings", "provenance",
    ],
    "res_arc_flash_worst": [
        "location_id", "scenario_id", "location_name", "scenario_name",
        "equipment_id", "access_area", "bus", "fault_case", "operating_condition",
        "arcing_current_case", "nominal_voltage_kv", "working_distance_mm", "phase_currents_ka",
        "phase_current_spread_percent", "bolted_fault_current_ka",
        "initial_arcing_current_ka", "clearing_time_s", "duration_capped",
        "incident_energy_cal_cm2", "arc_flash_boundary_mm", "controlling_device_ids",
        "segment_count", "segments", "status", "warnings", "provenance",
        "site_ppe", "limited_approach_boundary_mm", "restricted_approach_boundary_mm",
        "glove_class", "report_number", "revision", "issue_date", "label_count",
    ],
    "res_arc_flash_operation": [
        "location_id", "scenario_id", "arcing_current_case", "sequence", "event_time_s",
        "event_type", "device_id", "switch_id", "stage", "current_ka",
        "operating_fraction", "opening_time_s", "details",
    ],
    "res_measurement_est": [
        "position",
        "measurement_type",
        "target_index",
        "side",
        "name",
        "source_index",
        "observed",
        "sigma",
        "predicted",
        "residual",
        "standardized_residual",
        "normalized_residual",
        "accepted",
        "status",
        "rejection_reason",
        "units",
    ],
    "res_opf_constraint": [
        "table",
        "element",
        "kind",
        "actual",
        "limit",
        "residual",
        "dual",
        "satisfied",
    ],
    "res_cost": ["objective", "iterations", "algorithm", "message"],
    "res_contingency_case": [
        "name",
        "description",
        "status",
        "converged",
        "unsupplied_p_mw",
        "unsupplied_q_mvar",
        "voltage_violation_count",
        "line_violation_count",
        "trafo_violation_count",
        "error",
    ],
    "res_contingency_bus": [
        "case_id",
        "vm_pu",
        "va_degree",
        "p_mw",
        "q_mvar",
        "imbalance_percent",
        "under_voltage",
        "over_voltage",
        "voltage_violation",
        "is_isolated",
    ],
    "res_contingency_line": [
        "case_id",
        "i_from_ka",
        "ia_from_degree",
        "i_to_ka",
        "ia_to_degree",
        "i_ka",
        "p_from_mw",
        "p_to_mw",
        "q_from_mvar",
        "q_to_mvar",
        "pl_mw",
        "ql_mvar",
        "vm_from_pu",
        "vm_to_pu",
        "va_from_degree",
        "va_to_degree",
        "loading_percent",
        "rating_ka",
        "loading_violation",
    ],
    "res_contingency_trafo": [
        "case_id",
        "p_mw",
        "q_mvar",
        "i_ka",
        "vm_pu",
        "va_degree",
        "pl_mw",
        "ql_mvar",
        "loading_percent",
        "loading_violation",
    ],
    "res_protection": [
        "device_index",
        "switch_id",
        "protection_type",
        "trip_melt",
        "trip",
        "stage",
        "activation_parameter",
        "activation_parameter_value",
        "operating_time_s",
        "opening_time_s",
        "trip_melt_time_s",
        "trip_time_s",
        "operation_order",
        "operated",
        "elapsed_time_s",
        "stop_reason",
    ],
    "res_protection_coordination": [
        "coordination_group",
        "device_index",
        "switch_id",
        "protection_type",
        "trip_time_s",
        "asserted_order",
        "role",
        "primary_device_index",
        "primary_trip_time_s",
        "grading_margin_s",
        "minimum_margin_s",
        "margin_violation",
        "operation_order",
    ],
}

_COLUMN_DEFAULTS = {
    "controllable": False,
    "in_service": True,
    "status": "unknown",
}


def _ensure_dataframe_columns(df, columns, default=np.nan):
    if df is None:
        return pd.DataFrame(columns=columns)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    for column in columns:
        if column not in df.columns:
            df[column] = _COLUMN_DEFAULTS.get(column, default)
    return df


def upgrade_network_schema(net):
    """Add missing tables, flags, and result columns without overwriting values."""

    for table, columns in _SCHEMA_TABLES.items():
        if table not in net or net[table] is None:
            net[table] = pd.DataFrame(columns=columns)
        else:
            net[table] = _ensure_dataframe_columns(net[table], columns)

    for table, columns in _ELEMENT_TABLE_COLUMNS.items():
        if table in net and net[table] is not None:
            net[table] = _ensure_dataframe_columns(net[table], columns)

    for table, columns in _RESULT_TABLE_COLUMNS.items():
        if table in net and net[table] is not None:
            net[table] = _ensure_dataframe_columns(net[table], columns)

    for table, columns in _NORMALIZED_RESULT_TABLES.items():
        if table not in net or net[table] is None:
            net[table] = pd.DataFrame(columns=columns)
        else:
            net[table] = _ensure_dataframe_columns(net[table], columns)

    if "user_pf_options" not in net or net["user_pf_options"] is None:
        net["user_pf_options"] = dict()
    if "converged" not in net:
        net["converged"] = False
    if "OPF_converged" not in net:
        net["OPF_converged"] = False

    return net


class _CompatUnpickler(pickle.Unpickler):
    _MODULE_REMAP = {
        "numpy._core": "numpy.core",
    }

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            remapped = self._MODULE_REMAP.get(module)
            if remapped is None:
                for prefix, target in self._MODULE_REMAP.items():
                    if module.startswith(prefix + "."):
                        remapped = module.replace(prefix, target, 1)
                        break
            if remapped is None:
                raise
            return super().find_class(remapped, name)


def _pickle_load_compat(handle):
    return _CompatUnpickler(handle).load()


def create_empty_network(name="", sn_mva=1,rho_ohmm=100,f_hz=50, add_stdtypes=True):
    """
    This function initializes the pandapower datastructure.

    OPTIONAL:
        **name** (string, None) - name for the network

        **sn_mva** (float, 1e3) - reference apparent power for per unit system

        **rho_ohmm** (float, 100) - Soil resistivity in [Ohm*m]. Used in Carson-Clem formulation for \
            "configuration" line std types (if any).

        **f_hz** (float, 50.) - power system frequency in hertz

        **add_stdtypes** (boolean, True) - Includes standard types to net

    OUTPUT:
        **net** (attrdict) - PANDAPOWER attrdict with empty tables:

    EXAMPLE:
        net = create_empty_network()

    """
    
    structure_data = { "bus":  [('phase', 'u4'),
                                ('name', dtype(object)),
                                ('vn_kv', 'f8'),
                                ('grounded', 'bool'),
                                ('grounding_r_ohm', 'f8'),
                                ('grounding_x_ohm', 'f8'),
                                ('in_service', 'bool'),
                                ('type', dtype(object)),
                                ('zone', dtype(object))],
        "ext_grid": [("circuit", "i8"),
                     ("name", dtype(object)),
                     ("bus", "u4"),
                     ("from_phase", "u4"),
                     ("to_phase", "u4"),
                     ("vm_pu", "f8"),
                     ("va_degree", "f8"),
                     ("r_ohm", "f8"),
                     ("x_ohm", "f8"),
                     ("in_service", 'bool')],
        "ext_grid_sequence": [("sequence", "i8"),
                     ("name", dtype(object)),
                     ("bus", "u4"),
                     ("from_phase", "u4"),
                     ("to_phase", "u4"),
                     ("vm_pu", "f8"),
                     ("va_degree", "f8"),
                     ("r_ohm", "f8"),
                     ("x_ohm", "f8"),
                     ("in_service", 'bool')],
        "asymmetric_load": [("circuit", "i8"),
                            ("name", dtype(object)),
                            ("bus", "u4"),
                            ("from_phase", "u4"),
                            ("to_phase", "u4"),
                            ("p_mw", "f8"),
                            ("q_mvar", "f8"),
                            ("const_z_percent_p", "f8"),
                            ("const_i_percent_p", "f8"),
                            ("const_z_percent_q", "f8"),  # todo: is it important to have separate values for q?
                            ("const_i_percent_q", "f8"),  # todo: is it important to have separate values for q?
                            ("sn_mva", "f8"),
                            ("scaling", "f8"),
                            ("in_service", 'bool'),
                            ("type", dtype(object))],
        "asymmetric_sgen": [("circuit", "i8"),
                           ("name", dtype(object)),
                           ("bus", "i8"),
                           ("from_phase", "i8"),
                           ("to_phase", "i8"),
                           ("p_mw", "f8"),
                           ("q_mvar", "f8"),
                           ("vm_pu", "f8"),
                           ("const_z_percent_p", "f8"),
                           ("const_i_percent_p", "f8"),
                           ("const_z_percent_q", "f8"),  # todo: is it important to have separate values for q?
                           ("const_i_percent_q", "f8"),  # todo: is it important to have separate values for q?
                           ("sn_mva", "f8"),
                           ("scaling", "f8"),
                           ("in_service", 'bool'),
                           ("type", dtype(object)),
                           ("current_source", "bool"),
                           ("slack", "bool")],
        "asymmetric_gen": [("circuit", "i8"),
                           ("bus", "i8"),
                           ("from_phase", "i8"),
                           ("to_phase", "i8"),
                           ("p_mw", "f8"),
                           ("vm_pu", "f8"),
                           ("sn_mva", "f8"),
                           ("scaling", "f8"),
                           ("in_service", 'bool'),
                           ("name", dtype(object)),
                           ("type", dtype(object)),
                           ("current_source", "bool"),
                           ("slack", "bool")],
        # "asymmetric_shunt": [("circuit", "i8"),
        #                     ("name", dtype(object)),
        #                     ("bus", "i8"),
        #                     ("from_phase", "i8"),
        #                     ("to_phase", "i8"),
        #                     ("p_mw", "f8"),
        #                     ("q_mvar", "f8"),
        #                     ("vm_pu", "f8"),
        #                     ("const_z_percent_p", "f8"),
        #                     ("const_i_percent_p", "f8"),
        #                     ("const_z_percent_q", "f8"),
        #                     ("const_i_percent_q", "f8"),
        #                     ("in_service", 'bool')],
        "line": [("circuit", "i8"),
                 ("name", dtype(object)),
                 ("std_type", dtype(object)),
                 ("model_type", dtype(object)),
                 ("from_bus", "u4"),
                 ("from_phase", "u4"),
                 ("to_bus", "u4"),
                 ("to_phase", "u4"),
                 ("length_km", "f8"),
                 # ("r_ohm_per_km", "f8"),
                 # ("x_ohm_per_km", "f8"),
                 # ("c_nf_per_km", "f8"),
                 # ("g_us_per_km", "f8"),
                 # ("max_i_ka", "f8"),
                 # ("df", "f8"),
                 # ("parallel", "u4"),
                 ("type", dtype(object)),
                 ("in_service", 'bool')],
        "switch": [("circuit", "i8"),
                   ("bus", "u4"),
                   ("phase", "u4"),
                   ("element", "u4"),
                   ("et", dtype(object)),
                   ("type", dtype(object)),
                   ("closed", 'bool'),
                   ("name", dtype(object)),
                   ("r_ohm", "f8"),
                   ],
        "trafo1ph": [("bus", "i8"),
                     ("circuit", "i8"),
                     ("name", dtype(object)),
                     # ("std_type", dtype(object)),
                     ("from_phase", "u4"),
                     ("to_phase", "u4"),
                     ("vn_kv", "f8"),
                     ("sn_mva", "f8"),
                     ("vk_percent", "f8"),
                     ("vkr_percent", "f8"),
                     ("pfe_kw", "f8"),
                     ("i0_percent", "f8"),
                     # ("shift_degree", "f8"),
                     # ("tap_side", dtype(object)),
                     ("tap_neutral", "i4"),
                     ("tap_min", "i4"),
                     ("tap_max", "i4"),
                     ("tap_step_percent", "f8"),
                     # ("tap_step_degree", "f8"),
                     ("tap_pos", "i4"),
                     # ("tap_phase_shifter", 'bool'),
                     # ("parallel", "u4"),
                     # ("df", "f8"),
                     ("in_service", 'bool'),
                     # ("vrif_pu", "f8"),
                     # ("db_percent", "f8"),
                     # ("oltc_bus", "f8"),
                     # ("r_ohm", "f8"),
                     # ("x_ohm", "f8"),
                     # ("synchro", "f8")
                     ],
        "measurement": [("name", dtype(object)),
                        ("measurement_type", dtype(object)),
                        ("element_type", dtype(object)),
                        ("element", "uint32"),
                        ("value", "float64"),
                        ("std_dev", "float64"),
                        ("side", dtype(object))],
        "pwl_cost": [("power_type", dtype(object)),
                     ("element", "u4"),
                     ("et", dtype(object)),
                     ("points", dtype(object))],
        "poly_cost": [("element", "u4"),
                      ("et", dtype(object)),
                      ("cp0_eur", dtype("f8")),
                      ("cp1_eur_per_mw", dtype("f8")),
                      ("cp2_eur_per_mw2", dtype("f8")),
                      ("cq0_eur", dtype("f8")),
                      ("cq1_eur_per_mvar", dtype("f8")),
                      ("cq2_eur_per_mvar2", dtype("f8"))
                      ],
        'characteristic': [
            ('object', dtype(object))
        ],
        'controller': [
            ('object', dtype(object)),
            ('in_service', "bool"),
            ('order', "float64"),
            ('level', dtype(object)),
            ('initial_run', "bool"),
            ("recycle", dtype(object))
        ],
        'group': [
            ('name', dtype(object)),
            ('element_type', dtype(object)),
            ('element', dtype(object)),
            ('reference_column', dtype(object)),
        ],
        # geodata
        "line_geodata": [("coords", dtype(object))],
        "bus_geodata": [("x", "f8"), ("y", "f8"), ("coords", dtype(object))],
        # std_types
        'configuration_std_type': [("circuit", "i8"),
                                   ("conductor_outer_diameter_m", "f8"),
                                   ("gmr_coefficient", "f8"),
                                   ("r_dc_ohm_per_km", "f8"),
                                   ("g_us_per_km", "f8"),
                                   ("max_i_ka", "f8"),
                                   ("x_m", "f8"),
                                   ("y_m", "f8")],
        'sequence_std_type': [('name', dtype(object)),
                              ("r_ohm_per_km", "f8"),
                              ("x_ohm_per_km", "f8"),
                              ("r0_ohm_per_km", "f8"),
                              ("x0_ohm_per_km", "f8"),
                              ("c_nf_per_km", "f8"),
                              ("c0_nf_per_km", "f8"),
                              ("max_i_ka", "f8")],
        'matrix_std_type': [("circuit", "i8"),
                            ("name", dtype(object)),
                            ("max_i_ka", "f8"),
                            ("r_1_ohm_per_km", "f8"),
                            ("r_2_ohm_per_km", "f8"),
                            ("r_3_ohm_per_km", "f8"),
                            ("r_4_ohm_per_km", "f8"),
                            ("x_1_ohm_per_km", "f8"),
                            ("x_2_ohm_per_km", "f8"),
                            ("x_3_ohm_per_km", "f8"),
                            ("x_4_ohm_per_km", "f8"),
                            ("g_1_us_per_km", "f8"),
                            ("g_2_us_per_km", "f8"),
                            ("g_3_us_per_km", "f8"),
                            ("g_4_us_per_km", "f8"),
                            ("b_1_us_per_km", "f8"),
                            ("b_2_us_per_km", "f8"),
                            ("b_3_us_per_km", "f8"),
                            ("b_4_us_per_km", "f8")],

        # result tables
        "_empty_res_bus": [("vm_pu", "f8"),
                           ("va_degree", "f8"),
                           ("p_mw", "f8"),
                           ("q_mvar", "f8")],
        "_empty_res_ext_grid": [("p_mw", "f8"),
                                ("q_mvar", "f8")],
        "_empty_res_ext_grid_sequence": [("p_mw", "f8"),
                                         ("q_mvar", "f8")],
        "_empty_res_line": [("p_from_mw", "f8"),  # todo: define results format
                            ("q_from_mvar", "f8"),
                            ("p_to_mw", "f8"),
                            ("q_to_mvar", "f8"),
                            ("pl_mw", "f8"),
                            ("ql_mvar", "f8"),
                            ("i_from_ka", "f8"),
                            ("i_to_ka", "f8"),
                            ("i_ka", "f8"),
                            ("vm_from_pu", "f8"),
                            ("va_from_degree", "f8"),
                            ("vm_to_pu", "f8"),
                            ("va_to_degree", "f8"),
                            ("loading_percent", "f8")],
        # "_empty_res_trafo": [("p_hv_mw", "f8"),
        #                      ("q_hv_mvar", "f8"),
        #                      ("p_lv_mw", "f8"),
        #                      ("q_lv_mvar", "f8"),
        #                      ("pl_mw", "f8"),
        #                      ("ql_mvar", "f8"),
        #                      ("i_hv_ka", "f8"),
        #                      ("i_lv_ka", "f8"),
        #                      ("vm_hv_pu", "f8"),
        #                      ("va_hv_degree", "f8"),
        #                      ("vm_lv_pu", "f8"),
        #                      ("va_lv_degree", "f8"),
        #                      ("loading_percent", "f8")],
        "_empty_res_trafo": [("p_mw", "f8"),
                             ("q_mvar", "f8"),
                             ("pl_mw", "f8"),
                             ("ql_mvar", "f8"),
                             ("i_ka", "f8"),
                             ("vm_pu", "f8"),
                             ("va_degree", "f8"),
                             ("loading_percent", "f8")],
        "_empty_res_asymmetric_load": [("p_mw", "f8"),
                                       ("q_mvar", "f8")],
        "_empty_res_asymmetric_sgen": [("p_mw", "f8"),
                                      ("q_mvar", "f8"),
                                      ("va_degree", "f8"),
                                      ("vm_pu", "f8")],
        "_empty_res_asymmetric_gen": [("p_mw", "f8"),
                                      ("q_mvar", "f8"),
                                      ("va_degree", "f8"),
                                      ("vm_pu", "f8")],
        # internal
        "_ppc": None,
        "_ppc0": None,
        "_ppc1": None,
        "_ppc2": None,
        "_is_elements": None,
        "_pd2ppc_lookups": {"bus": None,
                            "gen": None,
                            "branch": None},
        "version": __version__,
        "format_version": __format_version__,
        "converged": False,
        "OPF_converged": False,
        "name": name,
        "f_hz": f_hz,
        "sn_mva": sn_mva,
        "rho_ohmm": rho_ohmm,
    }

    if hasattr(pandapowerNet, "create_dataframes"):
        # -------- RJ: Code version for post-July 17th develop branch of pandapower
        for key in structure_data.keys():
            if structure_data[key] is not None and isinstance(structure_data[key], list):
                structure_data[key] = dict(structure_data[key])

        net = pandapowerNet(pandapowerNet.create_dataframes(structure_data))
    else:
        # -------- RJ: Code version for previous version of pandapower
        net = pandapowerNet(structure_data)
    
    if add_stdtypes:
        add_basic_std_types(net)
    else:
        net.std_types = {"configuration": {}, "matrix": {}, "sequence": {}, "trafo": {}, "trafo3w": {}}
    # for mode in ["pf", "se", "sc", "pf_3ph"]:
    #     reset_results(net, mode)
    net['user_pf_options'] = dict()

    _set_multiindex(net)
    return upgrade_network_schema(net)


def _set_multiindex(net):
    for element, columns in _multi_index_columns.items():
        if element not in net:
            continue
        net[element].reset_index(inplace=True)        
        net[element].set_index(net[element].columns[columns].tolist(), inplace=True)
        # net[element].index.names = [None, *net[element].index.names[1:]]

def from_excel(filename, convert=True):
    """
    Load a pandapower network from an excel file

    INPUT:
        **filename** (string) - The absolute or relative path to the input file.

        **convert** (bool, True) - If True, converts the format of the net loaded from excel from
            the older version of pandapower to the newer version format

    OUTPUT:
        **net** (dict) - The pandapower format network

    EXAMPLE:

        >>> net1 = from_excel(os.path.join("C:", "example_folder", "example1.xlsx"))
        >>> net2 = from_excel("example2.xlsx") #relative path

    """

    if not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!" % filename)
    if not openpyxl_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "openpyxl")

    xls = dict()
    with pd.ExcelFile(filename, engine="openpyxl") as f:
        for sheet_name in f.sheet_names:
            xls[sheet_name] = f.parse(sheet_name=sheet_name, index_col=_multi_index_columns.get(sheet_name, 0))

    empty_multiconductor_net = create_empty_network()
    net = io_utils.from_dict_of_dfs(xls, net=empty_multiconductor_net)

    upgrade_network_schema(net)

    if convert:
        pass
        # convert_format(net)
    return net


def from_pickle(source):
    """
    Load a pickled pandapower/multiconductor network and apply schema upgrades.

    INPUT:
        **source** (path-like or binary file-like) - Pickle source.

    OUTPUT:
        **net** (pandapowerNet) - Upgraded network preserving existing values.

    WARNING:
        Pickle data can execute arbitrary code. Only load files from trusted sources.
    """

    if hasattr(source, "read"):
        net = _pickle_load_compat(source)
    else:
        with Path(source).expanduser().open("rb") as handle:
            net = _pickle_load_compat(handle)

    if not isinstance(net, pandapowerNet):
        if isinstance(net, dict):
            net = pandapowerNet(net)
        else:
            raise TypeError("Pickle payload does not contain a pandapower network mapping")

    return upgrade_network_schema(net)


if __name__ == "__main__":
    filename = os.path.join("multiconductor", "data", "IEEE_13bus_pp.xlsx")
    net = from_excel(filename)
