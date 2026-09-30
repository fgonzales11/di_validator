import pandas as pd
import warnings
import numbers
import numpy as np
import pandas as pd
from scipy import sparse
from numpy.linalg import inv


#PM: temporarily changed check required to false to support creation of 1ph_trafo_with_std_type and support p2p,p2n,mixed, added some std types



def create_std_type(net, data, name, element="configuration", overwrite=True, check_required=False):
    """
    Creates type data in the type database for multiconductor analysis.
    
    Standard types allowed are:
        - configuration: line data in configuration layout
        - matrix: line impedance matrix data
        - sequence: line impedance definition with symmetrical components
        - trafo: standard 3 phase - 2 windings transformer data
        
    The following is a list of the minimum required information set for each type:
        - configuration: conductor_outer_diameter_m, gmr_coefficient, r_dc_ohm_per_km, g_us_per_km, max_i_ka, x_m, y_m
        - matrix: max_i_ka, r_1_ohm_per_km, x_1_ohm_per_km, g_1_us_per_km, b_1_us_per_km
        - sequence: r_ohm_per_km, x_ohm_per_km, r0_ohm_per_km, x0_ohm_per_km, c_nf_per_km, c0_nf_per_km, max_i_ka
        - trafo: sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent

    The standard type is saved into the pandapower library of the given network.

    INPUT:
        **net** - The pandapower-like network

        **data** - dictionary of standard type parameters

        **name** - name of the standard type as string

        **element** - "configuration" (default), "matrix", "sequence", "trafo" or "trafo3w"

    EXAMPLE:

    >>> line_data = {"c_nf_per_km": 0, "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "max_i_ka": 0.142, "type": "cs", "q_mm2": 50, "alpha": 4.03e-3}
    >>> pandapower.create_std_type(net, line_data, "NAYY 4×50 SE", element='line')
    >>> # Three phase line creation:
    >>> pandapower.create_std_type(net, {"r_ohm_per_km": 0.1941, "x_ohm_per_km": 0.07476991,
                 "c_nf_per_km": 1160., "max_i_ka": 0.421,
                 "endtemp_degree": 70.0, "r0_ohm_per_km": 0.7766,
                 "x0_ohm_per_km": 0.2990796,
                 "c0_nf_per_km":  496.2}, name="unsymmetric_line_type",element = "line")
    >>> #Three-phase two-windings transformer creation
    >>> pp.create_std_type(net, {"sn_mva": 1.6,
         "vn_hv_kv": 10,
         "vn_lv_kv": 0.4,
         "vk_percent": 6,
         "vkr_percent": 0.78125,
         "pfe_kw": 2.7,
         "i0_percent": 0.16875,
         "shift_degree": 0,
         "vector_group": vector_group,
         "tap_side": "lv",
         "tap_neutral": 0,
         "tap_min": -2,
         "tap_max": 2,
         "tap_step_degree": 0,
         "tap_step_percent": 2.5,}, name='Unsymmetric_trafo_type', element="trafo")

    """

    if type(data) != dict:
        raise UserWarning("type data has to be given as a dictionary of parameters")

    if check_required:
        if element == "configuration":
            required = ["conductor_outer_diameter_m", "gmr_coefficient", "r_dc_ohm_per_km", "g_us_per_km", "max_i_ka", "x_m", "y_m"]
        elif element == "matrix":
            required = ["max_i_ka", "r_1_ohm_per_km", "x_1_ohm_per_km", "g_1_us_per_km", "b_1_us_per_km"]
        elif element == "sequence":
            required = ["r_ohm_per_km", "x_ohm_per_km", "r0_ohm_per_km", "x0_ohm_per_km", "c_nf_per_km", "c0_nf_per_km", "max_i_ka"]
        elif element == "trafo":
            required = ["sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent",
                        "pfe_kw", "i0_percent", "shift_degree"]
        elif element == "trafo3w":
            required = ["sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                        "vk_hv_percent", "vk_mv_percent", "vk_lv_percent", "vkr_hv_percent",
                        "vkr_mv_percent", "vkr_lv_percent", "pfe_kw", "i0_percent", "shift_mv_degree",
                        "shift_lv_degree"]
            
        else:
            raise ValueError("Unkown element type %s" % element)
        for par in required:
            if par not in data:
                raise UserWarning("%s is required as %s type parameter" % (par, element))
    library = net.std_types[element]
    if overwrite or not (name in library):
        library.update({name: data})


def create_std_types(net, data, element="configuration", overwrite=True, check_required=False):
    """
    Creates multiple standard types in the type database.

    INPUT:
        **net** - The pandapower-like network

        **data** - dictionary of standard type parameter sets

        **element** - "configuration", "matrix", "sequence", "trafo" or "trafo3w"

    EXAMPLE:

    >>> linetypes = {"typ1": {"r_ohm_per_km": 0.01, "x_ohm_per_km": 0.02, "c_nf_per_km": 10, "max_i_ka": 0.4, "type": "cs"},
                  "typ2": {"r_ohm_per_km": 0.015, "x_ohm_per_km": 0.01, "c_nf_per_km": 30, "max_i_ka": 0.3, "type": "cs"}}
    >>> mc.create_std_types(net, data=linetypes, element="line")

    """
    for name, typdata in data.items():
        create_std_type(net, data=typdata, name=name, element=element, overwrite=overwrite,check_required=check_required)


def copy_std_types(to_net, from_net, element="configuration", overwrite=True):
    """
    Transfers all standard types of one network to another.

    INPUT:

        **to_net** - The pandapower-like network to which the standard types are copied

        **from_net** - The pandapower-like network from which the standard types are taken

        **element** - "configuration", "matrix", "sequence", "trafo" or "trafo3w"

        **overwrite** - if True, overwrites standard types which already exist in to_net

    """
    for name, typdata in from_net.std_types[element].items():
        create_std_type(to_net, typdata, name, element=element, overwrite=overwrite)


def load_std_type(net, name, element="configuration"):
    """
    Loads standard type data from the linetypes data base. Issues a warning if
    linetype is unknown.

    INPUT:
        **net** - The pandapower-like network

        **name** - name of the standard type as string

        **element** - "configuration", "matrix", "sequence", "trafo" or "trafo3w"

    OUTPUT:
        **typedata** - dictionary containing type data

    """
    library = net.std_types[element]
    if name in library:
        return library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (element, name))


def std_type_exists(net, name, element="configuration"):
    """
    Checks if a standard type exists.

    INPUT:
        **net** - pandapower-like network

        **name** - name of the standard type as string

        **element** - type of element ("configuration", "matrix", "sequence", "trafo" or "trafo3w")

    OUTPUT:
        **exists** - True if standard type exists, False otherwise

    """
    library = net.std_types[element]
    return name in library


def delete_std_type(net, name, element="configuration"):
    """
    Deletes standard type parameters from database.

    INPUT:
        **net** - pandapower-like network

        **name** - name of the standard type as string

        **element** - type of element ("configuration", "matrix", "sequence", "trafo" or "trafo3w")

    """
    library = net.std_types[element]
    if name in library:
        del library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (element, name))


def available_std_types(net, element="configuration"):
    """
    Returns all standard types available for this network as a table.

    INPUT:
        **net** - pandapower-like network

        **element** - type of element ("configuration", "matrix", "sequence", "trafo" or "trafo3w")

    OUTPUT:
        **typedata** - table of standard type parameters

    """
    std_types = pd.DataFrame(net.std_types[element]).T
    try:
        return std_types.infer_objects()
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return std_types.convert_objects()

def basic_cfg_std_types():

    alpha_al = 4.03e-3
    alpha_cu = 3.93e-3

    types = {
        # CIGRE-LV (residential)
        "OH1":
        {"conductor_outer_diameter_m": 0.013,
         "gmr_coefficient": 0.53077,
         "r_dc_ohm_per_km": 0.387,
         "g_us_per_km": 0,
         "max_i_ka": 0.24,
         "x_m": [0, 0, 0, 0],
         "y_m": [8,7.7,7.4,7.1],
         "type": "ohl",
         "q_mm2": 50,
         "alpha": alpha_al}
        }

    return types

def basic_mat_std_types():

    types = {
        # IEEE-13bus types
        "601":
        {"max_i_ka": [1, 1, 1] ,
            'r_1_ohm_per_km': [0.21535,0.09695,0.0982],
            'r_2_ohm_per_km': [0.09695,0.20976,0.0954],
            'r_3_ohm_per_km': [0.0982,0.0954,0.21218],
            'x_1_ohm_per_km': [0.63263,0.31181,0.26327],
            'x_2_ohm_per_km': [0.31181,0.65121,0.23922],
            'x_3_ohm_per_km': [0.26327,0.23922,0.64313],
            'b_1_us_per_km': [3.91535,-1.2404,-0.78278],
            'b_2_us_per_km': [-1.2404,3.70398,-0.46097],
            'b_3_us_per_km': [-0.78278,-0.46097,3.50441],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
        
        "602":
        {"max_i_ka": [1, 1, 1],
            'r_1_ohm_per_km': [0.46774,0.0982,0.09695],
            'r_2_ohm_per_km': [0.0982,0.46457,0.0954],
            'r_3_ohm_per_km': [0.09695,0.0954,0.46215],
            'x_1_ohm_per_km': [0.73424,0.26327,0.31181],
            'x_2_ohm_per_km': [0.26327,0.74475,0.23922],
            'x_3_ohm_per_km': [0.31181,0.23922,0.75277],
            'b_1_us_per_km': [3.54195,-0.67228,-1.05065],
            'b_2_us_per_km': [-0.67228,3.21908,-0.40945],
            'b_3_us_per_km': [-1.05065,-0.40945,3.37141],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
        
        "603":
        {"max_i_ka": [1, 1],
            'r_1_ohm_per_km': [0.82623,0.1284],
            'r_2_ohm_per_km': [0.1284,0.82275],
            'x_1_ohm_per_km': [0.83723,0.28533],
            'x_2_ohm_per_km': [0.28533,0.84332],
            'b_1_us_per_km': [2.9271,-0.55929],
            'b_2_us_per_km': [-0.55929,2.89981],
            'g_1_us_per_km': [0,0],
            'g_2_us_per_km': [0,0]},
        
        "604":
        {"max_i_ka": [1, 1],
            'r_1_ohm_per_km': [0.82275,0.1284],
            'r_2_ohm_per_km': [0.1284,0.82623],
            'x_1_ohm_per_km': [0.84332,0.28533],
            'x_2_ohm_per_km': [0.28533,0.83723],
            'b_1_us_per_km': [2.89981,-0.55929],
            'b_2_us_per_km': [-0.55929,2.9271],
            'g_1_us_per_km': [0,0],
            'g_2_us_per_km': [0,0]},
        
        "605":
        {"max_i_ka": [1],
            'r_1_ohm_per_km': [0.8261],
            'x_1_ohm_per_km': [0.83748],
            'b_1_us_per_km': [2.80876],
            'g_1_us_per_km': [0]},
        
        "606":
        {"max_i_ka": [1, 1, 1],
            'r_1_ohm_per_km': [0.49608,0.19838,0.17707],
            'r_2_ohm_per_km': [0.19838,0.49043,0.19838],
            'r_3_ohm_per_km': [0.17707,0.19838,0.49608],
            'x_1_ohm_per_km': [0.27738,0.02039,-0.00889],
            'x_2_ohm_per_km': [0.02039,0.25115,0.02039],
            'x_3_ohm_per_km': [-0.00889,0.02039,0.27738],
            'b_1_us_per_km': [60.21734,0,0],
            'b_2_us_per_km': [0,60.21734,0],
            'b_3_us_per_km': [0,0,60.21734],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
        
        "607":
        {"max_i_ka": [1],
            'r_1_ohm_per_km': [0.83437],
            'x_1_ohm_per_km': [0.31846],
            'b_1_us_per_km': [55.30839],
            'g_1_us_per_km': [0]},
        
        "721":
        {"max_i_ka": [0.699,0.699,0.699],
            'r_1_ohm_per_km': [0.18181,0.04182,0.02094],
            'r_2_ohm_per_km': [0.04182,0.16442,0.04182],
            'r_3_ohm_per_km': [0.02094,0.04182,0.18181],
            'x_1_ohm_per_km': [0.1226,-0.02287,-0.02591],
            'x_2_ohm_per_km': [-0.02287,0.11806,-0.02287],
            'x_3_ohm_per_km': [-0.02591,-0.02287,0.1226],
            'b_1_us_per_km': [99.29033,0,0],
            'b_2_us_per_km': [0,99.29033,0],
            'b_3_us_per_km': [0,0,99.29033],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
            
        "722":
        {"max_i_ka": [0.483,0.483,0.483],
            'r_1_ohm_per_km': [0.29521,0.10122,0.07668],
            'r_2_ohm_per_km': [0.10122,0.27887,0.10122],
            'r_3_ohm_per_km': [0.07668,0.10122,0.29521],
            'x_1_ohm_per_km': [0.18473,-0.02026,-0.03772],
            'x_2_ohm_per_km': [-0.02026,0.1664,-0.02026],
            'x_3_ohm_per_km': [-0.03772,-0.02026,0.18473],
            'b_1_us_per_km': [79.43045,0,0],
            'b_2_us_per_km': [0,79.43045,0],
            'b_3_us_per_km': [0,0,79.43045],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
            
        "723":
            {"max_i_ka": [0.23,0.23,0.23],
            'r_1_ohm_per_km': [0.80381,0.30267,0.2849],
            'r_2_ohm_per_km': [0.30267,0.80915,0.30267],
            'r_3_ohm_per_km': [0.2849,0.30267,0.80381],
            'x_1_ohm_per_km': [0.41713,0.13117,0.09451],
            'x_2_ohm_per_km': [0.13117,0.39308,0.13117],
            'x_3_ohm_per_km': [0.09451,0.13117,0.41713],
            'b_1_us_per_km': [46.50385,0,0],
            'b_2_us_per_km': [0,46.50385,0],
            'b_3_us_per_km': [0,0,46.50385],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]},
            
        "724":
            {"max_i_ka": [0.156,0.156,0.156],
            'r_1_ohm_per_km': [1.3019,0.32336,0.30609],
            'r_2_ohm_per_km': [0.32336,1.30911,0.32336],
            'r_3_ohm_per_km': [0.30609,0.32336,1.3019],
            'x_1_ohm_per_km': [0.48206,0.17013,0.13192],
            'x_2_ohm_per_km': [0.17013,0.45969,0.17013],
            'x_3_ohm_per_km': [0.13192,0.17013,0.48206],
            'b_1_us_per_km': [37.43665,0,0],
            'b_2_us_per_km': [0,37.43665,0],
            'b_3_us_per_km': [0,0,37.43665],
            'g_1_us_per_km': [0,0,0],
            'g_2_us_per_km': [0,0,0],
            'g_3_us_per_km': [0,0,0]}

        }

    return types

def basic_seq_std_types():

    types = {
        # CIGRE-LV (residential)
        "UG1":
        {"r_ohm_per_km": 0.163,
         "x_ohm_per_km": 0.136,
         "r0_ohm_per_km": 0.490,
         "x0_ohm_per_km": 0.471,
         "c_nf_per_km": 240,
         "c0_nf_per_km": 240,
         "max_i_ka": 0.24}
        }

    return types

def basic_trafo_std_types():
    trafotypes = {
        # derived from Oswald - Transformatoren - Vorlesungsskript Elektrische Energieversorgung I
        # another recommendable references for distribution transformers is Werth:
        # Netzberechnung mit Erzeugungsprofilen
        "160 MVA 380/110 kV":
        {"i0_percent": 0.06,
            "pfe_kw": 60,
            "vkr_percent": 0.25,
            "sn_mva": 160,
            "vn_lv_kv": 110.0,
            "vn_hv_kv": 380.0,
            "vk_percent": 12.2,
            "shift_degree": 0,
            "vector_group": "Yy0",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "100 MVA 220/110 kV":
        {"i0_percent": 0.06,
            "pfe_kw": 55,
            "vkr_percent": 0.26,
            "sn_mva": 100,
            "vn_lv_kv": 110.0,
            "vn_hv_kv": 220.0,
            "vk_percent": 12.0,
            "shift_degree": 0,
            "vector_group": "Yy0",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},

        # compare to IFT Ingenieurbüro data and Schlabbach book
        "63 MVA 110/20 kV":
        {"i0_percent": 0.04,
            "pfe_kw": 22,
            "vkr_percent": 0.32,
            "sn_mva": 63,
            "vn_lv_kv": 20.0,
            "vn_hv_kv": 110.0,
            "vk_percent": 18,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "40 MVA 110/20 kV":
        {"i0_percent": 0.05,
            "pfe_kw": 18,
            "vkr_percent": 0.34,
            "sn_mva": 40,
            "vn_lv_kv": 20.0,
            "vn_hv_kv": 110.0,
            "vk_percent": 16.2,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "25 MVA 110/20 kV":
        {"i0_percent": 0.07,
            "pfe_kw": 14,
            "vkr_percent": 0.41,
            "sn_mva": 25,
            "vn_lv_kv": 20.0,
            "vn_hv_kv": 110.0,
            "vk_percent": 12,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "63 MVA 110/10 kV":
        {"sn_mva": 63,
            "vn_hv_kv": 110,
            "vn_lv_kv": 10,
            "vk_percent": 18,
            "vkr_percent": 0.32,
            "pfe_kw": 22,
            "i0_percent": 0.04,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "40 MVA 110/10 kV":
        {"sn_mva": 40,
            "vn_hv_kv": 110,
            "vn_lv_kv": 10,
            "vk_percent": 16.2,
            "vkr_percent": 0.34,
            "pfe_kw": 18,
            "i0_percent": 0.05,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        "25 MVA 110/10 kV":
        {"sn_mva": 25,
            "vn_hv_kv": 110,
            "vn_lv_kv": 10,
            "vk_percent": 12,
            "vkr_percent": 0.41,
            "pfe_kw": 14,
            "i0_percent": 0.07,
            "shift_degree": 150,
            "vector_group": "YNd5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -9,
            "tap_max": 9,
            "tap_step_degree": 0,
            "tap_step_percent": 1.5,
            "tap_phase_shifter": False},
        # Tafo20/0.4
        # 0.25 MVA 20/0.4 kV 0.45 Trafo Union
        "0.25 MVA 20/0.4 kV":
        {"sn_mva": 0.25,
            "vn_hv_kv": 20,
            "vn_lv_kv": 0.4,
            "vk_percent": 6,
            "vkr_percent": 1.44,
            "pfe_kw": 0.8,
            "i0_percent": 0.32,
            "shift_degree": 150,
            "vector_group": "Yzn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        # 0.4 MVA 20/0.4 kV Trafo Union
        "0.4 MVA 20/0.4 kV":
        {"sn_mva": 0.4, "vn_hv_kv": 20, "vn_lv_kv": 0.4,
            "vk_percent": 6,
            "vkr_percent": 1.425,
            "pfe_kw": 1.35,
            "i0_percent": 0.3375,
            "shift_degree": 150,
            "vector_group": "Dyn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        # 0.63 MVA 20/0.4 kV Trafo Union
        "0.63 MVA 20/0.4 kV":
        {"sn_mva": 0.63,
            "vn_hv_kv": 20,
            "vn_lv_kv": 0.4,
            "vk_percent": 6,
            "vkr_percent": 1.206,
            "pfe_kw": 1.65,
            "i0_percent": 0.2619,
            "shift_degree": 150,
            "vector_group": "Dyn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        # Tafo10/0.4:
        # 0.25 MVA 10/0.4 kV 0.4 Trafo Union wnr
        "0.25 MVA 10/0.4 kV":
        {"sn_mva": 0.25,
            "vn_hv_kv": 10,
            "vn_lv_kv": 0.4,
            "vk_percent": 4,
            "vkr_percent": 1.2,
            "pfe_kw": 0.6,
            "i0_percent": 0.24,
            "shift_degree": 150,
            "vector_group": "Dyn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        # 0.4 MVA 10/0.4 kV Trafo Union wnr
        "0.4 MVA 10/0.4 kV":
        {"sn_mva": 0.4,
            "vn_hv_kv": 10,
            "vn_lv_kv": 0.4,
            "vk_percent": 4,
            "vkr_percent": 1.325,
            "pfe_kw": 0.95,
            "i0_percent": 0.2375,
            "shift_degree": 150,
            "vector_group": "Dyn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        # 0.63 MVA 10/0.4 kV Trafo Union wnr
        "0.63 MVA 10/0.4 kV":
        {"sn_mva": 0.63,
            "vn_hv_kv": 10,
            "vn_lv_kv": 0.4,
            "vk_percent": 4,
            "vkr_percent": 1.0794,
            "pfe_kw": 1.18,
            "i0_percent": 0.1873,
            "shift_degree": 150,
            "vector_group": "Dyn5",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False},
        "0.63 MVA 10/0.4 kV Yy0": {
            "sn_mva": 0.63,
            "vn_hv_kv": 10,
            "vn_lv_kv": 0.4,
            "vk_percent": 4,
            "vkr_percent": 1.0794,
            "pfe_kw": 1.18,
            "i0_percent": 0.1873,
            "shift_degree": 0,
            "vector_group": "Yy0",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False
        },
        "0.63 MVA 10/0.4 kV Dd8": {
            "sn_mva": 0.63,
            "vn_hv_kv": 10,
            "vn_lv_kv": 0.4,
            "vk_percent": 4,
            "vkr_percent": 1.0794,
            "pfe_kw": 1.18,
            "i0_percent": 0.1873,
            "shift_degree": 0,
            "vector_group": "Dd8",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False
        },
        "0.63 MVA 10/0.4 kV Yd1": {
            "sn_mva": 0.63,
            "vn_hv_kv": 10.0,
            "vn_lv_kv": 0.4,
            "vk_percent": 4.0,
            "vkr_percent": 1.0794,
            "pfe_kw": 1.18,
            "i0_percent": 0.1873,
            "vector_group": "Yd1",
            "shift_degree": 30,
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -2,
            "tap_max": 2,
            "tap_step_degree": 0,
            "tap_step_percent": 2.5,
            "tap_phase_shifter": False
        },
        "Placeholder Standard Type 0.63 MVA 10/0.4 kV":
            {"sn_mva": 0.63,
             "vn_hv_kv": 10,
             "vn_lv_kv": 0.4,
             "vk_percent": 4,
             "vkr_percent": 1.0794,
             "pfe_kw": 1.18,
             "i0_percent": 0.1873,
             "tap_side": "hv",
             "tap_neutral": 0,
             "tap_min": -2,
             "tap_max": 2,
             "tap_step_degree": 0,
             "tap_step_percent": 2.5,
             "tap_phase_shifter": False},
    }
    return trafotypes

def basic_trafo3w_std_types():
    trafo3wtypes = {
        # generic trafo3w
        "63/25/38 MVA 110/20/10 kV":
        {"sn_hv_mva": 63,
            "sn_mv_mva": 25,
            "sn_lv_mva": 38,
            "vn_hv_kv": 110,
            "vn_mv_kv": 20,
            "vn_lv_kv": 10,
            "vk_hv_percent": 10.4,
            "vk_mv_percent": 10.4,
            "vk_lv_percent": 10.4,
            "vkr_hv_percent": 0.28,
            "vkr_mv_percent": 0.32,
            "vkr_lv_percent": 0.35,
            "pfe_kw": 35,
            "i0_percent": 0.89,
            "shift_mv_degree": 0,
            "shift_lv_degree": 0,
            "vector_group": "YN0yn0yn0",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -10,
            "tap_max": 10,
            "tap_step_percent": 1.2},
        "63/25/38 MVA 110/10/10 kV":
        {"sn_hv_mva": 63,
            "sn_mv_mva": 25,
            "sn_lv_mva": 38,
            "vn_hv_kv": 110,
            "vn_mv_kv": 10,
            "vn_lv_kv": 10,
            "vk_hv_percent": 10.4,
            "vk_mv_percent": 10.4,
            "vk_lv_percent": 10.4,
            "vkr_hv_percent": 0.28,
            "vkr_mv_percent": 0.32,
            "vkr_lv_percent": 0.35,
            "pfe_kw": 35,
            "i0_percent": 0.89,
            "shift_mv_degree": 0,
            "shift_lv_degree": 0,
            "vector_group": "YN0yn0yn0",
            "tap_side": "hv",
            "tap_neutral": 0,
            "tap_min": -10,
            "tap_max": 10,
            "tap_step_percent": 1.2}
    }
    return trafo3wtypes


def basic_std_types():
    return {
        "configuration" : basic_cfg_std_types(),
        "matrix"        : basic_mat_std_types(),
        "sequence"      : basic_seq_std_types(),
        "trafo"         : basic_trafo_std_types(),
        "trafo3w"       : basic_trafo3w_std_types(),
    }


def add_basic_std_types(net):
    """Adds basic standard types of the pandapower library to the net provided. These standard types
    are the same types that are available with output of `pandapower.create_empty_network()` and
    `pandapower.create_empty_network(add_stdtypes=True)` respectively.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net which should receive the basic standard types

    Returns
    -------
    tuple of dictionaries
        line, trafo and trafo3w types as dictionaries which have been added to the net.
    """

    if "std_types" not in net:
        net.std_types = {"configuration": {}, "matrix": {}, "sequence": {}, "trafo": {}, "trafo3w": {}}

    cfg_types = basic_cfg_std_types()
    mat_types = basic_mat_std_types()
    seq_types = basic_seq_std_types()
    trafotypes = basic_trafo_std_types()
    trafo3wtypes = basic_trafo3w_std_types()

    create_std_types(net, data=cfg_types, element="configuration")
    create_std_types(net, data=mat_types, element="matrix")
    create_std_types(net, data=seq_types, element="sequence")
    create_std_types(net, data=trafotypes, element="trafo")
    create_std_types(net, data=trafo3wtypes, element="trafo3w")
    return cfg_types, mat_types, seq_types, trafotypes
