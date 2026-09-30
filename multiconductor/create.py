import numbers
import warnings

import numpy as np
import pandas as pd
import itertools as it

from pandapower.create import _get_index_with_check
from multiconductor.pycci.std_types import load_std_type
SMALL_NUMBER = 1e-7


def create_bus(net, vn_kv, num_phases=4, name=None, grounded_phases=(0), grounding_r_ohm=0, grounding_x_ohm=0,
               in_service=True, type="b", zone=None, index=None):
    """
        Adds a new bus in net["bus"]

        INPUT:
            **net** - The net within this bus is added

            **vn_kv** (float) - The rated line-line voltage at the bus in kV

        OPTIONAL

            **name** (string) - A customized name for the bus

            **num_phases** (int, Default 4) - Number of phases at the bus (including neutral)

            **grounded_phases** (int, Default 0) - Number of the phase(s) connected to ground

            **grounding_r_ohm** (float) - Series resistance of grounding connection

            **grounding_r_ohm** (float) - Series reactance of grounding connection

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **type** (string) - Type of bus, for descriptive purpose

            **zone** (string) - Name of the zone including the bus, for descriptive purpose

            **index** (int, None) - Force a specified ID if it is available. Uses multi-index, requiring to number
                    both element and circuit(s). If None, the index one \
                    higher than the highest already existing index is selected.

            OUTPUT:
            **index** (int) - The unique ID of the created bus

            EXAMPLE:
            create_bus(net, vn_kv=0.4, num_phases=4, grounded_phases=3, grounding_r_ohm=10, grounding_x_ohm=2)

    """
    phases = np.arange(num_phases, dtype=np.int64)
    index, multi_index = _get_multi_index_with_check(net, "bus", (index, phases), ("index", "phase"))

    if grounded_phases is None:
        grounded_phases = []

    grounded = np.full_like(phases, False, bool)
    grounded[grounded_phases] = True
    grounding_r = np.full_like(phases, np.nan, np.float64)
    grounding_r[grounded] = grounding_r_ohm
    grounding_x = np.full_like(phases, np.nan, np.float64)
    grounding_x[grounded] = grounding_x_ohm

    entries = dict(zip(["name", "vn_kv", "grounded", "grounding_r_ohm", "grounding_x_ohm",
                        "type", "zone", "in_service"],
                       [name, vn_kv, grounded, grounding_r, grounding_x, type, zone, bool(in_service)]))

    net.bus = pd.concat([net.bus, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_line(net, std_type, model_type, from_bus, from_phase, to_bus, to_phase, length_km, name=None,
                in_service=True, index=None):
    """
        Creates a multi-phase line element using one of the standard model types among: configuration, matrix, sequence

        INPUT:
            **net** - The net within this line should be created

            **std_type** (string) - Name of standard type according to the model type selection.

            **model_type** (string) - Addresses the specific standard type model dataframe and must be chosen among:
            'configuration', 'matrix', 'sequence' (i.e. the three models available for a line representation)

            **from_bus** (int) - ID of the sending bus of the line

            **from_phase** (int) - ID of the phase in sending bus of the line

            **to_bus** (int) - ID of the receiving bus of the line

            **to_phase** (int) - ID of the phase in receiving bus of the line

            **length_km** (float) - The line length in km

        OPTIONAL:
            **name** (string, None) - A custom name for this line

            **in_service** (boolean, True) - True for in_service or False for out of service

            **index** (int, None) - Force a specified ID if it is available. Uses multi-index, requiring to number
                both element and circuit(s). If None, the index one \
                higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created line

        EXAMPLE:

        Create a three-phase (three-wired) line with length 2 km using a model based on symmetrical components theory
        ('sequence') and stored with index 'idx'

            create_line(net,std_type="type1",model_type="sequence",from_bus=0,from_phase=range(1,4),to_bus=1,to_phase=range(1,4),length_km=2)

    """
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)
    
    index, multi_index = _get_multi_index_with_check(net, "line", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['name', 'std_type', 'model_type', 'from_bus', 'from_phase', 'to_bus', 'to_phase',
                        'length_km', 'in_service'],
                       [name, std_type, model_type, from_bus, from_phase, to_bus, to_phase, length_km,
                        in_service]))

    net.line = pd.concat([net.line, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_switch(net, bus, phase, element, et, type=None, closed=True, name=None, r_ohm=0, index=None):
    """
    Adds a switch in the net["switch"] table.

    Switches can be either between two buses (bus-bus switch) or at the end of a line or transformer
    element (bus-element switch).

    Two buses that are connected through a closed bus-bus switches are fused in the power flow if
    the switch is closed or separated if the switch is open.

    An element that is connected to a bus through a bus-element switch is connected to the bus
    if the switch is closed or disconnected if the switch is open.

    INPUT:
        **net** - The net within which this switch should be created

        **bus** - The bus that the switch is connected to

        **phase** - List of phases at the switch bus connected by switch

        **element** - index of the element: bus id if et == "b", line id if et == "l", trafo id if \
            et == "t"

        **et** - (string) element type: "l" = switch between bus and line, "t" = switch between
            bus and transformer, "b" = switch between two buses

    OPTIONAL:
        **closed** (boolean, True) - switch position: False = open, True = closed (default)

        **type** (int, None) - indicates the type of switch: "LS" = Load Switch, "CB" = \
            Circuit Breaker, "LBS" = Load Break Switch or "DS" = Disconnecting Switch

        **name** (string, default None) - The name for this switch


    OUTPUT:
        **sid** - The unique switch_id of the created switch

    EXAMPLE:
        create_switch(net, bus = 0, element = 1, phase=range(1,4), et = 'l', closed= False)

    """
    if isinstance(phase,(list,tuple,range)):
        circuits = np.arange(len(phase), dtype=np.int64)
    elif isinstance(phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)
        
    index, multi_index = _get_multi_index_with_check(net, "switch", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['bus', 'phase', 'element', 'et', 'type', 'closed', 'name', 'r_ohm'],
                       [bus, phase, element, et, type, closed, name, r_ohm]))

    net.switch = pd.concat([net.switch, pd.DataFrame(index=multi_index, data=entries)])
    return index

def create_transformer_3ph(net, hv_bus, lv_bus, std_type, tap_pos=np.nan, name="tr",
                    in_service=True,index=None, shift_degree=0):
    """
        Adds a three-phase, two-windings transformer. The transformer data will
        be manipulated to undergo the multiconductor standard for generalized transformers.
        The resulting model will be stored in the net object as a "trafo1ph" object.

        INPUT:
            **net** - The net within which this switch should be created

            **hv_bus** (int) - Index of the higher voltage bus
            
            **lv_bus** (int) - Index of the lower voltage bus

            **std_type** (string) - Transformer std type from library
            
        OPTIONAL:

            **tap_pos** (int, default 0) - Current tap position

            **name** (string, None) - A custom name for this transformer

            **in_service** (boolean, True) - True for in_service or False for out of service, for each winding
            
        OUTPUT:
            
            **index** (int) - The unique ID of the created transformer

        EXAMPLE:
            
            mc.create_transformer(net, hv_bus = 0, lv_bus = 1, std_type = "0.4 MVA 10/0.4 kV", name = "trafo1")
            
    """
    
    name=name+'_3ph2w'
    NN=(hv_bus,lv_bus)
    
    if not hasattr(net, "std_types"):
        raise UserWarning('No std type library loaded in net.')
    elif not 'trafo' in net["std_types"]:
        raise UserWarning('Transformer standard types not in the loaded library.')
    
    std = load_std_type(net, std_type, "trafo")
    Sn=std["sn_mva"]    # Transformer rated power (per phase) [MVA]
    vn_hv=std["vn_hv_kv"]
    vn_lv=std["vn_lv_kv"]
    vk=std["vk_percent"]   # Short circuit voltage [%]
    vkr=std["vkr_percent"]   # Short circuit power [%]
    pfe=std["pfe_kw"]    # Open circuit losses (per phase) [kW]
    i0=std["i0_percent"]    # Open circuit current [%]
    Tap_min=std["tap_min"]    # Minimum tap position
    Tap_max=std["tap_max"]    # Maximum tap position
    Tap_side = std["tap_side"] # Tap side (currently only "lv" supported)
    Tap_n=std["tap_neutral"]    # Neutral tap position
    dU = std["tap_step_percent"]    # Voltage variation per tap [%]
    Tap_StepDeg = std["tap_step_degree"]    # Step size for voltage angle [�] (currently only 0 supported)
    if tap_pos is np.nan:
        Tap=Tap_n
    
    # Note: load vector group from std type. If not found, sets 'Yy' as default
    if 'vector_group' in std:
        vect_group=std["vector_group"]
    else:
        warnings.warn(message='Vector group not found in transformer std type. Assumed as Yy0.')
        vect_group='Yy0'

    vect_group = vect_group.replace('N', '')
    vect_group = vect_group.replace('n', '')
    hv_con=''.join([c for c in vect_group if c.isupper()])
    lv_con=''.join([c for c in vect_group if c.islower()])
    #group=int(''.join(c for c in vect_group if c.isdigit()))

    group_str = ''.join(c for c in vect_group if c.isdigit())
    if group_str:
        group = int(group_str)
    else:
        group = shift_degree
    
    if vect_group[0:2].isupper() or vect_group[0:2].islower():
        warnings.warn(message="Trafo 3ph 2w id:{i}, has wrong vect_group (all upper case or lower case letters). Assumed Yy connection.")
        hv_con='Y'
        lv_con='y'
    else:
        if vect_group[0].isupper():
            hv_con=vect_group[0]
            lv_con=vect_group[1]
        else:
            hv_con=vect_group[1]
            lv_con=vect_group[0]
    
    fr_ph_1,to_ph_1=tras_conn(hv_con.lower(), 0)
    fr_ph_2,to_ph_2=tras_conn(lv_con.lower(), group)
    
    v_r=(vn_hv,vn_lv)
    v_w=[]
    j=0
    for i in (hv_con,lv_con):
        if i.lower()=='y':
            # wye connection
            v_w.append(v_r[j]/np.sqrt(3))
        elif i.lower()=='d':
            # delta connection
            v_w.append(v_r[j])
        elif i.lower()=='z':
            warnings.warn(message='Connection type "z" not implemented. It is assumed as "y"')
            v_w.append(v_r[j] / np.sqrt(3))
        else:
            raise TypeError("Connection type not recognized in vector group.")
        j += 1

    # check consistency of open-circuit losses data (i.e. i0>=p0)
    if i0<pfe/(Sn*10):
        i0=pfe/(Sn*10)
    create_transformer1ph(net, NN, fr_ph_1+fr_ph_2, to_ph_1+to_ph_2, v_w, Sn/3, vk/2, vkr/2, pfe/6, i0/2, Tap_side, Tap_n, Tap_min, Tap_max, dU, Tap_StepDeg, Tap, name, in_service)

def create_transformer3w(net, hv_bus, mv_bus, lv_bus, std_type, tap_pos=np.nan, name="tr",
                    in_service=True,index=None):
    """
        Adds a three-phase, three-windings transformer. The transformer data will
        be manipulated to undergo the multiconductor standard for generalized transformers.
        The resulting model will be stored in the net object as a "trafo1ph" object.

        INPUT:
            **net** - The net within which this switch should be created

            **hv_bus** (int) - Index of the higher voltage bus
            
            **mv_bus** (int) - Index of the medium voltage bus
            
            **lv_bus** (int) - Index of the lower voltage bus

            **std_type** (string) - Transformer std type from library
            
        OPTIONAL:

            **tap_pos** (int, default 0) - Current tap position

            **name** (string, None) - A custom name for this transformer

            **in_service** (boolean, True) - True for in_service or False for out of service, for each winding
            
        OUTPUT:
            
            **index** (int) - The unique ID of the created transformer

        EXAMPLE:
            
            mc.create_transformer3w(net, hv_bus = 0, mv_bus = 1, lv_bus = 2, std_type = "63/25/38 MVA 110/20/10 kV", name = "trafo1")
            
    """
    
    name=name+'_3ph3w'
    NN=(hv_bus,mv_bus,lv_bus)
    
    if not hasattr(net, "std_types"):
        raise UserWarning('No std type library loaded in net.')
    elif not 'trafo' in net["std_types"]:
        raise UserWarning('Transformer standard types not in the loaded library.')
    
    std = load_std_type(net, std_type, "trafo3w")
    Sh=std["sn_hv_mva"]    # Transformer rated power (HV) [MVA]
    Sm=std["sn_mv_mva"]    # Transformer rated power (MV) [MVA]
    Sl=std["sn_lv_mva"]    # Transformer rated power (LV) [MVA]
    vh=std["vn_hv_kv"]    # HV side voltage [kV]
    vm=std["vn_mv_kv"]    # MV side voltage [kV]
    vl=std["vn_lv_kv"]    # LV side voltage [kV]
    vkh=std["vk_hv_percent"] # Short circuit voltage (HV-MV) [%]
    vkm=std["vk_mv_percent"] # Short circuit voltage (MV-LV) [%]
    vkl=std["vk_lv_percent"] # Short circuit voltage (HV-LV) [%]
    vkrh=std["vkr_hv_percent"] # Short circuit power (HV-MV) [%]
    vkrm=std["vkr_mv_percent"] # Short circuit power (MV-LV) [%]
    vkrl=std["vkr_lv_percent"] # Short circuit power (HV-LV) [%]
    pfe=std["pfe_kw"]    # Open circuit losses (per phase) [kW]
    i0=std["i0_percent"]    # Open circuit current [%]
    Tap_min=std["tap_min"]    # Minimum tap position
    Tap_max=std["tap_max"]    # Maximum tap position
    Tap_side = std["tap_side"] # Tap side (currently only "lv" supported)
    Tap_n=std["tap_neutral"]    # Neutral tap position
    dU = std["tap_step_percent"]    # Voltage variation per tap [%]
    Tap_StepDeg = std["tap_step_degree"]    # Step size for voltage angle [�]
    if tap_pos is np.nan:
        Tap=Tap_n
    
    # Note: load vector group from std type. If not found, sets 'YN0yn0yn0' as default
    if 'vector_group' in std:
        vect_group=std["vector_group"]
    else:
        warnings.warn(message='Vector group not found in transformer3w std type. Assumed as YN0yn0yn0.')
        vect_group='YN0yn0yn0'
    
    # Remove neutral specifications from vector group string
    vect_group=vect_group.replace('N','')
    vect_group=vect_group.replace('n','')
    
    conn=''.join([c for c in vect_group if not(c.isdigit())])
    conn=conn.lower()
    g_list=[]
    gr=''
    count=0
    for c in vect_group:
        count=count+1
        if c.isdigit():
            gr=gr+c
        elif len(gr)>0:
            g_list.append(int(gr))
            gr=''
        if count==len(vect_group):
            g_list.append(int(gr))
    
    if len(conn)<3:
        warnings.warn("Trafo 3ph-3w has missing connection specification on at least one side. It is assumed as wye.")
        conn=conn + ('y'*3-len(conn))
    elif len(conn)>3:
        raise UserWarning("Trafo 3ph-3w has unknown connection specification on at least one side.")
    
    if len(g_list)<3:
        warnings.warn("Trafo 3ph-3w has missing group on at least one side. It is assumed as 0.")
        g_list.extend([0]*3-len(g_list))
    elif len(conn)>3:
        raise UserWarning("Trafo 3ph-3w has unknown group on at least one side.")
    
    sn=[]
    sn.extend([Sh/3]*3)
    sn.extend([Sm/3]*3)
    sn.extend([Sl/3]*3)
    
    v_r=(vh,vm,vl)
    Zhm=vkh*Sh/min(Sh,Sm)
    Zml=vkm*Sh/min(Sm,Sl)
    Zlh=vkl*Sh/min(Sh,Sl)
    Zh=(Zhm+Zlh-Zml)/2
    Zm=(Zhm+Zml-Zlh)/2
    Zl=(Zml+Zlh-Zhm)/2
    zh=Zh
    zm=Zm*Sm/Sh
    zl=Zl*Sl/Sh
    Rhm=vkrh*Sh/min(Sh,Sm)
    Rml=vkrm*Sh/min(Sm,Sl)
    Rlh=vkrl*Sh/min(Sh,Sl)
    Rh=(Rhm+Rlh-Rml)/2
    Rm=(Rhm+Rml-Rlh)/2
    Rl=(Rml+Rlh-Rhm)/2
    rh=Rh
    rm=Rm*Sm/Sh
    rl=Rl*Sl/Sh
    
    z=[zh,zm,zl]
    r=[rh,rm,rl]
    v_w=[]
    j=0
    for i in conn:
        if i=='y':
            # wye connection
            v_w.extend([v_r[j]/np.sqrt(3)]*3)
        elif i=='d':
            # delta connection
            v_w.extend([v_r[j]]*3)
        else:
            raise TypeError("Connection type not recognized in vector group.")
        j += 1
    
    fph=[]
    tph=[]
    vk_list=[]
    vkr_list=[]
    pfe_list=[]
    i0_list=[]
    for j in range(len(g_list)):
        fr_ph,to_ph=tras_conn(conn[j], g_list[j])
        fph.extend(fr_ph)
        tph.extend(to_ph)
        vk_list.extend([z[j]]*3)
        vkr_list.extend([r[j]]*3)
        if j==0:
            pfe_list.extend([pfe/3]*3)
            i0_list.extend([i0]*3)
        else:
            pfe_list.extend([0]*3)
            i0_list.extend([1e-10]*3)
    
    # check consistency of open-circuit losses data (i.e. i0>=p0)
    if i0<(pfe/(1e3*Sh)):
        i0=pfe/(1e3*Sh)*100
        warnings.warn('The 3phase-3w transformer has an i0 too low. Forced to be equal to pfe ratio.')
    
    create_transformer1ph(net, NN, fph, tph, v_w, sn, vk_list, vkr_list, pfe_list, i0_list, Tap_side, Tap_n, Tap_min, Tap_max, dU, Tap_StepDeg, Tap, name, in_service)
    
def create_transformer1ph(net, buses, from_phase, to_phase, vn_kv, sn_mva, vk_percent, vkr_percent, pfe_kw,
                    i0_percent, tap_side='lv', tap_neutral=0, tap_min=0, tap_max=0, tap_step_percent=0, tap_step_degree=0, tap_pos=0, name=None,
                    in_service=True,index=None):
    """
        Adds a generic single- or multi-phase transformer with a generic number of windings in the net["trafo1ph"] table.

        Data is provided for each side of the transformer, which has to be singularly addressed (i.e. for a single-phase
        transformer, data has to be provided in two rows for primary and secondary sides, respectively.

        INPUT:
            **net** - The net within which this switch should be created

            **buses** (int) - List of indices of the buses connected by the transformer (number of buses will define number of windings)

            **from_phase** (int) - List of phases beginning windings connections at specific side (*)

            **to_phase** (int) - List of phases ending windings connections at specific side (*)

            **vn_kv** (float) - Winding connections' rated voltage in kV

            **sn_mva** (float) - Winding connections' rated power in MVA

            **vk_percent** (float) - Winding connections' short circuit voltage magnitude in percent with respect to rated voltage

            **vkr_percent** (float) - Winding connections' real component of short circuit voltage in percent with respect to rated voltage (i.e. corresponding to short-circuit power)

            **pfe_kw** (float) - Winding connections' iron core losses in kW

            **i0_percent** (float) -  Winding connections' magnetizing current with respect to rated current
            
            (*): list size must be equal to (number of buses)x(number of circuits)
            (**): list size should equal the number of terminals. If only 1 parameter is passed, it is used for all circuit ports.

        OPTIONAL:

            **tap_neutral** (int, default 0) - Neutral tap position

            **tap_min** (int, default 0) - Minimum tap position allowed

            **tap_max** (int, default 0) - Maximum tap position allowed

            **tap_step_percent** (float, default 0) - Percent variation of winding's voltage due to tap shift

            **tap_pos** (int, default 0) - Current tap position

            **name** (string, None) - A custom name for this transformer

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

            **in_service** (boolean, True) - True for in_service or False for out of service, for each winding

        OUTPUT:
            **index** (int) - The unique ID of the created transformer

        EXAMPLE:
            Create a three-phase, two-windings transformer having the following nameplate data:
                - rated three-phase power: 630 kVA
                - rated line-line voltages: 20/0.4 kV
                - short-circuit voltage: 6%
                - short-circuit power: 1%
                - iron core losses: 0.3 kW
                - magnetizing current: 0.1%

            Given the nameplate data above, the input data to the model should be provided according to these rules:
                - each side of each circuit is treated individually (6 rows for a 2w-3ph trafo)
                - each row has 1/6 of the overall three-phase rated power
                - short circuit impedance, defined by vkr_percent and vk_percent, which is in relative terms, has to be
                  divided by 2 since it is present at both sides
                - open-circuit power loss (iron losses) is defined in absolute terms (kW) and therefore has to be divided
                  by 6 to get the row value (same as rated power)
                - magnetizing current losses (i0) is defined in relative terms with respect to the rated power (which is
                  already divided by 6)

            create_trafo1ph(net, buses=(0,1), from_phase=range(1,4), to_phase=0, vn_kv=(11.547,0.23094),sn_mva=0.105, vk_percent=3, vkr_percent=0.5, pfe_kw=0.05, i0_percent=0.1)

    """
    circuits = np.arange(len(from_phase)/len(buses), dtype=np.int64)
    
    index, multi_index = _get_multi_index_with_check(net, "trafo1ph", (index, buses, circuits),
                                                     ("index", "bus", "circuit"))

    entries = dict(zip(['name', 'from_phase', 'to_phase', 'vn_kv', 'sn_mva', 'vk_percent', 'vkr_percent', 'pfe_kw',
                        'i0_percent', 'tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_step_percent', 'tap_step_degree', 'tap_pos', 'in_service'],
                       [name,
                        _trafo_variable_entry(buses, circuits, from_phase),
                        _trafo_variable_entry(buses, circuits, to_phase),
                        _trafo_variable_entry(buses, circuits, vn_kv),
                        sn_mva,
                        _trafo_variable_entry(buses, circuits, vk_percent),
                        _trafo_variable_entry(buses, circuits, vkr_percent),
                        _trafo_variable_entry(buses, circuits, pfe_kw),
                        _trafo_variable_entry(buses, circuits, i0_percent),
                        tap_side,
                        tap_neutral,
                        tap_min,
                        tap_max,
                        _trafo_variable_entry(buses, circuits, tap_step_percent),
                        _trafo_variable_entry(buses, circuits, tap_step_degree),
                        _trafo_variable_entry(buses, circuits, tap_pos),
                        in_service]))

    # # Check coherence between buses and phases in trafo and in net.bus
    # _check_branch_element(net, "Transformer", index, hv_bus, hv_phase)
    # _check_branch_element(net, "Transformer", index, lv_bus, lv_phase)
    
    net.trafo1ph = pd.concat([net.trafo1ph, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_transformer1ph_from_std_type(net, buses, std_type, from_phase, to_phase, tap_pos=np.nan,
                                        shift_degree=0, name=None,
                          in_service=True, index=None):
    """
        Adds a generic single- or multi-phase transformer with a generic number of windings in the net["trafo1ph"] table.

        Data is provided for each side of the transformer, which has to be singularly addressed (i.e. for a single-phase
        transformer, data has to be provided in two rows for primary and secondary sides, respectively.

        INPUT:
            **net** - The net within which this switch should be created

            **buses** (int) - List of indices of the buses connected by the transformer (number of buses will define number of windings)

            **from_phase** (int) - List of phases beginning windings connections at specific side (*)

            **to_phase** (int) - List of phases ending windings connections at specific side (*)

            **vn_kv** (float) - Winding connections' rated voltage in kV

            **sn_mva** (float) - Winding connections' rated power in MVA

            **vk_percent** (float) - Winding connections' short circuit voltage magnitude in percent with respect to rated voltage

            **vkr_percent** (float) - Winding connections' real component of short circuit voltage in percent with respect to rated voltage (i.e. corresponding to short-circuit power)

            **pfe_kw** (float) - Winding connections' iron core losses in kW

            **i0_percent** (float) -  Winding connections' magnetizing current with respect to rated current

            (*): list size must be equal to (number of buses)x(number of circuits)
            (**): list size should equal the number of terminals. If only 1 parameter is passed, it is used for all circuit ports.

        OPTIONAL:

            **tap_neutral** (int, default 0) - Neutral tap position

            **tap_min** (int, default 0) - Minimum tap position allowed

            **tap_max** (int, default 0) - Maximum tap position allowed

            **tap_step_percent** (float, default 0) - Percent variation of winding's voltage due to tap shift

            **tap_pos** (int, default 0) - Current tap position

            **name** (string, None) - A custom name for this transformer

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

            **in_service** (boolean, True) - True for in_service or False for out of service, for each winding

        OUTPUT:
            **index** (int) - The unique ID of the created transformer

        EXAMPLE:
            Create a three-phase, two-windings transformer having the following nameplate data:
                - rated three-phase power: 630 kVA
                - rated line-line voltages: 20/0.4 kV
                - short-circuit voltage: 6%
                - short-circuit power: 1%
                - iron core losses: 0.3 kW
                - magnetizing current: 0.1%

            Given the nameplate data above, the input data to the model should be provided according to these rules:
                - each side of each circuit is treated individually (6 rows for a 2w-3ph trafo)
                - each row has 1/6 of the overall three-phase rated power
                - short circuit impedance, defined by vkr_percent and vk_percent, which is in relative terms, has to be
                  divided by 2 since it is present at both sides
                - open-circuit power loss (iron losses) is defined in absolute terms (kW) and therefore has to be divided
                  by 6 to get the row value (same as rated power)
                - magnetizing current losses (i0) is defined in relative terms with respect to the rated power (which is
                  already divided by 6)

            create_trafo1ph(net, buses=(0,1), from_phase=range(1,4), to_phase=0, vn_kv=(11.547,0.23094),sn_mva=0.105, vk_percent=3, vkr_percent=0.5, pfe_kw=0.05, i0_percent=0.1)

    """

    NN = (buses)

    if not hasattr(net, "std_types"):
        raise UserWarning('No std type library loaded in net.')
    elif not 'trafo' in net["std_types"]:
        raise UserWarning('Transformer standard types not in the loaded library.')

    std = load_std_type(net, std_type, "trafo")
    Sn = std["sn_mva"]  # Transformer rated power (per phase) [MVA]
    vn_hv = std["vn_hv_kv"]
    vn_lv = std["vn_lv_kv"]
    vk = std["vk_percent"]  # Short circuit voltage [%]
    vkr = std["vkr_percent"]  # Short circuit power [%]
    pfe = std["pfe_kw"]  # Open circuit losses (per phase) [kW]
    i0 = std["i0_percent"]  # Open circuit current [%]
    Tap_min = std["tap_min"]  # Minimum tap position
    Tap_max = std["tap_max"]  # Maximum tap position
    Tap_n = std["tap_neutral"]  # Neutral tap position
    dU = std["tap_step_percent"]  # Voltage variation per tap [%]
    dD = std.get("tap_step_degree", 0)
    if tap_pos is np.nan:
        Tap = Tap_n
    v_r = (vn_hv, vn_lv)

    # check consistency of open-circuit losses data (i.e. i0>=p0)
    if i0 < pfe / (Sn * 10):
        i0 = pfe / (Sn * 10)

    from_phase= from_phase
    to_phase = to_phase
    sn_mva = Sn
    vn_kv = v_r
    vk_percent = vk / 2
    vkr_percent = vkr / 2
    pfe_kw = pfe /2
    i0_percent = i0 / 2
    tap_neutral = Tap_n
    tap_min = Tap_min
    tap_max = Tap_max
    tap_step_percent = dU
    tap_step_degree = dD
    tap_pos = Tap


    circuits = np.arange(len(from_phase) / len(buses), dtype=np.int64)

    index, multi_index = _get_multi_index_with_check(net, "trafo1ph", (index, buses, circuits),
                                                     ("index", "bus", "circuit"))

    entries = dict(zip(['name', 'from_phase', 'to_phase', 'vn_kv', 'sn_mva', 'vk_percent', 'vkr_percent', 'pfe_kw',
                        'i0_percent', 'tap_neutral', 'tap_min', 'tap_max', 'tap_step_percent', 'tap_step_degree', 'tap_pos', 'in_service'],
                       [name,
                        _trafo_variable_entry(buses, circuits, from_phase),
                        _trafo_variable_entry(buses, circuits, to_phase),
                        _trafo_variable_entry(buses, circuits, vn_kv),
                        sn_mva,
                        _trafo_variable_entry(buses, circuits, vk_percent),
                        _trafo_variable_entry(buses, circuits, vkr_percent),
                        _trafo_variable_entry(buses, circuits, pfe_kw),
                        _trafo_variable_entry(buses, circuits, i0_percent),
                        tap_neutral,
                        tap_min,
                        tap_max,
                        _trafo_variable_entry(buses, circuits, tap_step_percent),
                        _trafo_variable_entry(buses, circuits, tap_step_degree),
                        _trafo_variable_entry(buses, circuits, tap_pos),
                        in_service]))

    # # Check coherence between buses and phases in trafo and in net.bus
    # _check_branch_element(net, "Transformer", index, hv_bus, hv_phase)
    # _check_branch_element(net, "Transformer", index, lv_bus, lv_phase)

    net.trafo1ph = pd.concat([net.trafo1ph, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_asymmetric_load(net, bus, from_phase, to_phase, p_mw, q_mvar, const_z_percent_p=0, const_i_percent_p=0,
                           const_z_percent_q=0, const_i_percent_q=0, sn_mva=None, scaling=1, in_service=True, name=None,
                           type=None, index=None):
    """
        Adds a new generic load in net["asymmetric_load"]

        INPUT:
            **net** - The net within this line should be created

            **bus** (int) - Index of the load's bus

            **from_phase** (int) - Phase at the sending side of load connection. \
            Multiple parameters could be passed for multi-phase loads

            **to_phase** (int) - Phase at the receiving side of load connection. \
            Multiple parameters could be passed for multi-phase loads

            **p_mw** (float) - Load's active power in MW

            **q_mvar** (float) - Load's reactive power in MVAr

        OPTIONAL:
            **const_z_percent_p** (int) - Percent share of the load's active \
            power modelled as constant impedance. Default: 0

            **const_i_percent_p** (int) - Percent share of the load's active \
            power modelled as constant current. Default: 0

            **const_z_percent_q** (int) - Percent share of the load's reactive \
            power modelled as constant impedance. Default: 0

            **const_i_percent_q** (int) - Percent share of the load's reactive \
            power modelled as constant current. Default: 0

            **sn_mva** (float) - Load's apparent power in MVA

            **scaling** (float) - Scaling factor to active and reactive power. Default: 1

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **name** (string) - A customized name for the load

            **type** (string) - Type of load, to be used for applying load scaling profiles

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created load

        EXAMPLE:
            Define a three-phase load at bus 2, wye-connected at phases 1, 2, 3 and neutral point 0. \
            The load has a three-phase power of 15 MW + 6 MVAr

            create_asymmetric_load(net, bus=2, phase_from=range(1, 4), phase_to=0, p_mw=5, q_mvar=2)

    """
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)

    index, multi_index = _get_multi_index_with_check(net, "asymmetric_load", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase', 'p_mw', 'q_mvar',
                        'const_z_percent_p', 'const_i_percent_p', 'const_z_percent_q',
                        'const_i_percent_q', 'sn_mva', 'scaling', 'in_service', 'type'],
                       [name, bus, from_phase, to_phase, p_mw, q_mvar, const_z_percent_p, const_i_percent_p,
                        const_z_percent_q, const_i_percent_q, sn_mva, scaling, in_service, type]))

    net.asymmetric_load = pd.concat([net.asymmetric_load, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_asymmetric_sgen(net, bus, from_phase, to_phase, p_mw, q_mvar, const_z_percent_p=0, const_i_percent_p=0,
                           const_z_percent_q=0, const_i_percent_q=0, sn_mva=None, scaling=1, in_service=True, name=None,
                           type=None, slack=False, index=None):
    """
        Adds a new generic static generator in net["asymmetric_sgen"]. The static generator is used to refer to
        generators operating according to PQ model (i.e. active and reactive power are the input parameter, possibly
        varying according to voltage dependence specifications)

        INPUT:
            **net** - The net within this line should be created

            **bus** (int) - Index of the generator's bus

            **from_phase** (int) - Phase at the sending side of generator's connection. \
            Multiple parameters could be passed for multi-phase generators

            **to_phase** (int) - Phase at the receiving side of generator's connection. \
            Multiple parameters could be passed for multi-phase generators

            **p_mw** (float) - Generator's active power in MW

            **q_mvar** (float) - Generator's reactive power in MVAr

        OPTIONAL:
            **const_z_percent_p** (int) - Percent share of the generator's active power \
            modelled as constant impedance. Default: 0

            **const_i_percent_p** (int) - Percent share of the generator's active power \
            modelled as constant current. Default: 0

            **const_z_percent_q** (int) - Percent share of the generator's reactive power \
            modelled as constant impedance. Default: 0

            **const_i_percent_q** (int) - Percent share of the generator's reactive power \
            modelled as constant current. Default: 0

            **sn_mva** (float) - Generator's apparent power in MVA

            **scaling** (float) - Scaling factor to active and reactive power. Default: 1

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **name** (string) - A customized name for the generator

            **type** (string) - Type of generator, to be used for applying generator scaling profiles

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created generator

        EXAMPLE:
            Define a single-phase generator at bus 2 between phase 1 and neutral \
            point 0 having a real power of 6 kW with unitary power factor.

            create_asymmetric_sgen(net, bus=2, phase_from=1, phase_to=0, p_mw=0.006, q_mvar=0)

    """
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)
    
    index, multi_index = _get_multi_index_with_check(net, "asymmetric_sgen", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase', 'p_mw', 'q_mvar', 'const_z_percent_p',
                        'const_i_percent_p', 'const_z_percent_q', 'const_i_percent_q', 'sn_mva', 'scaling',
                        'in_service', 'type', 'slack'],
                       [name, bus, from_phase, to_phase, p_mw, q_mvar, const_z_percent_p, const_i_percent_p,
                        const_z_percent_q, const_i_percent_q, sn_mva, scaling, in_service, type, slack]))

    net.asymmetric_sgen = pd.concat([net.asymmetric_sgen, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_asymmetric_gen(net, bus, from_phase, to_phase, p_mw, vm_pu, sn_mva=None, scaling=1, in_service=True, name=None,
                          type=None, index=None):
    """
        Adds a new generic generator in net["asymmetric_gen"]. These generators are modelled according to a PV model
        (i.e. input data are active power and voltage magnitude).

        INPUT:
            **net** - The net within this line should be created

            **bus** (int) - Index of the generator's bus

            **from_phase** (int) - Phase at the sending side of generator's connection. \
            Multiple parameters could be passed for multi-phase generators

            **to_phase** (int) - Phase at the receiving side of generator's connection. \
            Multiple parameters could be passed for multi-phase generators

            **p_mw** (float) - Generator's active power in MW

            **vm_pu** (float) - Voltage magnitude controlled by generator

        OPTIONAL:
            **sn_mva** (float) - Generator's apparent power in MVA

            **scaling** (float) - Scaling factor to active and reactive power. Default: 1

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **name** (string) - A customized name for the generator

            **type** (string) - Type of generator, to be used for applying generator scaling profiles

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created generator

        EXAMPLE:
            Define a single-phase generator at bus 2 between phase 1 and neutral \
            point 0 having a real power of 6 kW with unitary power factor.

            create_asymmetric_sgen(net, bus=2, phase_from=1, phase_to=0, p_mw=0.006, q_mvar=0)

    """
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)
    
    index, multi_index = _get_multi_index_with_check(net, "asymmetric_gen", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase', 'p_mw', 'vm_pu', 'sn_mva', 'scaling',
                        'in_service', 'type'],
                       [name, bus, from_phase, to_phase, p_mw, vm_pu, sn_mva, scaling, in_service, type]))

    net.asymmetric_gen = pd.concat([net.asymmetric_gen, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_ext_grid(net, bus, from_phase, to_phase, vm_pu, va_degree, r_ohm, x_ohm, name=None, in_service=True,
                    index=None):
    """
        Adds a voltage reference source at the slack bus. The model can treat generic number of phases with specific
        impedance defined for each one

        INPUT:
            **net** - The net within this line should be created

            **bus** (int) - Index of the equivalent generator's bus

            **from_phase** (int) - Phase at the sending side of phase connection. \
            Multiple parameters could be passed for multi-phase elements

            **to_phase** (int) - Phase at the receiving side of phase connection. \
            Multiple parameters could be passed for multi-phase elements

            **vm_pu** (float) - Voltage magnitude set point in per unit

            **va_degree** (float) - Voltage angle set point in degrees

            **r_ohm** (float) - Internal series impedance resistive part in ohm

            **x_ohm** (float) - Internal series impedance reactive part in ohm

        OPTIONAL:
            **name** (string) - A customized name for the voltage source

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created voltage source

        EXAMPLE:
            Define a three-phase external grid connection at bus 0, wye-connected at phases 1, 2, 3 and neutral point 0.
            The voltage magnitude is 1 pu with angle 0. Each phase has an internal impedance of 0.01+j0.1 Ohm

            create_ext_grid(net, bus=0, phase_from=range(1, 4), phase_to=0, vm_pu=1, va_degree=0, r_ohm=0.01, x_ohm=0.1)

    """
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)
        
    index, multi_index = _get_multi_index_with_check(net, "ext_grid", (index, circuits), ("index", "circuit"))

    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase', 'vm_pu', 'va_degree', 'r_ohm', 'x_ohm', 'in_service'],
                       [name, bus, from_phase, to_phase, vm_pu, va_degree, r_ohm, x_ohm, in_service]))

    net.ext_grid = pd.concat([net.ext_grid, pd.DataFrame(index=multi_index, data=entries)])
    return index


def create_ext_grid_sequence(net, bus, from_phase, to_phase, vm_pu, va_degree, sn_mva, rx, x0x, r0x0,
                             z2z1=1, c=1.1, z1_weight=SMALL_NUMBER, name=None, in_service=True, index=None):
    """
        Adds a voltage reference source at the slack bus with an internal impedance branch defined according to
        sequence components parameters

        INPUT:
            **net** - The net within this line should be created

            **bus** (int) - Index of the equivalent generator's bus

            **from_phase** (int) - Phase at the sending side of phase connection. \
            Multiple parameters could be passed for multi-phase elements

            **to_phase** (int) - Phase at the receiving side of phase connection. \
            Multiple parameters could be passed for multi-phase elements

            **vm_pu** (float) - Voltage magnitude set point in per unit

            **va_degree** (float) - Voltage angle set point in degrees

            **sn_mva** (float) - Rated apparent power in MVA

            **rx** (float) - R/X ratio of the internal impedance

            **x0x** (float) - X0/X ratio of the internal impedance

            **r0x0** (float) - R0/X0 ratio of the internal impedance

        OPTIONAL:

            **z2z1** (float, 1) - Ratio between negative and positive sequence impedance magnitudes

            **c** (float, 1.1) - Voltage correction factor for internal impedance calculation

            **z1_weight** (float, 1e-7) - Weight of positive sequence impedance in the internal impedance branch. \
            Default is an arbitrary small number, 1e-7, to model an ideal branch for positive sequence component in \
            load flow simulation. Editing the parameter, the user can enable it in the model.

            **name** (string) - A customized name for the voltage source

            **in_service** (boolean, True) - Toggles bus (or specific phase) usage in the calculation

            **index** (int, None) - Force a specified ID if it is available. \
            Uses multi-index, requiring to number both element and circuit(s). \
            If None, the index one higher than the highest already existing index is selected.

        OUTPUT:
            **index** (int) - The unique ID of the created generator

        EXAMPLE:
            Define a three-phase external grid connection at bus 0, wye-connected \
            at phases 1, 2, 3 and neutral point 0. The voltage magnitude is 1 pu \
            with angle 0, Sn=15 MVA, R/X=0.1, X0/X=1, R0/X0=0.1

            create_ext_grid_sequence(net, bus=0, phase_from=range(1, 4), phase_to=0, vm_pu=1, va_degree=0, sn_mva=15, rx=0.1, x0x=1, r0x0=0.1)

    """
    if len(from_phase) != 3:
        raise UserWarning("sequence model for ext_grid is only possible with 3-circuit element")

    if isinstance(vm_pu, numbers.Number):
        vm_pu = np.array([0, vm_pu, 0], dtype=np.float64)
    elif hasattr(vm_pu, '__iter__'):
        vm_pu = np.array(vm_pu, dtype=np.float64)

    if 'bus' not in net:
        raise UserWarning("No bus has been created in the network yet. Please add buses first.")

    if isinstance(va_degree, numbers.Number):
        va_degree = np.array([0, va_degree, 0], dtype=np.float64)
    elif hasattr(vm_pu, '__iter__'):
        va_degree = np.array(va_degree, dtype=np.float64)

    z1_ohm = c / sn_mva * np.square(net.bus.at[(bus, from_phase[0]), "vn_kv"])
    x1_ohm = z1_ohm / np.sqrt(rx ** 2 + 1)
    r1_ohm = rx * x1_ohm
    x0_ohm = x0x * x1_ohm
    r0_ohm = r0x0 * x0_ohm
    z2_ohm = z2z1 * c * np.square(net.bus.at[(bus, from_phase[0]), "vn_kv"]) / sn_mva
    x2_ohm = z2_ohm / np.sqrt(rx ** 2 + 1)
    r2_ohm = np.sqrt(z2_ohm ** 2 - x2_ohm ** 2)

    r_ohm = np.vstack([r0_ohm*z1_weight, r1_ohm*z1_weight, r2_ohm*z1_weight]).flatten().real
    x_ohm = np.vstack([x0_ohm*z1_weight, x1_ohm*z1_weight, x2_ohm*z1_weight]).flatten().real
    # r_ohm = np.vstack([r0_ohm, SMALL_NUMBER, r1_ohm]).flatten().real
    # x_ohm = np.vstack([x0_ohm, SMALL_NUMBER, x1_ohm]).flatten().real

    circuits = np.arange(len(from_phase), dtype=np.int64)
    index, multi_index = _get_multi_index_with_check(net, "ext_grid_sequence", (index, circuits), ("index", "sequence"))

    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase', 'vm_pu', 'va_degree', 'r_ohm', 'x_ohm', 'in_service'],
                       [name, bus, from_phase, to_phase, vm_pu, va_degree, r_ohm, x_ohm, in_service]))

    net.ext_grid_sequence = pd.concat([net.ext_grid_sequence, pd.DataFrame(index=multi_index, data=entries)])
    return index




def create_asymmetric_shunt(net, bus, from_phase, to_phase, q_mvar, p_mw, control_mode=None,
                            closed=False, v_threshold_on=0.9, v_threshold_off=1.1, 
                            vn_kv=None, name=None, in_service=True, index=None):
    """
    Creates a asymmetric shunt element.

    INPUT:
        **net** - The net within this shunt should be created

        **bus** (int) - index of bus where the impedance starts
        
        **from_phase** (int) - Phase at the sending end of phase connection

        **to_phase** (int) -  Phase at the receiving end of phase connection

        **p_mw** - shunt active power in MW at v= 1.0 p.u. per step

        **q_mvar** - shunt reactive power in in MVAr at v= 1.0 p.u. per step


    OPTIONAL:
        **name** (str, None) - name of the shunt
        
        **control_mode** (str, None) - definition of control strategy
        
        **closed** (boolean, False) - shunt position: False = open, True = closed
        
        **v_threshold_on** (float, None) - Voltage threshold for which the capacitor will be activated
        
        **v_threshold_off** (float, None) - Voltage threshold for which the capacitor will be tripped/disconnected
        
        **vn_kv** (float, None) - rated voltage of the shunt. Defaults to rated voltage of
            connected bus

        **in_service** (boolean, True) - True for in_service or False for out of service
        
        **index** (int, None) - Force a specified ID if it is available. \
        Uses multi-index, requiring to number both element and circuit(s). \
        If None, the index one higher than the highest already existing index is selected.
        

    OUTPUT:
        **index** (int) - The unique ID of the created shunt

    EXAMPLE:
        TO DO
    """
   
    # to be adapted:
    if "asymmetric_shunt" not in net or net.asymmetric_shunt is None:
        net["asymmetric_shunt"] = pd.DataFrame(
            columns=['name', 'bus', 'from_phase', 'to_phase', 'p_mw', 'q_mvar', 
                     'control_mode', 'closed', 'v_threshold_on', 'v_threshold_off',
                     'max_q_mvar','max_p_mw', 'vn_kv', 'in_service'])
        net.asymmetric_shunt.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["index", "circuit"]) 
   
    if isinstance(from_phase,(list,tuple,range)):
        circuits = np.arange(len(from_phase), dtype=np.int64)
    elif isinstance(from_phase, numbers.Number):
        circuits = np.arange(1, dtype=np.int64)

    index, multi_index = _get_multi_index_with_check(net, "asymmetric_shunt", (index, circuits), ("index", "circuit"))
    
    max_q_mvar, max_p_mw = q_mvar, p_mw
    entries = dict(zip(['name', 'bus', 'from_phase', 'to_phase','p_mw', 'q_mvar', 
                        'control_mode', 'closed', 'v_threshold_on', 'v_threshold_off',
                        'max_q_mvar', 'max_p_mw', 'vn_kv', 'in_service'],
                        [name, bus, from_phase, to_phase, p_mw, q_mvar, 
                         control_mode, closed, v_threshold_on, v_threshold_off,
                         max_q_mvar, max_p_mw, vn_kv, in_service]))

    net.asymmetric_shunt = pd.concat([net.asymmetric_shunt, pd.DataFrame(index=multi_index, data=entries)])

    return index


def create_measurement(net, measurement_type, element_type, element, value, std_dev=None, name=None, side=None, index=None):

    """
    Creates a new measurement entry in net.measurement.

    Parameters
    ----------
    net : pandapowerNet
        Network in which the measurement is created.
    measurement_type : str
        Type of measurement, e.g. "p", "q", "i", "s", "pf".
    element_type : str
        Type of measured element, e.g. "line", "trafo1ph", "ext_grid".
    element : int
        Index of the measured element (row in the respective element table).
    value : float
        Measurement value (MW for P, MVAr for Q, kA for I, MVA for S, % for PF).
    std_dev : float, optional
        Standard deviation of the measurement (for state estimation etc.).
    name : str, optional
        Optional name for the measurement.
    side : str, optional
        For branches/transformers, side of the measurement (e.g. "from", "to", "hv", "lv").
    index : int, optional
        Optional explicit index. If None, the next free index is used.

    Returns
    -------
    index : int
        Index of the created measurement.
    """

    if "measurement" not in net or net.measurement is None:
        net["measurement"] = pd.DataFrame(
            columns=["name", "measurement_type", "element_type",
                     "element", "value", "std_dev", "side"]
        )

    cols = ["name", "measurement_type", "element_type", "element", "value", "std_dev", "side"]
    vals = [name, measurement_type, element_type,
            int(element), float(value), float(std_dev), side]
    entry = dict(zip(cols, vals))

    net.measurement = pd.concat([net.measurement, pd.DataFrame(index=[index], data=entry)])

    return index



def _trafo_variable_entry(buses, circuits, variable):
    if isinstance(variable, numbers.Number) or len(variable) == len(circuits)*len(buses):
        return variable
    elif len(variable) == len(circuits):
        return np.tile(variable, len(buses))
    elif len(variable) == len(buses):
        # return np.array([np.repeat(v, len(circuits)) for v in variable]).flatten()
        return np.repeat(variable, len(circuits))
    else:
        UserWarning(f"Variable  for trafo1ph has lenghth of {len(variable)}, supported are: single number, "
                    f"number of buses, number of circuits")


def get_length(data):
    if isinstance(data, (list, tuple)):
        return len(data)
    else:
        return 1


def _get_multi_index_with_check(net, table, indices, names):
    index, *ind = indices
    index = _get_index_with_check(net, table, index)
    mi = pd.MultiIndex.from_product(((index,), *ind), names=names)
    return index, mi


def _check_branch_element(net, element_name, index, node, phases, 
                          to_node, to_phase, node_name="bus", plural="es"):
    combinations=it.product(node,set(tuple(phases)))
    
    missing_nodes = set(combinations) - set(net[node_name].index.values)
    if missing_nodes:
        raise UserWarning("%s %d tries to attach to %s(%s) %s non-existing or not having required phases."
                          % (element_name.capitalize(), index, node_name, plural, missing_nodes))


def tras_conn(conn, group):
    # ports connection for three phase transformers based on typical
    # connection/group schemes
    if conn == 'z':
        conn = 'y'
        warnings.warn('Connection Type "z" not implemented yet. Using type "y"')

        # Mapping configuration: For each connection type, group keys are tuples
        # representing all groups sharing the same (from_phase, to_phase) tuple.
    config = {
        'y': {
            (0, 11): ([1, 2, 3], [0, 0, 0]),
            (1, 2): ([0, 0, 0], [2, 3, 1]),
            (3, 4): ([3, 1, 2], [0, 0, 0]),
            (5, 6): ([0, 0, 0], [1, 2, 3]),
            (7, 8): ([2, 3, 1], [0, 0, 0]),
            (9, 10): ([0, 0, 0], [3, 1, 2])
        },
        'd': {
            (0, 1): ([1, 2, 3], [2, 3, 1]),
            (2, 3): ([3, 1, 2], [2, 3, 1]),
            (4,): ([3, 2, 1], [1, 2, 3]),
            (5,): ([3, 1, 2], [1, 2, 3]),
            (6, 7): ([2, 3, 1], [1, 2, 3]),
            (8, 9): ([2, 3, 1], [3, 1, 2]),
            (10, 11): ([1, 2, 3], [3, 1, 2])
        }
    }

    # Validate that the connection type exists in our configuration.
    if conn not in config:
        raise ValueError(f'Connection type "{conn}" not implemented.')

    # For the given connection, iterate over the mapping entries and
    # return the corresponding phases if the group is found.
    for groups, (from_phase, to_phase) in config[conn].items():
        if group in groups:
            return from_phase, to_phase
    raise ValueError(f'Group "{group}" is not defined for connection type "{conn}".')