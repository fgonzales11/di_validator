import multiconductor as mc
from multiconductor.load_allocation.load_allocation import (build_measurement_graph, run_load_allocation, get_simulated_measurement_value, get_measurement_bus,
                                                            get_downstream_load_indices, filter_loads_under_other_measurement)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from multiconductor.pycci.std_types import create_std_type

def _create_matrix_type_line(mc_net, conductors, type_name, rvalue=0.02, xvalue=0.07):
    rmatrix = np.zeros(shape=(conductors, conductors))
    xmatrix = np.zeros(shape=(conductors, conductors))
    for i in range(0, conductors):
        rmatrix[i, i] = rvalue
        xmatrix[i, i] = xvalue

    mdata = {"r_1_ohm_per_km": rmatrix[:, 0],
             "x_1_ohm_per_km": xmatrix[:, 0],
             "g_1_us_per_km": np.zeros(conductors),
             "b_1_us_per_km": np.zeros(conductors),
             "max_i_ka": np.ones(conductors)
             }

    if conductors >= 2:
        mdata.update({"r_2_ohm_per_km": rmatrix[:, 1],
                      "x_2_ohm_per_km": xmatrix[:, 1],
                      "g_2_us_per_km": np.zeros(conductors),
                      "b_2_us_per_km": np.zeros(conductors)})
    if conductors >= 3:
        mdata.update({"r_3_ohm_per_km": rmatrix[:, 2],
                      "x_3_ohm_per_km": xmatrix[:, 2],
                      "g_3_us_per_km": np.zeros(conductors),
                      "b_3_us_per_km": np.zeros(conductors)})
    if conductors == 4:
        mdata.update({"r_4_ohm_per_km": rmatrix[:, 3],
                      "x_4_ohm_per_km": xmatrix[:, 3],
                      "g_4_us_per_km": np.zeros(conductors),
                      "b_4_us_per_km": np.zeros(conductors)})
    if conductors > 4:
        print("_create_matrix_type_line: Error: doesn't support more than 4 conductors")

    mc.pycci.std_types.create_std_types(mc_net, {type_name: mdata}, element="matrix")

def _create_trafo_types(mc_net):
    create_std_type(mc_net, {"sn_mva": 50,
                "vn_hv_kv": 115,
                "vn_lv_kv": 16,
                "vk_percent": .001,
                "vkr_percent": 0,
                "pfe_kw": 0,
                "i0_percent": .001,
                "shift_degree": 0,
                "vector_group": "Yyn0",
                "tap_side": "lv",
                "tap_neutral": 0,
                "tap_min": -16,
                "tap_max": 16,
                "tap_step_degree": 0,
                "tap_step_percent": 0.625,
                "tap_changer_type": "Ratio"}, name="50MVA_Yy_LDC", element="trafo")
    create_std_type(mc_net, {"sn_mva": 10,
                "vn_hv_kv": 115,
                "vn_lv_kv": 16,
                "vk_percent": 6,
                "vkr_percent": 0,
                "pfe_kw": 0,
                "i0_percent": 4,
                "shift_degree": 0,
                "vector_group": "Yyn0",
                "tap_side": "lv",
                "tap_neutral": 0,
                "tap_min": -16,
                "tap_max": 16,
                "tap_step_degree": 0,
                "tap_step_percent": 0.625,
                "tap_changer_type": "Ratio"}, name="10MVA_Yy_LDC", element="trafo")
    create_std_type(mc_net, {"sn_mva": 1,
                "vn_hv_kv": 16,
                "vn_lv_kv": 0.4,
                "vk_percent": 8,
                "vkr_percent": 0,
                "pfe_kw": 0,
                "i0_percent": 5,
                "shift_degree": 0,
                "vector_group": "Yyn0",
                "tap_side": "lv",
                "tap_neutral": 0,
                "tap_min": -16,
                "tap_max": 16,
                "tap_step_degree": 0,
                "tap_step_percent": 0.625,
                "tap_changer_type": "Ratio"}, name="1MVA_HVLV_Yy_LDC", element="trafo")

def add_line_p_measurement_value(net, line_idx, value_mw, side="from",
                                 meas_index=None, std_dev=0.01, name=None):


    mc.create_measurement(
        net,
        measurement_type="p",
        element_type="line",
        element=line_idx,
        value=float(value_mw),
        std_dev=std_dev,
        side=side,
        name=name,
        index=meas_index
    )


def add_trafo_p_measurement_value(net, trafo_idx, side="hv",
                                  value_mw=0.0, meas_index=None,
                                  std_dev=0.01, name=None):

    mc.create_measurement(net,
        measurement_type="p",
        element_type="trafo1ph",
        element=trafo_idx,
        value=float(value_mw),
        std_dev=std_dev,
        side=side,
        index=meas_index,
        name=name
    )

def get_measurement_nodes(net, mg):
    """
    Return a list of (bus, phase) nodes where measurements are located.
    """
    meas_nodes = []

    for idx in net.measurement.index:
        meas = net.measurement.loc[idx]
        if isinstance(meas, pd.DataFrame):
            meas = meas.iloc[0]

        bus = get_measurement_bus(net, meas)

        # all phase-nodes at that bus
        bus_phase_nodes = [n for n in mg.nodes if n[0] == bus]
        meas_nodes.extend(bus_phase_nodes)

    return meas_nodes


def create_net():
    net = mc.create_empty_network()
    bus0 = mc.create_bus(net, num_phases=4, vn_kv=115, name="bus0")
    bus1 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus1")
    bus2 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus2")
    bus3 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus3")

    grounding_r_ohm = 1E-7
    net.bus.at[(0, 0), 'grounded'] = True
    net.bus.at[(0, 0), 'grounding_r_ohm'] = grounding_r_ohm
    net.bus.at[(1, 0), 'grounded'] = True
    net.bus.at[(1, 0), 'grounding_r_ohm'] = grounding_r_ohm
    net.bus.at[(2, 0), 'grounded'] = False
    net.bus.at[(2, 0), 'grounding_r_ohm'] = grounding_r_ohm
    net.bus.at[(3, 0), 'grounded'] = False
    net.bus.at[(3, 0), 'grounding_r_ohm'] = grounding_r_ohm
    mc.create_ext_grid_sequence(
        net, bus=bus0, from_phase=(1, 2, 3), to_phase=0, vm_pu=1, va_degree=0,
        sn_mva=50, rx=0, x0x=1, r0x0=0, name="extgrid")

    _create_matrix_type_line(net, conductors=3, type_name="mat_line_3cond")
    _create_matrix_type_line(net, conductors=4, type_name="mat_line_4cond")
    _create_trafo_types(net)

    mc.create_transformer_3ph(net, hv_bus=bus0, lv_bus=bus1, std_type="10MVA_Yy_LDC")
    mc.create_line(net, model_type="matrix", std_type="mat_line_4cond", from_bus=bus1, from_phase=(0, 1, 2, 3),
                   to_bus=bus2, to_phase=(0, 1, 2, 3), length_km=12, name="Line1_2")
    mc.create_line(net, model_type="matrix", std_type="mat_line_4cond", from_bus=bus2, from_phase=(0, 1, 2, 3),
                   to_bus=bus3, to_phase=(0, 1, 2, 3), length_km=12, name="Line2_3")
    mc.create_asymmetric_load(net, bus3, from_phase=(1, 2, 3), to_phase=0, p_mw=6, q_mvar=1, name="Load1")

    return net


def create_5bus_radial_net():
    net = mc.create_empty_network()

    bus0 = mc.create_bus(net, num_phases=4, vn_kv=115, name="bus0")
    bus1 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus1")
    bus2 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus2")
    bus3 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus3")
    bus4 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus4")

    mc.create_ext_grid_sequence(
        net, bus=bus0, from_phase=(1, 2, 3), to_phase=0,
        vm_pu=1.0, va_degree=0,
        sn_mva=50, rx=0, x0x=1, r0x0=0,
        name="extgrid",
    )

    _create_matrix_type_line(net, conductors=4, type_name="mat_line_4cond")
    _create_trafo_types(net)

    mc.create_transformer_3ph(net, hv_bus=bus0, lv_bus=bus1, std_type="10MVA_Yy_LDC")

    mc.create_line(net, model_type="matrix", std_type="mat_line_4cond",
                   from_bus=bus1, from_phase=(0, 1, 2, 3),
                   to_bus=bus2, to_phase=(0, 1, 2, 3),
                   length_km=1.0, name="L1_2")

    mc.create_line(net, model_type="matrix", std_type="mat_line_4cond",
                   from_bus=bus2, from_phase=(0, 1, 2, 3),
                   to_bus=bus3, to_phase=(0, 1, 2, 3),
                   length_km=1.0, name="L2_3")

    mc.create_line(net, model_type="matrix", std_type="mat_line_4cond",
                   from_bus=bus2, from_phase=(0, 1, 2, 3),
                   to_bus=bus4, to_phase=(0, 1, 2, 3),
                   length_km=1.0, name="L2_4")

    mc.create_asymmetric_load(net, bus2, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=6.0, q_mvar=1.0, name="Load_bus2")

    mc.create_asymmetric_load(net, bus3, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=4.0, q_mvar=0.8, name="Load_bus3")

    mc.create_asymmetric_sgen(net, bus=bus4, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=(2.0, 2.0, 2.0), q_mvar=(0.0, 0.0, 0.0), name="sgen_bus4")

    return net


def create_20bus_radial_net():
    """
    20-bus radial system:

      bus0: HV bus with ext_grid
      bus1: LV bus
      Three LV feeders:

        Feeder 1: 1-2-3-4-5-6-7
        Feeder 2: 1-8-9-10-11-12-13
        Feeder 3: 1-14-15-16-17-18-19

      Asymmetric loads on all three feeders,
      sgen on feeder 1 and feeder 2

    """

    net = mc.create_empty_network()

    bus0 = mc.create_bus(net, num_phases=4, vn_kv=115, name="bus0_HV")
    bus1 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus1_LV")

    bus2 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus2_F1")
    bus3 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus3_F1")
    bus4 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus4_F1")
    bus5 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus5_F1")
    bus6 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus6_F1")
    bus7 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus7_F1_end")

    bus8 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus8_F2")
    bus9 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus9_F2")
    bus10 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus10_F2")
    bus11 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus11_F2")
    bus12 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus12_F2")
    bus13 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus13_F2_end")

    bus14 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus14_F3")
    bus15 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus15_F3")
    bus16 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus16_F3")
    bus17 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus17_F3")
    bus18 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus18_F3")
    bus19 = mc.create_bus(net, num_phases=4, vn_kv=16, name="bus19_F3_end")

    mc.create_ext_grid_sequence(
        net, bus=bus0, from_phase=(1, 2, 3), to_phase=0,
        vm_pu=1.0, va_degree=0.0,
        sn_mva=50.0, rx=0.0, x0x=1.0, r0x0=0.0,
        name="ext_grid"
    )

    _create_matrix_type_line(net, conductors=4, type_name="mat_line_4cond")
    _create_trafo_types(net)

    mc.create_transformer_3ph(net, hv_bus=bus0, lv_bus=bus1, std_type="10MVA_Yy_LDC")

    # line creation
    def create_feeder_lines(buses, name):
        for i in range(len(buses) - 1):
            mc.create_line(
                net, model_type="matrix", std_type="mat_line_4cond",
                from_bus=buses[i], from_phase=(0, 1, 2, 3),
                to_bus=buses[i + 1], to_phase=(0, 1, 2, 3),
                length_km=1.0,
                name=f"{name}_{i}"
            )

    # feeder 1: 1-2-3-4-5-6-7
    create_feeder_lines([bus1, bus2, bus3, bus4, bus5, bus6, bus7], "F1")

    # feeder 2: 1-8-9-10-11-12-13
    create_feeder_lines([bus1, bus8, bus9, bus10, bus11, bus12, bus13], "F2")

    # feeder 3: 1-14-15-16-17-18-19
    create_feeder_lines([bus1, bus14, bus15, bus16, bus17, bus18, bus19], "F3")

    # Feeder 1 loads
    mc.create_asymmetric_load(net, bus3, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=3.0, q_mvar=0.6, name="Load_F1_bus3")
    mc.create_asymmetric_load(net, bus5, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=2.0, q_mvar=0.4, name="Load_F1_bus5")
    mc.create_asymmetric_load(net, bus7, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=1.0, q_mvar=0.2, name="Load_F1_bus7")

    # Feeder 2 loads
    mc.create_asymmetric_load(net, bus9, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=2.0, q_mvar=0.4, name="Load_F2_bus9")
    mc.create_asymmetric_load(net, bus11, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=3.0, q_mvar=0.6, name="Load_F2_bus11")
    mc.create_asymmetric_load(net, bus13, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=1.0, q_mvar=0.2, name="Load_F2_bus13")

    # Feeder 3 loads
    mc.create_asymmetric_load(net, bus15, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=2.0, q_mvar=0.4, name="Load_F3_bus15")
    mc.create_asymmetric_load(net, bus17, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=2.0, q_mvar=0.4, name="Load_F3_bus17")
    mc.create_asymmetric_load(net, bus19, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=1.0, q_mvar=0.2, name="Load_F3_bus19")

    # sgen on two feeders
    mc.create_asymmetric_sgen(net, bus=bus7, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=(0.7, 0.7, 0.7), q_mvar=(0.0, 0.0, 0.0),
                              name="PV_F1_bus7")

    mc.create_asymmetric_sgen(net, bus=bus11, from_phase=(1, 2, 3), to_phase=0,
                              p_mw=(1.0, 1.0, 1.0), q_mvar=(0.0, 0.0, 0.0),
                              name="PV_F2_bus11")

    return net


def test_single_measurement():
    print("\nTest 1: single measurement")
    net = create_net()

    mc.run_pf(net)
    mg = build_measurement_graph(net)
    add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=8.0, meas_index=0)

    midx = 0
    loads = get_downstream_load_indices(net, mg, midx)
    loads = filter_loads_under_other_measurement(net, mg, loads, midx)
    print(f"measurement {midx} downstream loads: {loads}")

    run_load_allocation(
        net,
        mg,
        adjust_after_load_flow=True,
        tolerance=0.0001,
        cap_to_load_rating=False,
        measurement_indices=[0],
        verbose=True)

    print("\n asymmetric load results:")
    print(net.asymmetric_load[["p_mw", "q_mvar"]])

    mc.run_pf(net)
    meas = net.measurement.loc[0]
    if isinstance(meas, pd.DataFrame):
        meas = meas.iloc[0]

    target = float(meas["value"])
    sim = get_simulated_measurement_value(net, meas)
    print(f"\nTarget P = {target:.4f} MW, simulated P = {sim:.4f} MW")


def test_5bus_two_meas():
    net = create_5bus_radial_net()
    mg = build_measurement_graph(net)

    add_line_p_measurement_value(net, line_idx=1, side="from", value_mw=15.0, meas_index=1, name="P_L2_3_from")

    add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=24.0, meas_index=0, name="P_L1_2_from")
    measurement_nodes = get_measurement_nodes(net, mg)

    for midx in [1, 0]:
        loads = get_downstream_load_indices(net, mg, midx)
        loads = filter_loads_under_other_measurement(net, mg, loads, midx)
        print(f"measurement {midx} downstream loads:", loads)

    run_load_allocation(net, mg, adjust_after_load_flow=True, tolerance=0.5, measurement_indices=[1, 0], ignore_generators=False, verbose=True)

    mc.run_pf(net)
    for midx in [0, 1]:
        meas = net.measurement.loc[midx]
        if isinstance(meas, pd.DataFrame):
            meas = meas.iloc[0]
        sim = get_simulated_measurement_value(net, meas)
        print(f"  meas {midx}: {meas.name}  target={float(meas.value):6.2f} MW "f"sim={sim:6.2f} MW")

def test_5bus_trafo_measurement():
    """
    5-bus radial:
      - Trafo HV P measurement = 24 MW (idx 0)
      - Trafo LV P measurement = 15 MW (idx 1)
      - Line L2-3 from-side P measurement = 15 MW (idx 2)

    Runs load allocation for ignore_generators=False/True.
    """

    def add_meas_hv_lv(net, hv_val, lv_val):
        add_trafo_p_measurement_value(
            net, trafo_idx=0, side="hv",
            value_mw=hv_val, meas_index=0, name="P_trafo_hv"
        )
        add_trafo_p_measurement_value(
            net, trafo_idx=0, side="lv",
            value_mw=lv_val, meas_index=1, name="P_trafo_lv"
        )

    def get_meas(net, idx):
        m = net.measurement.loc[idx]
        return m.iloc[0] if isinstance(m, pd.DataFrame) else m

    for ignore_gen in (False, True):
        print("\n" + "=" * 70)
        print(f"=== 5-bus test: trafo HV+LV + line, ignore_generators={ignore_gen} ===")

        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)

        add_meas_hv_lv(net, hv_val=24.0, lv_val=15.0)

        add_line_p_measurement_value(
            net, line_idx=2, side="from",
            value_mw=10.0, meas_index=2, name="P_L2_3_from"
        )

        net.asymmetric_load["sn_mva"] = 10.0

        lv_downstream_loads = get_downstream_load_indices(net, mg, 1)
        print("LV trafo measurement downstream loads:", lv_downstream_loads)

        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.05,
            measurement_indices=[2, 1],
            cap_to_load_rating=True,
            ignore_generators=ignore_gen,
            verbose=True,
        )

        mc.run_pf(net)

        print("\nFinal comparison:")
        for midx in sorted(net.measurement.index):
            meas = get_meas(net, midx)
            sim = get_simulated_measurement_value(net, meas)
            print(
                f"  meas {midx}: type={meas.element_type:10}, side={meas.side:3}  "
                f"target={float(meas.value):7.3f} MW,  sim={sim:7.3f} MW"
            )


def test_20bus_load_allocation_extgrid_and_feeders():

    """
    20-bus net:
      - Feeder 1 measurement: line 0, from side, target 5 MW   (index 1)
      - Feeder 2 measurement: line 6, from side, target 6 MW   (index 2)
      - External grid measurement (index 0) is only read, not considered in load allocation
    2 cases:
      - adjust_after_load_flow = False / True
      - ignore_generators      = False / True
    """

    def add_measurements(net):
        add_line_p_measurement_value(
            net, line_idx=0, side="from",
            value_mw=5.0, meas_index=1, name="P_F1_head"
        )
        add_line_p_measurement_value(
            net, line_idx=6, side="from",
            value_mw=6.0, meas_index=2, name="P_F2_head"
        )

    def assign_load_ratings(net, sn_per_load=5.0):
        if "sn_mva" not in net.asymmetric_load.columns:
            net.asymmetric_load["sn_mva"] = np.nan
        net.asymmetric_load["sn_mva"] = sn_per_load

    def get_measurement_row(net, idx):
        m = net.measurement.loc[idx]
        return m.iloc[0] if isinstance(m, pd.DataFrame) else m

    for adjust_after_pf in (False, True):
        for ignore_gen in (False, True):
            print("\n" + "=" * 90)
            print(f"=== 20-bus test: adjust_after_load_flow={adjust_after_pf}, "
                  f"ignore_generators={ignore_gen} ===")

            net = create_20bus_radial_net()
            mg = build_measurement_graph(net)

            add_measurements(net)
            assign_load_ratings(net, sn_per_load=5.0)

            meas_order = [2, 1]

            run_load_allocation(
                net, mg,
                adjust_after_load_flow=adjust_after_pf,
                tolerance=0.5,
                measurement_indices=meas_order,
                cap_to_load_rating=True,
                ignore_generators=ignore_gen,
                verbose=True,
            )

            mc.run_pf(net)

            print("\nFinal comparison:")
            for midx in sorted(net.measurement.index):
                meas = get_measurement_row(net, midx)
                sim = get_simulated_measurement_value(net, meas)
                print(
                    f"  meas {midx}: type={meas.element_type}, side={meas.side} "
                    f"target={float(meas.value):7.3f} MW,  sim={sim:7.3f} MW"
                )

            print("\nFeeder head flows (res_line.p_from_mw):")
            try:
                print(f"  F1 head (line 0):   {float(net.res_line.loc[(0, 0), 'p_from_mw']):7.3f} MW")
                print(f"  F2 head (line 6):   {float(net.res_line.loc[(6, 0), 'p_from_mw']):7.3f} MW")
                print(f"  F3 head (line 12):  {float(net.res_line.loc[(12, 0), 'p_from_mw']):7.3f} MW")
            except Exception as e:
                print("[WARN] could not read some line flows:", e)

            print("=" * 90 + "\n")

# TODO: assert statements needed for testing
# test procedure: do load allocation once -> tweak loads -> run load allocation again whether it is getting back to the same p and q values +- tolerance
# maybe have a test for a simple grid that checks if load P&Q + P&Q losses match the measurement values
# TODO: add tests for I measurement

def test_cap_to_load_rating_effect():
    """
    Demonstrates the effect of cap_to_load_rating on a 5-bus radial network.
      - create five bus radial net (loads on bus2 & bus3).
      - Set each load sn_mva = 5 MVA (rating) and small initial P/Q.
      - Add a single P-measurement at the line (bus1 to  bus2) of 30 MW

    Expected behavior:
      - Without cap_to_load_rating: loads are scaled above 5 MW to match 30 MW.
      - With cap_to_load_rating: each load is clamped at around 5 MW, so the simulated value cannot reach 30 MW.

    """

    def run_case(cap_to_load_rating_flag):
        print("\n" + "=" * 80)
        print(f"cap_to_load_rating = {cap_to_load_rating_flag}")

        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)

        net.asymmetric_load["p_mw"] = 1.0
        net.asymmetric_load["q_mvar"] = 0.2
        net.asymmetric_load["sn_mva"] = 5.0

        add_line_p_measurement_value(
            net,
            line_idx=0,
            side="from",
            value_mw=30.0,
            meas_index=0,
            name="P_head_1_2"
        )

        run_load_allocation(
            net,
            mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[0],
            cap_to_load_rating=cap_to_load_rating_flag,
            ignore_generators=False,
            verbose=True
        )

        mc.run_pf(net)

        print("\nFinal asymmetric loads (per phase):")
        print(net.asymmetric_load[["p_mw", "q_mvar", "sn_mva"]])

        meas = net.measurement.loc[0]
        if isinstance(meas, pd.DataFrame):
            meas = meas.iloc[0]
        target = float(meas.value)
        sim = get_simulated_measurement_value(net, meas)
        print(f"\nHead measurement P target = {target:.3f} MW, simulated = {sim:.3f} MW")

    run_case(cap_to_load_rating_flag=False)
    run_case(cap_to_load_rating_flag=True)









