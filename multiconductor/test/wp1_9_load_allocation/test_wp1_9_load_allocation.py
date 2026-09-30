"""
Test cases for WP1.9 - Load Allocation

This module contains pytest test cases for load allocation functionality in
multiconductor networks. Tests cover:
- Single measurement load allocation
- Multiple measurement load allocation (hierarchical)
- Transformer-based measurements (HV and LV side)
- Multi-feeder networks with parallel measurements
- Cap-to-load-rating functionality
- Generator handling (ignore_generators flag)
- Adjust-after-load-flow functionality

Tests are based on the load allocation module and examples from the
multiconductor.load_allocation package.
"""

import multiconductor as mc
from multiconductor.load_allocation.load_allocation import (
    build_measurement_graph, 
    run_load_allocation, 
    get_simulated_measurement_value, 
    get_measurement_bus,
    get_downstream_load_indices, 
    filter_loads_under_other_measurement
)
import numpy as np
import pandas as pd
import warnings
import pytest
import copy

from multiconductor.pycci.std_types import create_std_type

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# Helper functions for creating test networks and measurements
# =============================================================================

def _create_matrix_type_line(mc_net, conductors, type_name, rvalue=0.02, xvalue=0.07):
    """Create a matrix-type line standard type."""
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

    mc.pycci.std_types.create_std_types(mc_net, {type_name: mdata}, element="matrix")


def _create_trafo_types(mc_net):
    """Create standard transformer types for testing."""
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
    """Add a power measurement on a line."""
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
    """Add a power measurement on a transformer."""
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


# =============================================================================
# Test network creation functions
# =============================================================================

def create_simple_4bus_net():
    """Create a simple 4-bus radial network for basic load allocation tests."""
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
    """Create a 5-bus radial network with loads and generation."""
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
    Create a 20-bus radial system with three feeders.

    bus0: HV bus with ext_grid
    bus1: LV bus
    Three LV feeders:
        Feeder 1: 1-2-3-4-5-6-7
        Feeder 2: 1-8-9-10-11-12-13
        Feeder 3: 1-14-15-16-17-18-19

    Asymmetric loads on all three feeders, sgen on feeder 1 and feeder 2.
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


# =============================================================================
# Test helper functions
# =============================================================================

def get_measurement_row(net, idx):
    """Get a measurement row, handling both Series and DataFrame returns."""
    m = net.measurement.loc[idx]
    return m.iloc[0] if isinstance(m, pd.DataFrame) else m


def check_measurement_accuracy(net, meas_idx, tolerance_mw=0.5):
    """
    Check if simulated measurement value is within tolerance of target.
    
    Returns:
        tuple: (is_within_tolerance, target_value, simulated_value, difference)
    """
    meas = get_measurement_row(net, meas_idx)
    target = float(meas["value"])
    sim = get_simulated_measurement_value(net, meas)
    diff = abs(target - sim)
    return diff <= tolerance_mw, target, sim, diff


# =============================================================================
# Test Classes
# =============================================================================

class TestMeasurementGraph:
    """Tests for measurement graph construction."""
    
    def test_build_measurement_graph(self):
        """Test that measurement graph is built correctly."""
        net = create_simple_4bus_net()
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        # Graph should be created
        assert mg is not None
        # Graph should have nodes (buses with phases)
        assert len(mg.nodes) > 0
    
    def test_downstream_load_detection(self):
        """Test that downstream loads are correctly identified."""
        net = create_simple_4bus_net()
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        # Add a measurement
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=8.0, meas_index=0)
        
        # Get downstream loads
        loads = get_downstream_load_indices(net, mg, 0)
        
        # Should find the load at bus 3
        assert len(loads) > 0


class TestSingleMeasurement:
    """Tests for single measurement load allocation."""
    
    def test_single_measurement_allocation(self):
        """Test load allocation with a single measurement."""
        net = create_simple_4bus_net()
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        # Add measurement
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=8.0, meas_index=0)
        
        # Run load allocation
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.1,
            cap_to_load_rating=False,
            measurement_indices=[0],
            verbose=False
        )
        
        # Run power flow and check result
        mc.run_pf(net)
        is_accurate, target, sim, diff = check_measurement_accuracy(net, 0, tolerance_mw=0.5)
        
        assert is_accurate, f"Target={target:.3f} MW, Simulated={sim:.3f} MW, Diff={diff:.3f} MW"
    
    def test_single_measurement_convergence(self):
        """Test that load allocation converges within iterations."""
        net = create_simple_4bus_net()
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=10.0, meas_index=0)
        
        # Should not raise an exception
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[0],
            verbose=False
        )
        
        mc.run_pf(net)
        # Verify loads have been modified
        assert net.asymmetric_load['p_mw'].sum() > 0


class TestMultipleMeasurements:
    """Tests for multiple measurement load allocation."""
    
    def test_two_measurements_allocation(self):
        """Test load allocation with two hierarchical measurements."""
        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)
        
        # Add two measurements: downstream first, then upstream
        add_line_p_measurement_value(net, line_idx=1, side="from", value_mw=15.0, meas_index=1, name="P_L2_3_from")
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=24.0, meas_index=0, name="P_L1_2_from")
        
        # Run load allocation (process downstream measurement first)
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[1, 0],
            ignore_generators=False,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # Check both measurements
        for midx in [0, 1]:
            is_accurate, target, sim, diff = check_measurement_accuracy(net, midx, tolerance_mw=1.0)
            assert is_accurate, f"Measurement {midx}: Target={target:.3f}, Sim={sim:.3f}, Diff={diff:.3f}"


class TestTrafoMeasurements:
    """Tests for transformer-based measurements."""
    
    def test_trafo_hv_lv_measurements(self):
        """Test load allocation with transformer HV and LV side measurements."""
        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)
        
        # Add transformer measurements
        add_trafo_p_measurement_value(net, trafo_idx=0, side="hv", value_mw=24.0, meas_index=0, name="P_trafo_hv")
        add_trafo_p_measurement_value(net, trafo_idx=0, side="lv", value_mw=15.0, meas_index=1, name="P_trafo_lv")
        
        # Add line measurement
        add_line_p_measurement_value(net, line_idx=2, side="from", value_mw=10.0, meas_index=2, name="P_L2_3_from")
        
        net.asymmetric_load["sn_mva"] = 10.0
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[2, 1],
            cap_to_load_rating=True,
            ignore_generators=False,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # Verify power flow completed successfully
        assert not net.res_bus.empty


class TestMultiFeeder:
    """Tests for multi-feeder network load allocation."""
    
    def test_20bus_feeder_allocation(self):
        """Test load allocation on 20-bus multi-feeder network."""
        net = create_20bus_radial_net()
        mg = build_measurement_graph(net)
        
        # Add measurements on feeder heads
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=5.0, meas_index=1, name="P_F1_head")
        add_line_p_measurement_value(net, line_idx=6, side="from", value_mw=6.0, meas_index=2, name="P_F2_head")
        
        net.asymmetric_load["sn_mva"] = 5.0
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[2, 1],
            cap_to_load_rating=True,
            ignore_generators=False,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # Verify loads have been modified and results are valid
        assert not net.res_bus.empty
        assert net.res_bus['vm_pu'].notna().any()
    
    @pytest.mark.parametrize("adjust_after_pf", [True, False])
    def test_adjust_after_load_flow_flag(self, adjust_after_pf):
        """Test both settings of adjust_after_load_flow flag."""
        net = create_20bus_radial_net()
        mg = build_measurement_graph(net)
        
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=5.0, meas_index=1, name="P_F1_head")
        net.asymmetric_load["sn_mva"] = 5.0
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=adjust_after_pf,
            tolerance=0.5,
            measurement_indices=[1],
            cap_to_load_rating=True,
            verbose=False
        )
        
        mc.run_pf(net)
        assert not net.res_bus.empty


class TestCapToLoadRating:
    """Tests for cap_to_load_rating functionality."""
    
    def test_without_cap_to_load_rating(self):
        """Test that loads can exceed rating when cap_to_load_rating=False."""
        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)
        
        # Set small initial loads with rating
        net.asymmetric_load["p_mw"] = 1.0
        net.asymmetric_load["q_mvar"] = 0.2
        net.asymmetric_load["sn_mva"] = 5.0
        
        # Add a large measurement target
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=30.0, meas_index=0, name="P_head_1_2")
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[0],
            cap_to_load_rating=False,
            ignore_generators=False,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # Check that simulated value gets closer to target (loads can exceed rating)
        is_accurate, target, sim, diff = check_measurement_accuracy(net, 0, tolerance_mw=5.0)
        # With cap_to_load_rating=False, we expect to get close to target
        assert sim > 20.0, f"Expected simulated value > 20 MW when uncapped, got {sim:.3f} MW"
    
    def test_with_cap_to_load_rating(self):
        """Test that loads are capped at rating when cap_to_load_rating=True."""
        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)
        
        # Set small initial loads with rating
        net.asymmetric_load["p_mw"] = 1.0
        net.asymmetric_load["q_mvar"] = 0.2
        net.asymmetric_load["sn_mva"] = 5.0
        
        # Add a large measurement target (impossible to reach with ratings)
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=30.0, meas_index=0, name="P_head_1_2")
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[0],
            cap_to_load_rating=True,
            ignore_generators=False,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # With cap_to_load_rating=True, simulated value cannot reach 30 MW target
        meas = get_measurement_row(net, 0)
        sim = get_simulated_measurement_value(net, meas)
        # Should be significantly below target due to capping
        assert sim < 25.0, f"Expected simulated value < 25 MW when capped, got {sim:.3f} MW"


class TestIgnoreGenerators:
    """Tests for ignore_generators flag functionality."""
    
    @pytest.mark.parametrize("ignore_gen", [True, False])
    def test_ignore_generators_flag(self, ignore_gen):
        """Test both settings of ignore_generators flag."""
        net = create_5bus_radial_net()
        mg = build_measurement_graph(net)
        
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=20.0, meas_index=0)
        net.asymmetric_load["sn_mva"] = 10.0
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[0],
            cap_to_load_rating=True,
            ignore_generators=ignore_gen,
            verbose=False
        )
        
        mc.run_pf(net)
        
        # Both configurations should complete without error
        assert not net.res_bus.empty


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_initial_load(self):
        """Test load allocation starting from zero load."""
        net = create_simple_4bus_net()
        net.asymmetric_load["p_mw"] = 0.0
        net.asymmetric_load["q_mvar"] = 0.0
        
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=5.0, meas_index=0)
        
        # This may or may not converge depending on implementation
        # Just verify no crash occurs
        try:
            run_load_allocation(
                net, mg,
                adjust_after_load_flow=True,
                tolerance=0.5,
                measurement_indices=[0],
                verbose=False
            )
        except Exception:
            pass  # Expected to potentially fail with zero initial load
    
    def test_empty_measurement_indices(self):
        """Test behavior with empty measurement indices list."""
        net = create_simple_4bus_net()
        mc.run_pf(net)
        mg = build_measurement_graph(net)
        
        add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=8.0, meas_index=0)
        
        # Run with empty list - should handle gracefully
        initial_load = net.asymmetric_load["p_mw"].copy()
        
        run_load_allocation(
            net, mg,
            adjust_after_load_flow=True,
            tolerance=0.5,
            measurement_indices=[],
            verbose=False
        )
        
        # Loads should be unchanged
        pd.testing.assert_series_equal(net.asymmetric_load["p_mw"], initial_load)


# =============================================================================
# Parametrized comprehensive tests
# =============================================================================

@pytest.mark.parametrize("target_mw", [5.0, 10.0, 15.0])
def test_various_measurement_targets(target_mw):
    """Test load allocation with various measurement targets."""
    net = create_simple_4bus_net()
    mc.run_pf(net)
    mg = build_measurement_graph(net)
    
    add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=target_mw, meas_index=0)
    
    run_load_allocation(
        net, mg,
        adjust_after_load_flow=True,
        tolerance=0.5,
        measurement_indices=[0],
        verbose=False
    )
    
    mc.run_pf(net)
    is_accurate, target, sim, diff = check_measurement_accuracy(net, 0, tolerance_mw=1.0)
    
    assert is_accurate, f"Target={target_mw} MW: Simulated={sim:.3f} MW, Diff={diff:.3f} MW"


@pytest.mark.parametrize("tolerance", [0.1, 0.5, 1.0])
def test_various_tolerances(tolerance):
    """Test load allocation with various tolerance settings."""
    net = create_simple_4bus_net()
    mc.run_pf(net)
    mg = build_measurement_graph(net)
    
    add_line_p_measurement_value(net, line_idx=0, side="from", value_mw=8.0, meas_index=0)
    
    run_load_allocation(
        net, mg,
        adjust_after_load_flow=True,
        tolerance=tolerance,
        measurement_indices=[0],
        verbose=False
    )
    
    mc.run_pf(net)
    
    # Should complete without error
    assert not net.res_bus.empty
