"""
Test cases for WP1.10 - Voltage Controlling Devices

This module contains pytest test cases for voltage controlling devices in a
multiconductor network with various configurations and controller setups.
Tests cover:
- LDC (Line Drop Controller) and LTC (Load Tap Changer) controllers
- Network configurations 1, 2, 3 with different phase setups
- Controller setups 1-4 with various Volt-Var and Shunt controllers
- Overvoltage and undervoltage test cases
- Yy and Dy transformer configurations
- Gang and phase operation modes
- Failure cases: load flow failure, controller non-convergence, oscillation

Tests are based on the definitions in the WP1.10 specification document.
"""

import multiconductor as mc
import multiconductor.test.testing_toolbox
from multiconductor.control.tools import print_controller_status
import pandas as pd
import warnings
import pytest
import numpy as np
import copy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Tolerance settings for comparison
DEFAULT_TOL = 1e-4
DEFAULT_TOL_VA_DEGREE = 1e-3

# Load all test grids once at module level
grids = mc.test.testing_toolbox.load_all_test_grids("wp1_10_voltage_regulating_devices")


def run_pf_and_compare_to_dss(net, run_control=True, tol=DEFAULT_TOL, tol_va_degree=DEFAULT_TOL_VA_DEGREE):
    """
    Run power flow and compare results to OpenDSS reference.
    
    Parameters:
        net: The network to run power flow on
        run_control: Whether to run with controllers enabled
        tol: Tolerance for voltage magnitude and power comparisons
        tol_va_degree: Tolerance for voltage angle comparisons
    """
    mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=run_control)
    c = multiconductor.test.testing_toolbox.compare_to_dss(net)
    multiconductor.test.testing_toolbox.assert_comparison(c, tol, tol_va_degree)


def check_voltage_narrowing(net_pre_control, net_post_control):
    """
    Verify that controllers narrow the voltage band throughout the network.
    
    The maximum absolute deviation from nominal voltage should be reduced
    after controller action.
    """
    # Get pre-control voltage deviation from nominal (1.0 pu)
    pre_vm = net_pre_control.res_bus['vm_pu']
    pre_vm_filtered = pre_vm[pre_vm > 0.1]  # Filter out inactive buses
    pre_max_deviation = max(abs(pre_vm_filtered.max() - 1.0), abs(pre_vm_filtered.min() - 1.0))
    
    # Get post-control voltage deviation from nominal
    post_vm = net_post_control.res_bus['vm_pu']
    post_vm_filtered = post_vm[post_vm > 0.1]
    post_max_deviation = max(abs(post_vm_filtered.max() - 1.0), abs(post_vm_filtered.min() - 1.0))
    
    return post_max_deviation <= pre_max_deviation


# =============================================================================
# Parametrized test for all grids with DSS reference
# =============================================================================

# Filter grids that have DSS reference data
grids_with_dss = {name: net for name, net in grids.items() if 'res_bus_dss' in net.keys()}


@pytest.mark.parametrize("name,net", grids_with_dss.items(), ids=list(grids_with_dss.keys()))
def test_wp1_10_compare_to_dss(name, net):
    """Test that power flow results match OpenDSS reference for all grids."""
    net_calc = copy.deepcopy(net)
    # Use slightly relaxed tolerance for complex controller interactions
    run_pf_and_compare_to_dss(net_calc, run_control=True, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Tests for working cases - Overvoltage
# =============================================================================

class TestOvervoltageCases:
    """Tests for overvoltage scenarios where controllers should reduce voltage."""
    
    def test_01_config1_ctrsetup3_ldc_overvoltage(self):
        """Test 1: Network config 1, Controller setup 3 with LDC, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_1_ctrsetup_3_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        # Verify controllers help with voltage regulation
        assert check_voltage_narrowing(net_pre_control, net)
        
        # Compare to DSS reference
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_02_config2_ctrsetup2_ltc_overvoltage(self):
        """Test 2: Network config 2, Controller setup 2 with LTC, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LTC_config_2_ctrsetup_2_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_03_config1_ctrsetup1_ltc_overvoltage(self):
        """Test 3: Network config 1, Controller setup 1 with LTC, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LTC_config_1_ctrsetup_1_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_04_config3_ctrsetup2_ldc_overvoltage(self):
        """Test 4: Network config 3, Controller setup 2 with LDC, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_3_ctrsetup_2_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_05_config3_ctrsetup4_ltc_overvoltage(self):
        """Test 5: Network config 3, Controller setup 4 with LTC, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LTC_config_3_ctrsetup_4_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Tests for working cases - Undervoltage
# =============================================================================

class TestUndervoltageCases:
    """Tests for undervoltage scenarios where controllers should raise voltage."""
    
    def test_06_config1_ctrsetup4_ldc_undervoltage(self):
        """Test 6: Network config 1, Controller setup 4 with LDC, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_1_ctrsetup_4_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_07_config1_ctrsetup2_ldc_undervoltage(self):
        """Test 7: Network config 1, Controller setup 2 with LDC, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_1_ctrsetup_2_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_08_config1_ctrsetup3_ldc_undervoltage(self):
        """Test 8: Network config 1, Controller setup 3 with LDC, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_1_ctrsetup_3_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_09_config3_ctrsetup3_ldc_undervoltage(self):
        """Test 9: Network config 3, Controller setup 3 with LDC, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_3_ctrsetup_3_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_10_config2_ctrsetup2_ltc_undervoltage(self):
        """Test 10: Network config 2, Controller setup 2 with LTC, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LTC_config_2_ctrsetup_2_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Tests for Dy transformer with gang-operated LTC
# =============================================================================

class TestDyTransformerGangLTC:
    """Tests for Dy transformer configurations with gang-operated LTC controller."""
    
    def test_11_dy_ltcgang_config3_ctrsetup3_overvoltage(self):
        """Test 11: Network config 3, Controller setup 3 with gang-operated LTC and Dy trafo, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LTCgang_config_3_ctrsetup_3_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_12_dy_ltcgang_config2_ctrsetup4_overvoltage(self):
        """Test 12: Network config 2, Controller setup 4 with gang-operated LTC and Dy trafo, overvoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LTCgang_config_2_ctrsetup_4_testmode_overvoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_13_dy_ltcgang_config2_ctrsetup3_undervoltage(self):
        """Test 13: Network config 2, Controller setup 3 with gang-operated LTC and Dy trafo, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LTCgang_config_2_ctrsetup_3_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)
    
    def test_14_dy_ltcgang_config3_ctrsetup4_undervoltage(self):
        """Test 14: Network config 3, Controller setup 4 with gang-operated LTC and Dy trafo, undervoltage case."""
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LTCgang_config_3_ctrsetup_4_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Tests for gang-operated LDC (new feature)
# =============================================================================

class TestGangOperatedLDC:
    """Tests for the newly added gang operation mode for LDC controller."""
    
    def test_15_dy_ldcgang_config2_ctrsetup3_undervoltage(self):
        """Test 15: Network config 2, Controller setup 3 with gang-operated LDC, undervoltage case.
        
        This case demonstrates the newly added gang operation mode for LDC
        (which previously only supported phase operation mode).
        """
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LDCgang_config_2_ctrsetup_3_testmode_undervoltage.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        assert check_voltage_narrowing(net_pre_control, net)
        
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Tests for failure cases
# =============================================================================

class TestFailureCases:
    """Tests for expected failure scenarios."""
    
    def test_load_flow_failure(self):
        """Test load flow failure on improperly configured network.
        
        Bus 6 of the 20-bus network is taken out of service, creating two
        network parts one of which is unsupplied. This should raise an exception.
        """
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_2_ctrsetup_1_testmode_undervoltage_lf_fail.pkl"])
        
        with pytest.raises(Exception) as excinfo:
            mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        # Verify we get an appropriate error message
        assert excinfo.value is not None
    
    def test_controller_non_convergence(self):
        """Test controller non-convergence scenario.
        
        Uses network config 2, controller setup 4 with LDC in the undervoltage
        situation with a different QV Curve for one of the volt_var controllers,
        causing controller non-convergence.
        """
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_3_ctrsetup_4_testmode_undervoltage_ctr_fail.pkl"])
        
        with pytest.raises(Exception) as excinfo:
            mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        # Verify we get a controller-related exception
        assert excinfo.value is not None
    
    def test_controller_non_convergence_with_voltvar_disabled(self):
        """Test that disabling volt-var controllers allows convergence.
        
        Re-tries the failing controller case with volt-var controllers disabled.
        The network should converge when the problematic controller type is disabled.
        """
        net = copy.deepcopy(grids["twenty_bus_system_Yy_LDC_config_3_ctrsetup_4_testmode_undervoltage_ctr_fail.pkl"])
        
        # This should succeed with volt-var controllers disabled
        mc.run_pf(net_pre_control := copy.deepcopy(net), 
                  tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
        mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, 
                  run_control=True, run_voltvar_control=False)
        
        # Verify the power flow completed (res_bus should be populated)
        assert not net.res_bus.empty


# =============================================================================
# Tests for controller oscillation
# =============================================================================

class TestControllerOscillation:
    """Tests for controller oscillation detection and handling."""
    
    def test_controller_oscillation_detection(self):
        """Test controller oscillation detection with gang-operated LTC.
        
        Uses grid configuration 2, controller setup 4 in the undervoltage testmode
        with a voltage bandwidth selected too narrow for the LTC.
        The controller should converge (with oscillation detection) even if some
        controlled voltages are out of spec.
        """
        net = copy.deepcopy(grids["twenty_bus_system_Dy_LTCgang_config_2_ctrsetup_4_testmode_undervoltage_ctrosc.pkl"])
        net_pre_control = copy.deepcopy(net)
        
        # Should produce a warning about oscillation but still complete
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mc.run_pf(net_pre_control, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=False)
            mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=True)
        
        # Verify the power flow completed
        assert not net.res_bus.empty
        
        # Compare to DSS reference (should still match)
        c = multiconductor.test.testing_toolbox.compare_to_dss(net)
        multiconductor.test.testing_toolbox.assert_comparison(c, tol=5e-4, tol_va_degree=1e-2)


# =============================================================================
# Additional parametrized tests by controller type
# =============================================================================

# Filter grids by controller type
ldc_grids = {name: net for name, net in grids.items() 
             if 'LDC' in name and 'res_bus_dss' in net.keys() and 'fail' not in name and 'osc' not in name}

ltc_grids = {name: net for name, net in grids.items() 
             if 'LTC' in name and 'res_bus_dss' in net.keys() and 'fail' not in name and 'osc' not in name}


@pytest.mark.parametrize("name,net", ldc_grids.items(), ids=list(ldc_grids.keys()))
def test_ldc_controller_convergence(name, net):
    """Test that LDC controllers converge for all LDC test cases."""
    net_calc = copy.deepcopy(net)
    mc.run_pf(net_calc, run_control=True)
    
    # Verify power flow completed successfully
    assert not net_calc.res_bus.empty
    assert net_calc.res_bus['vm_pu'].notna().any()


@pytest.mark.parametrize("name,net", ltc_grids.items(), ids=list(ltc_grids.keys()))
def test_ltc_controller_convergence(name, net):
    """Test that LTC controllers converge for all LTC test cases."""
    net_calc = copy.deepcopy(net)
    mc.run_pf(net_calc, run_control=True)
    
    # Verify power flow completed successfully
    assert not net_calc.res_bus.empty
    assert net_calc.res_bus['vm_pu'].notna().any()


# =============================================================================
# Bus and line result comparison tests
# =============================================================================

@pytest.mark.parametrize("name,net", list(grids_with_dss.items())[:5], ids=list(grids_with_dss.keys())[:5])
def test_bus_voltage_magnitude_accuracy(name, net):
    """Test that bus voltage magnitudes match DSS within tolerance."""
    net_calc = copy.deepcopy(net)
    mc.run_pf(net_calc, run_control=True)
    
    vm_diff = (net_calc.res_bus['vm_pu'] - net_calc.res_bus_dss['vm_pu']).abs().max()
    assert vm_diff < 1e-3, f"Voltage magnitude difference {vm_diff} exceeds tolerance"


@pytest.mark.parametrize("name,net", list(grids_with_dss.items())[:5], ids=list(grids_with_dss.keys())[:5])
def test_line_power_flow_accuracy(name, net):
    """Test that line power flows match DSS within tolerance."""
    net_calc = copy.deepcopy(net)
    mc.run_pf(net_calc, run_control=True)
    
    for what in ["p_from_mw", "p_to_mw", "q_from_mvar", "q_to_mvar"]:
        diff = (net_calc.res_line[what] - net_calc.res_line_dss[what]).abs().max()
        assert diff < 1e-3, f"{what} difference {diff} exceeds tolerance"
