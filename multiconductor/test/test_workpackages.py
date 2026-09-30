import multiconductor as mc
import pandas as pd
import warnings
import pytest
warnings.filterwarnings("ignore", category=FutureWarning)  
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import multiconductor.test.testing_toolbox
import multiconductor.control


def run_pf_and_compare_to_dss(net, run_control=False, tol=1e-5, tol_va_degree=1e-5):
    mc.run_pf(net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, run_control=run_control)
    c = multiconductor.test.testing_toolbox.compare_to_dss(net)
    multiconductor.test.testing_toolbox.assert_comparison(c, tol, tol_va_degree)

wp1_1_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_1_mc3ph")

@pytest.mark.parametrize("net", wp1_1_test_grids.values(), ids=list(wp1_1_test_grids.keys()))
def test_wp1_1(net):
    run_pf_and_compare_to_dss(net)


wp1_2_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_2_zip_loads")

@pytest.mark.parametrize("net", wp1_2_test_grids.values(), ids=list(wp1_2_test_grids.keys()))
def test_wp1_2(net):
    run_pf_and_compare_to_dss(net, tol_va_degree=4e-5)


wp1_3_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_3_multiphase")

@pytest.mark.parametrize("net", wp1_3_test_grids.values(), ids=list(wp1_3_test_grids.keys()))
def test_wp1_3(net):    
    run_pf_and_compare_to_dss(net, tol=2e-5, tol_va_degree=2e-4)


wp1_4_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_4_capacitor_modes")

@pytest.mark.parametrize("net", wp1_4_test_grids.values(), ids=list(wp1_4_test_grids.keys()))
def test_wp1_4(net):    
    run_pf_and_compare_to_dss(net, run_control=True, tol_va_degree=4e-5)


wp1_5_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_5_volt_var_control")

@pytest.mark.parametrize("net", wp1_5_test_grids.values(), ids=list(wp1_5_test_grids.keys()))
def test_wp1_5_voltvar(net):    
    l = []
    mc.run_pf(net, run_control=True)
    for idx in list(net.asymmetric_sgen[net.asymmetric_sgen.control_mode == "Volt/Var"].index.get_level_values(0).unique()):
        assert multiconductor.control.check_ctr_sgen_stable_state(net, sgen_first_lvl_idx=idx, verbose=True)

@pytest.mark.parametrize("net", wp1_5_test_grids.values(), ids=list(wp1_5_test_grids.keys()))
def test_wp1_5_compare_opendss_(net):    
    run_pf_and_compare_to_dss(net, run_control=True, tol=7e-5)


wp1_6_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_6_banked_transformer")

@pytest.mark.parametrize("net", wp1_6_test_grids.values(), ids=list(wp1_6_test_grids.keys()))
def test_wp1_6_banked_transformer(net):
    run_pf_and_compare_to_dss(net, run_control=True, tol=1e-3, tol_va_degree=3e-4)



wp1_7_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_7_ldc")

@pytest.mark.parametrize(("name","net"), wp1_7_test_grids.items(), ids=list(wp1_7_test_grids.keys()))
def test_wp1_7_LDC(net, name):    
    l = []
    mc.run_pf(net, run_control=True)
    mode = "locked_reverse" if "locked_reverse" in name else "locked_forward"    
    if "four" in name:
        ctr_bus = 2 if "bus2" in name else 3                    
    else:
        ctr_bus = 18 if "bus1" in name else 12                
    assert multiconductor.control.check_ldc_stable_state(net, ctr_bus, 0, [0.98,1.02], mode)


@pytest.mark.parametrize("net", wp1_7_test_grids.values(), ids=list(wp1_7_test_grids.keys()))
def test_wp1_7_compare_opendss(net):    
    l = []
    run_pf_and_compare_to_dss(net, run_control=True, tol=3e-5, tol_va_degree=8e-4)


wp1_8_test_grids = multiconductor.test.testing_toolbox.load_all_test_grids("wp1_8_ltc")

@pytest.mark.parametrize(("name","net"), wp1_8_test_grids.items(), ids=list(wp1_8_test_grids.keys()))
def test_wp1_8_LTC(net, name):    
    l = []
    mc.run_pf(net, run_control=True)
    v_bw = [1.03, 1.05] if ("load" in name or "undervoltage" in name) else [0.95, 0.97]
    mode = "phase" if "phase" in name else "gang"        
    assert multiconductor.control.check_ltc_stable_state(net, 0, v_bw, mode)    

@pytest.mark.parametrize("net", wp1_8_test_grids.values(), ids=list(wp1_8_test_grids.keys()))
def test_wp1_8_compare_opendss(net):    
    l = []
    run_pf_and_compare_to_dss(net, run_control=True, tol=3e-4, tol_va_degree=1e-2)
    
if __name__ == "__main__":
    
    pytest.main([__file__])    
