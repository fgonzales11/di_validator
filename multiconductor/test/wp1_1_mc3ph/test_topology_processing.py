import multiconductor as mc
from multiconductor.pycci.std_types import create_std_types
import pandas as pd
pd.set_option('display.float_format', '{:.6f}'.format) 
import warnings
import pytest
import numpy as np
import copy
warnings.filterwarnings("ignore", category=FutureWarning)  
warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
rtol=0
atol=5e-8

def test_bus_bus_switches():
    def create_net(num_switches=0):
        net = mc.create_empty_network(sn_mva=1)

        b0 = mc.create_bus(net, 20, grounding_r_ohm=0.001)
        mc.create_ext_grid_sequence(net, b0, from_phase=range(1,4), to_phase=0, vm_pu=1, va_degree=0, sn_mva=1e9, rx=0.1, x0x=1, r0x0=0.1, name="slack")

        create_std_types(net, {"lv_line": {"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.2, "r0_ohm_per_km": 0.05, "x0_ohm_per_km": 0.1, "c_nf_per_km": 100, "c0_nf_per_km": 50, "max_i_ka": 0.2}}, "sequence")

        bs = [mc.create_bus(net, 0.4) for _ in range(2*num_switches + 2)]
        i = 0
        #mc.create_switch(net, bus=b0, phase=[0,1,2,3], element=bs[i], et="b", closed=True)
        mc.create_transformer_3ph(net, b0, bs[i], "0.63 MVA 20/0.4 kV")
        # create num_switches consecutive busbus switches
        i += 1
        for s in range(num_switches):
            mc.create_switch(net, bus=bs[i-1], phase=[0,1,2,3], element=bs[i], et="b", closed=True)
            i += 1
        mc.create_line(net, std_type="lv_line", model_type="sequence", from_bus=bs[i-1], from_phase=range(1, 4), to_bus=bs[i], to_phase=range(1, 4), length_km=0.01, name="line")
        k = i
        i += 1
        # create num_switches busbus switches connected at bus k
        for s in range(num_switches):
            mc.create_switch(net, bus=bs[k], phase=[0,1,2,3], element=bs[i], et="b", closed=True)
            i += 1
        mc.create_asymmetric_load(net, bs[i-1], (1,2,3), (0,0,0), (.1, .2,.3), 0)
        return net

    net_without = create_net(0)
    net_with = create_net(1)
    # open busbus switches should be ignored
    mc.create_switch(net_with, bus=0, phase=[0,1,2,3], element=1, et="b", closed=False)
    mc.create_switch(net_with, bus=0, phase=[0,1,2,3], element=3, et="b", closed=False)

    mc.run_pf(net_without)
    mc.run_pf(net_with)
    rtol=0
    atol=5e-8
    # voltage at load should be the same
    lbus1 = net_with["asymmetric_load"]["bus"].values[0]
    lbus2 = net_without["asymmetric_load"]["bus"].values[0]
    assert np.allclose(net_with["res_bus"].loc[lbus1].vm_pu.values, net_without["res_bus"].loc[lbus2].vm_pu.values, rtol=rtol, atol=atol)
    assert np.allclose(net_with["res_bus"].loc[lbus1].va_degree.values, net_without["res_bus"].loc[lbus2].va_degree.values, rtol=rtol, atol=atol)


def test_open_line_switch():
    """tests that line charging current of line 2 is considered regardless of the switch state of the line switch
       #---o---o  oL       vs       #---o--- o  oL
    """
    net = mc.create_empty_network(sn_mva=1)
    [mc.create_bus(net, 20) for i in range(5)]
    mc.create_ext_grid(net, 0, from_phase=range(1,4), to_phase=0, vm_pu=1, va_degree=0, name="slack", r_ohm=1, x_ohm=1)
    mc.create_line(net, "UG1", "sequence", from_bus=0, from_phase=[1,2,3], to_bus=1, to_phase=[1,2,3], length_km=25, name="MV_line")
    l = mc.create_line(net, "UG1", "sequence", from_bus=1, from_phase=[1,2,3], to_bus=2, to_phase=[1,2,3], length_km=25, name="MV_line")
    
    netc = copy.deepcopy(net)
    mc.create_switch(netc, bus=2, element=l, phase=[1,2,3], et="l", closed=False)

    mc.run_pf(net)
    mc.run_pf(netc)

    assert np.allclose(net["res_bus"].loc[1].vm_pu.values, netc["res_bus"].loc[1].vm_pu.values, rtol=rtol, atol=atol)


def test_ideal_ext_grid():
    """this just tests if loadflow executes when r and x of ext_grid are 0"""
    net = mc.create_empty_network(sn_mva=1)
    mc.create_bus(net, 20)
    mc.create_bus(net, 20)
    mc.create_ext_grid(net, 0, from_phase=range(1,4), to_phase=0, vm_pu=1, va_degree=0, name="slack", r_ohm=0, x_ohm=0)
    create_std_types(net, {"lv_line": {"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.2, "r0_ohm_per_km": 0.05, "x0_ohm_per_km": 0.1, "c_nf_per_km": 100, "c0_nf_per_km": 50, "max_i_ka": 0.2}}, "sequence")
    mc.create_line(net, std_type="lv_line", model_type="sequence", from_bus=0, from_phase=range(1, 4), to_bus=1, to_phase=range(1, 4), length_km=5, name="line")
    mc.create_asymmetric_load(net, 1, (1, 2, 3), 0, (15, 5, 12.5), 0)
    mc.run_pf(net)
    assert net.model.is_ideal_ext_grid
    assert net.res_bus.loc[(0,1), "vm_pu"] == 1


if __name__ == "__main__":
    # test_zip_loads('five_bus_system_radial_delta_zip')
    pytest.main([__file__])





















