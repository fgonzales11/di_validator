import pandapower as pp
import numpy as np
from pathlib import Path
import pickle
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pickle import FALSE

def create_network(sn_mva, num_buses=2, load_type='symmetric', generation_type='symmetric',
                   load_bus=None, generation_bus=None, ext_grid_buses=None, create_lines=True,
                   hv_bus=None, lv_bus=None,
                   vector_group=None, shift_degree=None):
    """
    Create a power network with optional transformer and configurable load/generation types.

    Parameters:
    - sn_mva: Short-circuit power in MVA.
    - num_buses: Total number of buses.
    - load_type: 'symmetric' or 'asymmetric'.
    - generation_type: 'symmetric' or 'asymmetric'.
    - load_bus: Bus index or list for load placement.
    - generation_bus: Bus index or list for generator placement.
    - ext_grid_buses: List of bus indices for external grid (default: [0]).
    - create_lines: Whether to create lines between sequential buses.
    - hv_bus: High-voltage bus index (used if a transformer is present).
    - lv_bus: Low-voltage bus index (used if a transformer is present).
    - vector_group: Transformer vector group.
    - shift_degree: Transformer phase shift in degrees.

    Returns:
    - net: pandapower network object.
    """

    net = pp.create_empty_network()

    bus_vn_kv = []
    for i in range(num_buses):
        if hv_bus is not None and i == hv_bus:
            bus_vn_kv.append(10.0)
        elif lv_bus is not None and i == lv_bus:
            bus_vn_kv.append(0.4)
        else:
            bus_vn_kv.append(0.4)

    # Create buses
    buses = []
    for i in range(num_buses):
        buses.append(pp.create_bus(net, vn_kv=bus_vn_kv[i], name=f"Bus_{i}"))

    if load_bus is None:
        load_bus = [num_buses - 1]
    elif isinstance(load_bus, int):
        load_bus = [load_bus]

    if generation_bus is None:
        generation_bus = [num_buses - 1]
    elif isinstance(generation_bus, int):
        generation_bus = [generation_bus]

    ext_grid_buses = ext_grid_buses if ext_grid_buses is not None else [0]

    for bus in ext_grid_buses:
        pp.create_ext_grid(net, bus=bus, s_sc_max_mva=sn_mva, rx_max=0.1,
                           x0x_max=1, r0x0_max=0.1, name=f"ExtGrid_{bus}")

    if create_lines:
        for i in range(num_buses - 1):
            pp.create_line_from_parameters(
                net, from_bus=i, to_bus=i + 1, length_km=0.1,
                r_ohm_per_km=0.1, x_ohm_per_km=0.2,
                r0_ohm_per_km=0.05, x0_ohm_per_km=0.1,
                c_nf_per_km=100, c0_nf_per_km=50, max_i_ka=0.2,
                name=f"Line_{i}_{i + 1}"
            )

    for i, bus in enumerate(load_bus):
        if load_type == 'symmetric':
            pp.create_asymmetric_load(
                net, bus=bus, p_a_mw=0.01, p_b_mw=0.01, p_c_mw=0.01,
                q_a_mvar=0.005, q_b_mvar=0.005, q_c_mvar=0.005,
                name=f"Load_{i}_Bus_{bus}"
            )
        else:
            pp.create_asymmetric_load(
                net, bus=bus, p_a_mw=0.01, p_b_mw=0.02, p_c_mw=0.015,
                q_a_mvar=0.005, q_b_mvar=0.002, q_c_mvar=-0.007,
                name=f"Load_{i}_Bus_{bus}"
            )

    for i, bus in enumerate(generation_bus):
        if generation_type == 'symmetric':
            pp.create_asymmetric_sgen(
                net, bus=bus, p_a_mw=0.02, p_b_mw=0.02, p_c_mw=0.02,
                q_a_mvar=0.005, q_b_mvar=0.005, q_c_mvar=0.005,
                name=f"SGen_{i}_Bus_{bus}"
            )
        else:
            pp.create_asymmetric_sgen(
                net, bus=bus, p_a_mw=0.02, p_b_mw=0.01, p_c_mw=0.015,
                q_a_mvar=0.005, q_b_mvar=0.002, q_c_mvar=-0.003,
                name=f"SGen_{i}_Bus_{bus}"
            )

    if hv_bus is not None and lv_bus is not None:
        pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus,
            vn_hv_kv=10.0, vn_lv_kv=0.4, vk_percent=6.0, vkr_percent=0.5,
            vk0_percent=3.0, vkr0_percent=1.0, pfe_kw=14,
            vector_group=vector_group, shift_degree=shift_degree,
            mag0_percent=100.0, mag0_rx=0.0, si0_hv_partial=0.9,
            tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2,
            tap_pos=0, tap_step_percent=2.5, parallel=1, sn_mva=sn_mva,
            i0_percent=0.5, tap_changer_type="Ratio"
        )

    return net



def create_topology(net, connections, length_km=0.1):

    """
    Adds a topology (lines) to an existing pandapower network.

    Parameters:
    - net: An existing pandapower network.
    - connections: List of (from_bus, to_bus) tuples defining the topology.
    - length_km: Length of each line (default: 0.1 km).
    """

    for from_bus, to_bus in connections:
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.2,
            r0_ohm_per_km=0.05,
            x0_ohm_per_km=0.1,
            c_nf_per_km=100,
            c0_nf_per_km=50,
            max_i_ka=0.2,
            name=f"Line_{from_bus}_{to_bus}"
        )

# def run_power_flow_and_compare(net):

#     pp.runpp_3ph(net, tolerance_mva=1e-12)

#     # Convert to pp-os to pp-mc using converter function
#     mc_net = pp2mc(net)

#     # Run the power flow calculation in pp-mc
#     mc.run_pf(mc_net, tol_vmag_pu=1e-9, tol_vang_rad=1e-9, MaxIter=30)

#     _assert_bus_results(net, mc_net)
#     _assert_power_values(net, mc_net)
#     _assert_load_values(net, mc_net)


test_dir = Path(__file__).parent.resolve()

def save_test_grid(net, folder, name):
    with open(test_dir / folder / "grids" / name, "wb") as f:
        pickle.dump(net, f)


def load_test_grid(folder, name):
    with open(test_dir / folder / "grids" / name, "rb") as f:
        return pickle.load(f)


def load_all_test_grids(folder):
    nets = dict()
    for fn in (test_dir / folder / "grids").glob("*.pkl"):
        with open(fn, "rb") as f:
            nets[fn.name] = pickle.load(f)
    return nets


def compare_to_dss(net, threshold=1e-3):
    mask = net["res_bus"]["vm_pu"] > threshold
    max_va_degree_diff = (net["res_bus"]["va_degree"][mask] - net["res_bus_dss"]["va_degree"][mask]).abs().max()
    if max_va_degree_diff>180:
        max_va_degree_diff = 360-max_va_degree_diff
    c = {"max_vm_pu_diff": (net["res_bus"]["vm_pu"] - net["res_bus_dss"]["vm_pu"]).abs().max(),
         "max_va_degree_diff": max_va_degree_diff}
    for what in ["i_from_ka", "i_to_ka", "p_from_mw", "p_to_mw", "q_from_mvar", "q_to_mvar"]:
        c[f"max_{what}_diff"] = (net["res_line"][what] - net["res_line_dss"][what]).abs().max()
    return c

def assert_comparison(c, tol=1e-5, tol_va_degree=None, do_assert=True):
    for what, value in c.items():
        if np.isnan(value):
            continue
        tol_ = (tol_va_degree or 1e-3) if what == "max_va_degree_diff" else tol        
        if do_assert:
            assert value < tol_, f"{what} = {value} > {tol_}"
        else:
            if not value < tol_:
                return False 
    return True

def print_comparison(c):
    for v, k in c.items():
        print(f"Max overall {v[4:-5]} difference: {k}")
        
def print_sgen_stats(net, name, ctr_sgen, header=False, before_control=True):
    
    if header:
        print("---------------------------------")
        print(name)        
    
    if before_control:
        print("Before control")
    else:
        print("After control")
            
    bus_string = "Sgen bus: "
    v_pu_string = "vm_pu: "
    sn_mva_limit_string = "sn_mva_limit: "
    sn_mva_string = "sn_mva: "
    p_mw_string = "p_mw: "
    q_mvar_string = "q_mvar: "    
    for i in ctr_sgen:
        bus = net.asymmetric_sgen.loc[(i,1)]['bus']
        bus_string += str(bus)+" / "
        v_pu_string += "["+f"{net.res_bus.loc[(bus,1)]['vm_pu']:.4f}, {net.res_bus.loc[(bus,2)]['vm_pu']:.4f}, {net.res_bus.loc[(bus,3)]['vm_pu']:.4f}] / "
        sn_mva_limit_string += "["+f"{net.asymmetric_sgen.loc[(i,0)]['sn_mva']:.6f}, {net.asymmetric_sgen.loc[(i,1)]['sn_mva']:.6f}, {net.asymmetric_sgen.loc[(i,2)]['sn_mva']:.6f}] / "
        sn_mva = [sqrt( net.asymmetric_sgen.loc[(i,j)]['p_mw']**2 + net.asymmetric_sgen.loc[(i,j)]['q_mvar']**2) for j in [0,1,2] ]
        sn_mva_string += "["+f"{sn_mva[0]:.6f}, {sn_mva[1]:.6f}, {sn_mva[2]:.6f}] / "
        p_mw_string += "["+f"{net.asymmetric_sgen.loc[(i,0)]['p_mw']:.6f}, {net.asymmetric_sgen.loc[(i,1)]['p_mw']:.6f}, {net.asymmetric_sgen.loc[(i,2)]['p_mw']:.6f}] / "
        q_mvar_string += "["+f"{net.asymmetric_sgen.loc[(i,0)]['q_mvar']:.6f}, {net.asymmetric_sgen.loc[(i,1)]['q_mvar']:.6f}, {net.asymmetric_sgen.loc[(i,2)]['q_mvar']:.6f}] / "
   
    print(bus_string.removesuffix(" / "))
    print(v_pu_string.removesuffix(" / "))
    print(sn_mva_limit_string.removesuffix(" / "))
    print(sn_mva_string.removesuffix(" / "))
    print(p_mw_string.removesuffix(" / "))
    print(q_mvar_string.removesuffix(" / "))


def plot_control_step(grid, grid_pre_control, ylim=(0.85, 1.15), name='Bus Voltages', save_string=None):
   
    fig, ax = plt.subplots(figsize=(8, 8))

    shunt_indices = list(grid.asymmetric_shunt.index)
    shunt_buses = grid.asymmetric_shunt["bus"].values
    phases = grid.asymmetric_shunt["from_phase"].values

    unique_buses = list(sorted(set(shunt_buses)))
    cmap = cm.get_cmap('viridis', len(unique_buses))
    bus_colors = {bus: cmap(i) for i, bus in enumerate(unique_buses)}

    selected_phases_colors = {}
    for idx, bus, phase in zip(shunt_indices, shunt_buses, phases):
        selected_phases_colors[(bus, phase)] = bus_colors[bus]

    colors_default = 'lightgrey'
    x_values = [1, 2]

    for idx, vm_pu in grid.res_bus.vm_pu.items():
        if idx[1] == 0:
            continue
        y_data_pre_control = grid_pre_control.res_bus.vm_pu.loc[idx]
        y_data_grid = vm_pu

        color = selected_phases_colors.get(idx, colors_default)
        label_suffix = " - Controlled Bus" if idx in selected_phases_colors else ""

        ax.plot(
            x_values, [y_data_pre_control, y_data_grid],
            marker='o', linestyle='-', color=color,
            label=f'Bus {idx[0]}, Phase {idx[1]}{label_suffix}'
        )

    idx0 = next(iter(grid.asymmetric_shunt.index))
    ax.axhline(
        y=grid.asymmetric_shunt.v_threshold_on.at[idx0],
        color='red', linestyle='-.', label='v_threshold_on'
    )
    ax.axhline(
        y=grid.asymmetric_shunt.v_threshold_off.at[idx0],
        color='red', linestyle='-', label='v_threshold_off'
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(['pre control step', 'post control step'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel(name)
    ax.set_ylabel('Voltage (pu)')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)
    if save_string:
        plt.savefig(f'{save_string}.png', dpi=400, bbox_inches='tight')
    
    plt.show()
    
def print_minmax(mc_net):
    tr_loading_max = float(mc_net.res_trafo['loading_percent'].max())
    tr_direction = "reverse" if sum(mc_net.res_trafo.loc[0,0]['p_mw'])<0 else "forward"
    vm_max = mc_net.res_bus['vm_pu'].max()
    vm_argmax = mc_net.res_bus['vm_pu'].argmax()
    vm_busmax = mc_net.res_bus['vm_pu'].index[vm_argmax][0]
    vm_min = mc_net.res_bus[mc_net.res_bus['vm_pu']>0.1]['vm_pu'].min()
    vm_argmin = mc_net.res_bus[mc_net.res_bus['vm_pu']>0.1]['vm_pu'].argmin()
    vm_busmin = mc_net.res_bus[mc_net.res_bus['vm_pu']>0.1]['vm_pu'].index[vm_argmin][0]
    print("Max trafo loading: "+str(tr_loading_max)+" ("+tr_direction+") vm_pu min: "+str(vm_min)+" at bus "+str(int(vm_busmin))+", max: "+str(vm_max)+" at bus "+str(int(vm_busmax))) 

