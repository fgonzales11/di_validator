from dotted_dict import DottedDict

import numpy as np
import pandas as pd
#from numpy.linalg import inv
#from scipy.sparse import csc_matrix, vstack, linalg as sla

import numba
import scipy
import scipy.sparse

from multiconductor.studies import InvalidStudyInput


NUM_PHASES = 4


@numba.njit(cache=True)
def maxi(d):
    m = d[0]
    for n in d:
        if n > m:
            m = n
    return m


def initialize_terminal_lookup(net):
    model = net["model"]
    max_bus_index = maxi(net["bus"].index.codes[0]) + 1

    model["line_index_start"] = max_bus_index * NUM_PHASES
    max_line_index = max(a for a,_ in net["line"].index.values) + 1 if len(net["line"]) else 0
    model["ext_grid_index_start"] = model["line_index_start"] + max_line_index * NUM_PHASES * 2
    model["num_terminals"] = model["ext_grid_index_start"]

    _, eg = get_active_ext_grid_table(net, require_active=False)
    if eg is None or len(eg) == 0:
        model["is_ideal_ext_grid"] = False
    else:
        model["is_ideal_ext_grid"] = bool(np.all(eg["r_ohm"].values == 0) and np.all(eg["x_ohm"].values == 0))
        if not model["is_ideal_ext_grid"]:
            max_ext_grid_index = int(eg.index.get_level_values(0).max()) + 1
            model["num_terminals"] += max_ext_grid_index * NUM_PHASES

    model["load_index_start"] = model["num_terminals"]
    max_load_index = maxi(net["asymmetric_load"].index.codes[0]) + 1 if len(net["asymmetric_load"]) else 0
    model["num_terminals"] += max_load_index * NUM_PHASES * 2

    model["sgen_index_start"] = model["num_terminals"]
    max_sgen_index = maxi(net["asymmetric_sgen"].index.codes[0]) + 1 if len(net["asymmetric_sgen"]) else 0
    model["num_terminals"] += max_sgen_index * NUM_PHASES * 2

    model["gen_index_start"] = model["num_terminals"]
    max_gen_index = maxi(net["asymmetric_gen"].index.codes[0]) + 1 if len(net["asymmetric_gen"]) else 0
    model["num_terminals"] += max_gen_index * NUM_PHASES * 2

    if "asymmetric_shunt" in net:
        net.model.shunt_index_start = net.model.num_terminals
        lidxs = net.asymmetric_shunt.index.get_level_values(0).unique()
        max_shunt_index = (max(lidxs) + 1) if len(lidxs) else 0
        net.model.num_terminals += max_shunt_index * NUM_PHASES * 2


def info(net):
    r = []
    for i, b in net.bus.iterrows():
        t = get_bus_terminal(net.model, i[0], i[1])
        r.append([f"bus_{i[0]}_p{i[1]}", t, net.model.terminal_to_y_lookup[t], i[1]])
    for i, line_row in net.line.iterrows():
        t = get_line_to_terminal(net.model, i[0], line_row.to_phase)
        r.append([f"line_{i[0]}_to_p{line_row.to_phase}", t, net.model.terminal_to_y_lookup[t], line_row.to_phase])
        t = get_line_from_terminal(net.model, i[0], line_row.from_phase)
        r.append([f"line_{i[0]}_from_p{line_row.from_phase}", t, net.model.terminal_to_y_lookup[t], line_row.from_phase])
    for i, b in net.ext_grid.iterrows():
        t = get_ext_grid_terminal(net.model, i[0], i[1])
        r.append([f"ext_grid_{i[0]}_p{i[1]}", t, net.model.terminal_to_y_lookup[t], i[1]])
    for i, b in net.ext_grid_sequence.iterrows():
        t = get_ext_grid_terminal(net.model, i[0], i[1])
        r.append([f"ext_grid_sequence{i[0]}_p{i[1]}", t, net.model.terminal_to_y_lookup[t], i[1]])
    for i, b in net.asymmetric_load.iterrows():
        t = get_load_from_terminal(net.model, i[0], i[1])
        r.append([f"load{i[0]}_c{i[1]}_from", t, net.model.terminal_to_y_lookup[t], i[1]])
        t = get_load_to_terminal(net.model, i[0], i[1])
        r.append([f"load{i[0]}_c{i[1]}_to", t, net.model.terminal_to_y_lookup[t], i[1]])
    for i, b in net.asymmetric_gen.iterrows():
        t = get_gen_from_terminal(net.model, i[0], i[1])
        r.append([f"gen{i[0]}_c{i[1]}_from", t, net.model.terminal_to_y_lookup[t], i[1]])
        t = get_gen_to_terminal(net.model, i[0], i[1])
        r.append([f"gen{i[0]}_c{i[1]}_to", t, net.model.terminal_to_y_lookup[t], i[1]])

    return pd.DataFrame(r, columns=["element", "terminal", "terminal_to_y_lookup", "phase"])


def get_bus_terminal(model, bus, phase):
    return NUM_PHASES * bus + phase


def get_load_from_terminal(model, lidx, circuit):
    return model.load_index_start + (lidx * NUM_PHASES + circuit) * 2 + 1


def get_load_to_terminal(model, lidx, circuit):
    return model.load_index_start + (lidx * NUM_PHASES + circuit) * 2


def get_sgen_from_terminal(model, lidx, circuit):
    return model.sgen_index_start + (lidx * NUM_PHASES + circuit) * 2 + 1


def get_sgen_to_terminal(model, lidx, circuit):
    return model.sgen_index_start + (lidx * NUM_PHASES + circuit) * 2


def get_gen_from_terminal(model, lidx, circuit):
    return model.gen_index_start + (lidx * NUM_PHASES + circuit) * 2 + 1


def get_gen_to_terminal(model, lidx, circuit):
    return model.gen_index_start + (lidx * NUM_PHASES + circuit) * 2


def get_shunt_from_terminal(model, lidx, circuit):
    return model.shunt_index_start + (lidx * NUM_PHASES + circuit) * 2 + 1


def get_shunt_to_terminal(model, lidx, circuit):
    return model.shunt_index_start + (lidx * NUM_PHASES + circuit) * 2


def get_line_from_terminal(model, lidx, phase):
    return model.line_index_start + (lidx * NUM_PHASES + phase) * 2 + 1


def get_line_to_terminal(model, lidx, phase):
    return model.line_index_start + (lidx * NUM_PHASES + phase) * 2


def get_ext_grid_terminal(model, eixd, circ):
    return model.ext_grid_index_start + eixd * NUM_PHASES + circ


def get_active_ext_grid_table(net, require_active=True):
    ext_grid = net["ext_grid"].loc[net["ext_grid"]["in_service"].astype(bool)] if len(net["ext_grid"]) else net["ext_grid"]
    ext_grid_sequence = (
        net["ext_grid_sequence"].loc[net["ext_grid_sequence"]["in_service"].astype(bool)]
        if len(net["ext_grid_sequence"])
        else net["ext_grid_sequence"]
    )

    if len(ext_grid) and len(ext_grid_sequence):
        raise InvalidStudyInput("Only one active external-grid table is supported per power-flow solve.")
    if len(ext_grid_sequence) and len(ext_grid_sequence) != 3:
        raise InvalidStudyInput("Active ext_grid_sequence rows must contain all three sequence components.")

    if len(ext_grid):
        return "ext_grid", ext_grid.sort_index()
    if len(ext_grid_sequence):
        return "ext_grid_sequence", ext_grid_sequence.sort_index()
    if require_active:
        raise InvalidStudyInput("No active external-grid rows are available.")
    return None, None


def deal_with_ext_grid(net):
    _, ext_tab = get_active_ext_grid_table(net, require_active=False)
    if ext_tab is None or len(ext_tab) == 0:
        return []
    connections = []
    model = net["model"]
    for (eidx, circ), bus, phase in zip(ext_tab.index.values, ext_tab["bus"].values, ext_tab["from_phase"].values):
        bus_terminal = get_bus_terminal(model, bus, phase)
        if not model.terminal_in_service[bus_terminal]:
            continue
        if model.is_ideal_ext_grid:
            slack_terminal = bus_terminal
            connections.append((slack_terminal, slack_terminal))  # keep slack nodes without connections
        else:
            slack_terminal = get_ext_grid_terminal(model, eidx, circ)
            model.terminal_vn_kv[slack_terminal] = model.terminal_vn_kv[bus_terminal]
            connections.append((slack_terminal, int(bus_terminal)))
        model.terminal_is_slack[slack_terminal] = True
    return connections


@numba.njit(cache=True)
def collect_terminal_infos(bus, phases, vn_kv_, g_, gro_, gxo_ , _is_, terminal_vn_kv, terminal_in_service, terminal_is_ideally_grounded):
    for (b, p, vn_kv, g, gro, gxo , _is) in zip(bus, phases, vn_kv_, g_, gro_, gxo_ , _is_):
        terminal = 4 * b + p
        terminal_vn_kv[terminal] = vn_kv
        terminal_in_service[terminal] = _is
        if _is and g and gro == 0 and gxo == 0:
            terminal_is_ideally_grounded[terminal] = True


@numba.njit(cache=True)
def put(cx, cy, cv, cpos, t1, t2, v):
    cx[cpos] = t1
    cy[cpos] = t2
    cv[cpos] = v
    cpos += 1
    cx[cpos] = t2
    cy[cpos] = t1
    cv[cpos] = v
    cpos += 1
    return cpos


@numba.njit(cache=True)
def collect_line_connections(cx, cy, cv, cpos, lidx_, fb_, fp_, tb_, tp_, _is_, line_index_start, terminal_in_service):
    for lidx, fb, fp, tb, tp, _is in zip(lidx_, fb_, fp_, tb_, tp_, _is_):
        if not _is:
            continue
        bus_from_terminal = 4 * fb + fp
        line_from_terminal = line_index_start + (lidx * 4 + fp) * 2 + 1
        bus_to_terminal = 4 * tb + tp
        line_to_terminal =  line_index_start + (lidx * 4 + tp) * 2
        if not terminal_in_service[bus_from_terminal] or not terminal_in_service[bus_to_terminal]:
            continue
        cpos = put(cx, cy, cv, cpos, bus_from_terminal, line_from_terminal, 1)
        cpos = put(cx, cy, cv, cpos, bus_to_terminal, line_to_terminal, 1)
        #cpos = put(cx, cy, cv, cpos, line_from_terminal, line_to_terminal, 2)
    return cpos


@numba.njit(cache=True)
def collect_switch_connections(cx, cy, cv, cpos, tpos, bus_, phase_, element_, et_, _is_, rohm_, terminal_in_service):
    for bus, phase, element, et, closed, r_ohm in zip(bus_, phase_, element_, et_, _is_, rohm_):
        if et and closed:
            bus_terminal = 4 * bus + phase
            element_terminal = 4 * element + phase
            if not terminal_in_service[bus_terminal] or not terminal_in_service[element_terminal]:
                continue
            if r_ohm == 0:
                cpos = put(cx, cy, cv, cpos, bus_terminal, element_terminal, 1)
            else:
                cx[tpos] = bus_terminal
                cx[tpos - 1] = element_terminal
                tpos -= 2
    return cpos, tpos


@numba.njit(cache=True)
def collect_element_connections(cx, cy, cv, cpos, lidx_, circ_, bus_, fp_, tp_, _is_, element_index_start, terminal_in_service):
    for lidx, bus, circ, fp, tp, _is in zip(lidx_, bus_, circ_, fp_, tp_, _is_):
        if not _is:
            continue
        bus_from_terminal = 4 * bus + fp
        element_from_terminal = element_index_start + (lidx * 4 + circ) * 2 + 1
        bus_to_terminal = 4 * bus + tp
        element_to_terminal =  element_index_start + (lidx * 4 + circ) * 2
        if not terminal_in_service[bus_from_terminal] or not terminal_in_service[bus_to_terminal]:
            continue
        cpos = put(cx, cy, cv, cpos, bus_from_terminal, element_from_terminal, 1)
        cpos = put(cx, cy, cv, cpos, bus_to_terminal, element_to_terminal, 1)
        #cpos = put(cx, cy, cv, cpos, element_from_terminal, element_to_terminal, 2)
    return cpos


def build_y_index(net):
    initialize_terminal_lookup(net)
    model = net.model
    num_terminals = model.num_terminals

    model.terminal_in_service = terminal_in_service = np.ones(num_terminals, dtype=bool)
    model.terminal_vn_kv = terminal_vn_kv = np.ones(num_terminals, dtype=np.float64) * -1
    model.terminal_is_ideally_grounded = terminal_is_ideally_grounded = np.zeros(num_terminals, dtype=bool)
    model.terminal_is_slack = np.zeros(num_terminals, dtype=bool)

    bus = net["bus"]
    collect_terminal_infos(bus.index.codes[0],
                           bus.index.codes[1],
                           bus["vn_kv"].values,
                           bus["grounded"].values,
                           bus["grounding_r_ohm"].values,
                           bus["grounding_x_ohm"].values,
                           bus["in_service"].values,
                           terminal_vn_kv,
                           terminal_in_service,
                           terminal_is_ideally_grounded)
    
    cx = np.ones(num_terminals * 10, dtype=np.int32) * -10000000
    cy = np.empty(num_terminals * 10, dtype=np.int32)  # TODO think about the x10 more
    cv = np.empty(num_terminals * 10, dtype=np.int8)
    cpos = 0
    tpos = len(cv) - 1

    line_indizes = np.array([a for a,_ in net["line"].index.values])
    #line_indizes = net["line"].index.get_level_values(0).to_numpy()
    line = net["line"]
    cpos = collect_line_connections(cx, cy, cv, cpos,
                                    line_indizes,
                                    line["from_bus"].values,
                                    line["from_phase"].values,
                                    line["to_bus"].values,
                                    line["to_phase"].values,
                                    line["in_service"].values,
                                    model["line_index_start"],
                                    terminal_in_service)

    switch = net["switch"]
    r_ohm = switch["r_ohm"].values if "r_ohm" in switch else np.zeros_like(switch["closed"].values)
    cpos, tpos = collect_switch_connections(cx, cy, cv, cpos, tpos,
                                      switch["bus"].values,
                                      switch["phase"].values,
                                      switch["element"].values,
                                      switch["et"].values == "b",
                                      switch["closed"].values,
                                      r_ohm,
                                      terminal_in_service,
                                      )

    eg_connections = deal_with_ext_grid(net)
    for (f, t) in eg_connections:
        cx[tpos] = f
        cx[tpos - 1] = t
        tpos -= 2

#     line_from_set = set()
#     line = net.line
#     for (idx, _), fb, fp, tb, tp, in_service in zip(line.index, line["from_bus"].values, line["from_phase"].values,
#                                                     line["to_bus"].values, line["to_phase"].values, line["in_service"].values):
#         if not in_service:
#             continue
#         line_from_set.add((idx, fb))


    trafo = net["trafo1ph"]
    for (idx, bus, circ), fp, tp, in_service in zip(trafo.index, trafo["from_phase"].values,
                                                    trafo["to_phase"].values, trafo["in_service"].values):
        if not in_service:
            continue
        t1 = bus * 4 + fp
        t2 = bus * 4 + tp
        if not terminal_in_service[t1] or not terminal_in_service[t2]:
            continue
        cx[tpos] = t1
        cx[tpos - 1] = t2
        tpos -= 2
 
    # switch = net.switch
    # for bus, phase, element, et, closed in zip(switch["bus"], switch["phase"], switch["element"], switch["et"], switch["closed"]):
    #     if et == "b" and closed:
    #         bt1, bt2 = get_bus_terminal(model, bus, phase), get_bus_terminal(model, element, phase)
    #         C[bt1, bt2] = C[bt2, bt1] = 1
    #     if et == "l" and not closed:
    #         bt = get_bus_terminal(model, bus, phase)
    #         is_from_side = (element, bus) in line_from_set
    #         if is_from_side:
    #             lt = get_line_from_terminal(model, element, phase)
    #         else:
    #             lt = get_line_to_terminal(model, element, phase)
    #         C[bt, lt] = C[lt, bt] = 0

    for element, element_index_start in [(net["asymmetric_load"], model.load_index_start),
                                         (net["asymmetric_sgen"], model.sgen_index_start),
                                         (net["asymmetric_gen"], model.gen_index_start),
    #                                     (net.get("asymmetric_shunt", None), model.get("shunt_index_start", 0))
                                         ]:
# TODO fix shunt dtypes - 
#        if element is None or len(element) == 0:
#            continue
        
        cpos = collect_element_connections(cx, cy, cv, cpos,
                                    element.index.codes[0],
                                    element.index.codes[1],
                                    element["bus"].values,
                                    element["from_phase"].values,
                                    element["to_phase"].values,
                                    element["in_service"].values,
                                    element_index_start,
                                    terminal_in_service,
                                    )
# workaround
    if "asymmetric_shunt" in net and len(net["asymmetric_shunt"]) > 0:
        element = net["asymmetric_shunt"]
        cpos = collect_element_connections(cx, cy, cv, cpos,
                                element.index.codes[0],
                                element.index.codes[1],
                                element["bus"].astype(np.int64).values,
                                element["from_phase"].astype(np.int64).values,
                                element["to_phase"].astype(np.int64).values,
                                element["in_service"].astype(bool).values,
                                model.shunt_index_start,
                                terminal_in_service,
                                )

    model.C = C = scipy.sparse.coo_array((cv[:cpos], (cx[:cpos], cy[:cpos])), shape=(num_terminals, num_terminals)).tocsc()
    terminals = np.hstack([cx[:cpos], cx[tpos + 1:]])
    net.model.terminals = terminals
    terminal_to_y_lookup = model.terminal_to_y_lookup = find_islands(model.num_terminals, terminals, C.indptr, C.indices)
    model.y_size = np.max(terminal_to_y_lookup) + 1
    if model["debug_level"] > 0:
        net.bus["terminal"] = [get_bus_terminal(model, b, p) for b, p in net.bus.index]
        net.bus["y_index"] = [terminal_to_y_lookup[4 * b + p] for b, p in net.bus.index]

    #TODO is this really so compicated ?
    Ibase_y = np.zeros(model.y_size)
    model.y_fixed_voltage = np.ones((model.y_size, 1), dtype=complex) * -1

    sn_mva = net["sn_mva"]
    for terminal_idx in range(num_terminals):
        y_idx = terminal_to_y_lookup[terminal_idx]
        if y_idx != -1:
            vn_kv = terminal_vn_kv[terminal_idx]
            if vn_kv > -1:
                Ibase_y[y_idx] = sn_mva * 1e6 / ( vn_kv * 1e3 / np.sqrt(3))
            if terminal_is_ideally_grounded[terminal_idx]:
                model.y_fixed_voltage[y_idx] = 0
    model.Ibase_y = Ibase_y


@numba.njit(cache=True)
def find_islands(num_terminals, start_terminals, indptr, indices):
    visited = np.zeros(num_terminals, dtype=numba.boolean)
    islands = np.ones(num_terminals, dtype=numba.int32) * -1
#    visited = np.zeros(num_terminals, dtype=bool)
#    islands = np.ones(num_terminals, dtype=np.int32) * -1

    island_idx = 0
    
    stack = np.zeros(num_terminals, dtype=np.int32)
    stack_top = -1
    
    for node in start_terminals:
        if not visited[node]:
            visited[node] = True
            islands[node] = island_idx
            
            stack_top += 1
            stack[stack_top] = node
            
            while stack_top >= 0:
                v = stack[stack_top]
                stack_top -= 1
                
                start = indptr[v]
                end = indptr[v + 1]
                
                for i in range(start, end):
                    k = indices[i]
                    
                    if not visited[k]:
                        visited[k] = True
                        islands[k] = island_idx
                        
                        stack_top += 1
                        stack[stack_top] = k
            island_idx += 1
    
    return islands



def build_y_index_old(net):
    initialize_terminal_lookup(net)
    model = net.model
    num_terminals = model.num_terminals

    model.terminal_in_service = np.ones(num_terminals, dtype=bool)
    model.terminal_is_slack = np.zeros(num_terminals, dtype=bool)
    model.terminal_vn_kv = np.ones(num_terminals, dtype=np.float64) * -1
    model.terminal_is_ideally_grounded = np.zeros(num_terminals, dtype=bool)

    for (bus, phase), vn_kv, in_service, grounded, r_ohm, x_ohm in zip(
            net.bus.index.values,
            net.bus["vn_kv"].values,
            net.bus["in_service"].values,
            net.bus["grounded"].values,
            net.bus["grounding_r_ohm"].values,
            net.bus["grounding_x_ohm"].values):
        terminal = get_bus_terminal(model, bus, phase)
        model.terminal_vn_kv[terminal] = vn_kv
        model.terminal_in_service[terminal] = in_service

        if grounded and in_service and abs(r_ohm + 1j*x_ohm) == 0:
           model.terminal_is_ideally_grounded[terminal] = True

    eg_connections = deal_with_ext_grid(net)

    C = scipy.sparse.lil_matrix((net.model.num_terminals, net.model.num_terminals), dtype=np.int8)
    S = scipy.sparse.lil_matrix((net.model.num_terminals, net.model.num_terminals), dtype=np.int8)
    for (f, t) in eg_connections:
        S[f, t] = 2
        S[t, f] = 2

    line_from_set = set()
    line = net.line
    for (idx, _), fb, fp, tb, tp, in_service in zip(line.index, line["from_bus"].values, line["from_phase"].values,
                                                    line["to_bus"].values, line["to_phase"].values, line["in_service"].values):
        if not in_service:
            continue
        line_from_set.add((idx, fb))

        bus_from_terminal = get_bus_terminal(model, fb, fp)
        line_from_terminal = get_line_from_terminal(model, idx, fp)
        bus_to_terminal = get_bus_terminal(model, tb, tp)
        line_to_terminal = get_line_to_terminal(model, idx, tp)

        #model.terminal_vn_kv[line_from_terminal] = model.terminal_vn_kv[bus_from_terminal]
        #model.terminal_vn_kv[line_to_terminal] = model.terminal_vn_kv[bus_to_terminal]

        # connect terminals of lines to buses
        if model.terminal_in_service[bus_from_terminal]:
            C[bus_from_terminal, line_from_terminal] = 1
            C[line_from_terminal, bus_from_terminal] = 1
        if model.terminal_in_service[bus_to_terminal]:
            C[bus_to_terminal, line_to_terminal] = 1
            C[line_to_terminal, bus_to_terminal] = 1

        S[line_from_terminal, line_to_terminal] = 1
        S[line_to_terminal, line_from_terminal] = 1

    trafo = net.trafo1ph
    for (idx, bus, circ), fp, tp, in_service in zip(trafo.index, trafo["from_phase"].values,
                                            trafo["to_phase"].values, trafo["in_service"].values):
        if not in_service:
            continue
        t1 = get_bus_terminal(model, bus, fp)
        t2 = get_bus_terminal(model, bus, tp)
        S[t1, t2] = S[t2, t1] = 3
    # 
    # for (idx, bus, circ), fb, fp, tb, tp in zip(trafo.index, trafo["from_bus"], line["from_phase"], line["to_bus"], line["to_phase"]):
    #     C[bus_lookup_pos(model, fb, fp), line_lookup_pos(model, idx, fp, 1)] = 1
    #     C[line_lookup_pos(model, idx, fp, 1), bus_lookup_pos(model, fb, fp)] = 1
    #     C[bus_lookup_pos(model, tb, tp), line_lookup_pos(model, idx, tp, 0)] = 1
    #     C[line_lookup_pos(model, idx, tp, 0), bus_lookup_pos(model, tb, tp)] = 1

    switch = net.switch
    for bus, phase, element, et, closed in zip(switch["bus"], switch["phase"], switch["element"], switch["et"], switch["closed"]):
        if et == "b" and closed:
            bt1, bt2 = get_bus_terminal(model, bus, phase), get_bus_terminal(model, element, phase)
            C[bt1, bt2] = C[bt2, bt1] = 1
        if et == "l" and not closed:
            bt = get_bus_terminal(model, bus, phase)
            is_from_side = (element, bus) in line_from_set
            if is_from_side:
                lt = get_line_from_terminal(model, element, phase)
            else:
                lt = get_line_to_terminal(model, element, phase)
            C[bt, lt] = C[lt, bt] = 0

    load = net.asymmetric_load
    for (lidx, circuit), bus, fp, tp, in_service in zip(load.index, load["bus"], load["from_phase"], load["to_phase"], load["in_service"]):
        if not in_service:
            continue

        bus_from_terminal = get_bus_terminal(model, bus, fp)
        bus_to_terminal = get_bus_terminal(model, bus, tp)

        load_from_terminal = get_load_from_terminal(model, lidx, circuit)
        load_to_terminal = get_load_to_terminal(model, lidx, circuit)

        C[bus_from_terminal, load_from_terminal] = C[load_from_terminal, bus_from_terminal] = 1
        C[bus_to_terminal, load_to_terminal] = C[load_to_terminal, bus_to_terminal] = 1
        S[load_from_terminal, load_to_terminal] = 1

    sgen = net.asymmetric_sgen
    for (lidx, circuit), bus, fp, tp, in_service in zip(sgen.index, sgen["bus"], sgen["from_phase"], sgen["to_phase"], sgen["in_service"]):
        if not in_service:
            continue

        bus_from_terminal = get_bus_terminal(model, bus, fp)
        bus_to_terminal = get_bus_terminal(model, bus, tp)

        sgen_from_terminal = get_sgen_from_terminal(model, lidx, circuit)
        sgen_to_terminal = get_sgen_to_terminal(model, lidx, circuit)

        C[bus_from_terminal, sgen_from_terminal] = C[sgen_from_terminal, bus_from_terminal] = 1
        C[bus_to_terminal, sgen_to_terminal] = C[sgen_to_terminal, bus_to_terminal] = 1
        S[sgen_from_terminal, sgen_to_terminal] = 1

    if "asymmetric_shunt" in net:
        shunt = net.asymmetric_shunt
        for (lidx, circuit), bus, fp, tp, in_service in zip(shunt.index, shunt["bus"], shunt["from_phase"], shunt["to_phase"], shunt["in_service"]):
            if not in_service:
                continue

            bus_from_terminal = get_bus_terminal(model, bus, fp)
            bus_to_terminal = get_bus_terminal(model, bus, tp)

            sgen_from_terminal = get_shunt_from_terminal(model, lidx, circuit)
            sgen_to_terminal = get_shunt_to_terminal(model, lidx, circuit)

            C[bus_from_terminal, sgen_from_terminal] = C[sgen_from_terminal, bus_from_terminal] = 1
            C[bus_to_terminal, sgen_to_terminal] = C[sgen_to_terminal, bus_to_terminal] = 1
            S[sgen_from_terminal, sgen_to_terminal] = 1

    C = C.tocsc()

    ct = np.unique((C + S).nonzero()[0])

    net.model.C = C
    net.model.S = S
    net.model.terminal_to_y_lookup = find_islands(model.num_terminals, ct, C.indptr, C.indices)
    net.model.y_size = np.max(net.model.terminal_to_y_lookup) + 1
    net.bus["terminal"] = [get_bus_terminal(model, b, p) for b, p in net.bus.index]
    net.bus["y_index"] = [net.model.terminal_to_y_lookup[get_bus_terminal(model, b, p)]
                          for b, p in net.bus.index]

    #TODO is this really so compicated ?
    Ibase_y = np.zeros(net.model.y_size)
    model.y_fixed_voltage = np.ones((model.y_size, 1), dtype=complex) * -1

    for terminal_idx in range(net.model.num_terminals):
        y_idx = net.model.terminal_to_y_lookup[terminal_idx]
        if y_idx != -1:
            vn_kv = net.model.terminal_vn_kv[terminal_idx]
            if vn_kv > -1:
                Ibase_y[y_idx] = net.sn_mva * 1e6 / ( vn_kv * 1e3 / np.sqrt(3))
            if model.terminal_is_ideally_grounded[terminal_idx]:
                model.y_fixed_voltage[y_idx] = 0
    net.model.Ibase_y = Ibase_y


def _initialize_model(net, debug_level=0):
    """
        Initialize the network model including passive elements' admittance matrices.

        INPUT:
            **net** - The pandapower-like format network where elements are defined
            
            **sn_mva** (float, 1) - The system's base power in [MVA]. Default is 1.

            **rho_ohmm** (float, 100) - Soil resistivity in [Ohm*m]. Used in Carson-Clem formulation for \
            "configuration" line std types (if any).

            **f_hz** (float, 50) - System's frequency. Default is 50.

    """
    from .make_Y_elements import _make_yground, _make_ynet, _make_ysource ,_make_ytrafo, _make_yshunt, _make_yswitch
    net.model = DottedDict()
    net.model.debug_level = debug_level

    build_y_index(net)

    # Line standard types
    net.model.line_std_types = dict()
    # Source's admittance matrix
    _make_ysource(net)
    # Generalized transformers admittance matrix
    _make_ytrafo(net)
    # Branches admittance matrix:
    _make_ynet(net)
    # Earth connections admittance matrix:
    _make_yground(net)
    # Shunt admittance matrix
    _make_yshunt(net)
    _make_yswitch(net)
