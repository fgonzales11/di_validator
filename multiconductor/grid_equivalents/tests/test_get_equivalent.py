from __future__ import annotations

from copy import deepcopy
import pickle

import numpy as np
import pytest

import multiconductor as mc
from multiconductor.grid_equivalents import TerminalRef, get_equivalent, solve_equivalent
from multiconductor.pycci.std_types import create_std_type


def _build_three_bus_chain(with_internal_load: bool = False):
    net = mc.create_empty_network(sn_mva=1.0, add_stdtypes=False)
    create_std_type(
        net,
        {
            "max_i_ka": [1.0],
            "r_1_ohm_per_km": [1.0],
            "x_1_ohm_per_km": [0.0],
            "g_1_us_per_km": [0.0],
            "b_1_us_per_km": [0.0],
        },
        name="unit_line",
        element="matrix",
    )
    buses = [mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,)) for _ in range(3)]
    mc.create_ext_grid(net, bus=buses[0], from_phase=1, to_phase=0, vm_pu=1.0, va_degree=0.0, r_ohm=0.0, x_ohm=0.0)
    mc.create_line(net, "unit_line", "matrix", buses[0], [1], buses[1], [1], 1.0)
    mc.create_line(net, "unit_line", "matrix", buses[1], [1], buses[2], [1], 1.0)
    if with_internal_load:
        mc.create_asymmetric_load(
            net,
            bus=buses[1],
            from_phase=1,
            to_phase=0,
            p_mw=0.1,
            q_mvar=0.02,
            const_z_percent_p=0.0,
            const_i_percent_p=0.0,
            const_z_percent_q=0.0,
            const_i_percent_q=0.0,
        )
    return net


def _build_four_wire_chain():
    net = mc.create_empty_network(sn_mva=1.0, add_stdtypes=False)
    create_std_type(
        net,
        {
            "max_i_ka": [1.0, 1.0],
            "r_1_ohm_per_km": [0.40, 0.12],
            "r_2_ohm_per_km": [0.12, 0.35],
            "x_1_ohm_per_km": [0.16, 0.05],
            "x_2_ohm_per_km": [0.05, 0.14],
            "g_1_us_per_km": [0.0, 0.0],
            "g_2_us_per_km": [0.0, 0.0],
            "b_1_us_per_km": [0.0, 0.0],
            "b_2_us_per_km": [0.0, 0.0],
        },
        name="phase_neutral",
        element="matrix",
    )
    buses = [mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,)) for _ in range(3)]
    mc.create_ext_grid(
        net,
        bus=buses[0],
        from_phase=1,
        to_phase=0,
        vm_pu=1.0,
        va_degree=0.0,
        r_ohm=0.0,
        x_ohm=0.0,
    )
    mc.create_line(net, "phase_neutral", "matrix", buses[0], [1, 0], buses[1], [1, 0], 1.0)
    mc.create_line(net, "phase_neutral", "matrix", buses[1], [1, 0], buses[2], [1, 0], 1.0)
    return net


def _build_nonlinear_boundary_network():
    net = mc.create_empty_network(sn_mva=1.0, add_stdtypes=False)
    create_std_type(
        net,
        {
            "max_i_ka": [1.0],
            "r_1_ohm_per_km": [0.4],
            "x_1_ohm_per_km": [0.1],
            "g_1_us_per_km": [0.0],
            "b_1_us_per_km": [0.0],
        },
        name="zip_line",
        element="matrix",
    )
    buses = [mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,)) for _ in range(3)]
    mc.create_ext_grid(net, buses[0], 1, 0, 1.0, 0.0, 0.0, 0.0)
    mc.create_line(net, "zip_line", "matrix", buses[0], [1], buses[1], [1], 1.0)
    mc.create_line(net, "zip_line", "matrix", buses[1], [1], buses[2], [1], 1.0)
    mc.create_asymmetric_load(
        net,
        bus=buses[1],
        from_phase=1,
        to_phase=0,
        p_mw=0.12,
        q_mvar=0.03,
        const_z_percent_p=20.0,
        const_i_percent_p=30.0,
        const_z_percent_q=20.0,
        const_i_percent_q=30.0,
    )
    mc.create_asymmetric_load(
        net,
        bus=buses[2],
        from_phase=1,
        to_phase=0,
        p_mw=0.05,
        q_mvar=0.01,
        const_z_percent_p=10.0,
        const_i_percent_p=20.0,
        const_z_percent_q=10.0,
        const_i_percent_q=20.0,
    )
    return net


def _boundary_current_for_terminals(net, terminals):
    from multiconductor.grid_equivalents import core

    y_pass = (
        net.model.Y_tran
        + net.model.Y_network
        + net.model.Y_ground
        + net.model.Y_source
        + net.model.Y_shunt
        + net.model.Y_switch
    ).tocsr()
    injection = core._frozen_injection_vector(net)
    lookup = net.model.terminal_to_y_lookup
    boundary_y = np.array(
        [lookup[4 * terminal.bus + terminal.phase] for terminal in terminals],
        dtype=int,
    )
    return core._boundary_current_from_full_system(
        y_pass=y_pass,
        injection=injection,
        voltage=np.asarray(net.model.E).reshape(-1),
        boundary_y=boundary_y,
    )


def test_get_equivalent_matches_independent_schur_reduction_for_passive_chain():
    net = _build_three_bus_chain(with_internal_load=False)
    equivalent = get_equivalent(
        net,
        eq_type="multiport",
        boundary_buses=[TerminalRef(0, 0), TerminalRef(0, 1), TerminalRef(2, 0), TerminalRef(2, 1)],
    )

    work = deepcopy(net)
    mc.run_pf(work)
    y_pass = (
        work.model.Y_tran
        + work.model.Y_network
        + work.model.Y_ground
        + work.model.Y_source
        + work.model.Y_shunt
        + work.model.Y_switch
    ).toarray()
    lookup = work.model.terminal_to_y_lookup
    boundary_y = np.array(
        [
            lookup[0],
            lookup[1],
            lookup[8],
            lookup[9],
        ],
        dtype=int,
    )
    boundary_y = np.unique(boundary_y[boundary_y != -1])
    fixed = np.flatnonzero(np.isfinite(work.model.y_fixed_voltage.reshape(-1)) & (work.model.y_fixed_voltage.reshape(-1) != -1))
    free_internal = np.array(sorted(set(range(work.model.y_size)) - set(boundary_y.tolist()) - set(fixed.tolist())), dtype=int)
    fixed_internal = np.array(sorted(set(fixed.tolist()) - set(boundary_y.tolist())), dtype=int)
    fixed_voltage = work.model.y_fixed_voltage.reshape(-1)[fixed_internal]
    y_bb = y_pass[np.ix_(boundary_y, boundary_y)]
    y_br = y_pass[np.ix_(boundary_y, free_internal)]
    y_rb = y_pass[np.ix_(free_internal, boundary_y)]
    y_rr = y_pass[np.ix_(free_internal, free_internal)]
    y_bf = y_pass[np.ix_(boundary_y, fixed_internal)]
    y_rf = y_pass[np.ix_(free_internal, fixed_internal)]
    expected_y = y_bb - y_br @ np.linalg.solve(y_rr, y_rb)
    expected_i = -(y_bf @ fixed_voltage) - y_br @ np.linalg.solve(y_rr, -(y_rf @ fixed_voltage))

    assert np.allclose(equivalent.model.y_eq, expected_y)
    assert np.allclose(equivalent.model.i_offset, expected_i)
    assert equivalent.model.validation.current_mismatch_max < 1e-10
    assert equivalent.model.validation.passed is True
    assert len(equivalent.net.grid_equivalent) == 1


@pytest.mark.integration
def test_four_wire_mutual_neutral_passive_equivalent_is_exact():
    net = _build_four_wire_chain()
    terminals = [
        TerminalRef(0, 0),
        TerminalRef(0, 1),
        TerminalRef(2, 0),
        TerminalRef(2, 1),
    ]
    equivalent = get_equivalent(net, eq_type="multiport", boundary_buses=terminals)

    work = deepcopy(net)
    mc.run_pf(work)
    y_pass = (
        work.model.Y_tran
        + work.model.Y_network
        + work.model.Y_ground
        + work.model.Y_source
        + work.model.Y_shunt
        + work.model.Y_switch
    ).toarray()
    lookup = work.model.terminal_to_y_lookup
    boundary_y = np.array([lookup[0], lookup[1], lookup[8], lookup[9]], dtype=int)
    fixed = np.flatnonzero(
        np.isfinite(work.model.y_fixed_voltage.reshape(-1))
        & (work.model.y_fixed_voltage.reshape(-1) != -1)
    )
    free_internal = np.array(
        sorted(set(range(work.model.y_size)) - set(boundary_y.tolist()) - set(fixed.tolist())),
        dtype=int,
    )
    fixed_internal = np.array(
        sorted(set(fixed.tolist()) - set(boundary_y.tolist())),
        dtype=int,
    )
    fixed_voltage = work.model.y_fixed_voltage.reshape(-1)[fixed_internal]
    y_bb = y_pass[np.ix_(boundary_y, boundary_y)]
    y_br = y_pass[np.ix_(boundary_y, free_internal)]
    y_rb = y_pass[np.ix_(free_internal, boundary_y)]
    y_rr = y_pass[np.ix_(free_internal, free_internal)]
    y_bf = y_pass[np.ix_(boundary_y, fixed_internal)]
    y_rf = y_pass[np.ix_(free_internal, fixed_internal)]
    expected_y = y_bb - y_br @ np.linalg.solve(y_rr, y_rb)
    expected_i = -(y_bf @ fixed_voltage) - y_br @ np.linalg.solve(y_rr, -(y_rf @ fixed_voltage))

    assert np.allclose(equivalent.model.y_eq, expected_y, atol=1e-10)
    assert np.allclose(equivalent.model.i_offset, expected_i, atol=1e-10)
    assert equivalent.model.validation.current_mismatch_max < 1e-10

    perturb_voltage = np.array([0.0 + 0.0j, 1.05 + 0.0j, 0.0 + 0.0j, 0.95 + 0.0j])
    expected_current = expected_y @ perturb_voltage - expected_i
    perturbed = solve_equivalent(
        equivalent,
        boundary_conditions={
            "voltage": {
                TerminalRef(0, 0): 0.0 + 0.0j,
                TerminalRef(0, 1): perturb_voltage[1],
                TerminalRef(2, 0): 0.0 + 0.0j,
                TerminalRef(2, 1): perturb_voltage[3],
            }
        },
    )
    got_current = (
        perturbed.boundary_table["i_real_pu"].to_numpy()
        + 1j * perturbed.boundary_table["i_imag_pu"].to_numpy()
    )
    assert np.allclose(got_current, expected_current, atol=1e-10)


def test_get_equivalent_accepts_bus_inputs_and_preserves_source_network():
    net = _build_three_bus_chain(with_internal_load=True)
    original_bus = net.bus.copy(deep=True)
    equivalent = get_equivalent(net, eq_type="ward", boundary_buses=[0, 2])

    assert net.bus.equals(original_bus)
    assert equivalent.model.eq_type == "ward"
    assert equivalent.model.representation == "ward"
    assert sorted({terminal.bus for terminal in equivalent.model.boundary_terminals}) == [0, 2]
    assert equivalent.model.validation.current_mismatch_max < 1e-6
    assert equivalent.net.grid_equivalent.iloc[0]["representation"] == "ward"
    assert equivalent.model.provenance["compiled_representation"] == "multiport"
    assert equivalent.net.grid_equivalent.iloc[0]["provenance"]["settings"]["eq_type"] == "ward"
    assert equivalent.net.grid_equivalent.iloc[0]["provenance"]["network_hash"].startswith("sha256:")


def test_solve_equivalent_reproduces_base_point_and_voltage_perturbations():
    net = _build_three_bus_chain(with_internal_load=False)
    equivalent = get_equivalent(net, eq_type="multiport", boundary_buses=[0, 2])

    base = solve_equivalent(equivalent)
    assert np.isfinite(base.boundary_table.to_numpy()).all()
    assert base.residual_max == 0.0
    assert np.allclose(base.boundary_table["vr_pu"].to_numpy(), [1.0, 1.0])

    y_eq = equivalent.model.y_eq
    base_voltage = equivalent.model.base_voltage
    perturb_voltage = np.array([1.05 + 0j, 0.95 + 0j], dtype=np.complex128)
    expected_current = y_eq @ perturb_voltage - equivalent.model.i_offset

    perturbed = solve_equivalent(
        equivalent,
        boundary_conditions={
            "voltage": {
                TerminalRef(0, 1): perturb_voltage[0],
                TerminalRef(2, 1): perturb_voltage[1],
            }
        },
    )

    assert np.allclose(perturbed.boundary_table["vr_pu"].to_numpy(), perturb_voltage.real)
    assert np.allclose(perturbed.boundary_table["i_real_pu"].to_numpy(), expected_current.real)
    assert np.allclose(perturbed.boundary_table["i_imag_pu"].to_numpy(), expected_current.imag)
    assert not np.allclose(perturbed.boundary_table["vr_pu"].to_numpy(), base_voltage.real)


def test_solve_equivalent_roundtrips_through_reduced_net_payload():
    net = _build_three_bus_chain(with_internal_load=True)
    equivalent = get_equivalent(net, eq_type="xward", boundary_buses=[0, 2])

    serialized = pickle.loads(pickle.dumps(equivalent.net, protocol=pickle.HIGHEST_PROTOCOL))
    solved = solve_equivalent(
        serialized,
        boundary_conditions={
            "voltage": {
                (0, 1): {"vm_pu": 1.0, "va_degree": 0.0},
                (2, 1): (0.99, 0.01),
            }
        },
    )

    assert solved.model.eq_type == "xward"
    assert solved.model.representation == "xward"
    assert np.isfinite(solved.boundary_table.to_numpy()).all()
    assert len(solved.net.res_bus_equivalent) == len(solved.model.boundary_terminals)
    assert {"p_mw", "q_mvar", "i_ka"} <= set(solved.net.res_bus_equivalent.columns)
    assert solved.net.res_grid_equivalent.at[0, "residual_max"] < 1e-6
    assert solved.model.provenance["compiled_representation"] == "multiport"


def test_grid_equivalent_payload_uses_canonical_serializable_fields():
    net = _build_three_bus_chain(with_internal_load=True)
    equivalent = get_equivalent(net, eq_type="rei", boundary_buses=[0, 2])
    row = equivalent.net.grid_equivalent.iloc[0]

    assert set(equivalent.net.grid_equivalent.columns) == {
        "name",
        "eq_type",
        "representation",
        "boundary_terminals",
        "y_eq",
        "i_offset",
        "base_voltage",
        "validation",
        "provenance",
    }
    assert row["representation"] == "rei"
    assert row["provenance"]["compiled_representation"] == "multiport"
    assert row["boundary_terminals"][0] == {"bus": 0, "phase": 1, "side": None}
    assert {"real", "imag", "shape"} <= set(row["y_eq"])
    assert {"real", "imag", "shape"} <= set(row["i_offset"])
    assert {"real", "imag", "shape"} <= set(row["base_voltage"])


def test_solve_equivalent_reads_legacy_split_complex_payload():
    net = _build_three_bus_chain(with_internal_load=True)
    equivalent = get_equivalent(net, eq_type="xward", boundary_buses=[0, 2])
    legacy_net = pickle.loads(pickle.dumps(equivalent.net, protocol=pickle.HIGHEST_PROTOCOL))
    row = legacy_net.grid_equivalent.iloc[0].copy()
    legacy_net.grid_equivalent = legacy_net.grid_equivalent.drop(
        columns=["y_eq", "i_offset", "base_voltage"]
    )
    legacy_net.grid_equivalent["boundary_terminals"] = [[(0, 1), (2, 1)]]
    legacy_net.grid_equivalent["y_eq_real"] = [row["y_eq"]["real"]]
    legacy_net.grid_equivalent["y_eq_imag"] = [row["y_eq"]["imag"]]
    legacy_net.grid_equivalent["i_offset_real"] = [row["i_offset"]["real"]]
    legacy_net.grid_equivalent["i_offset_imag"] = [row["i_offset"]["imag"]]
    legacy_net.grid_equivalent["base_voltage_real"] = [row["base_voltage"]["real"]]
    legacy_net.grid_equivalent["base_voltage_imag"] = [row["base_voltage"]["imag"]]

    solved = solve_equivalent(legacy_net)

    assert solved.model.eq_type == "xward"
    assert solved.model.representation == "xward"
    assert np.isfinite(solved.boundary_table.to_numpy()).all()


def test_validation_false_returns_metrics_for_invalid_payload_and_true_raises(monkeypatch):
    net = _build_three_bus_chain(with_internal_load=False)
    from multiconductor.grid_equivalents import core

    original_reduce = core._reduce_system

    def inject_bad_reduce_system(**kwargs):
        y_eq, i_offset = original_reduce(**kwargs)
        y_eq = y_eq.copy()
        y_eq[0, 0] = np.nan
        return y_eq, i_offset

    monkeypatch.setattr(core, "_reduce_system", inject_bad_reduce_system)

    with pytest.raises(ValueError, match="Equivalent validation failed: non-finite entries"):
        get_equivalent(net, eq_type="multiport", boundary_buses=[0, 2], validate=True)

    equivalent = get_equivalent(net, eq_type="multiport", boundary_buses=[0, 2], validate=False)
    assert equivalent.model.validation.is_finite is False
    assert equivalent.model.validation.passed is False
    assert equivalent.net.grid_equivalent.iloc[0]["validation"]["passed"] is False


@pytest.mark.unit
def test_get_equivalent_rejects_overlapping_and_ambiguous_partitions():
    net = _build_three_bus_chain(with_internal_load=False)

    with pytest.raises(ValueError, match="internal_buses and external_buses overlap: \\[1\\]"):
        get_equivalent(
            net,
            eq_type="multiport",
            boundary_buses=[0, 2],
            internal_buses=[1],
            external_buses=[1],
        )

    with pytest.raises(ValueError, match="boundary_buses cannot also be internal_buses: \\[0\\]"):
        get_equivalent(
            net,
            eq_type="multiport",
            boundary_buses=[0, 2],
            internal_buses=[0, 1],
        )

    with pytest.raises(ValueError, match="boundary_buses cannot also be external_buses: \\[2\\]"):
        get_equivalent(
            net,
            eq_type="multiport",
            boundary_buses=[0, 2],
            external_buses=[2],
        )


@pytest.mark.integration
def test_get_equivalent_rejects_islanded_boundary_from_open_switch():
    net = mc.create_empty_network(sn_mva=1.0, add_stdtypes=False)
    create_std_type(
        net,
        {
            "max_i_ka": [1.0],
            "r_1_ohm_per_km": [1.0],
            "x_1_ohm_per_km": [0.0],
            "g_1_us_per_km": [0.0],
            "b_1_us_per_km": [0.0],
        },
        name="unit_line",
        element="matrix",
    )
    bus0 = mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,))
    bus1 = mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,))
    bus2 = mc.create_bus(net, vn_kv=11.0, num_phases=4, grounded_phases=(0,))
    mc.create_ext_grid(net, bus0, 1, 0, 1.0, 0.0, 0.0, 0.0)
    mc.create_line(net, "unit_line", "matrix", bus0, [1], bus1, [1], 1.0)
    mc.create_switch(net, bus1, [1], bus2, "b", closed=False)

    with pytest.raises(ValueError, match="Boundary terminals do not map to any connected Y nodes"):
        get_equivalent(net, eq_type="multiport", boundary_buses=[TerminalRef(bus2, 1)])


@pytest.mark.integration
def test_nonlinear_boundary_perturbations_meet_ge_certification_targets():
    net = _build_nonlinear_boundary_network()
    boundary = [TerminalRef(0, 1)]

    equivalent = get_equivalent(net, eq_type="multiport", boundary_buses=[0])
    restored = solve_equivalent(pickle.loads(pickle.dumps(equivalent.net, protocol=pickle.HIGHEST_PROTOCOL)))
    base = deepcopy(net)
    mc.run_pf(base)
    base_full_current = _boundary_current_for_terminals(base, boundary)
    base_equivalent_current = equivalent.model.y_eq @ equivalent.model.base_voltage - equivalent.model.i_offset
    restored_base_current = restored.model.y_eq @ restored.model.base_voltage - restored.model.i_offset
    assert np.allclose(restored_base_current, base_equivalent_current)
    base_relative_error = np.max(
        np.abs(base_equivalent_current - base_full_current)
        / np.maximum(np.abs(base_full_current), 1e-12)
    )
    assert base_relative_error <= 1e-4
    assert equivalent.model.validation.current_mismatch_max <= 1e-6

    max_relative_error = 0.0
    for vm_pu, va_degree in ((1.05, 0.0), (0.95, -2.0), (1.02, 1.0)):
        full = deepcopy(net)
        full.ext_grid.at[(0, 0), "vm_pu"] = vm_pu
        full.ext_grid.at[(0, 0), "va_degree"] = va_degree
        mc.run_pf(full)
        full_current = _boundary_current_for_terminals(full, boundary)
        boundary_voltage = np.array([vm_pu * np.exp(1j * np.deg2rad(va_degree))], dtype=np.complex128)
        equivalent_current = equivalent.model.y_eq @ boundary_voltage - equivalent.model.i_offset
        relative_error = np.max(
            np.abs(equivalent_current - full_current)
            / np.maximum(np.abs(full_current), 1e-12)
        )
        max_relative_error = max(max_relative_error, float(relative_error))

    assert max_relative_error <= 0.01, f"observed boundary-current relative error {max_relative_error:.4%}"
