"""Deterministic invariant tests for the multiconductor timeseries module.

This replaces the previous Hypothesis-based property tests with seeded,
repeatable parameter sets so the suite collects in environments without the
optional ``hypothesis`` dependency while preserving the same behavioral
contracts.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from multiconductor.timeseries.data_source import MC_CSVDataSource
from multiconductor.timeseries.output_writer import MC_OutputWriter
from multiconductor.timeseries.run_mc_timeseries import (
    _apply_time_step_data,
    _build_name_map,
)


def _missing_column_cases() -> list[list[str]]:
    required = list(MC_CSVDataSource.REQUIRED_COLUMNS)
    return [
        [required[0]],
        [required[-1]],
        required[2:4],
        [required[1], required[4], required[7]],
        required[:-1],
    ]


def _csv_round_trip_cases() -> list[list[dict[str, object]]]:
    cases: list[list[dict[str, object]]] = []
    for seed, n_timesteps, n_elements in ((7, 1, 1), (11, 2, 2), (23, 4, 3)):
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for t_index in range(n_timesteps):
            for e_index in range(n_elements):
                pq = rng.normal(loc=0.0, scale=25.0, size=6)
                rows.append(
                    {
                        "REPORTED_DTTM": f"TS_{seed}_{t_index}",
                        "NAME": f"E_{seed}_{e_index}",
                        "TYPE": "GROSS" if e_index % 2 == 0 else "GEN",
                        "MEASURE_VALUE_P_MW_APHASE": float(pq[0]),
                        "MEASURE_VALUE_P_MW_BPHASE": float(pq[1]),
                        "MEASURE_VALUE_P_MW_CPHASE": float(pq[2]),
                        "MEASURE_VALUE_Q_MVAR_APHASE": float(pq[3]),
                        "MEASURE_VALUE_Q_MVAR_BPHASE": float(pq[4]),
                        "MEASURE_VALUE_Q_MVAR_CPHASE": float(pq[5]),
                    }
                )
        cases.append(rows)
    return cases


def _timestamp_cases() -> list[list[str]]:
    return [
        ["TS_0001"],
        ["TS_0004", "TS_0001", "TS_0003"],
        ["TS_99", "TS_10", "TS_50", "TS_70", "TS_20"],
    ]


def _phase_value_cases() -> list[tuple[float, float, float, float, float, float]]:
    return [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
        (-5.5, 12.25, -18.75, 3.3, -4.4, 5.5),
        (1e-6, -1e-6, 2e-6, -3e-6, 4e-6, -5e-6),
        (125.0, -90.0, 45.5, -12.0, 8.0, 2.5),
    ]


def _res_bus_cases():
    cases = []
    for seed, bus_indices in ((5, [0]), (8, [0, 4]), (12, [1, 7, 11])):
        rng = np.random.default_rng(seed)
        bus_tuples = [(bus, phase) for bus in bus_indices for phase in (1, 2, 3)]
        bus_index = pd.MultiIndex.from_tuples(bus_tuples, names=["bus_idx", "phase"])
        bus_df = pd.DataFrame(index=bus_index, columns=["zone"], dtype=object)
        bus_df["zone"] = "default"
        res_bus = pd.DataFrame(
            {
                "vm_pu": rng.uniform(0.9, 1.1, size=len(bus_tuples)),
                "va_degree": rng.uniform(-15.0, 15.0, size=len(bus_tuples)),
            },
            index=bus_index,
        )
        cases.append((bus_indices, bus_df, res_bus))
    return cases


class _MockNet:
    """Minimal mock of a pandapowerNet for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def _make_asymmetric_table(name: str, bus: int = 0) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (0, 2)], names=["element_idx", "circuit"]
    )
    data = {
        "name": [name, name, name],
        "bus": [bus, bus, bus],
        "from_phase": [1, 2, 3],
        "to_phase": [0, 0, 0],
        "p_mw": [0.0, 0.0, 0.0],
        "q_mvar": [0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.mark.unit
@pytest.mark.parametrize("columns_to_remove", _missing_column_cases())
def test_property_1_missing_column_validation(tmp_path, columns_to_remove):
    """Missing required CSV columns are all named in the validation error."""
    remaining = [
        column
        for column in MC_CSVDataSource.REQUIRED_COLUMNS
        if column not in columns_to_remove
    ]
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text(",".join(remaining) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        MC_CSVDataSource(str(csv_path))

    error_msg = str(exc_info.value)
    for column in columns_to_remove:
        assert column in error_msg


@pytest.mark.unit
@pytest.mark.parametrize("rows", _csv_round_trip_cases())
def test_property_2_csv_data_round_trip(tmp_path, rows):
    """Every parsed row is returned unchanged at its original time step."""
    csv_path = tmp_path / "round_trip.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    data_source = MC_CSVDataSource(str(csv_path))

    for row in rows:
        values = data_source.get_time_step_values(row["REPORTED_DTTM"])
        payload = values[(row["NAME"], row["TYPE"])]
        assert math.isclose(payload["p_mw_a"], row["MEASURE_VALUE_P_MW_APHASE"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(payload["p_mw_b"], row["MEASURE_VALUE_P_MW_BPHASE"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(payload["p_mw_c"], row["MEASURE_VALUE_P_MW_CPHASE"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(payload["q_mvar_a"], row["MEASURE_VALUE_Q_MVAR_APHASE"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(payload["q_mvar_b"], row["MEASURE_VALUE_Q_MVAR_BPHASE"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(payload["q_mvar_c"], row["MEASURE_VALUE_Q_MVAR_CPHASE"], rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("timestamps", _timestamp_cases())
def test_property_3_time_step_order_preservation(tmp_path, timestamps):
    """Unique time steps preserve order of first appearance in the CSV."""
    rows = [
        {
            "REPORTED_DTTM": ts,
            "NAME": "ELEM1",
            "TYPE": "GROSS",
            "MEASURE_VALUE_P_MW_APHASE": 1.0,
            "MEASURE_VALUE_P_MW_BPHASE": 2.0,
            "MEASURE_VALUE_P_MW_CPHASE": 3.0,
            "MEASURE_VALUE_Q_MVAR_APHASE": 0.1,
            "MEASURE_VALUE_Q_MVAR_BPHASE": 0.2,
            "MEASURE_VALUE_Q_MVAR_CPHASE": 0.3,
        }
        for ts in timestamps
    ]
    csv_path = tmp_path / "time_steps.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    data_source = MC_CSVDataSource(str(csv_path))

    assert data_source.get_time_steps() == timestamps


@pytest.mark.unit
@pytest.mark.parametrize("phase_values", _phase_value_cases())
def test_property_4_gross_element_mapping(phase_values):
    """GROSS rows map to asymmetric_load with the correct phase ordering."""
    p_a, p_b, p_c, q_a, q_b, q_c = phase_values
    element_name = "LOAD_TEST"
    net = _MockNet(
        asymmetric_load=_make_asymmetric_table(element_name),
        asymmetric_sgen=_make_asymmetric_table("UNUSED_SGEN"),
    )

    _apply_time_step_data(
        net,
        {
            (element_name, "GROSS"): {
                "p_mw_a": p_a,
                "p_mw_b": p_b,
                "p_mw_c": p_c,
                "q_mvar_a": q_a,
                "q_mvar_b": q_b,
                "q_mvar_c": q_c,
            }
        },
        _build_name_map(net.asymmetric_load),
        _build_name_map(net.asymmetric_sgen),
    )

    table = net.asymmetric_load
    assert table.at[(0, 0), "p_mw"] == p_a
    assert table.at[(0, 0), "q_mvar"] == q_a
    assert table.at[(0, 1), "p_mw"] == p_b
    assert table.at[(0, 1), "q_mvar"] == q_b
    assert table.at[(0, 2), "p_mw"] == p_c
    assert table.at[(0, 2), "q_mvar"] == q_c


@pytest.mark.unit
@pytest.mark.parametrize("phase_values", _phase_value_cases())
def test_property_5_gen_element_mapping(phase_values):
    """GEN rows map to asymmetric_sgen with the correct phase ordering."""
    p_a, p_b, p_c, q_a, q_b, q_c = phase_values
    element_name = "GEN_TEST"
    net = _MockNet(
        asymmetric_load=_make_asymmetric_table("UNUSED_LOAD"),
        asymmetric_sgen=_make_asymmetric_table(element_name),
    )

    _apply_time_step_data(
        net,
        {
            (element_name, "GEN"): {
                "p_mw_a": p_a,
                "p_mw_b": p_b,
                "p_mw_c": p_c,
                "q_mvar_a": q_a,
                "q_mvar_b": q_b,
                "q_mvar_c": q_c,
            }
        },
        _build_name_map(net.asymmetric_load),
        _build_name_map(net.asymmetric_sgen),
    )

    table = net.asymmetric_sgen
    assert table.at[(0, 0), "p_mw"] == p_a
    assert table.at[(0, 0), "q_mvar"] == q_a
    assert table.at[(0, 1), "p_mw"] == p_b
    assert table.at[(0, 1), "q_mvar"] == q_b
    assert table.at[(0, 2), "p_mw"] == p_c
    assert table.at[(0, 2), "q_mvar"] == q_c


@pytest.mark.unit
@pytest.mark.parametrize("data", _res_bus_cases())
def test_property_6_output_writer_voltage_extraction(data):
    """Writer buffers exactly the per-phase bus voltages from res_bus."""
    bus_indices, bus_df, res_bus = data
    net = _MockNet(bus=bus_df, res_bus=res_bus, CIRCUIT_KEY="TEST_CKT")

    writer = MC_OutputWriter(net, ["t0"])
    writer.record_time_step(net, "t0", converged=True)

    for row_index, bus_idx in enumerate(sorted(bus_indices)):
        assert writer._voltage_buffer[row_index, writer._VM_A] == res_bus.loc[(bus_idx, 1), "vm_pu"]
        assert writer._voltage_buffer[row_index, writer._VA_A] == res_bus.loc[(bus_idx, 1), "va_degree"]
        assert writer._voltage_buffer[row_index, writer._VM_B] == res_bus.loc[(bus_idx, 2), "vm_pu"]
        assert writer._voltage_buffer[row_index, writer._VA_B] == res_bus.loc[(bus_idx, 2), "va_degree"]
        assert writer._voltage_buffer[row_index, writer._VM_C] == res_bus.loc[(bus_idx, 3), "vm_pu"]
        assert writer._voltage_buffer[row_index, writer._VA_C] == res_bus.loc[(bus_idx, 3), "va_degree"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("circuit_key", "n_steps"),
    [("A", 1), ("FEEDER_ALPHA", 3), ("ZONE_TEST", 5)],
)
def test_property_7_circuit_key_invariant(circuit_key, n_steps):
    """Every output row carries the exact network CIRCUIT_KEY."""
    bus_index = pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2), (0, 3)], names=["bus_idx", "phase"]
    )
    bus_df = pd.DataFrame(index=bus_index, columns=["zone"], dtype=object)
    bus_df["zone"] = "default"
    res_bus = pd.DataFrame(
        {"vm_pu": [1.0, 1.0, 1.0], "va_degree": [0.0, -120.0, 120.0]},
        index=bus_index,
    )
    net = _MockNet(bus=bus_df, res_bus=res_bus, CIRCUIT_KEY=circuit_key)
    time_steps = [f"t{i}" for i in range(n_steps)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        output_path = handle.name

    try:
        writer = MC_OutputWriter(net, time_steps, output_path=output_path)
        for time_step in time_steps:
            writer.record_time_step(net, time_step, converged=True)
        writer.finalize()

        result = pd.read_csv(output_path, dtype={"CIRCUIT_KEY": str})
        assert len(result) == n_steps
        assert all(result["CIRCUIT_KEY"] == circuit_key)
    finally:
        os.unlink(output_path)


@pytest.mark.unit
@pytest.mark.parametrize("n_buses", [1, 2, 5])
def test_property_8_non_convergence_nan(n_buses):
    """Non-converged time steps produce NaN voltage output for every bus."""
    bus_tuples = [(bus, phase) for bus in range(n_buses) for phase in (1, 2, 3)]
    bus_index = pd.MultiIndex.from_tuples(bus_tuples, names=["bus_idx", "phase"])
    bus_df = pd.DataFrame(index=bus_index, columns=["zone"], dtype=object)
    bus_df["zone"] = "default"
    res_bus = pd.DataFrame(
        {"vm_pu": [1.0] * len(bus_tuples), "va_degree": [0.0] * len(bus_tuples)},
        index=bus_index,
    )
    net = _MockNet(bus=bus_df, res_bus=res_bus, CIRCUIT_KEY="TEST_CKT")

    writer = MC_OutputWriter(net, ["t0"])
    writer.record_time_step(net, "t0", converged=False)

    for bus_index in range(n_buses):
        for column in range(6):
            assert np.isnan(writer._voltage_buffer[bus_index, column])
