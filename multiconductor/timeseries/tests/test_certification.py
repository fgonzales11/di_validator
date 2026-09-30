from __future__ import annotations

import importlib
import json
import os
import pickle
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

import multiconductor as mc
from multiconductor.pycci.cci_powerflow import LoadflowNotConverged, run_pf
from multiconductor.timeseries import (
    DFData,
    OutputWriter,
    ProfileBinding,
    ResultSpec,
    run_timeseries,
)

_runner_mod = importlib.import_module("multiconductor.timeseries.run_mc_timeseries")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_FEEDER_PATH = _REPO_ROOT / "networks" / "CKT_626_00490.pkl"
_REAL_FEEDER_ENV = "MC_TS_REAL_FEEDER_CERT"
_REAL_FEEDER_HOURS = 8760
_REAL_FEEDER_WARMUP_HOURS = 24
_REAL_FEEDER_CHUNK_SIZE = 168


def _process_peak_rss_bytes() -> int:
    """Return peak resident memory with platform-normalized byte units."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


class _MockNet:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def _make_mock_net(circuit_key: str = "CKT_CERT") -> _MockNet:
    load_index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (0, 2)], names=["element_idx", "circuit"]
    )
    asymmetric_load = pd.DataFrame(
        {
            "name": ["LOAD_1", "LOAD_1", "LOAD_1"],
            "bus": [0, 0, 0],
            "from_phase": [1, 2, 3],
            "to_phase": [0, 0, 0],
            "p_mw": [0.1, 0.2, 0.3],
            "q_mvar": [0.01, 0.02, 0.03],
        },
        index=load_index,
    )
    asymmetric_sgen = asymmetric_load.copy(deep=True)
    bus_index = pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2), (0, 3)], names=["bus_idx", "phase"]
    )
    bus = pd.DataFrame(index=bus_index)
    res_bus = pd.DataFrame(
        {
            "vm_pu": [1.0, 0.99, 0.98],
            "va_degree": [0.0, -120.0, 120.0],
        },
        index=bus_index,
    )
    return _MockNet(
        asymmetric_load=asymmetric_load,
        asymmetric_sgen=asymmetric_sgen,
        bus=bus,
        res_bus=res_bus,
        CIRCUIT_KEY=circuit_key,
    )


def _small_real_network():
    net = mc.create_empty_network(sn_mva=1.0)
    source = mc.create_bus(net, 20.0, grounding_r_ohm=0.001, name="Source")
    load_bus = mc.create_bus(net, 20.0, grounding_r_ohm=0.001, name="Load")
    mc.create_ext_grid(
        net,
        source,
        from_phase=range(1, 4),
        to_phase=0,
        vm_pu=1.0,
        va_degree=0.0,
        r_ohm=0.1,
        x_ohm=1.0,
        name="Slack",
    )
    mc.create_line(
        net,
        "UG1",
        "sequence",
        from_bus=source,
        from_phase=(1, 2, 3),
        to_bus=load_bus,
        to_phase=(1, 2, 3),
        length_km=1.0,
        name="Feeder",
    )
    mc.create_asymmetric_load(
        net,
        load_bus,
        from_phase=(1, 2, 3),
        to_phase=(0, 0, 0),
        p_mw=(0.10, 0.08, 0.12),
        q_mvar=(0.02, 0.015, 0.025),
        name="Load",
    )
    return net


def _load_real_feeder_net():
    with _REAL_FEEDER_PATH.open("rb") as handle:
        return pickle.load(handle)


def _real_feeder_load_elements(net, limit: int = 3) -> list[int]:
    elements = pd.unique(net.asymmetric_load.index.get_level_values(0)).tolist()
    return [int(element) for element in elements[:limit]]


def _scheduled_scale_frame(load_elements: list[int], n_steps: int) -> pd.DataFrame:
    rows: dict[str, list[object]] = {"TIME_STEP": list(range(n_steps))}
    for idx, _ in enumerate(load_elements):
        rows[f"scale_{idx}"] = [
            0.92
            + 0.08 * ((hour % 24) / 23.0)
            + 0.03 * (((hour // 24) % 30) / 29.0)
            + 0.01 * idx
            for hour in range(n_steps)
        ]
    return pd.DataFrame(rows)


def _real_feeder_bindings(load_elements: list[int]) -> list[ProfileBinding]:
    bindings: list[ProfileBinding] = []
    for idx, load_element in enumerate(load_elements):
        profile = f"scale_{idx}"
        bindings.extend(
            [
                ProfileBinding(
                    table="asymmetric_load",
                    variable="p_mw",
                    profile=profile,
                    element=load_element,
                    mode="scale",
                ),
                ProfileBinding(
                    table="asymmetric_load",
                    variable="q_mvar",
                    profile=profile,
                    element=load_element,
                    mode="scale",
                ),
            ]
        )
    return bindings


def _status_only_writer(net, time_steps: list[object], output_path: Path) -> OutputWriter:
    return OutputWriter(
        net,
        time_steps,
        result_specs=[],
        output_path=str(output_path),
        chunk_size=_REAL_FEEDER_CHUNK_SIZE,
    )


def _decode_index(raw: str):
    value = json.loads(raw)
    if isinstance(value, list):
        return tuple(value)
    return value


def _actual_rows(frame: pd.DataFrame, time_step: object) -> dict[tuple[str, object, str], float]:
    subset = frame.loc[frame["TIME_STEP"] == time_step]
    rows = {}
    for row in subset.itertuples(index=False):
        rows[(row.TABLE, _decode_index(row.INDEX), row.COLUMN)] = float(row.VALUE)
    return rows


def _expected_rows(net, result_specs: list[ResultSpec]) -> dict[tuple[str, object, str], float]:
    expected = {}
    for spec in result_specs:
        table = getattr(net, spec.table)
        label = spec.alias or spec.column
        for index, value in table[spec.column].items():
            expected[(spec.table, index, label)] = float(value)
    return expected


def _apply_phase_profiles(net, step_values):
    for phase in (1, 2, 3):
        circuit = phase - 1
        net.asymmetric_load.at[(0, circuit), "p_mw"] = step_values["load_abs"][phase]
        net.asymmetric_load.at[(0, circuit), "q_mvar"] *= step_values["load_scale"][phase]


def test_cert_real_solver_steps_match_snapshots_with_phase_bindings_and_custom_results(
    tmp_path,
):
    net = _small_real_network()
    baseline_p = net.asymmetric_load["p_mw"].copy(deep=True)
    baseline_q = net.asymmetric_load["q_mvar"].copy(deep=True)
    frame = pd.DataFrame(
        {
            "TIME_STEP": ["t0", "t1"],
            "load_abs": [
                {1: 0.11, 2: 0.09, 3: 0.13},
                {1: 0.09, 2: 0.07, 3: 0.10},
            ],
            "load_scale": [
                {1: 1.05, 2: 0.95, 3: 1.10},
                {1: 0.90, 2: 1.20, 3: 1.00},
            ],
        }
    )
    result_specs = [
        ResultSpec("res_bus", "vm_pu"),
        ResultSpec("res_bus", "va_degree"),
        ResultSpec("res_line", "loading_percent", alias="line_loading"),
    ]
    bindings = []
    for phase in (1, 2, 3):
        bindings.append(
            ProfileBinding(
                table="asymmetric_load",
                variable="p_mw",
                profile="load_abs",
                element=0,
                phase=phase,
                mode="absolute",
            )
        )
        bindings.append(
            ProfileBinding(
                table="asymmetric_load",
                variable="q_mvar",
                profile="load_scale",
                element=0,
                phase=phase,
                mode="scale",
            )
        )

    writer = run_timeseries(
        net,
        data_source=DFData(frame),
        bindings=bindings,
        result_specs=result_specs,
        output_path=str(tmp_path / "cert_real.csv"),
        auto_fix=False,
        verbose=False,
        tol_vmag_pu=1e-10,
        tol_vang_rad=1e-10,
        max_iter=80,
        warm_start=True,
    )

    assert set(writer.result_frame["COLUMN"]) == {"vm_pu", "va_degree", "line_loading"}
    for time_step in frame["TIME_STEP"]:
        snapshot = _small_real_network()
        _apply_phase_profiles(
            snapshot,
            DFData(frame).get_time_step_values(time_step),
        )
        run_pf(
            snapshot,
            tol_vmag_pu=1e-10,
            tol_vang_rad=1e-10,
            MaxIter=80,
        )
        actual = _actual_rows(writer.result_frame, time_step)
        expected = _expected_rows(snapshot, result_specs)
        assert actual.keys() == expected.keys()
        for key in actual:
            if pd.isna(expected[key]):
                assert pd.isna(actual[key])
            else:
                assert actual[key] == pytest.approx(expected[key], abs=1e-10)

    pdt.assert_series_equal(net.asymmetric_load["p_mw"], baseline_p)
    pdt.assert_series_equal(net.asymmetric_load["q_mvar"], baseline_q)


def test_cert_controller_order_and_topology_invalidation(monkeypatch):
    net = _make_mock_net("CKT_CTRL")
    net.controller = pd.DataFrame(
        {
            "object": ["first", "second"],
            "in_service": [True, True],
            "order": [10.0, 20.0],
            "level": [0, 0],
            "initial_run": [True, True],
            "recycle": [None, None],
        }
    )
    net.model = object()
    net._ppc = {"cached": True}
    net._ppc0 = {"cached": True}
    net._topology_hash = "hash"
    net._topology_cache = {"cached": True}

    seen_orders = []
    seen_run_control = []
    cache_presence = []

    def _solver(mock_net, **kwargs):
        seen_orders.append(tuple(mock_net.controller["order"].tolist()))
        seen_run_control.append(kwargs.get("run_control", False))
        cache_presence.append(
            tuple(
                hasattr(mock_net, attr)
                for attr in ("model", "_ppc", "_ppc0", "_topology_hash", "_topology_cache")
            )
        )
        mock_net.res_bus["vm_pu"] = [1.0, 1.0, 1.0]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    monkeypatch.setattr(_runner_mod, "run_pf", _solver)
    run_timeseries(
        net,
        data_source=DFData(pd.DataFrame({"TIME_STEP": ["t0", "t1"], "load_abs": [1.0, 1.1]})),
        bindings=[
            ProfileBinding(
                table="asymmetric_load",
                variable="p_mw",
                profile="load_abs",
                element=0,
                phase=1,
                mode="scale",
            )
        ],
        run_control=True,
        invalidate_topology_cache=True,
        auto_fix=False,
        verbose=False,
    )

    assert seen_orders == [(10.0, 20.0), (10.0, 20.0)]
    assert seen_run_control == [True, True]
    assert cache_presence == [
        (False, False, False, False, False),
        (False, False, False, False, False),
    ]


def test_cert_warm_start_reuses_only_injection_profile_models(monkeypatch):
    net = _make_mock_net("CKT_WARM")
    calls = []

    def _solver(mock_net, **kwargs):
        calls.append(kwargs)
        mock_net.model = _MockNet(E=[1.0, 1.0, 1.0])
        mock_net.res_bus["vm_pu"] = [1.0, 1.0, 1.0]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    monkeypatch.setattr(_runner_mod, "run_pf", _solver)
    frame = pd.DataFrame(
        {"TIME_STEP": ["t0", "t1", "t2"], "load_abs": [0.1, 0.2, 0.3]}
    )
    run_timeseries(
        net,
        data_source=DFData(frame),
        bindings=[
            ProfileBinding(
                table="asymmetric_load",
                variable="p_mw",
                profile="load_abs",
                element=0,
                phase=1,
                mode="absolute",
            )
        ],
        warm_start=True,
        auto_fix=False,
        verbose=False,
    )

    assert [call["init_model"] for call in calls] == [True, False, False]
    assert [call["warm_start"] for call in calls] == [False, True, True]


def test_cert_failure_policies_are_exact(monkeypatch, tmp_path):
    frame = pd.DataFrame({"TIME_STEP": ["t0", "t1", "t2"], "load_abs": [1.0, 2.0, 3.0]})
    bindings = [
        ProfileBinding(
            table="asymmetric_load",
            variable="p_mw",
            profile="load_abs",
            element=0,
            phase=1,
            mode="absolute",
        )
    ]

    def _make_policy_solver():
        calls = {"count": 0}

        def _solver(mock_net, **kwargs):
            call_index = calls["count"]
            calls["count"] += 1
            if call_index == 1:
                raise LoadflowNotConverged("policy divergence")
            mock_net.res_bus["vm_pu"] = [1.0 + call_index, 0.99, 0.98]
            mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

        return _solver

    monkeypatch.setattr(_runner_mod, "run_pf", _make_policy_solver())
    stop_writer = run_timeseries(
        _make_mock_net("CKT_STOP"),
        data_source=DFData(frame),
        bindings=bindings,
        output_path=str(tmp_path / "stop.csv"),
        failure_policy="stop",
        auto_fix=False,
        verbose=False,
    )
    assert stop_writer.status_frame["TIME_STEP"].tolist() == ["t0", "t1"]
    assert stop_writer.status_frame["STATUS"].tolist() == ["converged", "nonconverged"]

    monkeypatch.setattr(_runner_mod, "run_pf", _make_policy_solver())
    continue_writer = run_timeseries(
        _make_mock_net("CKT_CONTINUE"),
        data_source=DFData(frame),
        bindings=bindings,
        output_path=str(tmp_path / "continue.csv"),
        failure_policy="record_and_continue",
        auto_fix=False,
        verbose=False,
    )
    assert continue_writer.status_frame["TIME_STEP"].tolist() == ["t0", "t1", "t2"]
    assert continue_writer.status_frame["STATUS"].tolist() == [
        "converged",
        "nonconverged",
        "converged",
    ]
    assert continue_writer.result_frame["TIME_STEP"].unique().tolist() == ["t0", "t2"]
    assert len(continue_writer.result_frame) == 12

    monkeypatch.setattr(_runner_mod, "run_pf", _make_policy_solver())
    with pytest.raises(LoadflowNotConverged, match="policy divergence"):
        run_timeseries(
            _make_mock_net("CKT_RAISE"),
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(tmp_path / "raise.csv"),
            failure_policy="raise",
            auto_fix=False,
            verbose=False,
        )


def test_cert_interrupted_resume_is_identical_and_rejects_hash_mismatch(
    monkeypatch,
    tmp_path,
):
    frame = pd.DataFrame(
        {"TIME_STEP": ["t0", "t1", "t2", "t3"], "load_abs": [1.0, 2.0, 3.0, 4.0]}
    )
    bindings = [
        ProfileBinding(
            table="asymmetric_load",
            variable="p_mw",
            profile="load_abs",
            element=0,
            phase=1,
            mode="absolute",
        )
    ]

    def _deterministic_solver(mock_net, **kwargs):
        base = float(mock_net.asymmetric_load.at[(0, 0), "p_mw"])
        mock_net.res_bus["vm_pu"] = [1.0 + base * 0.01, 0.99 + base * 0.01, 0.98 + base * 0.01]
        mock_net.res_bus["va_degree"] = [base, -120.0, 120.0]

    full_path = tmp_path / "full.csv"
    monkeypatch.setattr(_runner_mod, "run_pf", _deterministic_solver)
    run_timeseries(
        _make_mock_net("CKT_FULL"),
        data_source=DFData(frame),
        bindings=bindings,
        output_path=str(full_path),
        chunk_size=1,
        auto_fix=False,
        verbose=False,
    )

    resume_path = tmp_path / "resume.csv"
    checkpoint_path = tmp_path / "checkpoint.json"

    def _interrupting_solver():
        calls = {"count": 0}

        def _solver(mock_net, **kwargs):
            if calls["count"] == 2:
                raise RuntimeError("simulated interruption")
            calls["count"] += 1
            _deterministic_solver(mock_net, **kwargs)

        return _solver

    monkeypatch.setattr(_runner_mod, "run_pf", _interrupting_solver())
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_timeseries(
            _make_mock_net("CKT_FULL"),
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(resume_path),
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=1,
            chunk_size=1,
            auto_fix=False,
            verbose=False,
        )

    monkeypatch.setattr(_runner_mod, "run_pf", _deterministic_solver)
    run_timeseries(
        _make_mock_net("CKT_FULL"),
        data_source=DFData(frame),
        bindings=bindings,
        output_path=str(resume_path),
        checkpoint_path=str(checkpoint_path),
        resume=True,
        chunk_size=1,
        auto_fix=False,
        verbose=False,
    )

    pdt.assert_frame_equal(pd.read_csv(full_path), pd.read_csv(resume_path))
    pdt.assert_frame_equal(
        pd.read_csv(full_path.with_name("full.status.csv")),
        pd.read_csv(resume_path.with_name("resume.status.csv")),
    )

    payload = json.loads(checkpoint_path.read_text())
    payload["time_steps_hash"] = "tampered"
    checkpoint_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Checkpoint mismatch for time_steps_hash"):
        run_timeseries(
            _make_mock_net("CKT_FULL"),
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(resume_path),
            checkpoint_path=str(checkpoint_path),
            resume=True,
            auto_fix=False,
            verbose=False,
        )


@pytest.mark.parametrize("mismatch_key", ["net_hash", "data_source_hash", "bindings_hash", "time_steps_hash"])
def test_cert_resume_rejects_each_checkpoint_identity_hash(
    mismatch_key,
    monkeypatch,
    tmp_path,
):
    frame = pd.DataFrame({"TIME_STEP": ["t0", "t1"], "load_abs": [1.0, 2.0]})
    bindings = [
        ProfileBinding(
            table="asymmetric_load",
            variable="p_mw",
            profile="load_abs",
            element=0,
            phase=1,
            mode="absolute",
        )
    ]
    checkpoint_path = tmp_path / "checkpoint.json"

    def _solver(mock_net, **kwargs):
        mock_net.res_bus["vm_pu"] = [1.0, 0.99, 0.98]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    monkeypatch.setattr(_runner_mod, "run_pf", _solver)
    run_timeseries(
        _make_mock_net("CKT_HASH"),
        data_source=DFData(frame),
        bindings=bindings,
        output_path=str(tmp_path / "hash.csv"),
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=1,
        auto_fix=False,
        verbose=False,
    )

    payload = json.loads(checkpoint_path.read_text())
    payload[mismatch_key] = "tampered"
    checkpoint_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=rf"Checkpoint mismatch for {mismatch_key}"):
        run_timeseries(
            _make_mock_net("CKT_HASH"),
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(tmp_path / "hash.csv"),
            checkpoint_path=str(checkpoint_path),
            resume=True,
            auto_fix=False,
            verbose=False,
        )


@pytest.mark.slow
def test_cert_slow_8760_step_lightweight_hook_has_bounded_peak_memory(
    monkeypatch,
    tmp_path,
):
    n_steps = 8760
    frame = pd.DataFrame(
        {
            "TIME_STEP": [f"t{step}" for step in range(n_steps)],
            "load_abs": [1.0] * n_steps,
        }
    )
    net = _make_mock_net("CKT_8760")
    writer = OutputWriter(
        net,
        frame["TIME_STEP"].tolist(),
        result_specs=[ResultSpec("res_bus", "vm_pu")],
        output_path=str(tmp_path / "8760.csv"),
        chunk_size=256,
    )

    def _solver(mock_net, **kwargs):
        mock_net.res_bus["vm_pu"] = [1.0, 0.99, 0.98]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    monkeypatch.setattr(_runner_mod, "run_pf", _solver)
    tracemalloc.start()
    run_timeseries(
        net,
        data_source=DFData(frame),
        bindings=[
            ProfileBinding(
                table="asymmetric_load",
                variable="p_mw",
                profile="load_abs",
                element=0,
                phase=1,
                mode="absolute",
            )
        ],
        output_writer=writer,
        auto_fix=False,
        verbose=False,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 64 * 1024 * 1024
    assert not writer._result_rows
    assert not writer._status_rows
    with (tmp_path / "8760.csv").open(encoding="utf-8") as result_file:
        result_row_count = sum(1 for _ in result_file) - 1
    with (tmp_path / "8760.status.csv").open(encoding="utf-8") as status_file:
        status_row_count = sum(1 for _ in status_file) - 1

    assert result_row_count == n_steps * len(net.res_bus)
    assert status_row_count == n_steps


@pytest.mark.real_feeder
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_REAL_FEEDER_ENV) != "1",
    reason=(
        f"set {_REAL_FEEDER_ENV}=1 to run the 8760-step packaged-feeder nightly "
        "certification on CKT_626_00490.pkl"
    ),
)
def test_cert_real_feeder_8760_step_status_only_run_has_bounded_memory_and_same_runner_baseline(
    tmp_path,
):
    net = _load_real_feeder_net()
    load_elements = _real_feeder_load_elements(net)
    frame = _scheduled_scale_frame(load_elements, _REAL_FEEDER_HOURS)
    bindings = _real_feeder_bindings(load_elements)
    output_path = tmp_path / "real_feeder_8760.csv"
    initial_peak_rss = _process_peak_rss_bytes()

    warmup_frame = frame.iloc[:_REAL_FEEDER_WARMUP_HOURS].copy()
    warmup_writer = _status_only_writer(
        _load_real_feeder_net(),
        warmup_frame["TIME_STEP"].tolist(),
        tmp_path / "real_feeder_warmup.csv",
    )
    warmup_started = time.perf_counter()
    run_timeseries(
        _load_real_feeder_net(),
        data_source=DFData(warmup_frame),
        bindings=bindings,
        output_writer=warmup_writer,
        auto_fix=False,
        verbose=False,
        max_iter=50,
    )
    warmup_elapsed = time.perf_counter() - warmup_started
    baseline_seconds_per_step = warmup_elapsed / _REAL_FEEDER_WARMUP_HOURS

    writer = _status_only_writer(net, frame["TIME_STEP"].tolist(), output_path)
    started = time.perf_counter()
    run_timeseries(
        net,
        data_source=DFData(frame),
        bindings=bindings,
        output_writer=writer,
        checkpoint_path=str(tmp_path / "real_feeder_8760.checkpoint.json"),
        checkpoint_every=_REAL_FEEDER_CHUNK_SIZE,
        auto_fix=False,
        verbose=False,
        max_iter=50,
    )
    elapsed = time.perf_counter() - started
    peak_rss_growth = max(0, _process_peak_rss_bytes() - initial_peak_rss)

    checkpoint = json.loads((tmp_path / "real_feeder_8760.checkpoint.json").read_text())
    assert checkpoint["next_index"] == _REAL_FEEDER_HOURS
    assert peak_rss_growth < 64 * 1024 * 1024
    assert writer.result_frame.empty
    assert writer.status_frame["STATUS"].eq("converged").all()
    assert len(writer.status_frame) == _REAL_FEEDER_HOURS
    assert sum(1 for _ in open(tmp_path / "real_feeder_8760.status.csv", "r", encoding="utf-8")) - 1 == _REAL_FEEDER_HOURS
    assert elapsed <= baseline_seconds_per_step * _REAL_FEEDER_HOURS * 1.2
