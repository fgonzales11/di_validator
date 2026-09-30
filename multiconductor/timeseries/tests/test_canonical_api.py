from __future__ import annotations

import importlib
import json

import pandas as pd
import pandas.testing as pdt
import pytest

import multiconductor as mc
from multiconductor.timeseries import DFData, ProfileBinding, run_timeseries
from multiconductor.studies import InvalidStudyInput, MulticonductorStudyError

_runner_mod = importlib.import_module("multiconductor.timeseries.run_mc_timeseries")


class _MockNet:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def _make_mock_net(circuit_key: str = "CKT_CANONICAL") -> _MockNet:
    tuples = [(0, 0), (0, 1), (0, 2)]
    index = pd.MultiIndex.from_tuples(tuples, names=["element_idx", "circuit"])
    asymmetric_load = pd.DataFrame(
        {
            "name": ["LOAD_1", "LOAD_1", "LOAD_1"],
            "bus": [0, 0, 0],
            "from_phase": [1, 2, 3],
            "to_phase": [0, 0, 0],
            "p_mw": [0.0, 0.0, 0.0],
            "q_mvar": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    bus_index = pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2), (0, 3)],
        names=["bus_idx", "phase"],
    )
    bus = pd.DataFrame(index=bus_index)
    res_bus = pd.DataFrame(
        {"vm_pu": [1.0, 1.0, 1.0], "va_degree": [0.0, -120.0, 120.0]},
        index=bus_index,
    )
    return _MockNet(
        asymmetric_load=asymmetric_load,
        asymmetric_sgen=asymmetric_load.copy(deep=True),
        bus=bus,
        res_bus=res_bus,
        CIRCUIT_KEY=circuit_key,
    )


def _small_unbalanced_network():
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
        name="Unbalanced load",
    )
    return net


def test_dfdata_wide_layout_exposes_profiles_by_time_step():
    frame = pd.DataFrame(
        {
            "TIME_STEP": ["t0", "t1"],
            "load_scale": [1.0, 1.1],
            "gen_scale": [0.8, 0.9],
        }
    )
    data = DFData(frame)

    assert data.get_time_steps() == ["t0", "t1"]
    assert data.get_time_step_values("t1") == {
        "load_scale": 1.1,
        "gen_scale": 0.9,
    }


def test_run_timeseries_applies_bindings_records_status_and_restores_inputs(tmp_path):
    net = _make_mock_net()
    original_p = net.asymmetric_load["p_mw"].copy()
    seen_values: list[float] = []

    def _solver(mock_net, **kwargs):
        seen_values.append(float(mock_net.asymmetric_load.loc[(0, 0), "p_mw"]))
        mock_net.res_bus["vm_pu"] = [1.01, 1.0, 0.99]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    frame = pd.DataFrame({"TIME_STEP": ["t0", "t1"], "load_abs": [2.0, 3.0]})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_runner_mod, "run_pf", _solver)
        writer = run_timeseries(
            net,
            data_source=DFData(frame),
            bindings=[
                ProfileBinding(
                    table="asymmetric_load",
                    variable="p_mw",
                    profile="load_abs",
                    element="LOAD_1",
                )
            ],
            output_path=str(tmp_path / "canonical.csv"),
            auto_fix=False,
            verbose=False,
        )

    assert seen_values == [2.0, 3.0]
    assert writer.status_frame["STATUS"].tolist() == ["converged", "converged"]
    assert set(writer.result_frame["COLUMN"]) == {"vm_pu", "va_degree"}
    pdt.assert_series_equal(net.asymmetric_load["p_mw"], original_p)


def test_run_timeseries_resume_and_checkpoint_validation(tmp_path):
    net = _make_mock_net("CKT_RESUME")
    frame = pd.DataFrame({"TIME_STEP": ["t0", "t1"], "load_abs": [1.0, 1.5]})
    bindings = [
        ProfileBinding(
            table="asymmetric_load",
            variable="p_mw",
            profile="load_abs",
            element="LOAD_1",
        )
    ]
    checkpoint = tmp_path / "checkpoint.json"

    def _solver(mock_net, **kwargs):
        mock_net.res_bus["vm_pu"] = [1.0, 1.0, 1.0]
        mock_net.res_bus["va_degree"] = [0.0, -120.0, 120.0]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_runner_mod, "run_pf", _solver)
        run_timeseries(
            net,
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(tmp_path / "initial.csv"),
            checkpoint_path=str(checkpoint),
            checkpoint_every=1,
            auto_fix=False,
            verbose=False,
        )

    payload = json.loads(checkpoint.read_text())
    payload["next_index"] = 1
    checkpoint.write_text(json.dumps(payload))

    resumed_net = _make_mock_net("CKT_RESUME")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_runner_mod, "run_pf", _solver)
        resumed = run_timeseries(
            resumed_net,
            data_source=DFData(frame),
            bindings=bindings,
            output_path=str(tmp_path / "resume.csv"),
            checkpoint_path=str(checkpoint),
            resume=True,
            auto_fix=False,
            verbose=False,
        )

    assert resumed.status_frame["TIME_STEP"].tolist() == ["t1"]

    payload["data_source_hash"] = "tampered"
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(InvalidStudyInput, match="Checkpoint mismatch"):
        run_timeseries(
            _make_mock_net("CKT_RESUME"),
            data_source=DFData(frame),
            bindings=bindings,
            checkpoint_path=str(checkpoint),
            resume=True,
            auto_fix=False,
            verbose=False,
        )


def test_canonical_timeseries_validation_uses_shared_study_error_root():
    with pytest.raises(InvalidStudyInput):
        ProfileBinding("asymmetric_load", "p_mw", "profile", mode="invented")

    assert issubclass(InvalidStudyInput, MulticonductorStudyError)


def test_run_timeseries_real_solver_smoke_restores_baseline(tmp_path):
    net = _small_unbalanced_network()
    base_p = net.asymmetric_load["p_mw"].copy()
    base_q = net.asymmetric_load["q_mvar"].copy()
    frame = pd.DataFrame({"TIME_STEP": ["t0", "t1"], "scale": [1.0, 1.05]})

    writer = run_timeseries(
        net,
        data_source=DFData(frame),
        bindings=[
            ProfileBinding(
                table="asymmetric_load",
                variable="p_mw",
                profile="scale",
                element="Unbalanced load",
                mode="scale",
            ),
            ProfileBinding(
                table="asymmetric_load",
                variable="q_mvar",
                profile="scale",
                element="Unbalanced load",
                mode="scale",
            ),
        ],
        output_path=str(tmp_path / "solver.csv"),
        auto_fix=False,
        verbose=False,
        tol_vmag_pu=1e-9,
        tol_vang_rad=1e-9,
        max_iter=50,
    )

    assert writer.status_frame["STATUS"].tolist() == ["converged", "converged"]
    assert not writer.result_frame.empty
    assert sorted(writer.status_frame["TIME_STEP"].tolist()) == ["t0", "t1"]
    pdt.assert_series_equal(net.asymmetric_load["p_mw"], base_p)
    pdt.assert_series_equal(net.asymmetric_load["q_mvar"], base_q)
