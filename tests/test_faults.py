import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from di_validator import store
from di_validator.api import app
from di_validator.events import run
from di_validator.faults.comtrade_io import import_pair, read_pair, sources
from di_validator.faults.numerics import (
    estimate_impedance_distance,
    estimate_peak_distance,
    fit_fundamental_phasors,
)
from di_validator.faults.pipeline import FaultConfig, PHASES, analyze_segment, validate_dataset
from di_validator.faults.synthetic import fault_frame
from di_validator.ingest import write_frame
from di_validator.notebooks import native_page


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DI_WORKSPACE", str(tmp_path))
    store.initialize()
    return tmp_path


@pytest.fixture
def recording(workspace):
    item = next(s for s in sources() if s["name"] == "1A_val1")
    return import_pair(item["cfg_path"], item["dat_path"], item["id"])


def test_all_imported_comtrade_pairs_and_native_precision(workspace):
    catalog = sources()
    assert len(catalog) == 99
    for item in catalog:
        before = [store.checksum(item[k]) for k in ["cfg_path", "dat_path"]]
        ds = import_pair(item["cfg_path"], item["dat_path"], item["id"])
        assert ds["rows"] == ds["valid_rows"] == 400
        assert ds["readings"] == 2400 and ds["sample_rate"] == 2000
        assert ds["quality"]["missing_readings"] == 0
        assert ds["clock_verified"] is False and ds["synthetic"]
        assert ds["comtrade"]["reported_start"].startswith("2026-")
        assert before == [store.checksum(item[k]) for k in ["cfg_path", "dat_path"]]
        native = pd.read_parquet(Path(ds["folder"]) / "samples/part-00000.parquet")
        np.testing.assert_array_equal(native.t_ns.to_numpy(), np.arange(400) * 500000)
        analog = ds["comtrade"]["original_channels"][0]
        first_raw = float(Path(item["dat_path"]).read_text().splitlines()[0].split(",")[2])
        assert native[analog["name"]].iloc[0] == pytest.approx(
            (first_raw * analog["a"] + analog["b"]) * 1000, abs=1e-9
        )
    assert len(store.listing("dataset")) == 99
    assert import_pair(catalog[0]["cfg_path"], catalog[0]["dat_path"], catalog[0]["id"])["id"]
    assert len(store.listing("dataset")) == 99  # Identical source import is idempotent.


def test_comtrade_catalog_preserves_legacy_identity_across_path_conventions(recording):
    legacy = {
        **recording,
        "comtrade": {**recording["comtrade"], "source_id": "legacy-windows-source"},
        "source": recording["source"].replace("/", "\\"),
    }
    store.put("dataset", legacy, recording["id"])
    entry = next(s for s in sources() if s["name"] == "1A_val1")
    assert entry["id"] == "legacy-windows-source"
    assert entry["dataset_id"] == recording["id"]


def test_notebook_distance_and_tensor_parity_and_chunk_invariance(recording):
    expected_path = store.ROOT / "fault-distance/output/notebook/1A_val1"
    expected = json.loads((expected_path / "fault_distance_estimate.json").read_text())
    config = dict(dataset_id=recording["id"], algorithm="fault-distance")
    first = run(config, batch_size=17)
    second = run(config, batch_size=400)
    a = first["fault_analyses"][0]
    assert a == second["fault_analyses"][0]
    assert a["onset"] == pytest.approx(0.0995, abs=1e-12)
    assert a["fault_type_heuristic"] == "1ph-G" and a["distance"]["fault_loop"] == "AG"
    # Source notebook used float32 COMTRADE calibration; the importer uses float64.
    assert a["distance"]["estimated_distance_km"] == pytest.approx(
        expected["estimated_distance_km"], abs=1e-6
    )
    assert a["distance"]["uncompensated_ground_estimate"]
    assert first["metrics"]["precision"] is None and first["metrics"]["mean_latency"] is None
    with np.load(Path(first["folder"]) / "segment-000-tensor.npz", allow_pickle=False) as actual:
        with np.load(expected_path / "fault_sample_processed.npz", allow_pickle=False) as original:
            np.testing.assert_allclose(actual["X"], original["X"], rtol=2e-5, atol=1e-5)
        assert actual["X"].shape == (12, 400) and np.isnan(actual["y"])
    with TestClient(app) as client:
        bundle = client.get(f"/api/v1/experiments/{first['id']}/export")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert {
                "manifest.json",
                "configuration.json",
                "dataset.json",
                "annotations.json",
                "report.html",
                "segment-000-tensor.npz",
                "segment-000-filtered.csv",
                "segment-000-distance-cycles.csv",
            } <= set(archive.namelist())
        page = client.get(
            f"/api/v1/notebooks/data/{recording['id']}", params={"start": 0, "end": 0.1995}
        ).json()
        assert len(page["rows"]) == 400 and page["metadata"]["clock_verified"] is False
        assert page["rows"][-1]["timestamp"] == "1970-01-01T00:00:00.199500+00:00"


@pytest.mark.parametrize("loop", ["AG", "BG", "CG", "AB", "BC", "CA", "POS"])
def test_distance_physics_for_all_loops(loop):
    fs, f0, distance = 2400, 60, 12.3
    currents = np.array([2 * np.exp(0.2j), 1.5 * np.exp(-2.1j), 1.1 * np.exp(2j)])
    z1, z0 = 0.1 + 0.4j, 0.3 + 1.2j
    voltages = currents * z1 * distance
    if loop.endswith("G"):
        phase = "ABC".index(loop[0])
        k0 = (z0 - z1) / (3 * z1)
        voltages[phase] = (currents[phase] + k0 * currents.sum()) * z1 * distance
    t = np.arange(240) / fs
    waveforms = np.sqrt(2) * np.real(np.r_[currents, voltages][:, None] * np.exp(2j * np.pi * f0 * t)) + 3
    np.testing.assert_allclose(
        fit_fundamental_phasors(waveforms, fs, f0), np.r_[currents, voltages], atol=1e-12
    )
    result = estimate_impedance_distance(
        waveforms[:3],
        waveforms[3:],
        fs,
        f0,
        0,
        len(t),
        dict(r1_ohm_km=0.1, x1_ohm_km=0.4, L_km=50, r0_ohm_km=0.3, x0_ohm_km=1.2),
        loop,
    )
    assert result["estimated_distance_km"] == pytest.approx(distance, abs=1e-10)
    assert result["ground_compensated"] == loop.endswith("G")


FEEDER = dict(line_length_km=5, nominal_voltage_kv=13.2, base_power_mva=10, r1_ohm_km=0.2, x1_ohm_km=0.3)
Z1, Z0 = 0.2 + 0.3j, 0.6 + 0.9j


def _analyze(frame, fs, f0, **settings):
    cfg = FaultConfig(channel_map={p: p for p in PHASES}, **{**FEEDER, **settings})
    return analyze_segment(frame, {"sample_rate": fs, "config": {"nominal_frequency": f0}}, cfg)[0]


@pytest.mark.parametrize("loop", ["AB", "BG", "POS"])
def test_four_samples_per_cycle_fault_cleared_in_two_cycles_uses_the_current_peak(loop):
    # Relay reports: onset detection lags the fault, and whole-cycle fits would land after clearing.
    frame = fault_frame(240, 60, loop, 1.5, Z1, Z0, fault_seconds=2 / 60)
    info = _analyze(frame, 240, 60, r0_ohm_km=0.6, x0_ohm_km=0.9, fault_loop=loop)
    distance = info["distance"]
    assert distance["status"] == "estimated" and distance["measurement"] == "peak-sample-pair"
    assert distance["estimated_distance_km"] == pytest.approx(1.5, abs=1e-9)
    assert distance["x_apparent_ohm"] == pytest.approx(Z1.imag * 1.5)  # k0-compensated for BG.


def test_implausible_distances_are_withheld_with_reason_and_impedance():
    beyond = _analyze(fault_frame(240, 60, "AB", 7.5, Z1, fault_seconds=2 / 60), 240, 60)["distance"]
    assert beyond["status"] == "out_of_range" and "estimated_distance_km" not in beyond
    assert beyond["apparent_distance_km"] == pytest.approx(7.5) and beyond["x_apparent_ohm"] == pytest.approx(2.25)
    behind = _analyze(fault_frame(240, 60, "AB", -1.0, Z1, fault_seconds=2 / 60), 240, 60)["distance"]
    assert behind["status"] == "behind_relay" and behind["reason"]
    # An evolving fault whose apparent distance moves 1 -> 2 -> 3 km across the averaged cycles.
    fs, f0, current = 2400, 60, np.array([3, -3, 0.0])
    t = np.arange(120) / fs
    d = np.repeat([1.0, 2.0, 3.0], 40)

    def wave(phasors):
        return np.sqrt(2) * np.real(phasors * np.exp(2j * np.pi * f0 * t))

    currents = wave(current[:, None] * np.ones(120))
    voltages = wave(current[:, None] * Z1 * d)
    evolving = estimate_impedance_distance(
        currents, voltages, fs, f0, 0, 120, dict(r1_ohm_km=0.2, x1_ohm_km=0.3, L_km=5), "AB", settle_cycles=0
    )
    assert evolving["status"] == "inconsistent" and evolving["apparent_distance_km"] == pytest.approx(2.0)


def test_cycles_after_breaker_opening_are_skipped():
    frame = fault_frame(2400, 60, "AB", 1.5, Z1, fault_seconds=4 / 60)
    guarded = _analyze(frame, 2400, 60)["distance"]
    assert guarded["status"] == "estimated" and guarded["estimated_distance_km"] == pytest.approx(1.5, abs=1e-6)
    short = fault_frame(2400, 60, "AB", 1.5, Z1, fault_seconds=3 / 60)
    guarded = _analyze(short, 2400, 60)["distance"]
    assert guarded["cycles_skipped_decayed"] >= 1
    assert guarded["estimated_distance_km"] == pytest.approx(1.5, rel=0.02)
    unguarded = _analyze(short, 2400, 60, fault_current_fraction=0)["distance"]
    assert unguarded["status"] != "estimated" or abs(unguarded["estimated_distance_km"] - 1.5) > 0.03


def test_peak_distance_needs_minimum_loop_current():
    quiet = np.zeros((3, 20))
    with pytest.raises(ValueError, match="minimum loop current"):
        estimate_peak_distance(quiet, quiet, 240, 0, dict(r1_ohm_km=0.2, x1_ohm_km=0.3, L_km=5), "AB")


def test_missing_data_and_incompatible_channels_do_not_produce_distance(recording):
    cfg = validate_dataset(recording, {})
    frame = pd.read_parquet(Path(recording["folder"]) / "samples/part-00000.parquet")
    clean = frame.copy()
    angle = 2 * np.pi * 50 * clean.offset.to_numpy()
    for channel in recording["channels"]:
        phase = "ABC".index(channel["phase"])
        clean[channel["name"]] = (100 if channel["kind"] == "current" else 10000) * np.sin(
            angle - phase * 2 * np.pi / 3
        )
    info, arrays = analyze_segment(clean, recording, cfg)
    assert info["status"] == "no_fault" and arrays is None
    assert analyze_segment(clean.iloc[:20], recording, cfg)[0]["status"] == "insufficient_data"
    bad = {
        **recording,
        "channels": [
            {**c, "kind": "current_rms"} if c["kind"] == "current" else c for c in recording["channels"]
        ],
    }
    with pytest.raises(ValueError, match="waveform"):
        validate_dataset(bad, {})
    with pytest.raises(ValueError, match="both r0"):
        validate_dataset(recording, dict(r0_ohm_km=0.3))
    frame.loc[198:200, recording["channels"][0]["name"]] = np.nan
    write_frame(frame, Path(recording["folder"]) / "samples/part-00000.parquet")
    with pytest.raises(ValueError, match="data gap"):
        run(dict(dataset_id=recording["id"], algorithm="fault-distance", parameters={"manual_onset": 0.0995}))
    first = run(dict(dataset_id=recording["id"], algorithm="fault-distance"), batch_size=17)
    second = run(dict(dataset_id=recording["id"], algorithm="fault-distance"), batch_size=401)
    assert first["fault_analyses"] == second["fault_analyses"]
    assert len(first["fault_analyses"]) == 2
    assert first["fault_analyses"][0]["end"] < 0.099
    assert first["fault_analyses"][1]["start"] > 0.1


def test_frozen_api_manual_onset_and_review_evidence(recording):
    with TestClient(app) as client:
        base = dict(dataset_id=recording["id"], asset_id=recording["asset_id"], partition="evaluation")
        assert (
            client.post(
                "/api/v1/annotations", json={**base, "kind": "review", "start": 0.08, "end": 0.15}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/annotations", json={**base, "kind": "fault", "start": 0.1, "end": 0.1}
            ).status_code
            == 200
        )
        response = client.post(
            "/api/v1/events/run", json={"dataset_id": recording["id"], "algorithm": "fault-distance"}
        )
        assert response.status_code == 200
        frozen = response.json()["config"]
        assert frozen["parameters"]["pipeline_checksum"]
        result = run(frozen)
        assert result["metrics"]["true_positives"] == 1
        assert result["metrics"]["onset_mae"] == pytest.approx(0.0005)
        assert result["metrics"]["reviewed_hours"] == pytest.approx(0.07 / 3600)
        manual = run({**frozen, "parameters": {**frozen["parameters"], "manual_onset": 0.1}})
        assert manual["events"] == [] and manual["metrics"]["precision"] is None
        assert manual["fault_analyses"][0]["inception_source"] == "manual"
        assert manual["fault_analyses"][0]["distance"]["status"] == "estimated"
        bad = {**frozen, "parameters": {**frozen["parameters"], "pipeline_checksum": "changed"}}
        with pytest.raises(ValueError, match="changed"):
            run(bad)
        assert client.get("/api/v1/faults/sources/no-such-source/cfg").status_code == 422
        assert client.get("/api/v1/faults/sources/no-such-source/bad").status_code == 404
        status = client.get("/api/v1/notebooks/status").json()
        assert "fault_distance.ipynb" in status["notebooks"]


def test_dat_gaps_and_truncation_are_not_reconstructed(workspace):
    item = next(s for s in sources() if s["name"] == "1A_val1")
    cfg = workspace / "gap.cfg"
    dat = workspace / "gap.dat"
    cfg.write_bytes(Path(item["cfg_path"]).read_bytes())
    lines = Path(item["dat_path"]).read_text().splitlines()
    for index in range(200, 400):
        fields = lines[index].split(",")
        fields[1] = str(int(fields[1]) + 5000)
        lines[index] = ",".join(fields)
    dat.write_text("\n".join(lines))
    imported = import_pair(cfg, dat, "gap-test")
    assert imported["quality"]["gaps"] == 1 and imported["quality"]["missing_intervals"] == 10
    assert native_page(imported["id"], start="0", end="1")["rows"][200]["offset"] == pytest.approx(0.105)
    dat.write_text("\n".join(lines[:-1]))
    with pytest.raises(ValueError, match="count/order"):
        read_pair(cfg, dat)
