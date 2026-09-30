from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from di_validator import store
from di_validator.api import app
from di_validator.faults.wavewin_io import import_file, import_sources, parse_cev, source, sources


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DI_WORKSPACE", str(tmp_path))
    store.initialize()
    return tmp_path


def _baycourt():
    return next(s for s in sources() if s["station"] == "Baycourt" and "15053255" in s["name"])


def test_all_wavewin_events_parse_and_import(workspace):
    catalog = sources()
    assert len(catalog) == 308
    stations = {s["station"] for s in catalog}
    assert stations == {
        "Baycourt",
        "Granada",
        "Gray St",
        "Himes",
        "Hyde Park",
        "Lois Ave",
        "Macdill",
        "Manhattan",
        "Matanzas",
    }
    sample = catalog[:5] + [_baycourt()]
    for item in sample:
        before = store.checksum(item["cev_path"])
        ds = import_file(item["cev_path"], item["id"])
        assert ds["format"] == "recording" and ds["asset_level"] == "terminal"
        assert ds["rows"] == ds["valid_rows"] and ds["rows"] > 0
        assert ds["readings"] == ds["rows"] * len(ds["channels"])
        assert ds["clock_verified"] is False and ds["synthetic"] is False
        assert before == store.checksum(item["cev_path"])
        phases = {(c["kind"], c["phase"]) for c in ds["channels"]}
        assert ("current", "A") in phases and ("voltage", "A") in phases
        assert ds["wavewin"]["import_version"] == "wavewin-cev-1"
        frame = pd.read_parquet(Path(ds["folder"]) / "samples/part-00000.parquet")
        assert len(frame) == ds["rows"]
        assert np.all(np.diff(frame["t_ns"].to_numpy()) > 0)


def test_wavewin_import_is_idempotent_and_catalog_links_dataset(workspace):
    item = _baycourt()
    first = import_file(item["cev_path"], item["id"])
    second = import_file(item["cev_path"], item["id"])
    assert first["id"] == second["id"]
    assert len(store.listing("dataset")) == 1
    entry = next(s for s in sources() if s["id"] == item["id"])
    assert entry["dataset_id"] == first["id"]


def test_baycourt_351_event_shape(workspace):
    item = _baycourt()
    frame, channels, aux, details = parse_cev(item["cev_path"])
    names = {c["name"] for c in channels}
    assert names == {"IA", "IB", "IC", "IN", "IG", "VA(kV)", "VB(kV)", "VC(kV)", "VS(kV)"}
    assert len(frame) == 120  # NUM_OF_CYC=30 * SAM/CYC_A=4
    assert details["event_type"] == "BG T"
    assert details["samples_per_cycle"] == 4.0
    assert "digital_word" in frame.columns


def test_unrecognized_cev_structure_raises(workspace, tmp_path):
    bad = tmp_path / "bad.CEV"
    bad.write_text('CEV 2"FID","0143"\r\n"NOT","A","DATE","HEADER"\r\n1,2,3,4\r\n', encoding="utf-8")
    with pytest.raises(ValueError, match="MONTH/DAY/YEAR"):
        parse_cev(bad)


def test_frozen_api_wavewin_endpoints(workspace):
    with TestClient(app) as client:
        catalog = client.get("/api/v1/faults/wavewin/sources").json()
        assert len(catalog) == 308
        item = next(s for s in catalog if s["station"] == "Baycourt" and "15053255" in s["name"])
        queued = client.post("/api/v1/faults/wavewin/imports", json={"source_ids": [item["id"]]})
        assert queued.status_code == 200
        from di_validator.worker import execute

        execute(queued.json()["id"])
        job = client.get("/api/v1/jobs/" + queued.json()["id"]).json()
        assert job["status"] == "completed"
        dataset_id = store.get("wavewin_import", job["result"]["id"])["dataset_ids"][0]
        assert client.get(f"/api/v1/faults/wavewin/sources/{item['id']}/cev").status_code == 200
        assert client.get("/api/v1/faults/wavewin/sources/no-such-source/cev").status_code == 422
        dataset = client.get(f"/api/v1/datasets/{dataset_id}").json()
        assert dataset["format"] == "recording"


def test_import_sources_detects_stale_checksum(workspace, tmp_path):
    item = _baycourt()
    config = {"sources": [{**item, "checksum": "stale"}]}
    with pytest.raises(ValueError, match="changed after queueing"):
        import_sources(config)


def test_wavewin_source_lookup_requires_known_id(workspace):
    with pytest.raises(ValueError, match="Choose a Wavewin event"):
        source("no-such-id")


def test_fault_distance_accepts_four_samples_per_cycle(workspace):
    from di_validator.events import run
    from di_validator.faults.pipeline import validate_dataset

    item = _baycourt()
    ds = import_file(item["cev_path"], item["id"])
    assert ds["sample_rate"] / ds["config"]["nominal_frequency"] == 4
    validate_dataset(ds, {})
    result = run(dict(dataset_id=ds["id"], algorithm="fault-distance"))
    analysis = result["fault_analyses"][0]
    assert analysis["status"] == "analyzed"
    assert analysis["distance"]["fault_loop"] == "BG"  # Relay reported "BG T" for this event.


def test_off_grid_events_are_hidden_and_refused(workspace):
    off_grid = next(
        Path(p)
        for p in (store.ROOT / "fault-distance/data/Wavewin_Logs").rglob("*.CEV")
        if "17340319" in p.name
    )
    assert all(s["cev_path"] != str(off_grid) for s in sources())
    with pytest.raises(ValueError, match="uniform grid"):
        import_file(off_grid, "off-grid-test")
    assert store.listing("dataset") == []


def test_network_model_lookup_and_safe_loading(tmp_path):
    import pickle

    from di_validator.faults.network_model import _load, feeder_id, model_path

    name = "260802,11100726,-5,MANHATTAN 81D WSA,WSA MANHATN 13112 (451),TEC,,1.000,1.31,BG T,Compressed Default.CEV"
    assert feeder_id({"source": f"C:/logs/Manhattan/{name}"}) == "13112"
    assert feeder_id({"source": "C:/other/recording.cfg"}) is None
    assert model_path("13112").name == "TECO_13112.pkl"
    assert model_path("99999") is None
    hostile = tmp_path / "hostile.pkl"
    hostile.write_bytes(pickle.dumps(Path))
    with pytest.raises(pickle.UnpicklingError, match="disallowed"):
        _load(hostile)


def test_wavewin_fault_located_on_feeder_network(workspace):
    from di_validator.events import run

    item = next(s for s in sources() if "11100726" in s["name"])
    ds = import_file(item["cev_path"], item["id"])
    analysis = run(dict(dataset_id=ds["id"], algorithm="fault-distance"))["fault_analyses"][0]
    network = analysis["network_location"]
    assert network["status"] == "located" and network["feeder"] == "13112" and network["loop"] == "BG"
    assert network["measurement"] == "peak sample-pair phasors"
    assert 1.0 < network["measured_reactance_ohm"] < network["modeled_reach_ohm"]
    best = network["candidates"][0]
    assert "B" in best["phases"] and 2.0 < best["distance_km"] < 3.5
    assert best["distance_mi"] == pytest.approx(best["distance_km"] / 1.609344)
    assert network["relay_location_mi"] == 1.31
    assert network["relay_location_km"] == pytest.approx(1.31 * 1.609344)
    assert network["loop_source"] == "relay event type"


def test_comtrade_analysis_has_no_network_location(workspace):
    from di_validator.events import run
    from di_validator.faults.comtrade_io import import_pair, sources as comtrade_sources

    item = next(s for s in comtrade_sources() if s["name"] == "1A_val1")
    ds = import_pair(item["cfg_path"], item["dat_path"], item["id"])
    analysis = run(dict(dataset_id=ds["id"], algorithm="fault-distance"))["fault_analyses"][0]
    assert "network_location" not in analysis


def test_equivalent_line_uses_longest_three_phase_path():
    from di_validator.faults.network_model import LOOPS, equivalent_line

    z1, zs = 0.2 + 0.3j, (0.6 + 0.9j + 2 * (0.2 + 0.3j)) / 3  # self impedance for z0 = 0.6+0.9j

    def bus(parent, km, phases=3):
        loops = {k: None for k in LOOPS}
        loops.update(AG=zs * km, BG=zs * km, CG=zs * km, AB=z1 * km, BC=z1 * km, CA=z1 * km, POS=z1 * km)
        if phases == 1:  # A single-phase lateral carries no positive-sequence path.
            loops.update(BG=None, CG=None, AB=None, BC=None, CA=None, POS=None)
        return dict(parent=parent, edge=None if parent is None else {}, distance_km=km, loops=loops)

    model = dict(nominal_kv=13.2, buses={0: bus(None, 0.0), 1: bus(0, 2.0), 2: bus(1, 3.0), 3: bus(1, 9.0, phases=1)})
    line = equivalent_line(model)
    assert line["line_length_km"] == 3.0 and line["nominal_voltage_kv"] == 13.2
    assert complex(line["r1_ohm_km"], line["x1_ohm_km"]) == pytest.approx(z1)
    assert complex(line["r0_ohm_km"], line["x0_ohm_km"]) == pytest.approx(0.6 + 0.9j)


def test_feeder_configuration_reads_model_and_relay_loop():
    from di_validator.faults.network_model import feeder_configuration

    name = "260802,11100726,-5,MANHATTAN 81D WSA,WSA MANHATN 13112 (451),TEC,,1.000,1.31,BG T,Compressed Default.CEV"
    settings, provenance = feeder_configuration({"source": f"C:/logs/Manhattan/{name}", "wavewin": {"event_type": "BG T"}})
    assert settings["fault_loop"] == "BG" and provenance["fault_loop_source"] == "relay event type"
    assert settings["nominal_voltage_kv"] == pytest.approx(13.2, rel=0.1)
    assert 0 < settings["line_length_km"] < 20 and settings["x1_ohm_km"] > 0 and settings["x0_ohm_km"] > 0
    assert provenance["model"].endswith("TECO_13112.pkl")
    settings, provenance = feeder_configuration({"source": "C:/logs/x/a,b,c,d,NO FEEDER 99999 (1),e", "wavewin": {}})
    assert "line_length_km" not in settings and provenance["line_source"] == "generic" and provenance["warning"]
