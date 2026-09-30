"""Input safety, immutable runs, independent evidence, and hosted boundaries."""

import json
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from di_validator import store
from di_validator.meter_lab import scenarios, service
from di_validator.meter_lab.api import router
from di_validator.meter_lab.schemas import ScenarioRequest, RunRequest, Mapping
from di_validator.meter_lab.schemas import Parameters


@pytest.fixture
def fixtures(monkeypatch):
    monkeypatch.setattr(
        scenarios,
        "fixture_root",
        lambda a: store.ROOT / "runtime" / ("pv-agent" if a == "pv" else "di-agent") / "fixtures",
    )


def uploaded(text):
    identifier = store.uid()
    path = scenarios.folder("uploads", identifier)
    path.mkdir(parents=True)
    (path / "source.csv").write_text(text)
    store.put("meter_lab_upload", {"id": identifier, "format": "csv"})
    return identifier


def test_samplewise_energy_and_gap_preview(fixtures):
    scenario = scenarios.create(ScenarioRequest(agent="pv", source_id="sign_changes"))
    scenarios.reference(scenario)
    expected = json.loads((scenarios.folder("scenarios", scenario["id"]) / "reference.json").read_text())[
        "intervals"
    ][0]
    assert expected["net_kw"] == pytest.approx(0)
    assert expected["import_kwh"] == pytest.approx(0.125)
    assert expected["export_kwh"] == pytest.approx(0.125)
    missing = scenarios.create(ScenarioRequest(agent="pv", source_id="absent_p"))
    assert all(row[1] is None for row in scenarios.preview(missing)["rows"])


def test_modified_pv_reference_keeps_integer_interval_boundaries(fixtures):
    scenario = scenarios.create(
        ScenarioRequest(agent="pv", source_id="sign_changes", parameters={"magnitude": 1.1})
    )
    scenarios.reference(scenario)
    result = json.loads((scenarios.folder("scenarios", scenario["id"]) / "reference.json").read_text())
    interval = result["intervals"][0]
    assert isinstance(interval["start"], int)
    assert interval["import_kwh"] == pytest.approx(0.1375)
    assert interval["export_kwh"] == pytest.approx(0.1375)


@pytest.mark.parametrize(
    "updates",
    [{"polarity": 0}, {"min_coverage": 0.89}, {"latitude": 34}, {"power_scale": "nan"}, {"unknown": 1}],
)
def test_invalid_configuration_rejected(updates):
    with pytest.raises(ValueError):
        scenarios.config("pv", updates)


@pytest.mark.parametrize(
    "asset,aggregate,rate", [("transformer", True, 1), ("meter", False, 1), ("meter", True, 1 / 3600)]
)
def test_pv_requires_one_second_aggregate_meter(asset, aggregate, rate):
    identifier = uploaded("t,p,q\n1743465600,1,.3\n1743469200,1,.3\n1743472800,1,.3\n")
    request = ScenarioRequest(
        agent="pv",
        source="upload",
        source_id=identifier,
        mapping=Mapping(
            timestamp="t",
            channels={"P": "p", "Q": "q"},
            units={"P": "kW", "Q": "kvar"},
            asset_level=asset,
            aggregate_power=aggregate,
            sample_rate=rate,
        ),
    )
    with pytest.raises(ValueError, match="one-second"):
        scenarios.build(request)


def test_coarse_observations_are_not_interpolated():
    identifier = uploaded("t,p,q\n1743465600,1,.3\n1743469200,1,.3\n1743472800,1,.3\n")
    request = ScenarioRequest(
        agent="pv",
        source="upload",
        source_id=identifier,
        mapping=Mapping(
            timestamp="t", channels={"P": "p", "Q": "q"}, units={"P": "kW", "Q": "kvar"}, aggregate_power=True
        ),
    )
    with pytest.raises(ValueError, match="cadence"):
        scenarios.build(request)


def test_units_validity_and_gaps_are_preserved():
    identifier = uploaded("t,p,q,pv\n1743465600,-2,.3,1\n1743465601,2,.3,0\n1743465610,1,.3,1\n")
    request = ScenarioRequest(
        agent="pv",
        source="upload",
        source_id=identifier,
        mapping=Mapping(
            timestamp="t",
            channels={"P": "p", "Q": "q"},
            units={"P": "kW", "Q": "kvar"},
            validity={"P": "pv"},
            aggregate_power=True,
        ),
    )
    result = scenarios.build(request)
    assert result["values"][:, 0].tolist() == [1743465600, 1743465601, 1743465610]
    assert result["values"][:, 1].tolist() == [-2000, 2000, 1000]
    assert result["values"][:, 3].tolist() == [1, 0, 1]
    assert result["total_samples"] == 3


def test_single_active_run_and_frozen_inputs(fixtures, monkeypatch):
    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    scenario = scenarios.create(ScenarioRequest(agent="fault", source_id="1A_val1"))
    request = RunRequest(scenario_id=scenario["id"])
    first = service.create(request)
    with pytest.raises(HTTPException) as error:
        service.create(request)
    assert error.value.status_code == 409
    path = scenarios.folder("runs", first["id"])
    assert store.checksum(path / "input.csv") == first["manifest"]["input_sha256"]
    service.atomic(path / "status.json", {"state": "completed", "run_id": first["id"]})
    second = service.create(request)
    assert second["id"] != first["id"]
    assert first["manifest"]["input_sha256"] == second["manifest"]["input_sha256"]


def test_registered_ids_cannot_escape_artifact_root():
    for value in ["../outside", "C:/Windows", "0" * 31, "0" * 32 + "/../anything"]:
        with pytest.raises(ValueError):
            scenarios.folder("runs", value)


def test_hosted_requests_cannot_control_wsl(monkeypatch):
    monkeypatch.setenv("DI_HOSTED", "1")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/api/v1/meter-lab/capabilities").json()["available"] is False
    for method, url, body in [
        ("GET", "/runs", None),
        ("POST", "/runs", {"scenario_id": "a" * 32}),
        ("POST", "/runs/" + "a" * 32 + "/control", {"action": "stop"}),
    ]:
        response = client.request(method, "/api/v1/meter-lab" + url, json=body)
        assert response.status_code == 503


def test_downsampling_preserves_extrema_missing_values_and_bound():
    data = np.column_stack([np.arange(10000), np.zeros(10000), np.zeros(10000)])
    data[131, 1] = 99
    data[7654, 1] = -42
    data[4900, 2] = np.nan
    rows = scenarios.bounded(data, 300)
    assert len(rows) <= 300
    assert any(r[1] == 99 for r in rows) and any(r[1] == -42 for r in rows)
    assert any(r[2] is None for r in rows)


def test_fault_gap_segmentation_and_frozen_reference(fixtures):
    scenario = scenarios.create(ScenarioRequest(agent="fault", source_id="gap"))
    reference = json.loads((scenarios.folder("scenarios", scenario["id"]) / "reference.json").read_text())
    assert len(reference["segments"]) > 1
    assert scenario["input_sha256"] == store.checksum(
        scenarios.folder("scenarios", scenario["id"]) / "input.csv"
    )


def test_constant_source_clipping_and_outage_boundaries(fixtures):
    result = scenarios.build(
        ScenarioRequest(
            agent="pv",
            source_id="pv_export",
            parameters=Parameters(
                start_offset_seconds=101,
                duration_seconds=1800,
                outage_start=100,
                outage_seconds=123,
            ),
        )
    )
    values = result["values"]
    assert result["total_samples"] == 1800
    assert sum(values[:, 1]) == 1800
    assert int(values[0, 0]) % 900 == 101
    assert np.all(values[:-1, 0] + values[:-1, 1] == values[1:, 0])
    assert sum(values[values[:, 4] == 0, 1]) == 122


def test_gap_markers_do_not_join_disconnected_waveforms():
    values = np.array([[0, 1], [0.01, 2], [0.1, 3], [0.11, 4]])
    marked = scenarios.gap_markers(values, 0.01)
    assert len(marked) == 5
    assert marked[2, 0] == pytest.approx(0.02)
    assert np.isnan(marked[2, 1])


def test_scope_window_inside_constant_block_preserves_values_and_gaps(monkeypatch):
    scenario_id = store.uid()
    folder = scenarios.folder("scenarios", scenario_id)
    folder.mkdir(parents=True)
    np.save(folder / "input.npy", np.array([
        [1000, 100, 2000, 300, 1, 1],
        [1120, 100, 4000, 500, 0, 1],
    ], dtype=float))
    store.put("meter_lab_scenario", {"id": scenario_id, "encoding": "pv-constant-seconds"})
    monkeypatch.setattr(service, "detail", lambda _: {
        "manifest": {"scenario_id": scenario_id, "mode": "replay"},
        "agent": "pv", "telemetry": {"scenario_time": 1150},
    })
    monkeypatch.setattr(service, "diagnostic_rows", lambda _: [])
    rows = service.series("scope", start=1050, end=1080)["inputs"]["rows"]
    assert rows == [[1050, 2, 0.3], [1080, 2, 0.3]]
    rows = service.series("scope", start=1080, end=1150)["inputs"]["rows"]
    assert rows == [
        [1080, 2, 0.3], [1099, 2, 0.3], [1100, None, None],
        [1120, None, 0.5], [1150, None, 0.5],
    ]
    assert service.series("scope", start=1300, end=1360)["inputs"]["rows"] == []


def test_warm_state_requires_compatible_time_and_configuration(fixtures, monkeypatch):
    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    early = scenarios.create(ScenarioRequest(agent="pv", source_id="pv_export"))
    with pytest.raises(ValueError, match="32-day"):
        service.create(RunRequest(scenario_id=early["id"], seed_fixture="warm_state"))
    changed = scenarios.create(
        ScenarioRequest(agent="pv", source_id="warm_tail", configuration={"polarity": -1})
    )
    with pytest.raises(ValueError, match="incompatible"):
        service.create(RunRequest(scenario_id=changed["id"], seed_fixture="warm_state"))


def test_polling_contracts_never_invoke_wsl(fixtures, monkeypatch):
    import subprocess

    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    scenario = scenarios.create(ScenarioRequest(agent="fault", source_id="1A_val1"))
    run = service.create(RunRequest(scenario_id=scenario["id"]))
    path = scenarios.folder("runs", run["id"])
    state = dict(run_id=run["id"], state="completed", total_samples=400, processed=400)
    service.atomic(path / "status.json", state)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Browser polling spawned a process"))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    base = "/api/v1/meter-lab/runs/" + run["id"]
    assert client.get(base).json()["telemetry"]["processed"] == 400
    response = client.get(base + "/series?start=0.05&end=0.06&max_points=100")
    assert response.status_code == 200
    rows = response.json()["inputs"]["rows"]
    assert rows and all(0.05 <= row[0] <= 0.06 for row in rows)
    assert client.get(base + "/outcomes").json()["rows"] == []
    first = json.dumps(state) + "\n"
    (path / "telemetry.jsonl").write_text(first + '{"partial":')
    page = client.get(base + "/telemetry").json()
    assert len(page["events"]) == 1
    assert client.get(base + "/telemetry?cursor=" + str(page["cursor"])).json()["events"] == []


def test_meter_form_and_sdk_clock_constraints(fixtures, monkeypatch):
    monkeypatch.setattr(service, "capabilities", lambda: {"available": True, "metrology_available": True})
    fault = scenarios.create(ScenarioRequest(agent="fault", source_id="1A_val1"))
    with pytest.raises(ValueError, match="polyphase"):
        service.create(RunRequest(scenario_id=fault["id"], meter_form="GENX_SP"))
    pv = scenarios.create(ScenarioRequest(agent="pv", source_id="sign_changes"))
    with pytest.raises(ValueError, match="real-time"):
        service.create(RunRequest(scenario_id=pv["id"], mode="metrology", speed="100"))


def test_overlapping_evidence_exports_are_immutable(fixtures, monkeypatch):
    import zipfile

    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    scenario = scenarios.create(ScenarioRequest(agent="fault", source_id="1A_val1"))
    run = service.create(RunRequest(scenario_id=scenario["id"]))
    path = scenarios.folder("runs", run["id"])
    service.atomic(path / "status.json", {"run_id": run["id"], "state": "completed"})
    first = service.export(run["id"])
    before = first.read_bytes()
    with zipfile.ZipFile(first) as opened:
        second = service.export(run["id"])
        assert first != second and first.read_bytes() == before
        assert opened.testzip() is None
    with zipfile.ZipFile(second) as opened:
        assert opened.testzip() is None


def test_windows_snapshot_reader_and_atomic_replacement_retry(tmp_path, monkeypatch):
    import os
    import threading

    if os.name != "nt":
        pytest.skip("Windows sharing semantics")
    path = tmp_path / "snapshot.json"
    path.write_text('{"value":1}')
    opened, release = threading.Event(), threading.Event()
    original = os.fdopen
    original_replace = type(path).replace
    result = []

    class HeldReader:
        def __init__(self, fd, mode):
            self.file = original(fd, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self):
            opened.set()
            assert release.wait(30)
            return self.file.read()

    monkeypatch.setattr(os, "fdopen", HeldReader)

    def replace_after_reader_releases(source, target):
        try:
            return original_replace(source, target)
        except PermissionError:
            # Some Windows providers reject replacing an open destination even
            # with FILE_SHARE_DELETE. The producer must retry after it closes.
            release.set()
            raise

    monkeypatch.setattr(type(path), "replace", replace_after_reader_releases)
    thread = threading.Thread(target=lambda: result.append(service.snapshot_bytes(path)))
    thread.start()
    assert opened.wait(10)
    try:
        service.atomic(path, {"value": 2})
    finally:
        release.set()
        thread.join(10)
    assert json.loads(result[0])["value"] == 1
    assert service.read(path)["value"] == 2


def test_synthetic_fault_has_known_distance_and_versioned_reference():
    scenario = scenarios.create(
        ScenarioRequest(
            agent="fault",
            source_id="synthetic_fault",
            parameters=Parameters(fault_type="AB", distance_km=12.3),
        )
    )
    scenarios.reference(scenario)
    folder = scenarios.folder("scenarios", scenario["id"])
    segment = json.loads((folder / "reference.json").read_text())["segments"][0]
    assert segment["distance"]["status"] == "estimated"
    assert segment["distance"]["estimated_distance_km"] == pytest.approx(12.3, abs=1e-6)
    assert scenario["configuration"]["known_distance_km"] == "12.3"
    (folder / "reference-version.txt").write_text("notebook-reactance-1")
    (folder / "reference.json").write_text("{}")
    scenarios.reference(scenario)  # A reference from an older pipeline is recomputed.
    assert json.loads((folder / "reference.json").read_text())["segments"][0]["distance"]["status"] == "estimated"


def test_wavewin_scenario_uses_feeder_line_relay_loop_and_network_placement(monkeypatch):
    from di_validator.faults.wavewin_io import import_file, sources

    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    item = next(s for s in sources() if "11100726" in s["name"])  # Relay: "BG T", LOCATION 1.31 mi.
    dataset = import_file(item["cev_path"], item["id"])
    scenario = scenarios.create(ScenarioRequest(agent="fault", source="dataset", source_id=dataset["id"]))
    configuration = scenario["configuration"]
    assert configuration["fault_loop"] == "BG" and float(configuration["line_length_km"]) < 20
    assert float(configuration["nominal_voltage_kv"]) < 35 and configuration["r0_ohm_km"] != "-1"
    assert scenario["feeder"]["model"].endswith("TECO_13112.pkl")
    scenarios.reference(scenario)
    folder = scenarios.folder("scenarios", scenario["id"])
    distance = json.loads((folder / "reference.json").read_text())["segments"][0]["distance"]
    assert distance["measurement"] == "peak-sample-pair"
    assert distance["status"] == "estimated" and 0 < distance["estimated_distance_km"] < 10
    placement = json.loads((folder / "network.json").read_text())[0]
    assert placement["status"] == "located" and placement["relay_location_mi"] == 1.31
    run = service.create(RunRequest(scenario_id=scenario["id"]))
    assert run["manifest"]["feeder"]["model"].endswith("TECO_13112.pkl")


def test_withheld_fault_alarm_is_decoded(monkeypatch, fixtures):
    monkeypatch.setattr(service, "capabilities", lambda: {"available": True})
    scenario = scenarios.create(ScenarioRequest(agent="fault", source_id="1A_val1"))
    run = service.create(RunRequest(scenario_id=scenario["id"]))
    alarms = ["98#FAULT#1#x-0#c#0.1#2ph#AB#out_of_range#unverified#compensated_or_phase", "98#FAULT#1#x-0#c#0.1#1ph-G#AG#0.8909#unverified#uncompensated"]
    rows = [
        {"table": "AgentEvents", "Id": i, "FeatureId": int("23030000", 16), "TimeStamp": 0, "payload": a}
        for i, a in enumerate(alarms)
    ]
    service.atomic(scenarios.folder("runs", run["id"]) / "outcomes.json", rows)
    withheld, estimated = [r["decoded"] for r in service.outcomes(run["id"])]
    assert withheld["distance_status"] == "out_of_range" and withheld["distance_km"] is None
    assert estimated["distance_status"] == "estimated" and estimated["distance_km"] == 0.8909
    assert estimated["ground"] == "uncompensated"
