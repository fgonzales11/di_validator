import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from di_validator.api import app
from di_validator.ingest import import_dataset
from di_validator.notebooks import native_page


def hourly(tmp_path):
    path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {"time": pd.date_range("2025-01-01", periods=4, freq="h"), "a": [-2, None, 4, 5], "b": [6, 7, 8, 9]}
    ).to_csv(path, index=False)
    return import_dataset(dict(path=str(path), timestamp_column="time", timezone="UTC", circuit="N"))


def test_native_pages_preserve_values_and_equal_timestamp_assets(tmp_path):
    dataset = hourly(tmp_path)
    selection = dict(asset_ids="transformer:N:a,transformer:N:b", limit=3)
    all_rows, cursor = [], None
    while True:
        page = native_page(dataset["id"], cursor=cursor, **selection)
        assert len(page["rows"]) <= 3
        all_rows.extend(page["rows"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(all_rows) == 8
    assert len({(r["timestamp"], r["asset_id"]) for r in all_rows}) == 8
    assert [r["value"] for r in all_rows if r["asset_id"].endswith(":a")] == [-2, None, 4, 5]
    assert page["metadata"]["interval_position"] == "unknown"
    assert page["metadata"]["resolution"] == "native"
    assert page["metadata"]["channels"][0]["unit"] == "kWh"


def test_native_recording_pages_retain_exact_nanoseconds_channels_and_gaps(tmp_path):
    source = tmp_path / "samples.csv"
    origin = pd.Timestamp("2025-01-01T00:00:00.000000123Z")
    times = [origin + pd.Timedelta(milliseconds=i) for i in [0, 1, 2, 4, 5]]
    pd.DataFrame(dict(time=[t.isoformat() for t in times], P=[1, 2, 3, None, 5], V=[120] * 5)).to_csv(
        source, index=False
    )
    dataset = import_dataset(
        dict(
            path=str(source),
            format="recording",
            timestamp_column="time",
            timezone="UTC",
            channels=[dict(name="P", kind="power", unit="kW"), dict(name="V", kind="voltage_rms", unit="V")],
        )
    )
    rows, cursor = [], None
    while True:
        page = native_page(dataset["id"], channels="P", limit=2, cursor=cursor)
        rows.extend(page["rows"])
        assert "V" not in page["columns"]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert [pd.Timestamp(r["timestamp"]).value for r in rows] == [t.value for t in times]
    assert [r["offset"] for r in rows] == [0, 0.001, 0.002, 0.004, 0.005]
    assert rows[3]["P"] is None
    assert not native_page(dataset["id"], start="0.003", end="0.0035")["rows"]
    spec = importlib.util.spec_from_file_location(
        "di_data", Path(__file__).resolve().parents[1] / "notebooks/files/di_data.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    converted = helper.DIClient.to_frame({**page, "rows": rows})
    assert converted.timestamp.iloc[0].value == origin.value
    empty_values = [{**r, "P": None} for r in rows]
    missing = helper.DIClient.to_frame({**page, "rows": empty_values})
    assert pd.api.types.is_numeric_dtype(missing.P)
    assert missing.P.isna().all()


def test_notebook_api_bounds_cursor_validation_and_registered_sources(tmp_path):
    dataset = hourly(tmp_path)
    with TestClient(app) as client:
        endpoint = "/api/v1/notebooks/data/" + dataset["id"]
        assert client.get(endpoint).status_code == 422
        assert client.get(endpoint, params={"limit": 100001}).status_code == 422
        assert (
            client.get(
                endpoint, params={"asset_ids": "transformer:N:a", "channels": "bad;SELECT"}
            ).status_code
            == 422
        )
        first = client.get(endpoint, params={"asset_ids": "transformer:N:a", "limit": 2}).json()
        changed = client.get(
            endpoint, params={"asset_ids": "transformer:N:b", "cursor": first["next_cursor"]}
        )
        assert changed.status_code == 422
        assert (
            client.get(endpoint, params={"asset_ids": "transformer:N:a", "cursor": "not-base64"}).status_code
            == 422
        )
        downloaded = client.get("/api/v1/notebooks/sources/" + dataset["id"])
        assert downloaded.content == Path(dataset["source"]).read_bytes()
        assert client.get("/api/v1/notebooks/status").json()["row_limit"] == 100000
    with pytest.raises(ValueError, match="unambiguous"):
        native_page(dataset["id"], asset_ids="transformer:N:a", start="not-a-time")


def test_unmapped_source_preserves_bytes_and_confines_reads(tmp_path, monkeypatch):
    from di_validator import store
    from di_validator.notebooks import SOURCE_READ_LIMIT, source_csv

    monkeypatch.setattr(store, "ROOT", tmp_path)
    directory = tmp_path / "data"
    directory.mkdir()
    source = directory / "anomaly.csv"
    original = (
        b"Timestamp,Load_kW,Event_Label,Anomaly_Type\r\n2025-10-10 16:38:02.049675,1.2,1,Cyber_Spike\r\n"
    )
    source.write_bytes(original)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(original)
    with TestClient(app) as client:
        result = client.get("/api/v1/notebooks/files/anomaly.csv")
        assert result.status_code == 200
        assert result.content == original
        assert result.headers["X-Source-SHA256"] == store.checksum(source)
        assert client.get("/api/v1/notebooks/files/missing.csv").status_code == 404
        with pytest.raises(HTTPException) as raised:
            source_csv("../outside.csv")
        assert raised.value.status_code == 404
        large = directory / "large.csv"
        with large.open("wb") as stream:
            stream.truncate(SOURCE_READ_LIMIT + 1)
        assert client.get("/api/v1/notebooks/files/large.csv").status_code == 413
    assert source.read_bytes() == original


def test_bootstrap_requires_mapping_for_new_recording_tables(tmp_path, monkeypatch):
    from di_validator import store

    monkeypatch.setattr(store, "ROOT", tmp_path)
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "power_anomaly_dataset_2022.csv").write_text(
        "Timestamp,Voltage,Event_Label\n2025-10-10,230,0\n"
    )
    (directory / "AMI_TEST.csv").write_text("REPORTED_DTTM,T1\n2025-01-01,1\n")
    with TestClient(app) as client:
        sources = {s["name"]: s for s in client.get("/api/v1/sources").json()}
        assert sources["power_anomaly_dataset_2022.csv"]["format"] == "mapping_required"
        assert sources["AMI_TEST.csv"]["format"] == "wide_ami"
        queued = client.post("/api/v1/bootstrap").json()
        assert len(queued) == 1
        assert Path(queued[0]["config"]["path"]).name == "AMI_TEST.csv"
        assert client.get("/api/v1/datasets").json() == []
