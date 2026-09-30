import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from di_validator import store
from di_validator.aggregation import run as aggregate
from di_validator.ingest import import_dataset, load_quality
from di_validator.query import interval_frame, recording_window, series


def test_dst_audit_source_integrity_and_power(tmp_path):
    path = tmp_path / "meter.csv"
    index = pd.date_range("2025-03-08", periods=72, freq="h")
    pd.DataFrame({"time": index, "'a'": np.arange(72) - 20, "b": np.r_[np.nan, np.ones(71)]}).to_csv(
        path, index=False
    )
    before = store.checksum(path)
    result = import_dataset(dict(path=str(path), timestamp_column="time", interval_seconds=3600, circuit="C"))
    assert result["rows"] == 72 and result["valid_rows"] == 71
    assert result["quality"]["excluded_timestamps"] == 1
    assert store.checksum(path) == before
    audit = json.loads((Path(result["folder"]) / "timestamp_audit.json").read_text())
    assert audit[0]["timestamp"].startswith("2025-03-09 02:")
    assert load_quality(result)[0]["negative_reads"] == 20
    output = series(result["id"], ["transformer:C:a"], power=True)
    assert output["unit"] == "kW" and output["traces"][0]["y"][0] == -20
    assert (
        import_dataset(dict(path=str(path), timestamp_column="time", interval_seconds=3600, circuit="C"))[
            "id"
        ]
        == result["id"]
    )


def test_duplicates_and_fall_clock(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("time,a,'a'\n2025-01-01,1,2\n")
    with pytest.raises(ValueError, match="unique"):
        import_dataset(dict(path=str(path), timestamp_column="time"))
    pd.DataFrame(
        {
            "time": [
                "2025-11-02 00:00",
                "2025-11-02 01:00",
                "2025-11-02 02:00",
                "2025-11-02 02:00",
                "2025-11-02 03:00",
            ],
            "a": [1, 2, 3, 4, 5],
        }
    ).to_csv(path, index=False)
    result = import_dataset(dict(path=str(path), timestamp_column="time", circuit="C"))
    assert result["quality"]["excluded_timestamps"] == 2
    assert result["quality"]["missing_intervals"] == 2
    values = series(result["id"], ["transformer:C:a"])["traces"][0]["y"]
    assert values == [1, None, None, 3, 5]


def test_recording_offset_import_and_exact_samples(tmp_path):
    path = tmp_path / "recording.csv"
    frame = pd.DataFrame({"seconds": np.arange(20000) / 1000, "P": np.sin(np.arange(20000) / 17)})
    frame.to_csv(path, index=False)
    result = import_dataset(
        dict(
            path=str(path),
            format="recording",
            timestamp_column="seconds",
            time_mode="offset_seconds",
            start_time="2025-01-01T00:00:00Z",
            channels=[dict(name="P", kind="power", unit="kW")],
        )
    )
    native = recording_window(result["id"], 1, 1.02)
    assert native["aggregation"] == "native samples"
    assert len(native["traces"][0]["x"]) == 21
    np.testing.assert_allclose(native["traces"][0]["y"], frame.P.iloc[1000:1021])
    broad = recording_window(result["id"], 0, 19.999, max_points=100)
    assert len(broad["traces"][0]["x"]) <= 100


def test_aggregation_inventory_coverage_and_boundary_requirement(tmp_path):
    path = tmp_path / "meters.csv"
    pd.DataFrame(
        {"t": pd.date_range("2025-01-01", periods=4, freq="h"), "m1": [1, 2, 3, 4], "m2": [3, np.nan, 1, 2]}
    ).to_csv(path, index=False)
    config = dict(path=str(path), timestamp_column="t", asset_level="meter", circuit="C", timezone="UTC")
    unknown = import_dataset(config)
    with pytest.raises(ValueError, match="conventions"):
        aggregate(dict(dataset_ids=[unknown["id"]]))
    result = import_dataset({**config, "interval_position": "start"})
    store.put("asset", dict(id="transformer:C:T", source_id="T", circuit="C", level="transformer"))
    relationships = [
        dict(
            meter_id=f"meter:C:m{i}",
            transformer_id="transformer:C:T",
            valid_from="2025-01-01",
            valid_to="2025-01-02",
        )
        for i in [1, 2]
    ]
    complete = aggregate(dict(dataset_ids=[result["id"]], relationship_snapshot=relationships))
    frame = interval_frame(complete)
    assert frame.value.iloc[0] == 4 and pd.isna(frame.value.iloc[1])
    assert frame.meter_coverage.iloc[1] == 0.5
    partial = aggregate(
        dict(dataset_ids=[result["id"]], relationship_snapshot=relationships, min_meter_coverage=0.5)
    )
    assert interval_frame(partial).value.iloc[1] == 2
