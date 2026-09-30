import numpy as np
import pandas as pd

from di_validator.events import Detector, evaluate, run, synthetic
from di_validator.query import files, recording_window


def normalize(events):
    return [{k: v for k, v in event.items() if k != "id"} for event in events]


def test_detector_chunk_invariance_and_ground_truth():
    dataset = synthetic(dict(duration_seconds=12))
    frame = pd.concat([pd.read_parquet(path) for path in files(dataset, "samples")], ignore_index=True)
    first = Detector(dataset, dict(dataset_id=dataset["id"]))
    first.process_chunk(frame)
    second = Detector(dataset, dict(dataset_id=dataset["id"]))
    for index in range(0, len(frame), 137):
        second.process_chunk(frame.iloc[index : index + 137])
    assert normalize(first.events) == normalize(second.events)
    assert {e["kind"] for e in first.events} == {
        "load_on",
        "load_off",
        "voltage_dip",
        "voltage_rise",
        "interruption",
    }
    result = run(dict(dataset_id=dataset["id"]), batch_size=149)
    assert result["metrics"]["true_positives"] == 5
    assert result["metrics"]["false_positives"] == 0
    assert result["metrics"]["f1"] == 1
    assert result["synthetic"]
    gap_view = recording_window(dataset["id"], 9.9, 10.2)
    assert any(v is None for v in gap_view["traces"][0]["y"])


def test_event_matching_one_to_one_review_coverage_and_gaps():
    truth = [
        dict(id="r", kind="review", start=0, end=10, asset_id="a"),
        dict(id="t", kind="load_on", start=2, end=2, asset_id="a"),
    ]
    predictions = [
        dict(id=str(i), kind="load_on", start=t, end=t, emitted_at=t + 0.02, asset_id="a")
        for i, t in enumerate([2, 2.01, 20])
    ]
    result = evaluate(predictions, truth, gaps=[(5, 6)])
    assert result["true_positives"] == 1 and result["false_positives"] == 1
    assert result["ignored_predictions"] == 1 and result["reviewed_hours"] == 9 / 3600
    assert result["false_alarms_per_hour"] == 400
    assert evaluate(predictions, [])["precision"] is None


def test_waveform_derivation_and_nan_barriers():
    dataset = synthetic(dict(duration_seconds=12))
    # Exercise waveform-derived RMS / power, rather than the supplied RMS and power channels.
    dataset["channels"] = dataset["channels"][:2]
    frame = pd.concat([pd.read_parquet(path) for path in files(dataset, "samples")], ignore_index=True)
    first = Detector(dataset, dict(dataset_id=dataset["id"]))
    first.process_chunk(frame)
    second = Detector(dataset, dict(dataset_id=dataset["id"]))
    for i in range(0, len(frame), 311):
        second.process_chunk(frame.iloc[i : i + 311])
    assert normalize(first.events) == normalize(second.events)
    assert any(e["kind"] == "voltage_dip" for e in first.events)
    # Missing chunks must discard derived filter state.
    missing = frame.iloc[:100].copy()
    missing["voltage"] = np.nan
    second.process_chunk(missing)
    assert second.warm == 0
