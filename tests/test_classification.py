import numpy as np
import pandas as pd

from di_validator import store
from di_validator.classification import asset_groups, freeze, run, split_groups
from di_validator.ingest import import_dataset
from di_validator.labels import import_registry, registry_for_assets


def test_grouped_splits_and_transitive_relationships():
    ids = [f"a{i}" for i in range(60)]
    relationships = [dict(meter_id=f"a{i}", transformer_id=f"a{i + 1}") for i in range(0, 60, 2)]
    groups = asset_groups(ids, relationships)
    y = np.tile([0, 0, 1, 1], 15)
    train, test, folds = split_groups(y, groups, 0.2, 42, 5)
    assert set(groups[train]).isdisjoint(groups[test])
    for tr, val in folds:
        assert set(groups[train][tr]).isdisjoint(groups[train][val])
    again = split_groups(y, groups, 0.2, 42, 5)
    np.testing.assert_array_equal(train, again[0])


def test_multicircuit_registry_no_false_negatives_outside_coverage(tmp_path):
    path = tmp_path / "registry.csv"
    rows = [
        dict(
            STRUCT_NUM="T",
            PROJECT_ID="P1",
            TECH_TYPE="PHOTOVOLTAIC",
            INTERCONNECTION_STATUS="PTO ISSUED",
            PERMTO_OPERATE_DT="2024-01-01",
            CONTRACT_END_DT="9999-12-31",
            CIRCUIT_KEY="C1",
            EDW_MODIFIED_DATE="2025-06-20",
        ),
        dict(
            STRUCT_NUM="T",
            PROJECT_ID="P2",
            TECH_TYPE="BATTERY",
            INTERCONNECTION_STATUS="PTO ISSUED",
            PERMTO_OPERATE_DT="2024-01-01",
            CONTRACT_END_DT="9999-12-31",
            CIRCUIT_KEY="C2",
            EDW_MODIFIED_DATE="2025-06-20",
        ),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    registry = import_registry(path)
    assets = [
        dict(id=f"transformer:{c}:T", source_id="T", circuit=c, level="transformer")
        for c in ["C1", "C2", "C3"]
    ]
    labeled, _ = registry_for_assets(registry, assets, "2025-01-01", "2025-06-19", True, tmp_path)
    assert [r["value"] for r in labeled] == [1, None, None]


def test_complete_benchmark_frozen_evidence_and_artifacts(tmp_path):
    rng = np.random.default_rng(3)
    index = pd.date_range("2025-01-01", periods=31 * 24, freq="h")
    data = {
        f"a{i}": 2 + i / 40 + 0.2 * np.sin(np.arange(len(index)) / 24) + rng.normal(0, 0.02, len(index))
        for i in range(40)
    }
    path = tmp_path / "features.csv"
    pd.DataFrame(data, index=index).to_csv(path, index_label="time")
    dataset = import_dataset(dict(path=str(path), timestamp_column="time", timezone="UTC", circuit="C"))
    for i in range(40):
        store.put(
            "label",
            dict(
                asset_id=f"transformer:C:a{i}",
                target="PV",
                value=int(i >= 20),
                evidence="fixture",
                valid_from="2024-01-01",
                valid_to="2026-01-01",
                verification="confirmed",
            ),
        )
    config = freeze(dict(dataset_ids=[dataset["id"]], label_policy="confirmed", models=["logistic"], folds=2))
    # New evidence must not modify the queued snapshot.
    store.put(
        "label",
        dict(
            asset_id="transformer:C:a1",
            target="PV",
            value=1,
            evidence="later revision",
            valid_from="2024-01-01",
            valid_to="2026-01-01",
            verification="confirmed",
        ),
    )
    result = run(config)
    assert result["counts"]["labeled"] == 40 and result["counts"]["positives"] == 20
    assert result["selected_model"] == "logistic"
    from pathlib import Path

    folder = Path(result["folder"])
    assert all(
        (folder / name).exists()
        for name in [
            "report.html",
            "manifest.json",
            "features.csv",
            "splits.csv",
            "predictions.csv",
            "logistic.joblib",
        ]
    )
    manifest = pd.read_csv(folder / "splits.csv")
    assert set(manifest.loc[manifest.partition.eq("train"), "group"]).isdisjoint(
        manifest.loc[manifest.partition.eq("test"), "group"]
    )
