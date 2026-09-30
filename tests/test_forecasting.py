import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from di_validator import store
from di_validator.api import app
from di_validator.forecasting.catalog import MODELS, RUNTIMES
from di_validator.forecasting.contracts import ForecastConfig
from di_validator.forecasting.engine import freeze, load_series, run, score, training_prefix


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    path = tmp_path / "data" / "sample.csv"
    values = 3 * np.sin(np.arange(240) * 2 * np.pi / 24) - 1 + np.arange(240) * 0.01
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=len(values), freq="h"),
            "power": values,
            "Event_Label": 0,
        }
    ).to_csv(path, index=False)
    return path


def config(**changes):
    return dict(
        source_file="sample.csv",
        channel="power",
        horizon=12,
        validation_windows=2,
        context_length=96,
        max_history=300,
        lags=12,
        season_length=24,
        models=["seasonal_naive", "random_forest", "ensemble"],
        parameters={"random_forest": {"n_estimators": 10, "max_depth": 4}},
        **changes,
    )


def rows(result):
    return json.loads((Path(result["folder"]) / "predictions.json").read_text())


def test_holdout_cannot_change_validation_or_holdout_predictions(source):
    first = run(config())
    # Preserve training bytes exactly; a CSV round trip can alter floating-point
    # ties and thereby change tree splits independently of the holdout.
    lines = source.read_text().splitlines()
    for i in range(len(lines) - 12, len(lines)):
        cells = lines[i].split(",")
        cells[1] = str(float(cells[1]) + 500)
        lines[i] = ",".join(cells)
    source.write_text("\n".join(lines) + "\n")
    second = run(config())
    assert first["selected_model"] == second["selected_model"]
    assert [m["validation"] for m in first["models"]] == [m["validation"] for m in second["models"]]
    for phase in ["validation", "holdout"]:
        a = [r["prediction"] for r in rows(first) if r["phase"] == phase]
        b = [r["prediction"] for r in rows(second) if r["phase"] == phase]
        np.testing.assert_array_equal(a, b)
    assert first["population_id"] != second["population_id"]
    assert first["source"]["timezone"] == "unverified"


def test_saved_model_reuse_and_artifact_integrity(source):
    first = run(config())
    saved = {
        **first["config"],
        "models": ["random_forest"],
        "parameters": {"random_forest": first["config"]["parameters"]["random_forest"]},
        "reuse_run_id": first["id"],
        "reuse_model": "random_forest",
        "source_snapshot": None,
        "code_version": None,
    }
    repeated = run(saved)
    expected = [
        r["prediction"] for r in rows(first) if r["phase"] == "future" and r["model"] == "random_forest"
    ]
    np.testing.assert_array_equal(expected, [r["prediction"] for r in rows(repeated)])
    assert repeated["mode"] == "inference"
    artifact = Path(first["folder"]) / "artifacts/random_forest/estimator.joblib"
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run(saved)


def test_source_snapshot_gaps_and_short_windows(source):
    frozen = freeze(config())
    table = pd.read_csv(source)
    table.loc[60, "power"] = None
    table.to_csv(source, index=False)
    with pytest.raises(ValueError, match="changed after"):
        run(frozen)
    with pytest.raises(ValueError, match="missing intervals"):
        load_series(ForecastConfig.model_validate(config()))
    cfg = ForecastConfig.model_validate(config(missing_policy="forward_fill"))
    series, _ = load_series(cfg)
    assert training_prefix(series, 61, cfg).iloc[-1] == series.iloc[59]
    assert pd.isna(series.iloc[60])  # The actual target stays missing.
    with pytest.raises(ValueError, match="Not enough history"):
        training_prefix(series, -1, cfg)
    table.iloc[:30].to_csv(source, index=False)
    with pytest.raises(ValueError, match="Not enough history"):
        run(config())


def test_no_huggingface_models_and_label_targets(source):
    assert not {"chronos", "timesfm", "tabpfn"} & MODELS.keys()
    packages = " ".join(p for group in RUNTIMES.values() for p in group["requirements"])
    assert (
        "huggingface" not in packages
        and "transformers" not in packages
        and "autogluon.timeseries" not in packages
    )
    with TestClient(app) as client:
        for model in ["chronos", "chronos-forecasting", "timesfm", "tabpfn"]:
            body = config()
            body["models"] = [model]
            body["parameters"] = {}
            assert client.post("/api/v1/forecasting/runs", json=body).status_code == 422
            assert client.post("/api/v1/forecasting/setup", json={"model": model}).status_code == 422
        bad = config()
        bad["channel"] = "Event_Label"
        assert client.post("/api/v1/forecasting/runs", json=bad).status_code == 422
        assert (
            "Event_Label"
            not in client.get("/api/v1/forecasting/source-columns", params={"name": "sample.csv"}).json()[
                "numeric"
            ]
        )


def test_metric_support_and_export_manifest(source):
    metrics = score([0, 0], [0, 0], [1, 1, 1, 1], 1)
    assert metrics["wape"] is None and metrics["mase"] is None and metrics["smape"] == 0
    result = run(config())
    splits = json.loads((Path(result["folder"]) / "split_manifest.json").read_text())
    assert all(pd.Timestamp(s["train_end"]) < pd.Timestamp(s["test_start"]) for s in splits)
    assert splits[-1]["phase"] == "holdout"
    with TestClient(app) as client:
        import io
        import zipfile

        response = client.get("/api/v1/forecasting/runs/" + result["id"] + "/export")
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            assert {
                "predictions.csv",
                "configuration.json",
                "split_manifest.json",
                "source_window.csv",
                "report.html",
                "artifact_checksums.json",
            } <= set(bundle.namelist())
            assert "artifacts/random_forest/estimator.joblib" in bundle.namelist()
            assert not any(p.startswith("holdout/") for p in bundle.namelist())


@pytest.mark.parametrize("freq", ["h", "ms"])
def test_interpolation_fills_only_bounded_training_gaps(freq):
    cfg = ForecastConfig.model_validate(config(missing_policy="interpolate"))
    original = pd.Series(np.linspace(-8, 8, 100), index=pd.date_range("2025-01-01", periods=100, freq=freq))
    series = original.copy()
    series.iloc[60:63] = np.nan
    filled = training_prefix(series, 96, cfg)
    np.testing.assert_allclose(filled.iloc[60:63], original.iloc[60:63], atol=1e-14)
    pd.testing.assert_series_equal(
        filled[series.iloc[:96].notna()], original.iloc[:96][series.iloc[:96].notna()]
    )
    assert series.iloc[60:63].isna().all()
    assert filled.attrs["preprocessing"]["filled"] == 3
    assert filled.attrs["preprocessing"]["gaps_filled"] == 1
    unseen = series.copy()
    unseen.iloc[96:] += 1000
    pd.testing.assert_series_equal(training_prefix(unseen, 96, cfg), filled)
    # A right endpoint just beyond the forecast origin may not be consulted.
    with pytest.raises(ValueError, match="Leading or trailing gaps"):
        training_prefix(series, 63, cfg)
    # An endpoint just outside the declared context is also ineligible.
    leading = original.copy()
    leading.iloc[4] = np.nan
    with pytest.raises(ValueError, match="Leading or trailing gaps"):
        training_prefix(leading, 100, cfg)
    long_gap = original.copy()
    long_gap.iloc[60:64] = np.nan
    with pytest.raises(ValueError, match="4 consecutive missing steps"):
        training_prefix(long_gap, 96, cfg)


def test_interpolation_coverage_and_preflight_fail_before_fitting(source, monkeypatch):
    cfg = ForecastConfig.model_validate(config(missing_policy="interpolate"))
    series = pd.Series(np.arange(100, dtype=float), index=pd.date_range("2025-01-01", periods=100, freq="h"))
    series.iloc[1:50:3] = np.nan
    with pytest.raises(ValueError, match="observed coverage"):
        training_prefix(series, 96, cfg)
    table = pd.read_csv(source)
    table.loc[227, "power"] = np.nan  # Trailing gap at the holdout forecast origin.
    table.to_csv(source, index=False)
    calls = []
    monkeypatch.setattr("di_validator.forecasting.engine.fitted_prediction", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="Leading or trailing gaps"):
        run(config(missing_policy="interpolate"))
    assert not calls


def test_interpolation_run_preserves_targets_and_exports_audit(source):
    table = pd.read_csv(source)
    table.loc[[160, 225, 232], "power"] = np.nan
    table = table.drop(index=161)  # Also reconstruct an omitted timestamp as missing.
    table.to_csv(source, index=False)
    checksum = store.checksum(source)
    result = run(config(missing_policy="interpolate"))
    assert store.checksum(source) == checksum
    assert result["source"]["missing"] == 4
    assert [p["filled"] for p in result["preprocessing"]] == [2, 2, 3, 4]
    assert all(m["validation"]["n"] == 23 and m["holdout"]["n"] == 11 for m in result["models"])
    assert sum(r["actual"] is None for r in rows(result) if r["phase"] == "holdout") == 3
    # The model sees only training interpolants, never the next window's targets.
    lines = source.read_text().splitlines()
    for i in range(len(lines) - 12, len(lines)):
        cells = lines[i].split(",")
        if cells[1]:
            cells[1] = str(float(cells[1]) + 500)
        lines[i] = ",".join(cells)
    source.write_text("\n".join(lines) + "\n")
    changed = run(config(missing_policy="interpolate"))
    assert changed["selected_model"] == result["selected_model"]
    assert [m["validation"] for m in result["models"]] == [m["validation"] for m in changed["models"]]
    np.testing.assert_array_equal(
        [r["prediction"] for r in rows(result) if r["phase"] == "holdout"],
        [r["prediction"] for r in rows(changed) if r["phase"] == "holdout"],
    )
    with TestClient(app) as client:
        import io
        import zipfile

        assert (
            client.get("/api/v1/forecasting/runs/" + result["id"]).json()["preprocessing"]
            == result["preprocessing"]
        )
        with zipfile.ZipFile(
            io.BytesIO(client.get("/api/v1/forecasting/runs/" + result["id"] + "/export").content)
        ) as bundle:
            assert json.loads(bundle.read("preprocessing.json")) == result["preprocessing"]
            audit = pd.read_csv(io.BytesIO(bundle.read("training_imputations.csv")))
            assert len(audit) == 11 and set(audit.policy) == {"interpolate"}
            assert pd.read_csv(io.BytesIO(bundle.read("source_window.csv"))).actual.isna().sum() == 4
            assert "Training preprocessing" in bundle.read("report.html").decode()
        preset = client.post(
            "/api/v1/presets",
            json={
                "kind": "forecast",
                "name": "Interpolate gaps",
                "config": config(missing_policy="interpolate"),
            },
        )
        assert preset.status_code == 200
        assert preset.json()["config"]["missing_policy"] == "interpolate"
