from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import store
from ..query import db, files, sql_name
from .catalog import MODELS, canonical, catalog, checked_parameters
from .contracts import ForecastConfig
from . import providers, runtime


def code_version():
    return store.digest({p.name: store.checksum(p) for p in Path(__file__).parent.glob("*.py")})


def source_path(name):
    root = (store.ROOT / "data").resolve()
    path = (root / name).resolve()
    if path.parent != root or path.name != name or path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError("Choose a source CSV from data/")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Import source files larger than 20 MiB before forecasting")
    return path


def source_columns(name):
    frame = pd.read_csv(source_path(name), nrows=100)
    numeric = [
        c for c in frame.select_dtypes("number") if "label" not in c.lower() and c.lower() != "anomaly_type"
    ]
    return dict(columns=frame.columns.tolist(), numeric=numeric)


def freeze(config):
    cfg = ForecastConfig.model_validate(config)
    models = list(dict.fromkeys(canonical(m) for m in cfg.models))
    if any(m not in MODELS for m in models):
        raise ValueError("Choose supported forecasting models")
    if "ensemble" in models and len(models) < 3:
        raise ValueError("An ensemble requires at least two constituent models")
    available = {m["id"]: m for m in catalog()}
    if any(not available[m]["ready"] for m in models if m != "ensemble"):
        raise ValueError("Install the selected model runtimes before starting a forecast")
    params = {canonical(k): v for k, v in cfg.parameters.items()}
    if set(params) - set(models):
        raise ValueError("Parameters must belong to selected models")
    params = {m: checked_parameters(m, params.get(m, {})) for m in models}
    if cfg.dataset_id:
        ds = store.get("dataset", cfg.dataset_id)
        if ds["format"] == "wide_ami":
            if not cfg.asset_id:
                raise ValueError("Choose an asset")
            store.get("asset", cfg.asset_id)
        elif cfg.asset_id and cfg.asset_id != ds["asset_id"]:
            raise ValueError("The recording asset does not match this dataset")
        if cfg.channel not in {c["name"] for c in ds["channels"]}:
            raise ValueError("Choose a declared measurement channel")
        snapshot = dict(dataset_id=ds["id"], source_checksum=ds["source_checksum"], synthetic=ds["synthetic"])
    else:
        path = source_path(cfg.source_file)
        if cfg.channel not in source_columns(cfg.source_file)["numeric"]:
            raise ValueError("Choose a numeric measurement; labels cannot be forecasting targets")
        snapshot = dict(source_file=path.name, source_checksum=store.checksum(path), synthetic=None)
    if cfg.reuse_run_id:
        saved = store.get("forecast", cfg.reuse_run_id)
        if saved["mode"] != "backtest" or cfg.reuse_model not in {
            m["model"] for m in saved["models"] if m["status"] == "completed"
        }:
            raise ValueError("Choose a successfully fitted model from a backtest")
        if cfg.horizon > saved["config"]["horizon"]:
            raise ValueError("Saved-model horizon cannot exceed its trained horizon")
        if cfg.reuse_model == "ensemble":
            raise ValueError(
                "Reuse individual fitted models; ensemble forecasts are exported in their original run"
            )
        # Saved model schema is authoritative; source selection may provide newer observations.
        params = {cfg.reuse_model: saved["config"]["parameters"][cfg.reuse_model]}
        models = [cfg.reuse_model]
        cfg.lags = saved["config"]["lags"]
        cfg.season_length = saved["config"]["season_length"]
    return {
        **cfg.model_dump(),
        "models": models,
        "parameters": params,
        "source_snapshot": snapshot,
        "code_version": code_version(),
    }


def _stamp(value, timezone):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None and timezone:
        stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return stamp


def load_series(cfg):
    if cfg.dataset_id:
        ds = store.get("dataset", cfg.dataset_id)
        channel = next(c for c in ds["channels"] if c["name"] == cfg.channel)
        recording = ds["format"] == "recording"
        conditions, params = [], []
        if not recording:
            conditions.append("asset_id = ?")
            params.append(cfg.asset_id)
        for value, operator in [(cfg.start, ">="), (cfg.end, "<=")]:
            if value:
                conditions.append(f"t_ns {operator} ?")
                params.append(_stamp(value, ds["timezone"]).value)
        measurement = sql_name(cfg.channel if recording else "value")
        sql = f"SELECT t_ns,{measurement} AS y FROM read_parquet(?)"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY t_ns DESC LIMIT ?"
        with db() as connection:
            frame = (
                connection.execute(
                    sql, [files(ds, "samples" if recording else "intervals"), *params, cfg.max_history]
                )
                .df()
                .sort_values("t_ns")
            )
        index = pd.DatetimeIndex(pd.to_datetime(frame.t_ns, utc=True))
        series = pd.Series(frame.y.to_numpy(dtype=float), index=index)
        step_ns = round(1e9 / ds["sample_rate"] if recording else ds["interval_seconds"] * 1e9)
        meta = dict(
            name=ds["name"],
            dataset_id=ds["id"],
            asset_id=ds.get("asset_id", cfg.asset_id),
            channel=cfg.channel,
            unit=channel["unit"],
            timezone=ds["timezone"],
            clock="UTC instants; original clock convention retained",
            interval_position=ds["interval_position"],
            source_checksum=ds["source_checksum"],
            synthetic=ds["synthetic"],
        )
    else:
        path = source_path(cfg.source_file)
        fingerprint = store.checksum(path)
        if cfg.source_snapshot and cfg.source_snapshot["source_checksum"] != fingerprint:
            raise ValueError("Source file changed after this run was queued. Create a new forecast.")
        frame = pd.read_csv(path, usecols=[cfg.timestamp_column, cfg.channel])
        index = pd.DatetimeIndex(pd.to_datetime(frame[cfg.timestamp_column], errors="raise"))
        if index.hasnans:
            raise ValueError("Resolve invalid source timestamps before forecasting")
        series = pd.Series(
            pd.to_numeric(frame[cfg.channel], errors="raise").to_numpy(dtype=float), index=index
        )
        if cfg.start:
            series = series[series.index >= _stamp(cfg.start, str(index.tz) if index.tz else None)]
        if cfg.end:
            series = series[series.index <= _stamp(cfg.end, str(index.tz) if index.tz else None)]
        series = series.iloc[-cfg.max_history :]
        if len(series) < 2:
            raise ValueError("Select at least two source samples")
        differences = np.diff(series.index.asi8)
        step_ns = (
            int(pd.Series(differences[differences > 0]).mode().iloc[0]) if (differences > 0).any() else 0
        )
        meta = dict(
            name=path.name,
            asset_id="source:" + path.name,
            channel=cfg.channel,
            unit=cfg.unit,
            source_checksum=fingerprint,
            timezone=str(index.tz) if index.tz else "unverified",
            clock="reported source timestamps",
            interval_position="unknown",
            synthetic=None,
        )
    if (
        len(series) < 2
        or not series.index.is_monotonic_increasing
        or series.index.has_duplicates
        or step_ns <= 0
    ):
        raise ValueError("Forecast input needs strictly increasing, unique timestamps")
    if ((series.index.asi8 - series.index.asi8[0]) % step_ns != 0).any():
        raise ValueError("Irregular sampling: import an explicitly resampled analysis dataset first")
    expected = int((series.index.asi8[-1] - series.index.asi8[0]) // step_ns + 1)
    if expected > cfg.max_history * 2:
        raise ValueError("Time range has too many gaps; select a shorter, contiguous window")
    series = series.reindex(
        pd.date_range(series.index[0], periods=expected, freq=pd.Timedelta(step_ns, unit="ns"))
    ).iloc[-cfg.max_history :]
    series = series.replace([np.inf, -np.inf], np.nan)
    coverage = float(series.notna().mean())
    if coverage < cfg.min_coverage:
        raise ValueError(
            f"Series coverage is {coverage:.1%}; requires {cfg.min_coverage:.1%}. Choose another asset/window."
        )
    if series.isna().any() and cfg.missing_policy == "reject":
        raise ValueError(
            f"Selected series contains {series.isna().sum()} missing intervals. Choose a contiguous window or enable interpolation for short interior gaps or bounded forward fill."
        )
    meta.update(
        step_ns=step_ns,
        resolution_seconds=step_ns / 1e9,
        start=series.index[0].isoformat(),
        end=series.index[-1].isoformat(),
        observations=len(series),
        missing=int(series.isna().sum()),
        coverage=coverage,
        missing_policy=cfg.missing_policy,
    )
    return series, meta


def training_prefix(series, stop, cfg):
    if stop <= 0 or stop > len(series):
        raise ValueError("Not enough history for the requested chronological forecast windows")
    # Slice before interpolation: neither endpoint may come from an evaluation
    # target or from outside the declared training context.
    original = series.iloc[max(0, stop - cfg.context_length) : stop]
    prefix = original.copy()
    missing = prefix.isna().to_numpy()
    gap_edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, ends = np.flatnonzero(gap_edges == 1), np.flatnonzero(gap_edges == -1)
    if cfg.missing_policy == "forward_fill":
        prefix = series.iloc[:stop].ffill(limit=cfg.max_gap_steps).iloc[-cfg.context_length :]
    elif cfg.missing_policy == "interpolate":
        if original.notna().mean() < cfg.min_coverage:
            raise ValueError(
                "Interpolation is not viable: training-window observed coverage is below the required minimum. Choose a better-covered window."
            )
        for start, end in zip(starts, ends, strict=True):
            if start == 0 or end == len(prefix):
                raise ValueError(
                    f"Interpolation is not viable for the gap at {prefix.index[start].isoformat()}: both bounding observations must be inside the training window. Leading or trailing gaps cannot be extrapolated; choose another window."
                )
            if end - start > cfg.max_gap_steps:
                raise ValueError(
                    f"Interpolation is not viable for the gap at {prefix.index[start].isoformat()}: {end - start} consecutive missing steps exceed the configured maximum of {cfg.max_gap_steps}."
                )
        # Use small relative integer timestamp differences, avoiding loss of
        # sub-second precision from floating-point absolute epoch timestamps.
        ticks = prefix.index.asi8
        for start, end in zip(starts, ends, strict=True):
            weight = (ticks[start:end] - ticks[start - 1]) / (ticks[end] - ticks[start - 1])
            prefix.iloc[start:end] = (1 - weight) * prefix.iloc[start - 1] + weight * prefix.iloc[end]
    if prefix.isna().any():
        raise ValueError(
            "Training window contains unresolved gaps; bounded forward fill cannot fill leading or long gaps"
        )
    if not np.isfinite(prefix.to_numpy()).all():
        raise ValueError("Training history must contain finite values after preprocessing")
    if len(prefix) < max(cfg.lags + 10, cfg.season_length * 2, 32):
        raise ValueError("Need more training history for the requested lags/seasonality and forecast windows")
    prefix.attrs["preprocessing"] = dict(
        policy=cfg.missing_policy,
        max_gap_steps=cfg.max_gap_steps,
        train_start=prefix.index[0].isoformat(),
        train_end=prefix.index[-1].isoformat(),
        observations=len(prefix),
        observed_coverage=float(original.notna().mean()),
        filled=int(missing.sum()),
        gaps_filled=len(starts),
    )
    prefix.attrs["filled_values"] = [
        dict(timestamp=stamp.isoformat(), value=float(value)) for stamp, value in prefix[missing].items()
    ]
    return prefix


def score(actual, predicted, training, season_length):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    good = np.isfinite(actual)
    if not good.any():
        raise ValueError("An evaluation window has no observed targets")
    a, p = actual[good], predicted[good]
    error = np.abs(a - p)
    denominator = np.abs(a) + np.abs(p)
    scale = np.mean(np.abs(np.asarray(training)[season_length:] - np.asarray(training)[:-season_length]))
    return dict(
        n=int(good.sum()),
        mae=float(error.mean()),
        rmse=float(np.sqrt(np.mean((a - p) ** 2))),
        smape=float(
            np.mean(np.divide(200 * error, denominator, out=np.zeros_like(error), where=denominator > 0))
        ),
        wape=float(100 * error.sum() / np.abs(a).sum()) if np.abs(a).sum() > 0 else None,
        mase=float(error.mean() / scale) if np.isfinite(scale) and scale > 0 else None,
    )


def fitted_prediction(model, prefix, cfg, step_ns, output, progress, reuse=None):
    progress.check()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = cfg.parameters[model]
    if MODELS[model]["group"] == "base":
        return providers.predict(
            model,
            prefix.to_numpy(),
            prefix.index.asi8,
            cfg.horizon,
            step_ns,
            settings,
            output,
            seed=cfg.seed,
            lags=cfg.lags,
            season_length=cfg.season_length,
            reuse=reuse,
        )
    data = output / "input.npz"
    np.savez_compressed(data, y=prefix.to_numpy(), times=prefix.index.asi8)
    result = runtime.request(
        model,
        dict(
            input=str(data),
            output=str(output),
            settings=settings,
            horizon=cfg.horizon,
            step_ns=step_ns,
            seed=cfg.seed,
            lags=cfg.lags,
            season_length=cfg.season_length,
            reuse=str(reuse) if reuse else None,
        ),
        output,
        progress,
        cfg.timeout_seconds,
    )
    return np.asarray(result["prediction"], dtype=float)


def prediction_rows(model, phase, index, prediction, actual=None, window=0, radius=None):
    rows = []
    for i, (stamp, value) in enumerate(zip(index, prediction, strict=True)):
        observed = float(actual[i]) if actual is not None and np.isfinite(actual[i]) else None
        rows.append(
            dict(
                model=model,
                phase=phase,
                window=window,
                step=i + 1,
                timestamp=stamp.isoformat(),
                prediction=float(value),
                actual=observed,
                lower=float(value - radius) if radius is not None else None,
                upper=float(value + radius) if radius is not None else None,
            )
        )
    return rows


def _finish(record, folder, rows, series, splits, training_windows):
    record["preprocessing"] = [
        dict(phase=phase, window=window, **prefix.attrs["preprocessing"])
        for phase, window, prefix in training_windows
    ]
    filled = [
        dict(phase=phase, window=window, policy=prefix.attrs["preprocessing"]["policy"], **value)
        for phase, window, prefix in training_windows
        for value in prefix.attrs["filled_values"]
    ]
    pd.DataFrame(filled, columns=["phase", "window", "policy", "timestamp", "value"]).to_csv(
        folder / "training_imputations.csv", index=False
    )
    (folder / "preprocessing.json").write_text(store.encode(record["preprocessing"]), encoding="utf-8")
    (folder / "predictions.json").write_text(store.encode(rows), encoding="utf-8")
    pd.DataFrame(rows).to_csv(folder / "predictions.csv", index=False)
    (folder / "split_manifest.json").write_text(store.encode(splits), encoding="utf-8")
    (folder / "configuration.json").write_text(store.encode(record["config"]), encoding="utf-8")
    (folder / "result.json").write_text(store.encode(record), encoding="utf-8")
    series.rename("actual").to_csv(folder / "source_window.csv", index_label="timestamp")
    artifacts = {
        p.relative_to(folder).as_posix(): store.checksum(p)
        for p in (folder / "artifacts").rglob("*")
        if p.is_file()
    }
    (folder / "artifact_checksums.json").write_text(store.encode(artifacts), encoding="utf-8")
    table = pd.DataFrame(
        [
            {
                "model": m["model"],
                "status": m["status"],
                "validation_MAE": m.get("validation", {}).get("mae"),
                "holdout_MAE": m.get("holdout", {}).get("mae"),
                "error": m.get("error", ""),
            }
            for m in record["models"]
        ]
    ).to_html(index=False, escape=True)
    (folder / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>DI forecasting report</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:3rem auto}td,th{padding:9px;text-align:left}pre{white-space:pre-wrap}</style>"
        f"<h1>{html.escape(record['name'])}</h1><p>{html.escape(record['selection_policy'])}</p>"
        "<p>Future forecasts use models refitted after evaluation. Bands use absolute validation residuals and do not guarantee coverage.</p>"
        + table
        + "<h2>Training preprocessing</h2><p>Imputed training values are estimates. Evaluation targets and original source readings remain unchanged.</p>"
        + pd.DataFrame(record["preprocessing"]).to_html(index=False, escape=True)
        + "<h2>Source and configuration</h2><pre>"
        + html.escape(json.dumps({"source": record["source"], "config": record["config"]}, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return store.put("forecast", record, record["id"])


def run(config, progress=None):
    cfg = ForecastConfig.model_validate(config)
    if not cfg.code_version:
        cfg = ForecastConfig.model_validate(freeze(config))
    if cfg.code_version != code_version():
        raise ValueError("Forecasting code changed after queueing; create a new run")
    progress = progress or store.Progress()
    progress(0.02, "Reading selected native measurements")
    series, source = load_series(cfg)
    run_id = store.uid()
    folder = store.workspace() / "forecasts" / run_id
    (folder / "artifacts").mkdir(parents=True)
    base = dict(
        id=run_id,
        name=cfg.name,
        created_at=store.now(),
        config=cfg.model_dump(),
        source=source,
        folder=str(folder),
        mode="inference" if cfg.reuse_run_id else "backtest",
        selected_model="",
        models=[],
        selection_policy="Lowest chronological validation MAE; final holdout is never used for selection.",
    )
    if cfg.reuse_run_id:
        parent = store.get("forecast", cfg.reuse_run_id)
        if parent["config"]["code_version"] != code_version():
            raise ValueError("Saved model adapter code differs; restore it or train a new model")
        for key in ["asset_id", "channel", "unit", "step_ns", "timezone"]:
            if source[key] != parent["source"][key]:
                raise ValueError(f"Saved-model {key} is incompatible with the selected source")
        if pd.Timestamp(source["end"]) < pd.Timestamp(parent["source"]["end"]):
            raise ValueError("Cannot apply a model to an origin earlier than its training cutoff")
        parent_folder = Path(parent["folder"])
        checks = json.loads((parent_folder / "artifact_checksums.json").read_text())
        for name, checksum in checks.items():
            if (
                name.startswith("artifacts/" + cfg.reuse_model + "/")
                and store.checksum(parent_folder / name) != checksum
            ):
                raise ValueError("Saved model artifact checksum differs")
        prefix = training_prefix(series, len(series), cfg)
        pred = fitted_prediction(
            cfg.reuse_model,
            prefix,
            cfg,
            source["step_ns"],
            folder / "artifacts" / cfg.reuse_model,
            progress,
            parent_folder / "artifacts" / cfg.reuse_model,
        )
        future = pd.date_range(
            series.index[-1], periods=cfg.horizon + 1, freq=pd.Timedelta(source["step_ns"], unit="ns")
        )[1:]
        base.update(
            selected_model=cfg.reuse_model,
            models=[dict(model=cfg.reuse_model, name=MODELS[cfg.reuse_model]["name"], status="completed")],
            population_id=store.digest([source, cfg.reuse_run_id]),
            selection_policy="Inference from a saved fitted model; no training or evaluation on this run.",
        )
        return _finish(
            base,
            folder,
            prediction_rows(cfg.reuse_model, "future", future, pred),
            series,
            [],
            [("inference", 0, prefix)],
        )
    h, n = cfg.horizon, len(series)
    origins = [n - h * (cfg.validation_windows + 1 - i) for i in range(cfg.validation_windows)]
    # Validate every required history before fitting any model. Reuse the same
    # repaired history across candidates, with a separate audit for each origin.
    prepared = {stop: training_prefix(series, stop, cfg) for stop in origins + [n - h, n]}
    # Same observed evaluation targets for all candidates; no automatic model-specific row dropping.
    for stop in origins + [n - h]:
        if series.iloc[stop : stop + h].notna().mean() < cfg.min_coverage:
            raise ValueError("An evaluation window falls below minimum coverage; choose another window")
    splits = [
        dict(
            window=i,
            phase="validation" if i < len(origins) else "holdout",
            train_start=series.index[max(0, stop - cfg.context_length)].isoformat(),
            train_end=series.index[stop - 1].isoformat(),
            test_start=series.index[stop].isoformat(),
            test_end=series.index[stop + h - 1].isoformat(),
        )
        for i, stop in enumerate(origins + [n - h])
    ]
    base["population_id"] = store.digest(
        {
            "source": source,
            "splits": splits,
            "values": hashlib.sha256(series.to_numpy().tobytes()).hexdigest(),
        }
    )
    validation, rows, models = {}, [], []
    candidates = [m for m in cfg.models if m != "ensemble"]
    for j, model in enumerate(candidates):
        progress(0.05 + 0.5 * j / len(candidates), f"Backtesting {MODELS[model]['name']}")
        predictions = []
        try:
            for i, stop in enumerate(origins):
                prefix = prepared[stop]
                pred = fitted_prediction(
                    model, prefix, cfg, source["step_ns"], folder / "backtests" / model / str(i), progress
                )
                predictions.append(pred)
            validation[model] = predictions
        except (ValueError, RuntimeError, ImportError, TimeoutError) as error:
            models.append(dict(model=model, name=MODELS[model]["name"], status="failed", error=str(error)))
    if "ensemble" in cfg.models:
        if len(validation) >= 2:
            validation["ensemble"] = [
                np.mean([v[i] for v in validation.values()], axis=0) for i in range(len(origins))
            ]
        else:
            models.append(
                dict(
                    model="ensemble",
                    name=MODELS["ensemble"]["name"],
                    status="failed",
                    error="Need at least two successful constituent models",
                )
            )
    if not validation:
        raise ValueError(
            "All forecasting models failed: " + "; ".join(m["model"] + ": " + m["error"] for m in models)
        )
    actual = np.concatenate([series.iloc[s : s + h].to_numpy() for s in origins])
    initial = prepared[origins[0]].to_numpy()
    for model, predictions in validation.items():
        points = np.concatenate(predictions)
        stats = score(actual, points, initial, cfg.season_length)
        residuals = np.abs(actual - points)
        residuals = residuals[np.isfinite(residuals)]
        radius = float(
            np.quantile(
                residuals, min(1, np.ceil((len(residuals) + 1) * 0.9) / len(residuals)), method="higher"
            )
        )
        models.append(
            dict(
                model=model,
                name=MODELS[model]["name"],
                status="completed",
                validation=stats,
                interval_radius=radius,
                interval_method="90% empirical validation residual band; coverage not guaranteed",
            )
        )
        for i, stop in enumerate(origins):
            rows += prediction_rows(
                model,
                "validation",
                series.index[stop : stop + h],
                predictions[i],
                series.iloc[stop : stop + h].to_numpy(),
                i,
            )
    success = [m for m in models if m["status"] == "completed"]
    selected = min(success, key=lambda m: (m["validation"]["mae"], m["model"]))["model"]
    base["selected_model"] = selected
    holdout, future_predictions = {}, {}
    train = prepared[n - h]
    full = prepared[n]
    for j, result in enumerate(success):
        model = result["model"]
        progress(0.6 + 0.35 * j / len(success), f"Evaluating and saving {MODELS[model]['name']}")
        if model == "ensemble":
            holdout[model] = np.mean(list(holdout.values()), axis=0)
            future_predictions[model] = np.mean(list(future_predictions.values()), axis=0)
            result["members"] = [m for m in validation if m != "ensemble"]
        else:
            # A holdout failure fails the run: never switch winners using holdout results.
            holdout[model] = fitted_prediction(
                model, train, cfg, source["step_ns"], folder / "holdout" / model, progress
            )
            future_predictions[model] = fitted_prediction(
                model, full, cfg, source["step_ns"], folder / "artifacts" / model, progress
            )
        result["holdout"] = score(
            series.iloc[-h:].to_numpy(), holdout[model], train.to_numpy(), cfg.season_length
        )
        result["selected"] = model == selected
        result["trained_through"] = source["end"]
        rows += prediction_rows(
            model,
            "holdout",
            series.index[-h:],
            holdout[model],
            series.iloc[-h:].to_numpy(),
            radius=result["interval_radius"],
        )
        future = pd.date_range(
            series.index[-1], periods=h + 1, freq=pd.Timedelta(source["step_ns"], unit="ns")
        )[1:]
        rows += prediction_rows(
            model, "future", future, future_predictions[model], radius=result["interval_radius"]
        )
    base["models"] = models
    progress(0.98, "Saving forecasts, fitted models, and provenance")
    training_windows = [("validation", i, prepared[stop]) for i, stop in enumerate(origins)]
    training_windows += [("holdout", 0, train), ("future", 0, full)]
    return _finish(base, folder, rows, series, splits, training_windows)
