"""Provider implementations runnable in isolated Python environments.

Only NumPy/pandas are imported eagerly. No API, database, or browser dependencies.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd

PACKAGES = {
    "random_forest": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "prophet": "prophet",
    "gluonts": "gluonts",
    "autogluon": "autogluon.tabular",
    "neuralprophet": "neuralprophet",
}


def environment():
    packages = {}
    for p in ["numpy", "pandas", "scikit-learn", "torch", "joblib", *PACKAGES.values()]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), packages=packages)


def calendar(tick):
    t = pd.Timestamp(int(tick))
    hour = t.hour + t.minute / 60 + t.second / 3600
    return [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * t.dayofweek / 7),
        np.cos(2 * np.pi * t.dayofweek / 7),
    ]


def lag_data(y, times, lags):
    # Target at i sees only y[:i]; no target-window values become regressors.
    x = np.array([np.r_[y[i - lags : i][::-1], calendar(times[i])] for i in range(lags, len(y))])
    return x, y[lags:]


def predict(
    model, y, times, horizon, step_ns, settings, output, *, seed=42, lags=24, season_length=24, reuse=None
):
    y, times = np.asarray(y, dtype=float), np.asarray(times, dtype=np.int64)
    if not np.isfinite(y).all() or len(y) != len(times):
        raise ValueError("Provider input must contain aligned, finite training observations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    existing = Path(reuse) if reuse else None
    current_environment = environment()
    if existing:
        previous = json.loads((existing / "model.json").read_text())
        if previous["environment"] != current_environment:
            raise ValueError(
                "Saved model environment differs. Restore its recorded versions or train a new model."
            )
        if previous["model"] != model or previous["lags"] != lags or previous["step_ns"] != step_ns:
            raise ValueError("Saved model feature schema or time resolution differs")
    state = dict(
        model=model,
        settings=settings,
        environment=current_environment,
        lags=lags,
        step_ns=int(step_ns),
        horizon=horizon,
        seed=seed,
        season_length=season_length,
    )
    random.seed(seed)
    np.random.seed(seed)
    future = times[-1] + np.arange(1, horizon + 1, dtype=np.int64) * step_ns
    freq = pd.tseries.frequencies.to_offset(pd.Timedelta(step_ns, unit="ns")).freqstr
    dates = pd.to_datetime(times)
    if model == "seasonal_naive":
        prediction = np.resize(y[-min(season_length, len(y)) :], horizon)
    elif model in {"random_forest", "xgboost", "lightgbm", "autogluon"}:
        import joblib

        columns = [f"lag_{i + 1}" for i in range(lags)] + [
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
        ]
        if existing:
            if model == "autogluon":
                from autogluon.tabular import TabularPredictor

                estimator = TabularPredictor.load(str(existing / "predictor"), require_version_match=True)
            else:
                estimator = joblib.load(existing / "estimator.joblib")
        else:
            x, target = lag_data(y, times, lags)
            x = pd.DataFrame(x, columns=columns)
            if len(target) < 10:
                raise ValueError("Need at least ten training targets after creating lag features")
            if model == "random_forest":
                from sklearn.ensemble import RandomForestRegressor

                estimator = RandomForestRegressor(**settings, random_state=seed, n_jobs=1)
            elif model == "xgboost":
                from xgboost import XGBRegressor

                estimator = XGBRegressor(
                    **settings, random_state=seed, n_jobs=2, objective="reg:squarederror"
                )
            elif model == "lightgbm":
                from lightgbm import LGBMRegressor

                estimator = LGBMRegressor(**settings, random_state=seed, n_jobs=2, verbosity=-1)
            else:
                from autogluon.tabular import TabularPredictor

                table = x.assign(forecast_target=target)
                validation_rows = max(2, min(100, len(table) // 5))
                estimator = TabularPredictor(
                    label="forecast_target",
                    problem_type="regression",
                    eval_metric="mean_absolute_error",
                    path=str(output / "predictor"),
                    verbosity=1,
                )
                estimator.fit(
                    table.iloc[:-validation_rows],
                    tuning_data=table.iloc[-validation_rows:],
                    hyperparameters={"GBM": {}, "RF": {}, "XT": {}, "XGB": {}},
                    time_limit=settings["time_limit"],
                    num_cpus=2,
                )
            if model != "autogluon":
                estimator.fit(x, target)
        prediction, history = [], list(y)
        for tick in future:
            row = pd.DataFrame([np.r_[history[-lags:][::-1], calendar(tick)]], columns=columns)
            value = float(np.asarray(estimator.predict(row))[0])
            prediction.append(value)
            history.append(value)
        if model != "autogluon":
            joblib.dump(estimator, output / "estimator.joblib")
        elif existing:
            import shutil

            shutil.copytree(existing / "predictor", output / "predictor", dirs_exist_ok=True)
    elif model == "prophet":
        from prophet import Prophet
        from prophet.serialize import model_from_json, model_to_json

        estimator = (
            model_from_json((existing / "prophet.json").read_text())
            if existing
            else Prophet(**settings, uncertainty_samples=0)
        )
        if not existing:
            estimator.fit(pd.DataFrame(dict(ds=dates, y=y)), seed=seed)
        prediction = estimator.predict(pd.DataFrame(dict(ds=pd.to_datetime(future))))["yhat"].to_numpy()
        (output / "prophet.json").write_text(model_to_json(estimator), encoding="utf-8")
    elif model == "gluonts":
        import torch
        from lightning.pytorch import seed_everything
        from gluonts.dataset.common import ListDataset
        from gluonts.model.predictor import Predictor
        from gluonts.torch.model.deepar import DeepAREstimator

        torch.set_num_threads(2)
        seed_everything(seed, workers=True)
        data = ListDataset([{"start": pd.Period(dates[0], freq=freq), "target": y}], freq=freq)
        if existing:
            estimator = Predictor.deserialize(existing / "predictor")
        else:
            estimator = DeepAREstimator(
                freq=freq,
                prediction_length=horizon,
                context_length=min(len(y) // 2, max(lags, horizon * 2)),
                hidden_size=settings["hidden_size"],
                num_batches_per_epoch=settings["num_batches_per_epoch"],
                lags_seq=list(range(1, min(lags, 24) + 1)),
                batch_size=16,
                trainer_kwargs={
                    "max_epochs": settings["epochs"],
                    "accelerator": "cpu",
                    "devices": 1,
                    "enable_progress_bar": False,
                    "logger": False,
                    "default_root_dir": str(output / "training"),
                },
            ).train(data)
        # Fix sampling independently of how much RNG state training consumed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        prediction = next(iter(estimator.predict(data))).mean[:horizon]
        (output / "predictor").mkdir(exist_ok=True)
        estimator.serialize(output / "predictor")
    elif model == "neuralprophet":
        import torch
        from neuralprophet import NeuralProphet, set_random_seed, save, load

        torch.set_num_threads(2)
        set_random_seed(seed)
        table = pd.DataFrame(dict(ds=dates, y=y))
        if existing:
            estimator = load(str(existing / "neuralprophet.np"), map_location="cpu")
        else:
            estimator = NeuralProphet(
                n_lags=lags,
                n_forecasts=horizon,
                epochs=settings["epochs"],
                learning_rate=settings["learning_rate"],
                accelerator="cpu",
                yearly_seasonality=False,
                weekly_seasonality="auto",
                daily_seasonality="auto",
            )
            estimator.fit(table, freq=freq, progress=None)
        future_table = estimator.make_future_dataframe(
            table, periods=estimator.n_forecasts, n_historic_predictions=False
        )
        raw = estimator.predict(future_table, raw=True)
        # Raw format: one forecast-origin row, with step0..step(H-1) columns.
        prediction = raw[[f"step{i}" for i in range(horizon)]].iloc[-1].to_numpy(dtype=float)
        save(estimator, str(output / "neuralprophet.np"))
    else:
        raise ValueError(f"Unknown forecasting provider: {model}")
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if len(prediction) != horizon or not np.isfinite(prediction).all():
        raise ValueError(f"{model} returned an invalid forecast; expected {horizon} finite predictions")
    (output / "model.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return prediction
