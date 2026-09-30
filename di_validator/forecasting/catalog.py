"""Lightweight provider descriptions; no ML framework imports in HTTP handlers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from .. import store

MODELS = {
    "seasonal_naive": dict(
        name="Seasonal naive", group="base", module=None, package=None, mode="baseline", defaults={}
    ),
    "random_forest": dict(
        name="Random Forest",
        group="base",
        module="sklearn",
        package="scikit-learn",
        mode="train",
        defaults={"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2},
    ),
    "xgboost": dict(
        name="XGBoost",
        group="base",
        module="xgboost",
        package="xgboost",
        mode="train",
        defaults={"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
    ),
    "lightgbm": dict(
        name="LightGBM",
        group="base",
        module="lightgbm",
        package="lightgbm",
        mode="train",
        defaults={"n_estimators": 150, "num_leaves": 31, "learning_rate": 0.05},
    ),
    "prophet": dict(
        name="Prophet",
        group="base",
        module="prophet",
        package="prophet",
        mode="train",
        defaults={"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    ),
    "gluonts": dict(
        name="GluonTS · DeepAR",
        group="gluonts",
        module="gluonts",
        package="gluonts",
        mode="train",
        defaults={"epochs": 5, "num_batches_per_epoch": 20, "hidden_size": 40},
    ),
    "autogluon": dict(
        name="AutoGluon tree ensemble",
        group="autogluon_trees",
        module="autogluon.tabular",
        package="autogluon.tabular",
        mode="automl",
        defaults={"time_limit": 60},
    ),
    "neuralprophet": dict(
        name="NeuralProphet",
        group="neuralprophet",
        module="neuralprophet",
        package="neuralprophet",
        mode="train",
        defaults={"epochs": 20, "learning_rate": 0.01},
    ),
    "ensemble": dict(
        name="Ensemble · equal-weight mean",
        group="base",
        module=None,
        package=None,
        mode="ensemble",
        defaults={},
    ),
}
ALIASES = {"randomforecast": "random_forest", "randomforest": "random_forest"}
RUNTIMES = {
    "gluonts": {
        "python": "3.13",
        "requirements": ["numpy<3", "pandas<3", "scikit-learn", "joblib", "gluonts[torch]==0.17.0"],
    },
    "autogluon_trees": {"python": "3.13", "requirements": ["autogluon.tabular[lightgbm,xgboost]==1.6.3"]},
    "neuralprophet": {
        "python": "3.11",
        "requirements": [
            "neuralprophet==0.9.0",
            "torch==2.5.1",
            "pytorch-lightning==2.5.0.post0",
            "torchmetrics==1.6.1",
            "pandas<3",
            "scikit-learn",
            "joblib",
        ],
    },
}


def canonical(model):
    return ALIASES.get(model, model)


def runtime_root(group):
    return Path(os.environ.get("DI_FORECAST_RUNTIMES", store.workspace() / "forecast-runtimes")) / group


def interpreter(model):
    group = MODELS[canonical(model)]["group"]
    if group == "base":
        return Path(sys.executable)
    return runtime_root(group) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def catalog():
    result = []
    for identifier, spec in MODELS.items():
        version, reason, ready = None, "", True
        if spec["group"] == "base":
            if spec["module"]:
                ready = importlib.util.find_spec(spec["module"]) is not None
                if ready:
                    version = importlib.metadata.version(spec["package"])
                else:
                    reason = "Run uv sync --frozen to install the base forecasting dependencies."
        else:
            marker = runtime_root(spec["group"]) / "di-runtime.json"
            state = json.loads(marker.read_text()) if marker.exists() else {}
            provider = state.get("providers", {}).get(identifier, {})
            ready = interpreter(identifier).exists() and provider.get("import_ok", False)
            version = provider.get("version")
            reason = provider.get("error", "Install the isolated runtime.") if not ready else ""
        result.append(
            dict(
                id=identifier,
                **spec,
                ready=ready,
                version=version,
                reason=reason,
                runtime=str(interpreter(identifier)),
                aliases=[a for a, b in ALIASES.items() if b == identifier],
            )
        )
    return result


def checked_parameters(model, parameters):
    spec = MODELS[model]
    if set(parameters) - set(spec["defaults"]):
        raise ValueError(
            f"Unsupported parameters for {model}: {sorted(set(parameters) - set(spec['defaults']))}"
        )
    result = {**spec["defaults"], **parameters}
    for key, value in result.items():
        default = spec["defaults"][key]
        if isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 100000:
                raise ValueError(f"{model}.{key} must be a finite positive number")
            if isinstance(default, int) and not isinstance(value, int):
                raise ValueError(f"{model}.{key} must be an integer")
        elif not isinstance(value, str) or len(value) > 2048:
            raise ValueError(f"Invalid {model}.{key}")
    if model == "prophet" and result["seasonality_mode"] not in {"additive", "multiplicative"}:
        raise ValueError("Choose additive or multiplicative seasonality")
    return result
