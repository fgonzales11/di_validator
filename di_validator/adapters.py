"""Algorithm extension points for trusted, locally installed Python modules.

Set DI_PLUGINS=your_package.your_module (comma-separated) in BOTH API and worker
environments. Modules call register_classifier / register_event_detector on import.
All classification estimators receive the same training-only preprocessing pipeline.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable, Protocol

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSIFIER_NAMES = {
    "baseline": "Majority baseline",
    "logistic": "Logistic regression",
    "svm": "RBF SVM",
    "forest": "Random forest",
    "boosting": "Gradient boosting",
}
PARAMETERS = {
    "logistic": {"C"},
    "svm": {"C", "gamma"},
    "forest": {"n_estimators", "max_depth", "min_samples_leaf"},
    "boosting": {"max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization"},
}
CUSTOM_CLASSIFIERS: dict[str, Callable] = {}
EVENT_DETECTORS: dict[str, dict] = {}
_loaded: set[str] = set()


class EventProcessor(Protocol):
    events: list[dict]

    def process_chunk(self, frame: pd.DataFrame) -> list[dict]: ...


def register_classifier(identifier: str, name: str, factory: Callable, parameters=()):
    """factory(seed) returns a cloneable sklearn classifier, with predict and a ranking score.

    Parameters are classifier parameters (not pipeline parameter names).
    IDs, labels and screening diagnostics are never supplied as feature columns.
    """
    if identifier in CLASSIFIER_NAMES:
        raise ValueError(f"Algorithm identifier already registered: {identifier}")
    CLASSIFIER_NAMES[identifier] = name
    PARAMETERS[identifier] = set(parameters)
    CUSTOM_CLASSIFIERS[identifier] = factory


def register_event_detector(identifier: str, name: str, factory: Callable, required_channel_kinds: set[str]):
    """factory(dataset, settings) returns a causal EventProcessor.

    Event records use id, asset_id, kind, start, end, emitted_at (seconds from origin).
    The processor must carry state between chunks and reset it at missing samples/gaps.
    """
    if identifier == "baseline-events" or identifier in EVENT_DETECTORS:
        raise ValueError(f"Algorithm identifier already registered: {identifier}")
    EVENT_DETECTORS[identifier] = dict(name=name, factory=factory, channels=required_channel_kinds)


def load_plugins():
    for module in filter(None, (v.strip() for v in os.environ.get("DI_PLUGINS", "").split(","))):
        if module not in _loaded:
            importlib.import_module(module)
            _loaded.add(module)


def pipelines(seed):
    from .reference_analysis import build_classifiers

    load_plugins()
    result = build_classifiers(seed)
    for identifier, factory in CUSTOM_CLASSIFIERS.items():
        result[CLASSIFIER_NAMES[identifier]] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", factory(seed)),
            ]
        )
    return result
