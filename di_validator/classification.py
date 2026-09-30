from __future__ import annotations

import html
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from . import store
from .adapters import CLASSIFIER_NAMES as MODEL_NAMES, PARAMETERS as ALLOWED_PARAMS, load_plugins, pipelines
from .labels import manual_for_window, registry_for_assets
from .query import interval_frame
from .reference_analysis import (
    accuracy_interval,
    extract_customer_features,
    make_heuristic_labels,
    positive_scores,
)
from .schemas import ExperimentConfig


def freeze(config):
    load_plugins()
    cfg = ExperimentConfig.model_validate(config).model_dump()
    for ds_id in cfg["dataset_ids"]:
        store.get("dataset", ds_id)
    cfg["label_snapshot"] = store.listing("label")
    cfg["relationship_snapshot"] = store.listing("relationship")
    cfg["code_version"] = code_version()
    registries = store.listing("registry")
    cfg["registry_version"] = registries[0]["id"] if registries else None
    if cfg["target"] == "PV" and cfg["label_policy"] == "heuristic":
        raise ValueError("PV benchmarks use registry or confirmed evidence; heuristic mode is for EV")
    if cfg["target"] == "EV" and cfg["label_policy"] not in {"confirmed", "heuristic"}:
        raise ValueError("EV benchmarks require confirmed labels or explicit heuristic agreement mode")
    if cfg["label_policy"] in {"registry", "provisional"} and not cfg["registry_version"]:
        raise ValueError("Import a PV registry first")
    if any(m not in MODEL_NAMES or m == "baseline" for m in cfg["models"]):
        raise ValueError("Select supported learned models; the majority baseline is always included")
    for model, params in cfg["parameters"].items():
        if model not in ALLOWED_PARAMS or not isinstance(params, dict) or set(params) - ALLOWED_PARAMS[model]:
            raise ValueError(f"Unsupported parameters for {model}")
    return cfg


def asset_groups(ids, relationships):
    parents = {}

    def root(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = root(parents[x])
        return parents[x]

    # Group every historical relationship conservatively, including topology changes.
    for r in relationships:
        parents[root(r["meter_id"])] = root(r["transformer_id"])
    return np.array([root(i) for i in ids])


def split_groups(y, groups, test_size, seed, folds):
    if len(set(y)) != 2 or pd.Series(y).value_counts().min() < 5:
        raise ValueError(
            f"Need at least five labeled assets in each class; found {pd.Series(y).value_counts().to_dict()}. Add labels or explicitly select exploratory evidence."
        )
    # Choose group-only splits by closeness to requested size and prevalence, never by model performance.
    n_groups = len(set(groups))
    n = min(n_groups, max(2, round(1 / test_size)))
    outer = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    candidates = [
        (tr, te)
        for tr, te in outer.split(np.zeros(len(y)), y, groups)
        if len(set(y[tr])) == 2 and len(set(y[te])) == 2
    ]
    if not candidates:
        raise ValueError("No valid grouped holdout contains both classes; add independent labeled groups")
    tr, te = min(
        candidates, key=lambda s: abs(len(s[1]) / len(y) - test_size) + abs(y[s[1]].mean() - y.mean())
    )
    group_support = min(len(set(groups[tr][y[tr] == c])) for c in [0, 1])
    k = min(folds, group_support)
    while k >= 2:
        inner = list(
            StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
                np.zeros(len(tr)), y[tr], groups[tr]
            )
        )
        if all(len(set(y[tr][a])) == 2 and len(set(y[tr][b])) == 2 for a, b in inner):
            return tr, te, inner
        k -= 1
    raise ValueError("Too few independent groups for two-class training cross-validation")


def metrics(y, pred, score):
    lo, hi = accuracy_interval(y, pred)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        accuracy_low=float(lo),
        accuracy_high=float(hi),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, score)),
        average_precision=float(average_precision_score(y, score)),
    )


def code_version():
    import sys

    paths = list(Path(__file__).parent.glob("*.py"))
    for module in filter(None, __import__("os").environ.get("DI_PLUGINS", "").split(",")):
        file = getattr(sys.modules.get(module.strip()), "__file__", None)
        if file:
            paths.append(Path(file))
    return store.digest({str(p): store.checksum(p) for p in paths})


def environment():
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={
            p: importlib.metadata.version(p)
            for p in ["pandas", "numpy", "scikit-learn", "scipy", "pyarrow", "duckdb"]
        },
        code_sha256=code_version(),
        features_sha256=store.checksum(Path(__file__).with_name("reference_analysis.py")),
    )


def run(config, progress=None):
    load_plugins()
    progress = progress or store.Progress()
    cfg = ExperimentConfig.model_validate(config)
    if cfg.code_version and cfg.code_version != code_version():
        raise ValueError(
            "Analysis code changed after this job was queued. Create a new experiment or restore the original code version."
        )
    experiment_id = store.uid()
    folder = store.workspace() / "experiments" / experiment_id
    folder.mkdir(parents=True)
    datasets = [store.get("dataset", i) for i in cfg.dataset_ids]
    if any(d["format"] != "wide_ami" or d["interval_seconds"] != 3600 for d in datasets):
        raise ValueError("These classifier features require hourly interval-energy data")
    if len({d["timezone"] for d in datasets}) != 1:
        raise ValueError("Classifier cohorts must share a declared local timezone")
    timezone = datasets[0]["timezone"]

    def timestamp(value):
        t = pd.Timestamp(value)
        return t.tz_localize(timezone) if t.tzinfo is None else t.tz_convert(timezone)

    start = timestamp(cfg.start) if cfg.start else max(timestamp(d["start"]) for d in datasets)
    end = timestamp(cfg.end) if cfg.end else min(timestamp(d["end"]) for d in datasets)
    registry = store.get("registry", cfg.registry_version) if cfg.registry_version else None
    if cfg.label_policy in {"registry", "provisional"}:
        end = min(end, timestamp(registry["cutoff"]) - pd.Timedelta(hours=1))
    if (end - start).total_seconds() < 30 * 86400 - 3600:
        raise ValueError("At least 30 days are required after applying the evidence cutoff")
    feature_parts, label_parts, audit, evidence_rows, quality_rows = [], [], [], [], []
    hashes, seen_assets = {}, set()
    for i, dataset in enumerate(datasets):
        progress(0.03 + i / len(datasets) * 0.3, f"Extracting features: {dataset['name']}")
        selected = (
            [a for a in cfg.asset_ids if a.startswith(f"{dataset['asset_level']}:{dataset['circuit']}:")]
            if cfg.asset_ids
            else None
        )
        if cfg.asset_ids and not selected:
            continue
        long = interval_frame(dataset, selected, start.isoformat(), end.isoformat())
        if long.empty:
            continue
        wide = long.pivot(index="t_ns", columns="asset_id", values="value")
        wide.index = pd.to_datetime(wide.index, utc=True).tz_convert(timezone)
        wide = wide.reindex(pd.date_range(start, end, freq="h"))
        if dataset["channels"][0]["unit"] == "Wh":
            wide /= 1000
        eligible = []
        for asset in wide:
            reason = ""
            observed = wide[asset].notna().sum()
            fingerprint = store.digest(pd.util.hash_pandas_object(wide[asset], index=True).tolist())
            if asset in seen_assets:
                raise ValueError(
                    "The same asset appears in multiple selected dataset versions; select one version"
                )
            seen_assets.add(asset)
            if observed / len(wide) < cfg.min_coverage:
                reason = "insufficient coverage"
            elif wide[asset].nunique() <= 1:
                reason = "empty or constant profile"
            elif fingerprint in hashes:
                reason = f"duplicate profile of {hashes[fingerprint]}"
            hashes[fingerprint] = asset
            quality_rows.append(
                dict(asset_id=asset, coverage=float(observed / len(wide)), exclusion_reason=reason)
            )
            if not reason:
                eligible.append(asset)
        wide = wide[eligible]
        if wide.empty:
            continue
        features = extract_customer_features(wide)

        def family(column):
            if column.startswith(("ramp", "residual", "moving_residual")):
                return "ramps"
            if column.startswith(("hour_", "month_", "autocorrelation")):
                return "shape"
            if column.startswith(("night", "morning", "midday", "evening", "weekend", "summer", "cool")):
                return "calendar"
            return "load"

        features = features[[c for c in features if family(c) in cfg.feature_groups]]
        features.index.name = "asset_id"
        feature_parts.append(features)
        assets = [store.get("asset", a) for a in eligible]
        values = pd.Series(np.nan, index=eligible)
        evidence = {a: "unknown" for a in eligible}
        if cfg.label_policy in {"registry", "provisional"}:
            rows, audits = registry_for_assets(
                registry, assets, start, end, cfg.label_policy == "provisional", folder
            )
            audit.extend(audits)
            for row in rows:
                values.loc[row["asset_id"]] = row["value"]
                evidence[row["asset_id"]] = row["evidence"]
        elif cfg.label_policy == "heuristic":
            rules, diagnostics = make_heuristic_labels(wide)
            diagnostics.to_csv(folder / f"heuristic-{i}.csv")
            values = rules.has_ev.astype(float)
            evidence = {a: "heuristic_unvalidated" for a in eligible}
        for asset in eligible:
            # Heuristic agreement keeps a single target definition; confirmed overrides apply only to evidence modes.
            confirmed = manual_for_window(cfg.label_snapshot or [], asset, cfg.target, start, end)
            if confirmed and cfg.label_policy != "heuristic":
                values.loc[asset] = confirmed["value"]
                evidence[asset] = f"confirmed: {confirmed['evidence']}"
            evidence_rows.append(
                dict(
                    asset_id=asset,
                    value=None if pd.isna(values.loc[asset]) else int(values.loc[asset]),
                    evidence=evidence[asset],
                    dataset_id=dataset["id"],
                )
            )
        label_parts.append(values)
    if not feature_parts:
        raise ValueError("No eligible assets in the selected window")
    features, labels = pd.concat(feature_parts), pd.concat(label_parts)
    features.to_csv(folder / "features.csv")
    pd.DataFrame(quality_rows).to_csv(folder / "quality.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(folder / "labels.csv", index=False)
    known = labels.dropna().index
    x, y = features.loc[known], labels.loc[known].to_numpy(dtype=int)
    groups = asset_groups(known, cfg.relationship_snapshot or [])
    tr, te, splits = split_groups(y, groups, cfg.test_size, cfg.seed, cfg.folds)
    manifest = pd.DataFrame({"asset_id": known, "group": groups, "label": y, "partition": "test", "fold": -1})
    manifest.loc[tr, "partition"] = "train"
    for fold, (_, val) in enumerate(splits):
        manifest.loc[tr[val], "fold"] = fold
    manifest.to_csv(folder / "splits.csv", index=False)
    classifiers = pipelines(cfg.seed)
    selected_models = ["baseline"] + list(dict.fromkeys(cfg.models))
    cv_rows, results, fitted, thresholds = [], [], {}, {}
    with threadpool_limits(limits=2):
        for m, code in enumerate(selected_models):
            progress(
                0.35 + 0.45 * m / len(selected_models), f"Training cross-validation: {MODEL_NAMES[code]}"
            )
            estimator = classifiers[MODEL_NAMES[code]]
            estimator.set_params(**{f"model__{k}": v for k, v in cfg.parameters.get(code, {}).items()})
            oof = np.zeros(len(tr))
            fold_metrics = []
            for fold, (train, val) in enumerate(splits):
                progress.check()
                model = clone(estimator).fit(x.iloc[tr[train]], y[tr[train]])
                score = positive_scores(model, x.iloc[tr[val]])
                oof[val] = score
                entry = metrics(y[tr[val]], model.predict(x.iloc[tr[val]]), score)
                fold_metrics.append(entry)
                cv_rows.append(dict(model=code, fold=fold, **entry))
            threshold = 0.5 if hasattr(estimator, "predict_proba") else 0.0
            if cfg.threshold_policy == "training_f1" and code != "baseline":
                precision, recall, cuts = precision_recall_curve(y[tr], oof)
                f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
                threshold = float(cuts[np.argmax(f1)])
            thresholds[code] = threshold
            fitted[code] = clone(estimator).fit(x.iloc[tr], y[tr])
            results.append(
                dict(
                    model=code,
                    name=MODEL_NAMES[code],
                    cv={k: float(np.mean([r[k] for r in fold_metrics])) for k in fold_metrics[0]},
                    cv_std={k: float(np.std([r[k] for r in fold_metrics], ddof=1)) for k in fold_metrics[0]},
                )
            )
        winner = max(
            [r for r in results if r["model"] != "baseline"],
            key=lambda r: (r["cv"]["average_precision"], r["cv"]["balanced_accuracy"]),
        )["model"]
        # Winner and thresholds are frozen before evaluating any holdout labels.
        predictions = []
        evidence_map = {r["asset_id"]: r for r in evidence_rows}
        for result in results:
            code = result["model"]
            model = fitted[code]
            score = positive_scores(model, x.iloc[te])
            pred = (
                model.predict(x.iloc[te]) if code == "baseline" else (score >= thresholds[code]).astype(int)
            )
            result["test"] = metrics(y[te], pred, score)
            result["confusion"] = confusion_matrix(y[te], pred, labels=[0, 1]).tolist()
            fpr, tpr, _ = roc_curve(y[te], score)
            precision, recall, _ = precision_recall_curve(y[te], score)
            result["curves"] = dict(
                fpr=fpr.tolist(), tpr=tpr.tolist(), precision=precision.tolist(), recall=recall.tolist()
            )
            result["threshold"] = thresholds[code]
            result["selected"] = code == winner
            for asset, truth, predicted, s in zip(known[te], y[te], pred, score):
                predictions.append(
                    dict(
                        asset_id=asset,
                        target=cfg.target,
                        model=code,
                        label=int(truth),
                        prediction=int(predicted),
                        score=float(s),
                        score_type="uncalibrated_model_probability"
                        if hasattr(model, "predict_proba")
                        else "decision_score",
                        evidence=evidence_map[asset]["evidence"],
                        dataset_id=evidence_map[asset]["dataset_id"],
                    )
                )
            joblib.dump(model, folder / f"{code}.joblib")
    pd.DataFrame(cv_rows).to_csv(folder / "cv_scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / "predictions.csv", index=False)
    (folder / "predictions.json").write_text(store.encode(predictions), encoding="utf-8")
    result = dict(
        id=experiment_id,
        kind="classification",
        name=cfg.name,
        config=cfg.model_dump(),
        effective_start=start.isoformat(),
        effective_end=end.isoformat(),
        target=cfg.target,
        label_policy=cfg.label_policy,
        selected_model=winner,
        models=results,
        counts=dict(
            eligible=len(features),
            labeled=len(known),
            unknown=len(features) - len(known),
            train=len(tr),
            test=len(te),
            positives=int(y.sum()),
            negatives=int((y == 0).sum()),
        ),
        split_id=store.digest(manifest.to_dict("records")),
        evidence_id=store.digest(evidence_rows),
        dataset_ids=cfg.dataset_ids,
        folder=str(folder),
        environment=environment(),
        registry_audit=audit,
        beats_baseline=next(r for r in results if r["model"] == winner)["cv"]["average_precision"]
        > results[0]["cv"]["average_precision"],
        interpretation={
            "heuristic": "Agreement with unvalidated EV screening rules",
            "provisional": "Exploratory registry evidence versus provisional negatives",
            "registry": "Registry-supported positives and confirmed labels",
            "confirmed": "Confirmed-label validation",
        }[cfg.label_policy],
    )
    (folder / "manifest.json").write_text(store.encode(result), encoding="utf-8")
    from .reports import classification_figures

    classification_figures(result, folder)
    report = "<!doctype html><html><meta charset='utf-8'><title>DI Validator report</title><style>body{font:16px system-ui;max-width:1100px;margin:48px auto;color:#1e293b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>"
    report += f"<h1>{html.escape(cfg.name)}</h1><p>{html.escape(result['interpretation'])}</p><p>{html.escape(start.isoformat())} to {html.escape(end.isoformat())}</p>"
    report += pd.DataFrame(
        [dict(model=r["name"], selected=r["selected"], **r["test"]) for r in results]
    ).to_html(index=False)
    report += "<img src='holdout-figures.svg' alt='Holdout ROC, precision-recall, and confusion matrix' style='width:100%'>"
    report += (
        f"<h2>Reproducibility manifest</h2><pre>{html.escape(json.dumps(result, indent=2))}</pre></html>"
    )
    (folder / "report.html").write_text(report, encoding="utf-8")
    progress(0.98, "Saving holdout results and reproducibility bundle")
    return store.put("experiment", result, experiment_id)
