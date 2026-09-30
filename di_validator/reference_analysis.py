"""Customer-level EV/PV screening and honest, reproducible model evaluation.

Used by Electric_Vehicle_Detection.ipynb. No meter IDs, labels, or screening
rule outputs are supplied as model features. Heuristic labels are not truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def normalize_id(value):
    return str(value).strip().strip("\"'").strip()


def load_readings(path, min_coverage=0.90, timestamp_timezone=None):
    """Read wide hourly AMI; preserve missing observations and audit exclusions.

    Naive timestamps are assumed to be local meter clock time unless a source
    timezone is supplied. A timezone-aware index is converted to Los Angeles.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1].")
    # pandas renames duplicate CSV headers; check the original header first.
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    ids = [normalize_id(c) for c in header[1:]]
    if not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Customer IDs must be nonempty and unique after normalization.")
    raw = pd.read_csv(path)
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], errors="raise"))
    if timestamps.isna().any() or timestamps.has_duplicates:
        raise ValueError("Missing or duplicate timestamps: reconcile these at the source.")
    if timestamps.tz is None and timestamp_timezone:
        timestamps = timestamps.tz_localize(timestamp_timezone, ambiguous="raise", nonexistent="raise")
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("America/Los_Angeles")
    readings = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    readings = readings.replace([np.inf, -np.inf], np.nan)
    readings.columns = pd.Index(ids, name="customer_id")
    readings.index = timestamps
    readings = readings.sort_index()
    if len(readings) < 24 * 30:
        raise ValueError("At least 30 days of hourly readings are needed for these features.")
    hourly = pd.date_range(readings.index.min(), readings.index.max(), freq="h")
    if not readings.index.isin(hourly).all():
        raise ValueError("Readings must lie on a regular hourly grid.")
    readings = readings.reindex(hourly)  # Missing timestamps become NaN, never zero.
    quality = pd.DataFrame(
        {
            "observed_reads": readings.notna().sum(),
            "expected_reads": len(readings),
            "coverage": readings.notna().mean(),
            "unique_values": readings.nunique(),
            "negative_reads": readings.lt(0).sum(),
        }
    )
    quality["exclusion_reason"] = ""
    quality.loc[quality.coverage < min_coverage, "exclusion_reason"] = "insufficient coverage"
    quality.loc[quality.unique_values <= 1, "exclusion_reason"] = "empty or constant profile"
    # Exact duplicate profiles can otherwise leak across the customer split.
    duplicate = readings.T.duplicated(keep="first")
    quality.loc[duplicate, "exclusion_reason"] = "duplicate profile"
    quality["eligible"] = quality.exclusion_reason.eq("")
    if not quality.eligible.any():
        raise ValueError("No customers meet data quality requirements.")
    return readings.loc[:, quality.eligible], quality


def _fraction(mask, values):
    """Fraction of observed values satisfying a condition, excluding missing data."""
    return mask.sum() / values.notna().sum().replace(0, np.nan)


def extract_customer_features(readings):
    """One row per customer. Every summary uses only that customer's own year.

    No population-fitted transformations here; imputation/scaling occur inside
    each training fold. Units are the source meter units, which must be verified
    before interpreting values as kW. Hourly kWh equals mean kW numerically.
    """
    x = readings
    hour, month = x.index.hour, x.index.month
    f = pd.DataFrame(index=x.columns)
    f["mean_load"] = x.mean()
    f["std_load"] = x.std()
    for q in [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        f[f"load_p{int(q * 100):02d}"] = x.quantile(q)
    # Relative scales avoid imposing unverified kW thresholds on these meters.
    scale = x.abs().median().clip(lower=1e-6)
    f["coefficient_of_variation"] = x.std() / scale
    f["peak_to_median"] = x.quantile(0.99) / scale
    f["zero_fraction"] = _fraction(x.eq(0), x)
    f["negative_fraction"] = _fraction(x.lt(0), x)
    delta = x.diff()  # NaN across missing reads; gaps are not treated as ramps.
    f["ramp_std"] = delta.std()
    f["ramp_abs_p95"] = delta.abs().quantile(0.95)
    residual = x - x.rolling(10, min_periods=10).mean()
    f["moving_residual_std"] = residual.std()
    f["moving_residual_p95"] = residual.quantile(0.95)
    f["moving_residual_p05"] = residual.quantile(0.05)
    windows = {
        "night": (hour >= 22) | (hour <= 5),
        "morning": (hour >= 6) & (hour <= 9),
        "midday": (hour >= 10) & (hour <= 15),
        "evening": (hour >= 17) & (hour <= 21),
    }
    for name, mask in windows.items():
        v = x.loc[mask]
        f[f"{name}_mean_relative"] = v.mean() / scale
        f[f"{name}_p95_relative"] = v.quantile(0.95) / scale
        f[f"{name}_variability"] = v.std() / scale
    f["midday_export_fraction"] = _fraction(x.loc[windows["midday"]].lt(0), x.loc[windows["midday"]])
    weekend = x.index.dayofweek >= 5
    f["weekend_weekday_ratio"] = x.loc[weekend].mean() / x.loc[~weekend].mean().abs().clip(lower=1e-6)
    summer = np.isin(month, [6, 7, 8, 9])
    cool = np.isin(month, [10, 11, 12, 1, 2, 3, 4])
    f["summer_cool_ratio"] = x.loc[summer].mean() / x.loc[cool].mean().abs().clip(lower=1e-6)
    for season, mask in [("summer", summer), ("cool", cool)]:
        for name in ["night", "midday", "evening"]:
            f[f"{season}_{name}_relative"] = x.loc[mask & windows[name]].mean() / scale
    for h in range(24):
        f[f"hour_{h:02d}_relative"] = x.loc[hour == h].mean() / scale
    for m in range(1, 13):
        f[f"month_{m:02d}_relative"] = x.loc[month == m].mean() / scale
    for lag in [1, 24, 168]:
        f[f"autocorrelation_{lag}h"] = x.corrwith(x.shift(lag))
    # Keep the original notebook's idea of standardized residual/ramp histograms.
    edges = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
    for prefix, v in [("residual", residual), ("ramp", delta)]:
        z = v.div(v.std().replace(0, np.nan), axis=1)
        for b, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            f[f"{prefix}_hist_{b}"] = _fraction(z.ge(left) & z.lt(right), z)
    return f.replace([np.inf, -np.inf], np.nan).astype(float)


def make_heuristic_labels(readings):
    """Transparent, fixed rules for review candidates; NOT verified ownership.

    Zero means 'rule did not trigger', never 'confirmed absence'. Rule diagnostics
    are returned separately, so IDs and the exact rule flags cannot be features.
    """
    dates = readings.index.normalize()
    hour = readings.index.hour
    cool_months = {10, 11, 12, 1, 2, 3, 4}
    diagnostic_rows = []
    for customer, s in readings.items():
        day_observed = s.groupby(dates).count().ge(20)
        good_days = day_observed.index[day_observed]
        cool_days = sum(d.month in cool_months for d in good_days)
        baseline = s.rolling(24, min_periods=18).quantile(0.25).shift(1)
        iqr = s.quantile(0.75) - s.quantile(0.25)
        elevation = max(1.5 * iqr, 0.75 * s.abs().median(), 1e-6)
        high = ((s - baseline) >= elevation).to_numpy()
        # Contiguous blocks must have observable boundaries, so outages do not
        # create artificial charging events. Hourly data misses shorter events.
        boundaries = np.diff(np.r_[False, high, False].astype(int))
        starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        event_dates = set()
        for start, stop in zip(starts, stops):
            if not (2 <= stop - start <= 8) or start == 0 or stop >= len(s):
                continue
            if not (hour[start] >= 18 or hour[start] <= 6):
                continue
            block = s.iloc[start:stop]
            before, after = s.iloc[start - 1], s.iloc[stop]
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            if s.index[start].normalize() not in good_days:
                continue
            if (
                block.iloc[0] - before >= 0.5 * elevation
                and block.iloc[-1] - after >= 0.5 * elevation
                and block.std(ddof=0) <= 0.4 * elevation
            ):
                event_dates.add(s.index[start].normalize())
        cool_events = sum(d.month in cool_months for d in event_dates)
        night_event_fraction = len(event_dates) / max(len(good_days), 1)
        cool_event_fraction = cool_events / max(cool_days, 1)
        # PV screening: repeated low midday import relative to both shoulders.
        daily = {}
        for name, mask, required in [
            ("morning", (hour >= 6) & (hour <= 9), 3),
            ("midday", (hour >= 10) & (hour <= 15), 5),
            ("evening", (hour >= 17) & (hour <= 20), 3),
        ]:
            group = s.loc[mask].groupby(dates[mask])
            daily[name] = group.mean().where(group.count() >= required)
        daily = pd.DataFrame(daily).dropna()
        shoulder = daily[["morning", "evening"]].min(axis=1)
        # Avoid a trivial zero/zero trough on near-zero or inactive meters.
        valid = shoulder > max(0.1 * s.abs().median(), 1e-6)
        valley = daily.loc[valid, "midday"] / shoulder.loc[valid]
        valley_fraction = float(valley.lt(0.55).mean()) if len(valley) else np.nan
        valley_median = float(valley.median()) if len(valley) else np.nan
        export_fraction = float(s.loc[(hour >= 10) & (hour <= 15)].dropna().lt(0).mean())
        diagnostic_rows.append(
            {
                "customer_id": customer,
                "night_event_days": len(event_dates),
                "cool_event_days": cool_events,
                "night_event_fraction": night_event_fraction,
                "cool_event_fraction": cool_event_fraction,
                "valid_pv_days": len(valley),
                "midday_valley_fraction": valley_fraction,
                "median_midday_shoulder_ratio": valley_median,
                "midday_export_fraction": export_fraction,
                "has_ev": int(night_event_fraction >= 0.04 and cool_event_fraction >= 0.02),
                "has_pv": int(
                    export_fraction >= 0.05
                    or (len(valley) >= 60 and valley_fraction >= 0.35 and valley_median < 0.80)
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("customer_id")
    labels = diagnostics[["has_ev", "has_pv"]].astype("Int64")
    diagnostics["label_source"] = "heuristic_unvalidated"
    return labels, diagnostics


def read_confirmed_labels(path, customer_ids):
    """Explicitly requested label files fail loudly; unknowns stay unknown."""
    labels = pd.read_csv(path, dtype={"customer_id": "string"})
    required = {"customer_id", "has_ev", "has_pv"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Label CSV needs columns: {sorted(required)}")
    if labels.customer_id.isna().any():
        raise ValueError("Label CSV contains missing customer IDs.")
    labels["customer_id"] = labels.customer_id.map(normalize_id)
    if labels.customer_id.eq("").any() or labels.customer_id.duplicated().any():
        raise ValueError("Label customer IDs must be nonempty and unique.")
    labels = labels.set_index("customer_id")
    for col in ["has_ev", "has_pv"]:
        values = pd.to_numeric(labels[col], errors="raise")
        if not values.dropna().isin([0, 1]).all():
            raise ValueError(f"{col} must contain 0, 1, or blank for unknown.")
        labels[col] = values.astype("Int64")
    extra = labels.index.difference(customer_ids)
    return labels[["has_ev", "has_pv"]].reindex(customer_ids), extra


def registry_pv_labels(
    path, customer_ids, period_start, period_end, no_record_as_negative=True, expected_circuit=None
):
    """Join DER evidence at STRUCT_NUM, never at an inferred individual meter.

    Positive: an unambiguous PV project with PTO ISSUED, a real PTO date on/before
    the feature-window start, and no contract end within that window. Pending,
    cancelled, storage-only, and first installations during the window stay
    unknown unless another eligible PV project establishes prior PV presence.
    No DER record may be used as an EXPLICITLY PROVISIONAL negative.
    """
    wanted = {
        "STRUCT_NUM",
        "PROJECT_ID",
        "TECH_TYPE",
        "INTERCONNECTION_STATUS",
        "PERMTO_OPERATE_DT",
        "CONTRACT_END_DT",
        "SERV_PNT_ID",
        "CIRCUIT_KEY",
        "EDW_MODIFIED_DATE",
    }
    records = pd.read_csv(path, dtype="string", usecols=lambda c: c in wanted)
    required = {"STRUCT_NUM", "PROJECT_ID", "TECH_TYPE", "INTERCONNECTION_STATUS", "PERMTO_OPERATE_DT"}
    if not required.issubset(records.columns):
        raise ValueError(f"DER CSV needs columns: {sorted(required)}")
    records["customer_id"] = records.STRUCT_NUM.map(lambda x: normalize_id(x) if pd.notna(x) else "")
    records["technology"] = records.TECH_TYPE.str.strip().str.upper().fillna("UNKNOWN")
    records["status"] = records.INTERCONNECTION_STATUS.str.strip().str.upper().fillna("UNKNOWN")
    records["pto_date"] = pd.to_datetime(records.PERMTO_OPERATE_DT, errors="coerce")
    # 1900-01-01 is a source-system placeholder, including on pending records.
    records.loc[records.pto_date.le(pd.Timestamp("1900-01-01")), "pto_date"] = pd.NaT
    start, end = pd.Timestamp(period_start), pd.Timestamp(period_end)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    if start > end:
        raise ValueError("DER label feature window must have start <= end.")
    if expected_circuit and "CIRCUIT_KEY" in records:
        circuits = set(records.CIRCUIT_KEY.dropna().str.strip())
        if circuits != {expected_circuit}:
            raise ValueError(f"DER circuit mismatch: expected {expected_circuit}, found {sorted(circuits)}")
    records["is_pv"] = records.technology.isin(
        [
            "PHOTOVOLTAIC",
            "SOLAR PV",
            "SOLAR",
            "PHOTOVOLTAIC PANELS",
            "SOLAR PV (EQUIPMENT NOT LISTED)",
        ]
    )
    records["is_storage"] = records.technology.isin(["BATTERY", "ENERGY STORAGE", "STORAGE"])
    keys = ["customer_id", "PROJECT_ID", "technology"]
    # Duplicate inverter/application rows cannot turn one installation into many.
    conflicts = records.groupby(keys, dropna=False).status.transform("nunique").gt(1)
    records["project_status_conflict"] = conflicts
    records["issued_pv"] = (
        records.is_pv
        & records.status.eq("PTO ISSUED")
        & ~conflicts
        & records.PROJECT_ID.notna()
        & records.pto_date.notna()
    )
    contract = records.get("CONTRACT_END_DT", pd.Series(pd.NA, index=records.index, dtype="string"))
    open_ended = contract.isna() | contract.str.startswith("9999-12-31").fillna(False)
    contract_end = pd.to_datetime(contract.where(~open_ended), errors="coerce")
    valid_through_window = open_ended | contract_end.gt(end)
    records["pv_present_through_window"] = (
        records.issued_pv & records.pto_date.le(start) & valid_through_window
    )
    records["pv_first_issued_in_window"] = (
        records.issued_pv & records.pto_date.gt(start) & records.pto_date.le(end)
    )
    groups = {key: group for key, group in records.groupby("customer_id")}
    rows = []
    for customer in customer_ids:
        group = groups.get(normalize_id(customer))
        if group is None:
            rows.append(
                {
                    "customer_id": customer,
                    "registry_records": 0,
                    "pv_projects": 0,
                    "storage_projects": 0,
                    "service_points": 0,
                    "pending_pv_records": 0,
                    "cancelled_pv_records": 0,
                    "earliest_issued_pv_date": pd.NaT,
                    "has_pv": 0 if no_record_as_negative else pd.NA,
                    "label_basis": "no_der_record_provisional_negative"
                    if no_record_as_negative
                    else "no_der_record_unknown",
                }
            )
            continue
        positive = bool(group.pv_present_through_window.any())
        if positive:
            basis = "pv_pto_before_feature_window"
        elif group.project_status_conflict.any():
            basis = "conflicting_project_status_unknown"
        elif group.pv_first_issued_in_window.any():
            basis = "first_pv_pto_during_window_unknown"
        elif (group.issued_pv & group.pto_date.gt(end)).any():
            basis = "pv_pto_after_window_unknown"
        elif group.is_pv.any():
            basis = "pv_pending_cancelled_or_unverified_unknown"
        elif group.is_storage.any():
            basis = "storage_without_verified_pv_unknown"
        else:
            basis = "other_der_without_verified_pv_unknown"
        rows.append(
            {
                "customer_id": customer,
                "registry_records": len(group),
                "pv_projects": group.loc[group.is_pv, "PROJECT_ID"].nunique(),
                "storage_projects": group.loc[group.is_storage, "PROJECT_ID"].nunique(),
                "service_points": group.SERV_PNT_ID.nunique() if "SERV_PNT_ID" in group else 0,
                "pending_pv_records": int((group.is_pv & group.status.eq("PENDING")).sum()),
                "cancelled_pv_records": int((group.is_pv & group.status.eq("CANCELLED")).sum()),
                "earliest_issued_pv_date": group.loc[group.issued_pv, "pto_date"].min(),
                "has_pv": 1 if positive else pd.NA,
                "label_basis": basis,
            }
        )
    summary = pd.DataFrame(rows).set_index("customer_id")
    summary["has_pv"] = summary.has_pv.astype("Int64")
    summary["label_source"] = (
        "der_registry_with_provisional_negatives" if no_record_as_negative else "der_registry_positive_only"
    )
    matched = records.customer_id.isin(customer_ids)
    audit = {
        "source_records": len(records),
        "source_structures": int(records.loc[records.customer_id.ne(""), "customer_id"].nunique()),
        "matched_records": int(matched.sum()),
        "unmatched_records": int((~matched).sum()),
        "missing_structure_records": int(records.customer_id.eq("").sum()),
        "matched_structures": int(summary.registry_records.gt(0).sum()),
        "structures_with_multiple_service_points": int(summary.service_points.gt(1).sum()),
        "project_status_conflict_rows": int(conflicts.sum()),
        "feature_start": str(start),
        "feature_end": str(end),
        "negative_assumption": "no DER record is provisional absence"
        if no_record_as_negative
        else "unmatched is unknown",
        "join_granularity": "STRUCT_NUM; any qualifying PV at the AMI structure",
    }
    counts = records.groupby(["technology", "status"]).size().rename("records").reset_index()
    return summary.has_pv.copy(), summary, audit, counts


def build_classifiers(seed=42):
    """Modest fixed configurations for hundreds of customer examples."""
    classifiers = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            C=1, class_weight="balanced", max_iter=4000, random_state=seed
        ),
        "RBF SVM": SVC(C=1, gamma="scale", class_weight="balanced", random_state=seed),
        "Random forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            max_bins=32,
            min_samples_leaf=12,
            l2_regularization=2,
            learning_rate=0.06,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    result = {}
    for name, classifier in classifiers.items():
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if name in {"Logistic regression", "RBF SVM"}:
            steps.append(("scale", StandardScaler()))
        result[name] = Pipeline(steps + [("model", classifier)])
    return result


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def positive_scores(estimator, x):
    # SVM decision values are ranking scores, deliberately not called probabilities.
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, list(estimator.classes_).index(1)]
    return estimator.decision_function(x)


def accuracy_interval(y, pred, z=1.959963984540054):
    """Wilson 95% interval; sampling uncertainty only, not heuristic label error."""
    n = len(y)
    p = accuracy_score(y, pred)
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


@dataclass
class BenchmarkResult:
    target: str
    label_source: str
    train_ids: pd.Index
    test_ids: pd.Index
    folds: pd.DataFrame
    cv: pd.DataFrame
    test: pd.DataFrame
    predictions: pd.DataFrame
    models: dict
    selected_name: str
    beats_baseline: bool
    fold_manifest: pd.DataFrame


def benchmark_target(
    features, labels, target, label_source, test_size=0.20, seed=42, max_folds=5, classifiers=None
):
    """Select on training CV only; holdout scores never choose the winner.

    Separate, stratified customer splits are used for EV and PV. A customer can
    be positive for both. There is no interval-level random split and no fitting
    on unknown labels. Insufficient class support raises an actionable error.
    """
    if not features.index.is_unique or not labels.index.is_unique:
        raise ValueError("Features and labels need unique customer IDs.")
    if not 0 < test_size < 1 or max_folds < 2:
        raise ValueError("Need 0 < test_size < 1 and at least 2 CV folds.")
    y = labels.reindex(features.index).dropna()
    if not y.isin([0, 1]).all():
        raise ValueError("Target labels must be binary 0/1 or missing.")
    y = y.astype(int)
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError(
            f"{target}: need at least 5 labeled customers in EACH class; found {counts.to_dict()}."
        )
    train_ids, test_ids = train_test_split(y.index, test_size=test_size, stratify=y, random_state=seed)
    train_ids, test_ids = pd.Index(train_ids), pd.Index(test_ids)
    assert train_ids.intersection(test_ids).empty
    x_train, x_test = features.loc[train_ids], features.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(f"{target}: adjust test_size to retain both classes in each partition.")
    n_folds = min(max_folds, int(y_train.value_counts().min()))
    if n_folds < 2:
        raise ValueError(f"{target}: too few minority training customers for cross-validation.")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x_train, y_train))
    fold_manifest = pd.DataFrame(
        {"customer_id": train_ids, "label": y_train.to_numpy(), "validation_fold": -1}
    )
    for fold, (_, val_idx) in enumerate(splits, start=1):
        fold_manifest.loc[val_idx, "validation_fold"] = fold
    classifiers = build_classifiers(seed) if classifiers is None else classifiers
    cv_rows, cv_summary, fitted = [], [], {}
    with threadpool_limits(limits=2):
        for name, estimator in classifiers.items():
            scores = cross_validate(
                estimator,
                x_train,
                y_train,
                cv=splits,
                scoring=SCORING,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            summary = {"model": name}
            for metric in SCORING:
                v = scores[f"test_{metric}"]
                summary[f"cv_{metric}"] = float(v.mean())
                summary[f"cv_{metric}_std"] = float(v.std(ddof=1))
            summary["cv_fit_seconds"] = float(scores["fit_time"].sum())
            cv_summary.append(summary)
            for fold in range(n_folds):
                cv_rows.append(
                    {
                        "model": name,
                        "fold": fold + 1,
                        **{metric: scores[f"test_{metric}"][fold] for metric in SCORING},
                    }
                )
            fitted[name] = clone(estimator).fit(x_train, y_train)
        cv_table = pd.DataFrame(cv_summary).set_index("model")
        candidates = cv_table.drop(index="Majority baseline", errors="ignore")
        if candidates.empty:
            raise ValueError("Provide at least one non-baseline classifier.")
        # Freeze selection BEFORE looking at the test results.
        selected = candidates.sort_values(
            ["cv_average_precision", "cv_balanced_accuracy"], ascending=False, kind="stable"
        ).index[0]
        beats_baseline = (
            "Majority baseline" not in cv_table.index
            or cv_table.loc[selected, "cv_average_precision"]
            > cv_table.loc["Majority baseline", "cv_average_precision"]
        )
        test_rows, predictions = [], []
        for name, estimator in fitted.items():
            pred = estimator.predict(x_test)
            score = positive_scores(estimator, x_test)
            low, high = accuracy_interval(y_test, pred)
            test_rows.append(
                {
                    "model": name,
                    "train_accuracy": accuracy_score(y_train, estimator.predict(x_train)),
                    "accuracy": accuracy_score(y_test, pred),
                    "accuracy_low": low,
                    "accuracy_high": high,
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, score),
                    "average_precision": average_precision_score(y_test, score),
                    "test_customers": len(test_ids),
                    "test_positives": int(y_test.sum()),
                    "selected_on_training_cv": name == selected,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "customer_id": test_ids,
                        "model": name,
                        "label": y_test.to_numpy(),
                        "prediction": pred,
                        "score": score,
                        "score_type": "model_probability_uncalibrated"
                        if hasattr(estimator, "predict_proba")
                        else "decision_score",
                        "label_source": label_source,
                        "target": target,
                    }
                )
            )
    result = BenchmarkResult(
        target,
        label_source,
        train_ids,
        test_ids,
        pd.DataFrame(cv_rows),
        cv_table,
        pd.DataFrame(test_rows).set_index("model"),
        pd.concat(predictions, ignore_index=True),
        fitted,
        selected,
        beats_baseline,
        fold_manifest,
    )
    return result
