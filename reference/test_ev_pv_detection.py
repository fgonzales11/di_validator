"""Guard label integrity, customer isolation, and training-only model selection."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent))
import ev_pv_detection as detection


def hourly_profiles(days=90):
    index = pd.date_range('2025-01-01', periods=days * 24, freq='h')
    rng = np.random.default_rng(19)
    return pd.DataFrame({
        'a': 2 + .3 * np.sin(np.arange(len(index)) / 9) + rng.normal(0, .02, len(index)),
        'b': 3 + .4 * np.cos(np.arange(len(index)) / 7) + rng.normal(0, .02, len(index)),
    }, index=index)


def test_readings_audit_missing_values_and_duplicates(tmp_path):
    frame = hourly_profiles(31)
    frame.loc[frame.index[:200], 'b'] = np.nan
    frame['copy_a'] = frame.a
    frame['constant'] = 4.0
    frame.columns = ["'a'", "'b'", 'copy_a', 'constant']
    path = tmp_path / 'ami.csv'
    frame.to_csv(path, index_label='REPORTED_DTTM')
    x, quality = detection.load_readings(path)
    assert x.columns.tolist() == ['a']
    assert quality.loc['b', 'exclusion_reason'] == 'insufficient coverage'
    assert quality.loc['copy_a', 'exclusion_reason'] == 'duplicate profile'
    assert quality.loc['constant', 'exclusion_reason'] == 'empty or constant profile'


def test_readings_preserve_missing_hours_and_normalize_timezone(tmp_path):
    frame = hourly_profiles(31).drop(hourly_profiles(31).index[30])
    path = tmp_path / 'ami.csv'
    frame.to_csv(path)
    x, quality = detection.load_readings(path, timestamp_timezone='UTC')
    assert str(x.index.tz) == 'America/Los_Angeles'
    assert len(x) == 31 * 24
    assert x.isna().sum().eq(1).all()
    assert quality.observed_reads.eq(31 * 24 - 1).all()


def test_readings_reject_normalized_duplicate_ids_and_duplicate_times(tmp_path):
    path = tmp_path / 'ami.csv'
    path.write_text('time,a,\'a\'\n2025-01-01,1,2\n')
    with pytest.raises(ValueError, match='unique'):
        detection.load_readings(path)
    frame = hourly_profiles(31)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path)
    with pytest.raises(ValueError, match='duplicate timestamps'):
        detection.load_readings(path)


def test_feature_rows_are_independent_of_other_customers():
    frame = hourly_profiles()
    original = detection.extract_customer_features(frame)
    frame['b'] *= 1e6
    altered = detection.extract_customer_features(frame)
    assert_frame_equal(original.loc[['a']], altered.loc[['a']])
    assert not set(['customer_id', 'has_ev', 'has_pv', 'label_source']).intersection(original.columns)
    assert original.index.tolist() == ['a', 'b']
    assert not np.isinf(original.to_numpy()).any()


def test_screening_rules_find_controlled_patterns_without_changing_readings():
    # Synthetic fixtures test rule behavior only; never used for reported accuracy.
    index = pd.date_range('2025-01-01', periods=90 * 24, freq='h')
    frame = pd.DataFrame({'flat': np.ones(len(index)), 'ev': np.ones(len(index)),
                          'pv': np.full(len(index), 2.0), 'both': np.full(len(index), 2.0)}, index=index)
    charging = np.isin(index.hour, [20, 21, 22])
    midday = (index.hour >= 10) & (index.hour <= 15)
    frame.loc[charging, ['ev', 'both']] += 4
    frame.loc[midday, ['pv', 'both']] = .1
    original = frame.copy()
    labels, diagnostics = detection.make_heuristic_labels(frame)
    assert labels.loc['flat'].tolist() == [0, 0]
    assert labels.loc['ev', 'has_ev'] == 1
    assert labels.loc['pv', 'has_pv'] == 1
    assert labels.loc['both'].tolist() == [1, 1]
    assert diagnostics.label_source.eq('heuristic_unvalidated').all()
    assert_frame_equal(frame, original)


def test_confirmed_labels_align_by_id_and_keep_unknown(tmp_path):
    path = tmp_path / 'labels.csv'
    path.write_text("customer_id,has_ev,has_pv\n'b',1,\na,0,1\nexternal,1,0\n")
    labels, extra = detection.read_confirmed_labels(path, pd.Index(['a', 'b', 'c']))
    assert labels.loc['a'].tolist() == [0, 1]
    assert pd.isna(labels.loc['b', 'has_pv'])
    assert labels.loc['c'].isna().all()
    assert list(extra) == ['external']
    path.write_text('customer_id,has_ev,has_pv\na,2,1\n')
    with pytest.raises(ValueError, match='0, 1'):
        detection.read_confirmed_labels(path, pd.Index(['a']))
    path.write_text("customer_id,has_ev,has_pv\na,0,1\n'a',1,1\n")
    with pytest.raises(ValueError, match='unique'):
        detection.read_confirmed_labels(path, pd.Index(['a']))


def small_classifiers():
    models = detection.build_classifiers()
    forest = models['Random forest']
    forest.set_params(model__n_estimators=12)
    return {name: models[name] for name in ['Majority baseline', 'Logistic regression', 'Random forest']}


def test_selection_and_preprocessing_ignore_test_labels_and_values(monkeypatch):
    rng = np.random.default_rng(7)
    ids = pd.Index([f'customer-{i}' for i in range(60)])
    x = pd.DataFrame(rng.normal(size=(60, 4)), index=ids, columns=list('abcd'))
    x.loc[ids[:4], 'b'] = np.nan
    x.loc[ids[48:], 'a'] = 1e8  # Holdout values must not determine imputer or scaler.
    y = pd.Series(np.tile([0, 1], 30), index=ids)
    monkeypatch.setattr(detection, 'train_test_split', lambda *args, **kwargs: (ids[:48], ids[48:]))
    first = detection.benchmark_target(x, y, 'EV', 'test_fixture', classifiers=small_classifiers(), max_folds=3)
    changed_labels = y.copy()
    changed_labels.loc[ids[48:]] = 1 - changed_labels.loc[ids[48:]]
    second = detection.benchmark_target(x, changed_labels, 'EV', 'test_fixture', classifiers=small_classifiers(), max_folds=3)
    assert first.train_ids.intersection(first.test_ids).empty
    assert first.selected_name == second.selected_name
    assert_frame_equal(first.cv.drop(columns='cv_fit_seconds'), second.cv.drop(columns='cv_fit_seconds'))
    np.testing.assert_array_equal(first.predictions.score, second.predictions.score)
    estimator = first.models['Logistic regression']
    np.testing.assert_allclose(estimator.named_steps['impute'].statistics_, x.loc[ids[:48]].median())
    imputed = estimator.named_steps['impute'].transform(x.loc[ids[:48]])
    np.testing.assert_allclose(estimator.named_steps['scale'].mean_, imputed.mean(axis=0))
    assert set(first.fold_manifest.customer_id) == set(ids[:48])
    assert first.fold_manifest.validation_fold.between(1, 3).all()


def test_too_few_or_single_class_labels_have_no_accuracy():
    x = pd.DataFrame({'value': range(20)})
    for y in [pd.Series([0] * 20), pd.Series([0] * 16 + [1] * 4), pd.Series([pd.NA] * 20)]:
        with pytest.raises(ValueError, match='at least 5'):
            detection.benchmark_target(x, y, 'PV', 'heuristic_unvalidated')


def test_unknown_customers_excluded_and_rare_class_reduces_cv_folds():
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(30, 3)))
    y = pd.Series([0] * 20 + [1] * 5 + [pd.NA] * 5, dtype='Int64')
    models = detection.build_classifiers()
    models = {name: models[name] for name in ['Majority baseline', 'Logistic regression']}
    result = detection.benchmark_target(x, y, 'PV', 'confirmed_test_fixture', classifiers=models)
    assert len(result.train_ids) == 20 and len(result.test_ids) == 5
    assert set(result.train_ids).union(result.test_ids) == set(range(25))
    assert result.folds.fold.nunique() == 4
    assert result.predictions.label.nunique() == 2


def test_wilson_interval_respects_small_sample_uncertainty():
    low, high = detection.accuracy_interval([1] * 10, [1] * 10)
    assert .70 < low < .75
    assert high == pytest.approx(1)


def test_der_pv_labels_require_pv_status_and_temporally_valid_pto(tmp_path):
    def record(structure, project, technology='PHOTOVOLTAIC', status='PTO ISSUED',
               pto='2024-01-01', contract_end='9999-12-31', service='s1'):
        return {'STRUCT_NUM': structure, 'PROJECT_ID': project, 'TECH_TYPE': technology,
                'INTERCONNECTION_STATUS': status, 'PERMTO_OPERATE_DT': pto,
                'CONTRACT_END_DT': contract_end, 'SERV_PNT_ID': service, 'CIRCUIT_KEY': 'test-circuit'}
    rows = [record("'both'", 'p1'), record("'both'", 'p1'),  # duplicate inverter row
            record('both', 'battery1', technology='BATTERY', service='s2'),
            record('pending', 'p2', status='PENDING', pto='2024-04-01'),
            record('cancelled', 'p3', status='CANCELLED'),
            record('storage', 'p4', technology='BATTERY'),
            record('during', 'p5', pto='2025-03-01'),
            record('after', 'p6', pto='2025-09-01'),
            record('sentinel', 'p7', pto='1900-01-01'),
            record('expired', 'p8', contract_end='2025-02-01'),
            record('conflict', 'p9'), record('conflict', 'p9', status='CANCELLED')]
    path = tmp_path / 'der.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    ids = pd.Index(['both', 'pending', 'cancelled', 'storage', 'during', 'after',
                    'sentinel', 'expired', 'conflict', 'no_record'])
    labels, audit, stats, counts = detection.registry_pv_labels(
        path, ids, '2025-01-01', '2025-06-19 23:00', expected_circuit='test-circuit')
    assert labels['both'] == 1 and labels['no_record'] == 0
    assert labels.drop(['both', 'no_record']).isna().all()
    assert audit.loc['both', 'pv_projects'] == 1
    assert audit.loc['both', 'storage_projects'] == 1
    assert audit.loc['both', 'service_points'] == 2
    assert stats['structures_with_multiple_service_points'] == 1
    assert stats['project_status_conflict_rows'] == 2
    assert audit.loc['during', 'label_basis'] == 'first_pv_pto_during_window_unknown'
    assert audit.loc['no_record', 'label_basis'] == 'no_der_record_provisional_negative'
    assert counts.records.sum() == len(rows)


def test_der_unmatched_can_remain_unknown_and_wrong_circuit_is_rejected(tmp_path):
    path = tmp_path / 'der.csv'
    pd.DataFrame([{'STRUCT_NUM': 'known', 'PROJECT_ID': 'p1', 'TECH_TYPE': 'PHOTOVOLTAIC',
                   'INTERCONNECTION_STATUS': 'PTO ISSUED', 'PERMTO_OPERATE_DT': '2024-01-01',
                   'CIRCUIT_KEY': 'wrong-circuit'}]).to_csv(path, index=False)
    labels, _, _, _ = detection.registry_pv_labels(path, ['known', 'missing'], '2025-01-01',
                                                  '2025-06-19', no_record_as_negative=False)
    assert labels['known'] == 1 and pd.isna(labels['missing'])
    with pytest.raises(ValueError, match='circuit mismatch'):
        detection.registry_pv_labels(path, ['known'], '2025-01-01', '2025-06-19', expected_circuit='expected')
