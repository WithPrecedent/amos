"""Tests the audits, which check a fitted model's assumptions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_regression, requires

import amos
from amos import assumptions, audits


def _verdict(dataset: amos.Dataset, table: str, condition: str) -> dict:
    """Returns the row of a table of verdicts for a condition."""
    rows = dataset.tables[table]
    return rows[rows['condition'] == condition].iloc[0].to_dict()


def _fitted(data: pd.DataFrame, **settings: object) -> amos.Dataset:
    """Returns `data` with a linear regression fitted to every row."""
    dataset = amos.Dataset(data, label = 'target', seed = SEED, **settings)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        amos.models.Linear().apply(dataset)
    return dataset


def test_audit_residuals_passes_normal_errors() -> None:
    dataset = _fitted(make_regression(rows = 300))
    audits.AuditResiduals().apply(dataset)
    row = _verdict(dataset, 'audit_residuals', 'normal_errors')
    assert row['technique'] == 'linear'
    assert row['passed']
    # The rows are also in the table of every check.
    assert 'audit_residuals' in set(dataset.tables['assumptions']['check'])


def test_audit_errors_finds_errors_whose_variance_changes() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.normal(size = 500)})
    spread = 1 + 2 * data['x'].abs()
    data['target'] = data['x'] + rng.normal(size = 500) * spread
    dataset = _fitted(data)
    with pytest.warns(amos.AmosWarning, match = 'constant variance'):
        audits.AuditErrors().apply(dataset)
    row = _verdict(dataset, 'audit_errors', 'constant_variance')
    assert row['passed'] is False
    assert row['p_value'] < 0.05


def test_audit_independence_finds_correlated_errors_in_order() -> None:
    rng = np.random.default_rng(SEED)
    rows = 400
    errors = np.zeros(rows)
    for i in range(1, rows):
        errors[i] = 0.8 * errors[i - 1] + rng.normal()
    data = pd.DataFrame({'x': rng.normal(size = rows), 'day': range(rows)})
    data['target'] = data['x'] + errors
    # Shuffled, so that only the order puts the rows next to each other.
    data = data.sample(frac = 1, random_state = SEED)
    dataset = _fitted(data, order = 'day')
    assert 'day' not in dataset.features
    with pytest.warns(amos.AmosWarning, match = 'next to each other'):
        audits.AuditIndependence().apply(dataset)
    row = _verdict(dataset, 'audit_independence', 'independent_errors')
    assert row['passed'] is False
    assert row['statistic'] < 1.5


def test_audit_independence_finds_errors_correlated_within_groups() -> None:
    rng = np.random.default_rng(SEED)
    groups = np.repeat(np.arange(20), 20)
    data = pd.DataFrame({'x': rng.normal(size = 400), 'court': groups})
    shift = rng.normal(scale = 2, size = 20)[groups]
    data['target'] = data['x'] + shift + rng.normal(size = 400)
    dataset = _fitted(data, groups = ['court'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        audits.AuditIndependence().apply(dataset)
    row = _verdict(dataset, 'audit_independence', 'independent_errors')
    assert row['passed'] is False
    assert 'intraclass correlation' in row['detail']


def test_audit_independence_needs_an_order_or_groups() -> None:
    dataset = _fitted(make_regression())
    audits.AuditIndependence().apply(dataset)
    row = _verdict(dataset, 'audit_independence', 'independent_errors')
    assert row['passed'] is None
    assert 'without an order or groups' in row['detail']


def test_audit_assumptions_runs_only_the_audits_that_the_model_needs() -> None:
    dataset = _fitted(make_regression())
    audits.AuditAssumptions().apply(dataset)
    conditions = set(dataset.tables['audit_assumptions']['condition'])
    assert conditions == {
        'normal_errors', 'constant_variance', 'independent_errors'}
    dataset = amos.Dataset(make_regression(), label = 'target', seed = SEED)
    amos.models.RandomForest().apply(dataset, n_estimators = 10)
    audits.AuditAssumptions().apply(dataset)
    assert dataset.tables['audit_assumptions'].empty


def test_an_audit_of_an_assumption_the_model_does_not_make() -> None:
    dataset = amos.Dataset(make_regression(), label = 'target', seed = SEED)
    amos.models.RandomForest().apply(dataset, n_estimators = 10)
    audits.AuditResiduals().apply(dataset)
    row = _verdict(dataset, 'audit_residuals', 'normal_errors')
    # It is tested, but it is not the model's, so it is never a violation.
    assert pd.isna(row['technique'])


def test_an_audit_raises_when_told() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.normal(size = 400)})
    data['target'] = data['x'] + rng.exponential(size = 400) ** 2
    dataset = _fitted(data)
    dataset.assumption_violations = 'raise'
    with pytest.raises(amos.AssumptionError, match = 'normal errors'):
        audits.AuditResiduals().apply(dataset)


def test_an_audit_needs_a_fitted_model() -> None:
    dataset = amos.Dataset(make_regression(), label = 'target')
    with pytest.raises(ValueError, match = 'fitted model'):
        audits.AuditAssumptions().apply(dataset)


def test_the_tests_of_errors() -> None:
    rng = np.random.default_rng(SEED)
    errors = rng.normal(size = 300)
    x = rng.normal(size = (300, 2))
    statistic, p_value, share = assumptions.breusch_pagan_test(errors, x)
    assert p_value > 0.01
    assert share < 0.05
    order = pd.Series(range(300))
    watson, _, p_value, rho = assumptions.serial_correlation_test(
        pd.Series(errors), order)
    assert 1.5 < watson < 2.5
    assert abs(rho) < 0.2
    groups = pd.Series(np.repeat(np.arange(30), 10))
    icc, _, p_value = assumptions.clustering_test(errors, groups)
    assert icc < 0.1
    assert assumptions.clustering_test(errors[:1], groups[:1]) is None


def test_audit_proportional_hazards_finds_an_effect_that_changes() -> None:
    requires('statsmodels')
    rng = np.random.default_rng(SEED)
    rows = 600
    x = rng.binomial(1, 0.5, rows).astype(float)
    # The hazard of x doubles early and is the same later: early times come
    # from a higher rate for x, and later ones from the same rate.
    early = rng.exponential(1 / (1 + 3 * x))
    late = 1 + rng.exponential(1.0, rows)
    days = np.where(early < 1, early, late)
    data = pd.DataFrame({
        'x': x, 'z': rng.normal(size = rows), 'days': days,
        'event': np.ones(rows, dtype = int)})
    dataset = amos.Dataset(
        data, label = 'days', seed = SEED, groups = ['event'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        amos.models.Cox().apply(dataset, event = 'event')
        audits.AuditProportionalHazards().apply(dataset)
    row = _verdict(
        dataset, 'audit_proportional_hazards', 'proportional_hazards')
    assert row['technique'] == 'cox'
    assert row['passed'] is False
    assert "['x']" in row['detail']
    # It is not a Cox model's, so it cannot be tested.
    fitted = _fitted(make_regression())
    audits.AuditProportionalHazards().apply(fitted)
    row = _verdict(
        fitted, 'audit_proportional_hazards', 'proportional_hazards')
    assert row['passed'] is None


def test_the_tests_of_causes_and_time() -> None:
    rng = np.random.default_rng(SEED)
    z = rng.normal(size = 400)
    f_value, p_value = assumptions.first_stage_test(
        z + rng.normal(size = 400), z[:, None], np.empty((400, 0)))
    assert f_value > 100
    assert p_value < 0.001
    assert assumptions.density_test(np.zeros(5), 0.0) is None
    treated = np.array([0, 1] * 50)
    assert assumptions.overlap_test(treated, np.empty((100, 0))) == 0
    values = rng.normal(size = (300, 3))
    values[rng.random(300) < 0.3, 1] = np.nan
    statistic, p_value, freedom = assumptions.little_test(values)
    assert freedom == 2
    assert p_value > 0.01
    assert assumptions.little_test(np.full((5, 2), np.nan)) is None
