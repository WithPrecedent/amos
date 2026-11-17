"""Tests the pretests, which check assumptions before techniques are applied."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, make_regression, requires

import amos
from amos import assumptions, pretests


def _row(table: pd.DataFrame, technique: str | None, condition: str) -> dict:
    """Returns the row of a table of verdicts for a technique's condition."""
    if technique is None:
        rows = table[table['technique'].isna()]
    else:
        rows = table[table['technique'] == technique]
    return rows[rows['condition'] == condition].iloc[0].to_dict()


def test_a_pretest_without_techniques_describes_the_data() -> None:
    dataset = amos.Dataset(make_regression(), label = 'target', seed = SEED)
    pretests.CheckNormality().apply(dataset)
    table = dataset.tables['check_normality']
    assert set(table['condition']) == {'normal_errors', 'normal_features'}
    assert table['technique'].isna().all()
    assert _row(table, None, 'normal_errors')['passed']
    assert dataset.history[-1]['technique'] == 'check_normality'


def test_check_assumptions_checks_every_condition_of_the_techniques() -> None:
    data = make_regression()
    data['x0'] *= 1000
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    pretests.CheckAssumptions().apply(dataset, techniques = ['linear', 'knn'])
    table = dataset.tables['check_assumptions']
    linear = set(table.loc[table['technique'] == 'linear', 'condition'])
    # Constant variance and independence are only audited, after a fit.
    assert linear == {
        'linearity', 'normal_errors', 'no_multicollinearity',
        'numeric_features', 'no_missing_values', 'more_rows_than_features'}
    assert not _row(table, 'knn', 'similar_scales')['passed']
    assert _row(table, 'linear', 'linearity')['passed']
    assert dataset.history[-1]['failed'] == [
        "'knn' needs features on similar scales (scale them first)"]
    with pytest.raises(KeyError):
        pretests.CheckAssumptions().apply(dataset, techniques = 'no_such')


def test_check_assumptions_checks_the_techniques_of_the_workflow() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {
            'steps': ['split', 'check', 'model'],
            'split_techniques': 'train_test',
            'check_techniques': 'check_assumptions',
            'model_techniques': 'linear'}}
    project = amos.Project.create(settings, item = make_regression())
    table = project.result.tables['check_assumptions']
    assert set(table['technique']) == {'linear'}


def test_check_multicollinearity_finds_a_copy_of_a_feature() -> None:
    data = make_regression()
    data['copy'] = data['x0'] * 2 + 1
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckMulticollinearity().apply(dataset, techniques = 'ols')
    row = _row(dataset.tables['check_multicollinearity'], 'ols',
        'no_multicollinearity')
    assert not row['passed']
    assert 'variance inflation factor' in row['detail']


def test_check_linearity_finds_a_curve() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.uniform(-3, 3, 300)})
    data['target'] = data['x'] ** 2 + rng.normal(scale = 0.3, size = 300)
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckLinearity().apply(dataset, techniques = 'linear')
    row = _row(dataset.tables['check_linearity'], 'linear', 'linearity')
    assert not row['passed']
    assert 'RESET' in row['detail']
    assert row['p_value'] < 0.05


def test_check_linearity_tests_the_log_odds_of_two_classes() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.uniform(0.1, 10, 600)})
    odds = 4 * np.log(data['x']) - 4
    data['target'] = (rng.logistic(size = 600) < odds).astype(int)
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckLinearity().apply(dataset, techniques = 'logit')
    row = _row(dataset.tables['check_linearity'], 'logit', 'linearity')
    assert 'Box-Tidwell' in row['detail']
    assert not row['passed']
    dataset = amos.Dataset(
        make_numeric(rows = 300, classes = 3), label = 'target')
    pretests.CheckLinearity().apply(dataset, techniques = 'logit')
    row = _row(dataset.tables['check_linearity'], 'logit', 'linearity')
    assert row['passed'] is None


def test_check_label_and_check_sample_size() -> None:
    dataset = amos.Dataset(
        make_numeric(rows = 300, classes = 3), label = 'target')
    pretests.CheckLabel().apply(dataset, techniques = ['logit_sm', 'mnlogit'])
    table = dataset.tables['check_label']
    assert _row(table, 'logit_sm', 'binary_label')['detail'] == (
        'is for 2 classes, not 3')
    assert _row(table, 'mnlogit', 'multiclass_label')['passed']
    rng = np.random.default_rng(SEED)
    wide = pd.DataFrame(rng.normal(size = (60, 40)))
    wide.columns = [f'x{i}' for i in range(40)]
    wide['target'] = rng.normal(size = 60)
    dataset = amos.Dataset(wide, label = 'target')
    pretests.CheckSampleSize().apply(dataset, techniques = ['knn', 'ols'])
    table = dataset.tables['check_sample_size']
    assert not _row(table, 'knn', 'max_features')['passed']
    assert _row(table, 'ols', 'more_rows_than_features')['passed']


def test_check_dispersion_finds_overdispersed_counts() -> None:
    requires('statsmodels')
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.normal(size = 400)})
    data['target'] = rng.negative_binomial(1, 1 / (1 + np.exp(1 + data['x'])))
    dataset = amos.Dataset(data, label = 'target', task = 'regress')
    pretests.CheckDispersion().apply(dataset, techniques = 'poisson')
    row = _row(dataset.tables['check_dispersion'], 'poisson', 'equidispersion')
    assert not row['passed']
    assert 'overdispersed' in row['detail']


def test_check_features_names_the_columns_that_are_not_numbers() -> None:
    data = make_regression()
    data['court'] = 'a'
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckFeatures().apply(dataset, techniques = 'linear')
    row = _row(dataset.tables['check_features'], 'linear', 'numeric_features')
    assert not row['passed']
    assert "['court']" in row['detail']


def test_the_tests_of_linearity() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size = (200, 2))
    y = x @ [1.0, 2.0] + rng.normal(size = 200)
    f_value, p_value, gain = assumptions.reset_test(x, y)
    assert p_value > 0.01
    assert 0 <= gain < 0.02
    assert assumptions.reset_test(x[:4], y[:4]) is None
