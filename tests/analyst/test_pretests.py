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
    assert row['passed'] is False
    assert 'variance inflation factor' in row['detail']


def test_check_linearity_finds_a_curve() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({'x': rng.uniform(-3, 3, 300)})
    data['target'] = data['x'] ** 2 + rng.normal(scale = 0.3, size = 300)
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckLinearity().apply(dataset, techniques = 'linear')
    row = _row(dataset.tables['check_linearity'], 'linear', 'linearity')
    assert row['passed'] is False
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
    assert row['passed'] is False
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
    assert row['passed'] is False
    assert 'overdispersed' in row['detail']


def test_check_features_names_the_columns_that_are_not_numbers() -> None:
    data = make_regression()
    data['court'] = 'a'
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckFeatures().apply(dataset, techniques = 'linear')
    row = _row(dataset.tables['check_features'], 'linear', 'numeric_features')
    assert row['passed'] is False
    assert "['court']" in row['detail']


def test_the_tests_of_linearity() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size = (200, 2))
    y = x @ [1.0, 2.0] + rng.normal(size = 200)
    f_value, p_value, gain = assumptions.reset_test(x, y)
    assert p_value > 0.01
    assert 0 <= gain < 0.02
    assert assumptions.reset_test(x[:4], y[:4]) is None


""" Assumptions That Can Be Partly Tested """


def test_check_missingness_tells_values_missing_at_random() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size = (600, 3))
    data = pd.DataFrame(x, columns = ['a', 'b', 'c'])
    data['b'] += data['a']
    data['target'] = data['a'] + rng.normal(size = 600)
    random = data.copy()
    random.loc[rng.random(600) < 0.2, 'c'] = np.nan
    dataset = amos.Dataset(random, label = 'target')
    pretests.CheckMissingness().apply(dataset, techniques = 'mean_impute')
    row = _row(dataset.tables['check_missingness'], 'mean_impute',
        'missing_completely_at_random')
    assert row['passed']
    # Values of "c" are missing when "a" is high.
    related = data.copy()
    related.loc[related['a'] > 0.5, 'c'] = np.nan
    dataset = amos.Dataset(related, label = 'target')
    pretests.CheckMissingness().apply(dataset, techniques = 'mean_impute')
    row = _row(dataset.tables['check_missingness'], 'mean_impute',
        'missing_completely_at_random')
    assert row['passed'] is False
    assert "Little's test" in row['detail']


def test_check_instrument_finds_a_weak_instrument() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'z': rng.normal(size = 500), 'w': rng.normal(size = 500)})
    data['strong'] = data['z'] + rng.normal(size = 500)
    data['weak'] = 0.02 * data['z'] + rng.normal(size = 500)
    data['target'] = data['strong'] + rng.normal(size = 500)
    dataset = amos.Dataset(data, label = 'target')
    check = pretests.CheckInstrument()
    check.apply(
        dataset, techniques = 'partially_linear_iv', treatment = 'strong',
        instrument = 'z')
    table = dataset.tables['check_instrument']
    assert _row(table, 'partially_linear_iv', 'valid_instrument')['passed']
    check.apply(
        dataset, techniques = 'partially_linear_iv', treatment = 'weak',
        instrument = 'z')
    row = _row(dataset.tables['check_instrument'], 'partially_linear_iv',
        'valid_instrument')
    assert row['passed'] is False
    assert row['statistic'] < 10
    # Without its parameters, the instrument cannot be tested.
    check.apply(dataset, techniques = 'partially_linear_iv')
    row = _row(dataset.tables['check_instrument'], 'partially_linear_iv',
        'valid_instrument')
    assert row['passed'] is None


def _panel(diverging: bool) -> pd.DataFrame:
    """Returns 100 units in 6 periods, half of them treated from period 4."""
    rng = np.random.default_rng(SEED)
    units = np.repeat(np.arange(100), 6)
    time = np.tile(np.arange(6), 100)
    treated = units < 50
    label = time + 2 * treated + rng.normal(size = 600)
    if diverging:
        label = label + treated * time
    return pd.DataFrame({
        'unit': units, 'time': time, 'x': rng.normal(size = 600),
        'first': np.where(treated, 4.0, np.nan), 'target': label})


def test_check_parallel_trends_compares_the_trends_before_treatment() -> None:
    check = pretests.CheckParallelTrends()
    for diverging, expected in ((False, True), (True, False)):
        dataset = amos.Dataset(_panel(diverging), label = 'target')
        check.apply(
            dataset, techniques = 'difference_in_differences', time = 'time',
            treatment = 'first')
        row = _row(dataset.tables['check_parallel_trends'],
            'difference_in_differences', 'parallel_trends')
        assert row['passed'] is expected


def test_check_continuity_finds_rows_bunched_above_a_cutoff() -> None:
    rng = np.random.default_rng(SEED)
    score = rng.normal(size = 2000)
    data = pd.DataFrame({'score': score, 'target': rng.normal(size = 2000)})
    dataset = amos.Dataset(data, label = 'target')
    check = pretests.CheckContinuity()
    check.apply(
        dataset, techniques = 'regression_discontinuity', running = 'score')
    table = dataset.tables['check_continuity']
    assert _row(table, 'regression_discontinuity', 'continuity')['passed']
    # Rows just below the cutoff moved just above it.
    data['score'] = np.where(
        (score > -0.2) & (score < 0), score + 0.2, score)
    dataset = amos.Dataset(data, label = 'target')
    check.apply(
        dataset, techniques = 'regression_discontinuity', running = 'score')
    row = _row(dataset.tables['check_continuity'], 'regression_discontinuity',
        'continuity')
    assert row['passed'] is False


""" Assumptions That Can Be Tested """


def test_check_overlap_finds_extreme_propensity_scores() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame(rng.normal(size = (600, 2)), columns = ['a', 'b'])
    data['target'] = rng.normal(size = 600)
    check = pretests.CheckOverlap()
    for strength, expected in ((1, True), (6, False)):
        data['treated'] = (
            strength * data['a'] + rng.logistic(size = 600) > 0).astype(int)
        dataset = amos.Dataset(data, label = 'target')
        check.apply(
            dataset, techniques = 'propensity_weighting',
            treatment = 'treated')
        row = _row(dataset.tables['check_overlap'], 'propensity_weighting',
            'overlap')
        assert row['passed'] is expected


def test_check_stationarity_finds_a_random_walk() -> None:
    requires('statsmodels')
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'steady': rng.normal(size = 300),
        'walk': np.cumsum(rng.normal(size = 300)),
        'target': rng.normal(size = 300)})
    dataset = amos.Dataset(data, label = 'target')
    pretests.CheckStationarity().apply(dataset, techniques = 'pcmci')
    row = _row(dataset.tables['check_stationarity'], 'pcmci', 'stationarity')
    assert row['passed'] is False
    assert "['walk']" in row['detail']


def test_check_proportional_odds_compares_the_cutoffs() -> None:
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(rng.normal(size = (1500, 2)), columns = ['a', 'b'])
    latent = x['a'] - 0.5 * x['b'] + rng.logistic(size = 1500)
    check = pretests.CheckProportionalOdds()
    # "b" matters more above the middle cutoff than below it.
    shifted = np.where(latent > 0.5, latent + 1.5 * x['b'], latent)
    for values, expected in ((latent, True), (shifted, False)):
        data = x.assign(target = np.digitize(values, [-1, 0.5, 2]))
        dataset = amos.Dataset(data, label = 'target')
        check.apply(dataset, techniques = 'ordinal_regression')
        row = _row(dataset.tables['check_proportional_odds'],
            'ordinal_regression', 'proportional_odds')
        assert row['passed'] is expected


def test_an_inferer_tests_its_assumptions_with_its_parameters() -> None:
    requires('doubleml')
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame(rng.normal(size = (400, 2)), columns = ['a', 'b'])
    data['treated'] = (6 * data['a'] + rng.logistic(size = 400) > 0).astype(int)
    data['target'] = data['treated'] + data['a'] + rng.normal(size = 400)
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    with pytest.warns(amos.AmosWarning, match = 'propensity scores'):
        amos.inferers.InteractiveRegression().apply(
            dataset, treatment = 'treated', outcome_model = 'linear',
            treatment_model = 'logit', n_folds = 2)
    table = dataset.tables['assumptions']
    assert _row(table, 'interactive_regression', 'overlap')['passed'] is False
