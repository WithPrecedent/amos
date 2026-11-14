"""Tests the describers module."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos
from amos import describers

# The "is_" explorer of each distribution, by the name of the distribution.
TESTS: dict[str, str] = {
    family: f'is_{family}' for family in describers._FAMILIES}


def test_association_rules(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    describers.AssociationRules().apply(
        mixed, min_support = 0.05, min_threshold = 0.5)
    table = mixed.tables['association_rules']
    assert {'antecedents', 'consequents', 'support', 'confidence',
            'lift'} <= set(table.columns)
    assert table['lift'].is_monotonic_decreasing
    assert (table['confidence'] >= 0.5).all()
    assert (table['support'] >= 0.05).all()
    items = {
        i for side in ('antecedents', 'consequents')
        for rule in table[side] for i in rule.split(', ')}
    # The values of "region", "member" (where it is true), and the label.
    assert 'outcome = yes' in items
    assert 'member' in items
    assert items <= {
        'member', 'outcome = no', 'outcome = yes', 'region = east',
        'region = north', 'region = south', 'region = west'}
    south = table[
        (table['antecedents'] == 'region = south')
        & (table['consequents'] == 'outcome = yes')]
    data = mixed.data
    share = (data.loc[data['region'] == 'south', 'outcome'] == 'yes').mean()
    assert south['confidence'].iloc[0] == pytest.approx(share)


def test_association_rules_of_chosen_columns(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    describers.AssociationRules().apply(
        mixed, columns = ['region', 'outcome'], min_support = 0.05,
        metric = 'lift', min_threshold = 1.2, max_len = 2)
    table = mixed.tables['association_rules']
    assert not table.empty
    assert (table['lift'] >= 1.2).all()
    assert not table['antecedents'].str.contains(', ').any()
    assert 'member' not in set(table['antecedents'])


def test_association_rules_without_common_items(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    with pytest.raises(ValueError, match = 'lower "min_support"'):
        describers.AssociationRules().apply(
            mixed, columns = ['region'], min_support = 0.9)


def test_correlations(classified: amos.Dataset) -> None:
    describers.Correlations().apply(classified, method = 'spearman')
    table = classified.tables['correlations']
    assert list(table.columns) == [*classified.features, 'target']
    assert table.loc['x0', 'x0'] == pytest.approx(1.0)


def test_describe(mixed: amos.Dataset) -> None:
    describers.Describe().apply(mixed)
    table = mixed.tables['describe']
    assert list(table.index) == list(mixed.data.columns)
    assert mixed.history[-1] == {'technique': 'describe', 'table': 'describe'}


def test_frequencies(mixed: amos.Dataset) -> None:
    describers.Frequencies().apply(mixed)
    table = mixed.tables['frequencies']
    assert list(table.columns) == ['column', 'value', 'count', 'share']
    assert set(table['column']) == {'region', 'member'}
    regions = table[table['column'] == 'region']
    assert regions['count'].sum() == 200
    assert regions['share'].sum() == pytest.approx(1.0)


def test_frequencies_of_all_includes_dummies(mixed: amos.Dataset) -> None:
    mixed.data['flag'] = (mixed.data['income'] > 30000).astype('int8')
    mixed.kinds['flag'] = 'booleans'
    describers.Frequencies().apply(mixed, columns = 'all')
    table = mixed.tables['frequencies']
    assert set(table['column']) == {'region', 'member', 'flag'}
    describers.Frequencies().apply(mixed, columns = 'region')
    assert set(mixed.tables['frequencies']['column']) == {'region'}


def test_frequencies_without_categorical_columns(
    classified: amos.Dataset) -> None:
    describers.Frequencies().apply(classified)
    assert classified.tables['frequencies'].empty


def test_label_balance(mixed: amos.Dataset) -> None:
    describers.LabelBalance().apply(mixed)
    table = mixed.tables['label_balance']
    assert list(table.index) == ['no', 'yes']
    assert table['count'].sum() == 200


def test_missing_values(mixed: amos.Dataset) -> None:
    describers.MissingValues().apply(mixed)
    table = mixed.tables['missing_values']
    assert table.loc['age', 'missing'] == mixed.data['age'].isna().sum()
    assert table.loc['income', 'share'] == 0


def test_summarize(mixed: amos.Dataset) -> None:
    describers.Summarize().apply(mixed)
    table = mixed.tables['summarize']
    assert list(table.index) == ['age', 'income', 'member']
    assert list(table.columns) == [
        'count', 'missing', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max',
        'skew', 'kurtosis']
    assert table.loc['age', 'count'] + table.loc['age', 'missing'] == 200


def test_summarize_includes_a_numeric_label(regressed: amos.Dataset) -> None:
    describers.Summarize().apply(regressed)
    assert 'target' in regressed.tables['summarize'].index


def _samples(rows: int = 400) -> pd.DataFrame:
    """Returns a column drawn from each distribution, and a column of text.

    Args:
        rows: the number of rows.

    Returns:
        The data, with a column named after each distribution ("uniform"
            has numbers and "dice" whole numbers), "flag" (booleans), and
            "text".

    """
    rng = np.random.default_rng(SEED)
    half = rows // 2
    return pd.DataFrame({
        'normal': rng.normal(10, 2, rows),
        'poisson': rng.poisson(3, rows),
        'uniform': rng.uniform(2, 7, rows),
        'dice': rng.integers(1, 7, rows),
        'exponential': rng.exponential(4, rows),
        'log_normal': rng.lognormal(1, 0.6, rows),
        'student': rng.standard_t(3, rows) * 2 + 5,
        'bernoulli': rng.integers(0, 2, rows),
        'binomial': rng.binomial(10, 0.3, rows),
        'beta': rng.beta(2, 5, rows),
        'bimodal': np.r_[rng.normal(0, 1, half), rng.normal(6, 1, half)],
        'flag': rng.random(rows) < 0.3,
        'text': rng.choice(['a', 'b'], rows)})


@pytest.mark.parametrize('family', list(TESTS))
def test_each_distribution_fits_its_own_sample(family: str) -> None:
    dataset = amos.Dataset(_samples(), seed = SEED)
    amos.library.borrow(TESTS[family])().apply(dataset)
    table = dataset.tables[TESTS[family]]
    assert list(table.columns) == [TESTS[family]]
    assert table[TESTS[family]].dtype == bool
    assert list(table.index) == list(dataset.data.columns)
    assert table.loc[family, TESTS[family]]
    assert not table.loc['text', TESTS[family]]


@pytest.mark.parametrize(('family', 'others'), [
    ('normal', ['exponential', 'uniform', 'bimodal', 'poisson']),
    ('poisson', ['normal', 'uniform', 'exponential', 'dice']),
    ('uniform', ['normal', 'exponential', 'beta']),
    ('exponential', ['normal', 'uniform', 'log_normal']),
    ('log_normal', ['normal', 'exponential', 'uniform']),
    ('student', ['exponential', 'uniform', 'bimodal']),
    ('bernoulli', ['poisson', 'dice', 'normal']),
    ('binomial', ['uniform', 'dice', 'normal']),
    ('beta', ['normal', 'uniform', 'exponential']),
    ('bimodal', ['normal', 'poisson', 'dice', 'bernoulli', 'binomial'])])
def test_each_distribution_rejects_other_samples(
    family: str,
    others: list[str]) -> None:
    data = _samples()
    for other in others:
        assert not describers.fits_distribution(data[other], family), other


def test_skew_explorers() -> None:
    dataset = amos.Dataset(_samples(), seed = SEED)
    dataset.data['mirrored'] = -dataset.data['exponential']
    describers.IsLeftSkewed().apply(dataset, columns = ['mirrored', 'normal'])
    describers.IsRightSkewed().apply(dataset, columns = 'exponential')
    left = dataset.tables['is_left_skewed']['is_left_skewed']
    assert left.to_dict() == {'mirrored': True, 'normal': False}
    assert dataset.tables['is_right_skewed'].loc[
        'exponential', 'is_right_skewed']
    assert not describers.is_skewed(dataset.data['mirrored'], 'right')
    with pytest.raises(ValueError, match = 'direction'):
        describers.is_skewed(
            dataset.data['normal'], 'up')  # type: ignore[arg-type]


def test_distributions_names_each_sample() -> None:
    dataset = amos.Dataset(_samples(), seed = SEED)
    describers.Distributions().apply(dataset)
    table = dataset.tables['distributions']
    assert list(table.columns) == ['distribution', 'closest', 'skew']
    assert list(table.index) == list(dataset.data.columns)
    expected = {
        name: name for name in describers._FAMILIES}
    expected.update({'dice': 'uniform', 'flag': 'bernoulli', 'text': 'none'})
    assert table['distribution'].to_dict() == expected
    assert table.loc['text', 'skew'] == 'none'
    assert table.loc['normal', 'skew'] == 'symmetric'
    assert table.loc['exponential', 'skew'] == 'right'
    assert dataset.history[-1] == {
        'technique': 'distributions', 'table': 'distributions'}


def test_distributions_of_chosen_columns() -> None:
    dataset = amos.Dataset(_samples(), seed = SEED)
    describers.Distributions().apply(dataset, columns = 'poisson')
    assert list(dataset.tables['distributions'].index) == ['poisson']
    describers.Distributions().apply(dataset, columns = ['beta', 'normal'])
    assert list(dataset.tables['distributions'].index) == ['beta', 'normal']


def test_distributions_without_a_passing_fit() -> None:
    rng = np.random.default_rng(SEED)
    # A uniform peak on top of a normal spread fits none of the tests, but a
    # normal distribution is still the closest.
    values = np.r_[rng.normal(0, 3, 500), rng.uniform(-0.5, 0.5, 500)]
    dataset = amos.Dataset(pd.DataFrame({'x': values}), seed = SEED)
    describers.Distributions().apply(dataset)
    row = dataset.tables['distributions'].loc['x']
    assert row['distribution'] == 'none'
    assert row['closest'] in describers._FAMILIES


def test_distributions_without_diptest(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(describers.importlib.util, 'find_spec', lambda _: None)
    dataset = amos.Dataset(_samples()[['bimodal', 'normal']], seed = SEED)
    with caplog.at_level(logging.WARNING):
        describers.Distributions().apply(dataset)
    table = dataset.tables['distributions']
    assert 'bimodal' not in set(table['closest'])
    assert table.loc['normal', 'distribution'] == 'normal'
    assert 'diptest is not installed' in caplog.text


def test_too_few_values_fit_nothing() -> None:
    values = pd.Series([1.0, 2.0, 3.0, np.nan])
    for family in describers._FAMILIES:
        assert not describers.fits_distribution(values, family)
    assert not describers.is_skewed(values, 'left')
    # Any 0s and 1s are a Bernoulli distribution.
    assert describers.fits_distribution(pd.Series([0, 1, 1]), 'bernoulli')


def test_bimodal_ignores_ties_of_whole_numbers() -> None:
    data = _samples()
    for name in ('poisson', 'binomial', 'dice', 'bernoulli'):
        assert not describers.fits_distribution(data[name], 'bimodal'), name


def test_fit_distribution_recovers_parameters() -> None:
    data = _samples(rows = 5000)
    student = describers.fit_distribution(data['student'], 'student')
    assert student.args == pytest.approx((3, 5, 2), rel = 0.15)
    binomial = describers.fit_distribution(data['binomial'], 'binomial')
    assert binomial.args == pytest.approx((10, 0.3), rel = 0.15)
    given = describers.fit_distribution(
        data['binomial'], 'binomial', trials = 12)
    assert given.args[0] == 12
    mixture = describers.fit_distribution(
        data['bimodal'], 'bimodal', seed = SEED)
    assert sorted(mixture.means) == pytest.approx([0, 6], abs = 0.15)
    assert mixture.cdf(np.array([-50.0, 50.0])) == pytest.approx([0, 1])


def test_binomial_trials_bound_the_values() -> None:
    values = _samples()['binomial']
    assert describers.fits_distribution(values, 'binomial', trials = 10)
    assert not describers.fits_distribution(
        values, 'binomial', trials = int(values.max()) - 1)
    dataset = amos.Dataset(_samples(), seed = SEED)
    describers.IsBinomial().apply(dataset, columns = 'binomial', trials = 10)
    assert dataset.tables['is_binomial'].loc['binomial', 'is_binomial']


def test_distribution_errors() -> None:
    values = _samples()['normal']
    with pytest.raises(ValueError, match = 'not a distribution'):
        describers.fits_distribution(values, 'gamma')
    with pytest.raises(ValueError, match = 'trials'):
        describers.fit_distribution(values, 'binomial', trials = 0)
    with pytest.raises(ValueError, match = 'do not suit'):
        describers.fit_distribution(values, 'poisson')


def test_best_distribution() -> None:
    data = _samples()
    assert describers.best_distribution(data['poisson']) == 'poisson'
    assert describers.best_distribution(
        data['bimodal'], seed = SEED) == 'bimodal'
    assert describers.best_distribution(data['text']) is None
