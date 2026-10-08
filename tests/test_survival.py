"""Tests survival analysis: Kaplan-Meier curves, Cox models, and concordance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos


@pytest.fixture(autouse = True)
def _statsmodels() -> None:
    """Skips the tests if statsmodels cannot be imported."""
    requires('statsmodels')


def _survival(split: bool = False) -> amos.Dataset:
    """Returns times until rearrest, which are shorter for more priors."""
    rng = np.random.default_rng(5)
    rows = 300
    prior = rng.poisson(2, rows)
    age = rng.normal(35, 8, rows)
    days = rng.exponential(400 / (1 + prior), rows).round() + 1
    data = pd.DataFrame({
        'prior': prior.astype(float),
        'age': age,
        'program': rng.choice(['yes', 'no'], rows),
        'arrested': (rng.random(rows) < 0.75).astype(int),
        'days': days})
    dataset = amos.Dataset(
        data, label = 'days', seed = SEED, groups = ['program', 'arrested'])
    if split:
        amos.splitters.TrainTest().apply(dataset)
    return dataset


def test_kaplan_meier_is_the_product_limit() -> None:
    dataset = _survival()
    amos.describers.KaplanMeier().apply(dataset, event = 'arrested')
    table = dataset.tables['kaplan_meier']
    assert list(table.columns) == ['group', 'time', 'survival', 'at_risk']
    assert set(table['group']) == {'all'}
    days = dataset.y.to_numpy()
    arrested = dataset.data['arrested'].to_numpy() == 1
    expected = [
        np.prod([
            1 - (arrested & (days == t)).sum() / (days >= t).sum()
            for t in np.unique(days[arrested]) if t <= time])
        for time in table['time']]
    np.testing.assert_allclose(table['survival'], expected)
    assert list(table['at_risk']) == [(days >= t).sum() for t in table['time']]
    assert table['time'].iloc[0] == 0
    assert table['survival'].iloc[0] == 1
    assert table['time'].iloc[-1] == days.max()
    assert table['survival'].is_monotonic_decreasing


def test_survival_steps_have_confidence_intervals() -> None:
    dataset = _survival()
    curve = amos.describers.survival_curves(dataset, 'arrested')['all']
    steps = amos.describers.survival_steps(curve)
    assert (steps['ci_lower'] <= steps['survival']).all()
    assert (steps['survival'] <= steps['ci_upper']).all()
    assert steps['ci_lower'].between(0, 1).all()
    assert steps['ci_upper'].between(0, 1).all()


def test_kaplan_meier_by_group() -> None:
    dataset = _survival()
    amos.describers.KaplanMeier().apply(
        dataset, event = 'arrested', group = 'program')
    assert set(dataset.tables['kaplan_meier']['group']) == {'no', 'yes'}


def test_survival_curves() -> None:
    requires('matplotlib')
    dataset = _survival()
    amos.plots.SurvivalCurves().apply(
        dataset, event = 'arrested', group = 'program')
    axes = dataset.figures['survival_curves'].axes[0]
    assert len(axes.get_lines()) >= 2
    assert axes.get_xlabel() == 'days'


def test_cox_reports_hazard_ratios() -> None:
    dataset = _survival(split = True)
    amos.models.Cox().apply(dataset, event = 'arrested')
    table = dataset.tables['cox_coefficients']
    assert list(table.index) == ['prior', 'age']
    assert list(table.columns) == [
        'coefficient', 'hazard_ratio', 'standard_error', 'statistic',
        'p_value', 'ci_lower', 'ci_upper']
    assert table.loc['prior', 'hazard_ratio'] == pytest.approx(
        np.exp(table.loc['prior', 'coefficient']))
    # More priors means a higher hazard of rearrest.
    assert table.loc['prior', 'hazard_ratio'] > 1
    assert dataset.model.event == 'arrested'
    assert dataset.predictions.notna().all()


def test_cox_penalizer_shrinks_the_coefficients() -> None:
    dataset = _survival(split = True)
    amos.models.Cox().apply(dataset, event = 'arrested')
    amos.models.Cox(name = 'ridge').apply(
        dataset, event = 'arrested', penalizer = 0.5)
    plain = dataset.tables['cox_coefficients']
    ridge = dataset.tables['ridge_coefficients']
    assert abs(ridge.loc['prior', 'coefficient']) < abs(
        plain.loc['prior', 'coefficient'])
    assert ridge['standard_error'].notna().all()
    assert ridge['p_value'].between(0, 1).all()


def test_concordance_uses_the_cox_event() -> None:
    dataset = _survival(split = True)
    amos.models.Cox().apply(dataset, event = 'arrested')
    amos.metrics.Concordance().apply(dataset)
    rows = dataset.predictions.index
    expected = amos.metrics.concordance_index(
        dataset.y.loc[rows],
        dataset.predictions,
        dataset.data.loc[rows, 'arrested'])
    assert dataset.metrics['concordance'] == pytest.approx(expected)
    assert dataset.metrics['concordance'] > 0.5


@pytest.mark.parametrize(('times', 'scores', 'observed', 'expected'), [
    # Every pair is ordered correctly.
    ([1, 2, 3, 4], [1, 3, 2, 4], [1, 0, 1, 1], 1.0),
    # The first row is predicted to outlast the third, which it does not.
    ([1, 2, 3, 4], [2, 3, 1, 4], [1, 0, 1, 1], 0.75),
    # A row censored when another has its event is compared with it, and a
    # tie in the predictions counts as half.
    ([1, 1, 2, 3], [1, 1, 3, 2], [1, 0, 1, 0], 0.625),
])
def test_concordance_index_counts_pairs(
    times: list[float],
    scores: list[float],
    observed: list[int],
    expected: float) -> None:
    assert amos.metrics.concordance_index(
        times, scores, observed) == pytest.approx(expected)


def test_concordance_index_needs_pairs_to_compare() -> None:
    # Two events at the same time are not compared.
    with pytest.raises(ValueError, match = 'no two rows'):
        amos.metrics.concordance_index([1, 1], [1, 2], [1, 1])


def test_cox_needs_an_event_column_that_exists() -> None:
    dataset = _survival(split = True)
    with pytest.raises(KeyError, match = 'event column'):
        amos.models.Cox().apply(dataset, event = 'missing')


def test_survival_curves_count_the_rows_at_risk() -> None:
    requires('matplotlib')
    dataset = _survival()
    amos.plots.SurvivalCurves().apply(
        dataset, event = 'arrested', group = 'program', at_risk = True)
    curves, table = dataset.figures['survival_curves'].axes
    names = [t.get_text() for t in table.get_yticklabels()]
    assert sorted(names) == ['no', 'yes']
    # At a time of 0, every row is at risk.
    first = [t for t in table.texts if t.get_position()[0] == 0]
    assert sum(int(t.get_text()) for t in first) == len(dataset.data)
    days = dataset.data['days']
    for text in table.texts:
        tick, row = text.get_position()
        group = dataset.data['program'] == names[int(row)]
        assert int(text.get_text()) == int((days[group] >= tick).sum())
