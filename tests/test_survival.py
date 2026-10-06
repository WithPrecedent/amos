"""Tests survival analysis: Kaplan-Meier curves, Cox models, and concordance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos


@pytest.fixture(autouse = True)
def _lifelines() -> None:
    """Skips the tests if lifelines cannot be imported."""
    requires('lifelines')


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


def test_kaplan_meier_matches_lifelines() -> None:
    lifelines = pytest.importorskip('lifelines')
    dataset = _survival()
    amos.describers.KaplanMeier().apply(dataset, event = 'arrested')
    table = dataset.tables['kaplan_meier']
    assert list(table.columns) == ['group', 'time', 'survival', 'at_risk']
    assert set(table['group']) == {'all'}
    fitter = lifelines.KaplanMeierFitter().fit(
        dataset.y, dataset.data['arrested'])
    np.testing.assert_allclose(
        table['survival'], fitter.survival_function_.iloc[:, 0])
    assert table['survival'].is_monotonic_decreasing
    assert table['at_risk'].iloc[0] == len(dataset.data)


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


def test_concordance_uses_the_cox_event() -> None:
    lifelines = pytest.importorskip('lifelines.utils')
    dataset = _survival(split = True)
    amos.models.Cox().apply(dataset, event = 'arrested')
    amos.metrics.Concordance().apply(dataset)
    rows = dataset.predictions.index
    expected = lifelines.concordance_index(
        dataset.y.loc[rows],
        dataset.predictions,
        dataset.data.loc[rows, 'arrested'])
    assert dataset.metrics['concordance'] == pytest.approx(expected)
    assert dataset.metrics['concordance'] > 0.5


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
