"""Tests the fairness metrics and the fairness evaluator (fairlearn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires, techniques

import amos
from amos.evaluators import Scorecard

GROUP_METRICS = [
    (n, k) for n, k in techniques('metric')
    if issubclass(k, amos.GroupMetric)]


@pytest.fixture(autouse = True)
def _fairlearn() -> None:
    """Skips the tests if fairlearn cannot be imported."""
    requires('fairlearn')


def _fitted(labels: tuple[str, str] = ('no', 'yes')) -> amos.Dataset:
    """Returns a fitted classifier of data with a "race" group."""
    rng = np.random.default_rng(3)
    rows = 300
    race = rng.choice(['a', 'b'], rows)
    x = rng.normal(size = (rows, 3))
    score = x[:, 0] + 0.8 * (race == 'a') + rng.normal(0, 1, rows)
    data = pd.DataFrame(x, columns = ['x0', 'x1', 'x2'])
    data['race'] = race
    data['outcome'] = np.where(score > 0.4, labels[1], labels[0])
    dataset = amos.Dataset(
        data, label = 'outcome', seed = SEED, groups = ['race'])
    amos.splitters.Stratified().apply(dataset)
    return amos.models.SkLogit().apply(dataset)


@pytest.mark.parametrize(('name', 'kind'), GROUP_METRICS)
def test_group_metrics_match_fairlearn(
    name: str,
    kind: type[amos.Metric]) -> None:
    fairness = pytest.importorskip('fairlearn.metrics')
    dataset = _fitted()
    kind().apply(dataset)
    function = getattr(fairness, kind.contents.split('.')[-1])
    rows = dataset.predictions.index
    expected = function(
        (dataset.y.loc[rows] == 'yes').astype(int),
        (dataset.predictions == 'yes').astype(int),
        sensitive_features = dataset.data.loc[rows, 'race'])
    assert dataset.metrics[name] == pytest.approx(expected)


def test_differences_are_better_when_lower() -> None:
    dataset = _fitted()
    metric = amos.metrics.DemographicParity()
    assert metric.score(dataset) == pytest.approx(-metric.measure(dataset))
    ratio = amos.metrics.DemographicParityRatio()
    assert ratio.score(dataset) == pytest.approx(ratio.measure(dataset))
    assert 0 <= ratio.measure(dataset) <= 1


def test_a_group_can_be_named() -> None:
    dataset = _fitted()
    dataset.data['half'] = np.arange(len(dataset.data)) % 2
    by_race = amos.metrics.DemographicParity().measure(dataset)
    by_half = amos.metrics.DemographicParity().measure(dataset, group = 'half')
    assert by_race != by_half


def test_group_metrics_need_a_group(fitted: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'needs a group'):
        amos.metrics.EqualizedOdds().apply(fitted)


def test_group_metrics_need_two_classes(multiclass: amos.Dataset) -> None:
    multiclass.data['half'] = np.arange(len(multiclass.data)) % 2
    amos.models.SkLogit().apply(multiclass)
    with pytest.raises(ValueError, match = 'two classes'):
        amos.metrics.DemographicParity().apply(multiclass, group = 'half')


def test_fairness_table() -> None:
    dataset = _fitted()
    amos.evaluators.Fairness().apply(dataset)
    table = dataset.tables['fairness']
    assert table.index.name == 'race'
    assert list(table.index) == ['a', 'b', 'difference', 'ratio']
    assert list(table.columns) == [
        'count', 'selection_rate', 'accuracy', 'true_positive_rate',
        'false_positive_rate']
    assert table.loc[['a', 'b'], 'count'].sum() == len(dataset.test)
    gap = abs(table.loc['a', 'selection_rate'] - table.loc['b', 'selection_rate'])
    assert table.loc['difference', 'selection_rate'] == pytest.approx(gap)
    assert table.loc['difference', 'selection_rate'] == pytest.approx(
        amos.metrics.DemographicParity().measure(dataset))


def test_the_scorecard_adds_fairness_metrics_for_groups() -> None:
    dataset = _fitted(labels = ('0', '1'))
    table = Scorecard.create(dataset).table
    assert list(table.columns[-2:]) == ['demographic_parity', 'equalized_odds']


def test_the_scorecard_has_no_fairness_metrics_without_groups(
    fitted: amos.Dataset) -> None:
    table = Scorecard.create(fitted).table
    assert 'demographic_parity' not in table.columns
