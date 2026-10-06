"""Tests the workers module: the Experiment design."""

from __future__ import annotations

import dataclasses

import pytest

import amos
import chrisjen


def test_experiment_keeps_the_best_path_and_compares_them(
    classified: amos.Dataset) -> None:
    experiment = amos.Experiment(
        name = 'analyst', criteria = amos.metrics.Accuracy())
    experiment.populate([[
        amos.models.Baseline(name = 'baseline'),
        amos.models.Logit(name = 'logit')]])
    result = experiment.apply(classified)
    assert experiment.winner == 'logit'
    assert result.model is not None
    table = result.tables['analyst_comparison']
    assert list(table.index) == ['logit', 'baseline']
    assert list(table['rank']) == [1, 2]
    assert table.index.name == 'path'
    assert table.loc['logit', 'accuracy'] == pytest.approx(
        experiment.scores['logit'])
    assert result.history[-1] == {
        'technique': 'analyst',
        'winner': 'logit',
        'criterion': 'accuracy',
        'paths': 2}
    branches = {branch.label: branch for branch in result.branches}
    assert set(branches) == {'baseline', 'logit'}
    assert branches['logit'].steps == {'model': 'logit'}
    assert branches['logit'].score == experiment.scores['logit']
    assert branches['logit'].result.predictions is not None


def test_experiment_reports_the_real_value_of_a_lower_is_better_metric(
    classified: amos.Dataset) -> None:
    experiment = amos.Experiment(
        name = 'analyst', criteria = amos.metrics.LogLoss())
    experiment.populate([[
        amos.models.Baseline(name = 'baseline'),
        amos.models.Logit(name = 'logit')]])
    result = experiment.apply(classified)
    table = result.tables['analyst_comparison']
    assert experiment.winner == 'logit'
    assert (table['log_loss'] > 0).all()
    assert (table['score'] < 0).all()


def test_experiment_includes_metrics_from_each_path(
    classified: amos.Dataset) -> None:
    experiment = amos.Experiment(
        name = 'analyst', criteria = amos.metrics.Accuracy())
    experiment.populate([
        [amos.models.Baseline(name = 'baseline'),
         amos.models.Logit(name = 'logit')],
        amos.evaluators.Scorecard(name = 'scorecard')])
    result = experiment.apply(classified)
    table = result.tables['analyst_comparison']
    assert {'f1', 'roc_auc', 'log_loss'} <= set(table.columns)
    assert len(table) == 2


def test_experiment_works_with_any_criteria_and_item() -> None:
    @dataclasses.dataclass
    class Double(chrisjen.Technique):
        def implement(self, item, **kwargs):
            return item * 2

    @dataclasses.dataclass
    class Triple(chrisjen.Technique):
        def implement(self, item, **kwargs):
            return item * 3

    experiment = amos.Experiment(
        name = 'numbers', criteria = chrisjen.Criteria(contents = float))
    experiment.populate([[Double(name = 'double'), Triple(name = 'triple')]])
    assert experiment.apply(2) == 6
    assert experiment.winner == 'triple'
    assert experiment.scores == {'double': 4.0, 'triple': 6.0}
    table = experiment.tabulate()
    assert list(table.index) == ['triple', 'double']
    assert list(table.columns) == ['rank', 'score']


def test_experiment_needs_paths() -> None:
    experiment = amos.Experiment(
        name = 'empty', criteria = amos.metrics.Accuracy())
    with pytest.raises(ValueError, match = 'no paths'):
        experiment.apply(1)
