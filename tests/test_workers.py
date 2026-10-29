"""Tests the workers module: the Experiment design."""

from __future__ import annotations

import dataclasses

import pandas as pd
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


def test_experiment_that_conserves_memory_keeps_only_the_winner(
    classified: amos.Dataset) -> None:
    def run(conserve_memory: bool) -> tuple[amos.Experiment, amos.Dataset]:
        experiment = amos.Experiment(
            name = 'analyst',
            criteria = amos.metrics.LogLoss(),
            conserve_memory = conserve_memory)
        experiment.populate([
            [amos.models.Baseline(name = 'baseline'),
             amos.models.Logit(name = 'logit')],
            amos.evaluators.Scorecard(name = 'scorecard')])
        return experiment, experiment.apply(classified)

    keeping, kept = run(conserve_memory = False)
    conserving, result = run(conserve_memory = True)
    assert conserving.winner == keeping.winner == 'logit > scorecard'
    assert list(conserving.results) == ['logit > scorecard']
    assert conserving.results['logit > scorecard'] is result
    assert conserving.scores == keeping.scores
    assert set(conserving.outcomes) == {
        'baseline > scorecard', 'logit > scorecard'}
    # The table and the branches are the same either way.
    pd.testing.assert_frame_equal(
        result.tables['analyst_comparison'], kept.tables['analyst_comparison'])
    assert result.history[-1]['paths'] == 2
    branches = {
        branch.steps['model']: branch for branch in result.branches}
    assert branches['baseline'].result.predictions is not None
    assert branches['baseline'].result.metrics['log_loss'] > 0
    # Only the winner's whole dataset can be found through its branch.
    assert branches['logit'].dataset is result
    assert branches['baseline'].dataset is None


def test_conserve_memory_in_the_general_section_reaches_experiments() -> None:
    settings = {
        'general': {'label': 'target', 'conserve_memory': True},
        'study_project': {'study_workers': 'analyst, keeper'},
        'analyst': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'model_techniques': 'logit, baseline'},
        'keeper': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'conserve_memory': False,
            'model_techniques': 'logit, baseline'}}
    idea = chrisjen.Idea.create(settings)

    class Holder:
        pass

    holder = Holder()
    holder.idea = idea  # type: ignore[attr-defined]
    assert amos.Experiment.build('analyst', holder).conserve_memory
    assert not amos.Experiment.build('keeper', holder).conserve_memory


def test_experiment_works_with_any_criteria_and_item() -> None:
    @dataclasses.dataclass
    class Double(chrisjen.Technique):
        def apply(self, item, **kwargs):
            return item * 2

    @dataclasses.dataclass
    class Triple(chrisjen.Technique):
        def apply(self, item, **kwargs):
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
