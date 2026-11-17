"""Tests the assumptions and needs of techniques, and how amos acts on them."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_regression

import amos
from amos import conditions


def make_skewed(rows: int = 400) -> pd.DataFrame:
    """Returns a regression whose label is not linear and whose errors skew."""
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame(rng.normal(size = (rows, 3)), columns = ['a', 'b', 'c'])
    data['target'] = (
        np.exp(1.5 * data['a']) + data['b'] + rng.exponential(size = rows))
    return data


def _amos_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Returns the messages of the warnings from amos."""
    return [
        str(w.message) for w in caught
        if issubclass(w.category, amos.AmosWarning)]


""" The Conditions """


def test_conditions_are_named_in_snake_case() -> None:
    catalog = conditions.catalog()
    assert catalog['normal_errors'] is amos.assumptions.NormalErrors
    assert catalog['no_missing_values'] is amos.needs.NoMissingValues
    assert conditions.borrow('linearity').kind == 'assumption'
    assert conditions.borrow('binary_label').kind == 'need'
    with pytest.raises(KeyError, match = 'not an assumption or need'):
        conditions.borrow('no_such_condition')


def test_every_condition_that_a_technique_names_exists() -> None:
    for name, kind in amos.library.all.items():
        for condition in conditions.conditions_of(kind):
            assert condition in conditions.catalog(), (name, condition)


def test_conditions_of_lists_assumptions_and_then_needs() -> None:
    names = conditions.conditions_of(amos.models.KNN)
    assert names == [
        'similar_scales', 'numeric_features', 'no_missing_values',
        'max_rows', 'max_features']
    assert conditions.conditions_of(amos.cleaners.DropColumns) == []


def test_a_user_can_add_an_assumption() -> None:

    @dataclasses.dataclass(frozen = True)
    class TestPositiveLabel(amos.Assumption):
        """The label is greater than 0."""

        description = 'the label is greater than 0'
        pretested_by = 'check_label'

        def pretest(self, item, technique, found, cache):
            passed = bool((item.y > 0).all())
            return self._verdict(
                technique, self.pretested_by, passed,
                'has a positive label' if passed else
                'assumes a positive label')

    assert conditions.borrow('test_positive_label') is TestPositiveLabel

    @dataclasses.dataclass
    class TestLogLinear(amos.models.Linear):
        """A linear model of a positive label."""

        assumptions = ('test_positive_label',)

    data = make_regression()
    data['target'] -= 1000
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    with pytest.warns(amos.AmosWarning, match = 'assumes a positive label'):
        TestLogLinear().apply(dataset)


""" The Dataset """


def test_the_order_is_not_a_feature() -> None:
    data = make_regression()
    data['day'] = range(len(data))
    dataset = amos.Dataset(data, label = 'target', order = 'day')
    assert 'day' not in dataset.features
    with pytest.raises(ValueError, match = 'cannot also be'):
        amos.Dataset(data, label = 'target', order = 'target')


def test_the_policy_must_be_one_that_amos_knows() -> None:
    dataset = amos.Dataset(make_regression(), label = 'target')
    assert dataset.assumption_violations == 'warn'
    dataset = amos.Dataset.create(
        make_regression(), label = 'target', assumption_violations = 'RAISE')
    assert dataset.assumption_violations == 'raise'
    with pytest.raises(ValueError, match = 'assumption_violations'):
        amos.Dataset(make_regression(), assumption_violations = 'sometimes')


""" Before a Technique Is Applied """


def test_a_technique_warns_of_the_assumptions_that_fail() -> None:
    dataset = amos.Dataset(make_skewed(), label = 'target', seed = SEED)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always')
        amos.models.Linear().apply(dataset)
    messages = _amos_warnings(caught)
    assert any('assumes normal errors' in m for m in messages)
    assert any('linear in the features' in m for m in messages)
    table = dataset.tables['assumptions']
    assert set(table['technique']) == {'linear'}
    failed = table.loc[table['passed'] == False, 'condition']  # noqa: E712
    assert {'normal_errors', 'linearity'} <= set(failed)
    # The violations are in the model's own entry in the history.
    entry = dataset.history[-1]
    assert entry['technique'] == 'linear'
    assert entry['role'] == 'model'
    assert any('normal errors' in v for v in entry['violations'])
    # The model is still fitted.
    assert dataset.model is not None


def test_a_technique_raises_or_ignores_as_it_is_told() -> None:
    dataset = amos.Dataset(
        make_skewed(), label = 'target', assumption_violations = 'raise')
    with pytest.raises(amos.AssumptionError, match = 'normal errors'):
        amos.models.Linear().apply(dataset)
    assert dataset.model is None
    dataset = amos.Dataset(
        make_skewed(), label = 'target', assumption_violations = 'ignore')
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always')
        amos.models.Linear().apply(dataset)
    assert _amos_warnings(caught) == []
    assert 'assumptions' in dataset.tables


def test_a_hard_need_is_never_let_pass() -> None:
    data = make_regression()
    data.loc[0, 'x0'] = np.nan
    dataset = amos.Dataset(
        data, label = 'target', assumption_violations = 'ignore')
    with pytest.raises(amos.AssumptionError, match = 'missing values'):
        amos.models.Linear().apply(dataset)


def test_a_technique_that_another_applies_is_not_checked_again() -> None:
    dataset = amos.Dataset(make_skewed(), label = 'target', seed = SEED)
    amos.splitters.TrainTest().apply(dataset)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always')
        amos.auto.AutoModel().apply(
            dataset, choices = ['linear', 'random_forest'], folds = 1)
    assert dataset.model_name == 'random_forest'
    assert not any('linear' in m for m in _amos_warnings(caught))
    assert 'assumptions' not in dataset.tables


""" Experiments and Reports """


def _experiment(policy: str) -> tuple[amos.Experiment, amos.Dataset]:
    """Returns an experiment of a linear model and a forest, and its result."""
    dataset = amos.Dataset(
        make_skewed(), label = 'target', seed = SEED,
        assumption_violations = policy)
    amos.splitters.TrainTest().apply(dataset)
    experiment = amos.Experiment(
        name = 'analyst', criteria = amos.metrics.R2())
    experiment.populate([[
        amos.models.Linear(name = 'linear'),
        amos.models.RandomForest(name = 'random_forest')]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = experiment.apply(dataset)
    return experiment, result


def test_an_experiment_reports_the_violations_of_each_path() -> None:
    experiment, result = _experiment('warn')
    assert 'linear' in experiment.violations
    table = result.tables['analyst_comparison']
    assert 'normal errors' in table.loc['linear', 'violations']
    assert table.loc['random_forest', 'violations'] == ''
    assert 'linear' in result.history[-1]['path_violations']


def test_an_experiment_leaves_out_the_paths_that_fail_when_told() -> None:
    experiment, result = _experiment('skip')
    assert experiment.winner == 'random_forest'
    assert list(experiment.scores) == ['random_forest']
    assert 'normal errors' in experiment.excluded['linear']
    table = result.tables['analyst_comparison']
    assert list(table.index) == ['random_forest', 'linear']
    assert pd.isna(table.loc['linear', 'rank'])
    assert table.loc['linear', 'violations'].startswith('left out:')
    assert result.history[-1]['excluded_paths'] == experiment.excluded


def test_an_experiment_that_leaves_out_every_path_raises() -> None:
    dataset = amos.Dataset(
        make_skewed(), label = 'target', assumption_violations = 'skip')
    amos.splitters.TrainTest().apply(dataset)
    experiment = amos.Experiment(
        name = 'analyst', criteria = amos.metrics.R2())
    experiment.populate([[amos.models.Linear(name = 'linear')]])
    with pytest.raises(amos.AssumptionError, match = 'left out every path'):
        experiment.apply(dataset)


def test_the_report_lists_the_violations() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {
            'design': 'experiment',
            'criterion': 'r2',
            'steps': ['split', 'model'],
            'split_techniques': 'train_test',
            'model_techniques': ['linear', 'random_forest']}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        project = amos.Project.create(settings, item = make_skewed())
    report = amos.Findings().generate(project)
    assert 'assumptions not met (assumption_violations = warn):' in report
    assert "'linear' assumes normal errors" in report
