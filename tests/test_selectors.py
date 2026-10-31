"""Tests the selectors module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_mixed, make_numeric, package_of, requires

import amos
import chrisjen
from amos import evaluators, selectors, transformers

# `feature_group` keeps the features that it is given, so it is tested on its
# own.
SELECTORS = [
    (n, k) for n, k in sorted(amos.library.get_genre('selector').items())
    if n != 'feature_group']
# Parameters that make a selector keep fewer than every feature of the
# `classified` dataset, which has 6.
FEWER: dict[str, dict[str, int]] = {
    'exhaustive_selection': {'cv': 3},
    'k_best': {'k': 3},
    'recursive_elimination_cv': {'cv': 3},
    'sequential_selection': {'cv': 3},
    'sequential_selection_sk': {'cv': 3}}


def _split_mixed() -> amos.Dataset:
    """Returns the mixed dataset, split, with its missing ages filled."""
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    return transformers.MedianImpute().apply(dataset)


def _group_settings(
    groups: dict[str, object],
    techniques: list[str]) -> dict[str, object]:
    """Returns the settings of an experiment that compares `techniques`."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'feature_groups': groups,
        'study_project': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'steps': ['split', 'select', 'model'],
            'split_techniques': 'stratified',
            'select_techniques': techniques,
            'model_techniques': 'logit'}}


def _features_by_group(project: amos.Project) -> dict[str, list[str]]:
    """Returns the features that each branch of an experiment kept.

    Args:
        project: an applied project whose experiment has a "select" step.

    Returns:
        The features of the dataset that each branch made, by the name of
            its "select" technique.

    """
    found = {}
    for branch in project.result.branches:
        source = branch.source() if branch.source else None
        assert source is not None
        found[branch.steps['select']] = source.features
    return found


@pytest.mark.parametrize(('name', 'kind'), SELECTORS)
def test_selectors_keep_some_features(
    name: str,
    kind: type[amos.Selector],
    classified: amos.Dataset) -> None:
    requires(package_of(kind))
    classified.data['constant'] = 1.0
    kind().apply(classified, **FEWER.get(name, {}))
    assert 0 < len(classified.features) < 7
    assert set(classified.features) <= {f'x{i}' for i in range(6)}
    assert classified.history[-1]['technique'] == name


def test_selectors_are_a_genre_of_transformer() -> None:
    operations = chrisjen.library['vertex']['operation']
    assert 'selector' in operations['transformer']
    assert chrisjen.library.classify('k_best') == 'selector'
    assert chrisjen.library.classify('pca_reduce') == 'reducer'


def test_selectors_raise_an_error_if_they_keep_no_features() -> None:
    noise = pd.DataFrame(
        np.random.default_rng(0).normal(size = (200, 4)),
        columns = ['a', 'b', 'c', 'd'])
    noise['target'] = np.tile([0, 1], 100)
    dataset = amos.Dataset(noise, label = 'target', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    with pytest.raises(ValueError, match = 'kept none of the 4 features'):
        selectors.SelectFwe().apply(dataset, alpha = 1e-12)


def test_generic_univariate_takes_a_mode(classified: amos.Dataset) -> None:
    selectors.GenericUnivariate().apply(classified, mode = 'k_best', param = 2)
    assert len(classified.features) == 2
    assert classified.fitted['generic_univariate'].score_func.__name__ == (
        'f_classif')


def test_recursive_elimination_keeps_every_feature_if_asked_for_more(
    classified: amos.Dataset) -> None:
    selectors.RecursiveElimination().apply(
        classified, n_features_to_select = 50)
    assert len(classified.features) == 6


def test_recursive_elimination_cv_adds_the_score_of_each_size(
    classified: amos.Dataset) -> None:
    selectors.RecursiveEliminationCV().apply(classified, cv = 3)
    table = classified.tables['recursive_elimination_cv_search']
    assert list(table.columns) == ['size', 'score', 'std']
    assert sorted(table['size']) == [1, 2, 3, 4, 5, 6]
    assert table['score'].is_monotonic_decreasing
    tool = classified.fitted['recursive_elimination_cv']
    assert table['size'].iloc[0] == tool.n_features_
    assert tool.scoring == 'accuracy'


def test_selectors_build_a_model_of_the_library_by_name(
    regressed: amos.Dataset) -> None:
    selectors.SelectFromModel().apply(regressed, estimator = 'random_forest')
    estimator = regressed.fitted['select_from_model'].estimator
    assert type(estimator).__name__ == 'RandomForestRegressor'
    assert estimator.random_state == SEED


def test_selectors_reject_a_model_for_another_task(
    regressed: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'can classify'):
        selectors.RecursiveElimination().apply(regressed, estimator = 'logit')


def test_sequential_selection_sk_scores_by_the_task(
    regressed: amos.Dataset) -> None:
    selectors.SequentialSelectionSK().apply(
        regressed, n_features_to_select = 2, cv = 3)
    tool = regressed.fitted['sequential_selection_sk']
    assert type(tool.estimator).__name__ == 'LinearRegression'
    assert tool.scoring == 'r2'
    assert len(regressed.features) == 2


def test_feature_group_keeps_the_listed_features(
    classified: amos.Dataset) -> None:
    selectors.FeatureGroup().apply(classified, columns = ['x3', 'x1'])
    assert classified.features == ['x1', 'x3']
    entry = classified.history[-1]
    assert entry['technique'] == 'feature_group'
    assert entry['listed'] == ['x3', 'x1']
    assert entry['kept'] == ['x1', 'x3']
    assert entry['removed'] == ['x0', 'x2', 'x4', 'x5']
    assert 'target' in classified.data.columns


def test_the_scorecard_lists_a_group_that_is_not_in_the_library(
    classified: amos.Dataset) -> None:
    # A group registered only in a project's own library.
    selectors.FeatureGroup(name = 'mine').apply(
        classified, columns = ['x0', 'x1'])
    with pytest.raises(ValueError, match = 'not in the library'):
        chrisjen.library.classify('mine')
    steps = evaluators.Scorecard()._history_steps(classified)
    assert steps['selector'] == 'mine'


def test_feature_group_keeps_the_features_made_from_a_listed_one() -> None:
    dataset = _split_mixed()
    transformers.OneHot().apply(dataset, columns = ['region'])
    made = [c for c in dataset.features if c.startswith('region_')]
    assert len(made) > 1
    selectors.FeatureGroup().apply(dataset, columns = ['age', 'region'])
    assert dataset.features == ['age', *made]


def test_feature_group_rejects_names_that_are_not_features(
    classified: amos.Dataset) -> None:
    with pytest.raises(KeyError, match = r"\['target', 'missing'\]"):
        selectors.FeatureGroup().apply(
            classified, columns = ['x0', 'target', 'missing'])
    with pytest.raises(ValueError, match = 'has no features'):
        selectors.FeatureGroup().apply(classified)


def test_feature_groups_in_the_settings_are_techniques() -> None:
    data = make_numeric()
    settings = _group_settings(
        {'first': ['x0', 'x1'], 'last': 'x5'}, ['first', 'last'])
    project = amos.Project.create(settings, item = data)
    assert project.name == 'study'
    assert chrisjen.library.all['first'] is selectors.FeatureGroup
    assert chrisjen.library.classify('last') == 'selector'
    assert _features_by_group(project) == {
        'first': ['x0', 'x1'], 'last': ['x5']}


def test_the_feature_groups_section_is_not_a_worker() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'feature_groups': {'first': ['x0', 'x1']},
        'analyst': {
            'steps': ['split', 'select', 'model'],
            'split_techniques': 'stratified',
            'select_techniques': 'first',
            'model_techniques': 'logit'}}
    project = amos.Project.create(settings, item = make_numeric())
    assert project.name == 'analyst'
    assert 'feature_groups' not in project.idea.workers
    assert project.result.features == ['x0', 'x1']


def test_feature_groups_can_be_numbered_techniques() -> None:
    settings = _group_settings({}, ['feature_group1', 'feature_group2'])
    settings['feature_group1_parameters'] = {'columns': ['x0', 'x1']}
    settings['feature_group2_parameters'] = {'columns': ['x2']}
    project = amos.Project.create(settings, item = make_numeric())
    assert _features_by_group(project) == {
        'feature_group1': ['x0', 'x1'], 'feature_group2': ['x2']}


@pytest.mark.parametrize(('groups', 'error', 'match'), [
    ({'k_best': ['x0']}, ValueError, 'KBest class'),
    ({'empty': []}, ValueError, 'lists no features'),
    ({'numbers': [1, 2]}, TypeError, 'a list of them'),
    ('x0', TypeError, 'must be a section')])
def test_feature_groups_are_checked(
    groups: object,
    error: type[Exception],
    match: str) -> None:
    settings = _group_settings({}, ['k_best'])
    settings['feature_groups'] = groups
    with pytest.raises(error, match = match):
        amos.Project.create(settings, item = make_numeric())


def test_sequential_selection_keeps_the_best_features(
    classified: amos.Dataset) -> None:
    requires('mlxtend')
    selectors.SequentialSelection().apply(classified, k_features = 3, cv = 3)
    assert len(classified.features) == 3
    table = classified.tables['sequential_selection_search']
    assert list(table.columns) == ['features', 'size', 'score', 'std']
    assert sorted(table['size']) == [1, 2, 3]
    assert table['score'].is_monotonic_decreasing
    chosen = table.loc[table['size'] == 3, 'features'].iloc[0]
    assert chosen.split(', ') == classified.features
    selector = classified.fitted['sequential_selection']
    assert selector.estimator.random_state == SEED
    assert selector.scoring == 'accuracy'


def test_selectors_score_by_the_task(regressed: amos.Dataset) -> None:
    requires('mlxtend')
    selectors.SequentialSelection().apply(
        regressed, k_features = 2, forward = False, cv = 3)
    selector = regressed.fitted['sequential_selection']
    assert type(selector.estimator).__name__ == 'LinearRegression'
    assert selector.scoring == 'r2'
    assert len(regressed.features) == 2


def test_selectors_keep_groups_in_one_fold(classified: amos.Dataset) -> None:
    requires('mlxtend')
    classified.data['case'] = np.arange(len(classified.data)) // 3
    classified.groups = ['case']
    selectors.SequentialSelection().apply(classified, k_features = 2, cv = 3)
    folds = classified.fitted['sequential_selection'].cv
    cases = classified.data.loc[classified.y_train.index, 'case'].to_numpy()
    assert len(folds) == 3
    for train, test in folds:
        assert not set(cases[train]) & set(cases[test])


def test_exhaustive_selection_tries_every_set(
    classified: amos.Dataset) -> None:
    requires('mlxtend')
    selectors.ExhaustiveSelection().apply(
        classified, columns = ['x0', 'x1', 'x2'], max_features = 5, cv = 3)
    table = classified.tables['exhaustive_selection_search']
    # Every set of 1 to 3 of the 3 columns.
    assert len(table) == 7
    assert table['score'].is_monotonic_decreasing
    kept = [c for c in classified.features if c in {'x0', 'x1', 'x2'}]
    assert ', '.join(kept) == table['features'].iloc[0]
    assert {'x3', 'x4', 'x5'} <= set(classified.features)


def test_selectors_take_a_model_and_text_classes() -> None:
    requires('mlxtend')
    dataset = _split_mixed()
    transformers.OneHot().apply(dataset)
    selectors.SequentialSelection().apply(
        dataset, k_features = 2, cv = 3,
        estimator = 'sklearn.tree.DecisionTreeClassifier')
    selector = dataset.fitted['sequential_selection']
    assert type(selector.estimator).__name__ == 'DecisionTreeClassifier'
    assert selector.estimator.random_state == SEED
    # The dates are not chosen from, so they are kept.
    assert 'joined' in dataset.features
    assert len(dataset.numerics + dataset.booleans) == 2


def test_k_best_keeps_every_feature_if_there_are_fewer_than_k(
    classified: amos.Dataset) -> None:
    selectors.KBest().apply(classified, k = 50)
    assert len(classified.features) == 6


def test_k_best_scores_by_the_task(regressed: amos.Dataset) -> None:
    selectors.KBest().apply(regressed, k = 2)
    tool = regressed.fitted['k_best']
    assert tool.score_func.__name__ == 'f_regression'
    assert len(regressed.features) == 2


def test_k_best_imports_a_scoring_function(classified: amos.Dataset) -> None:
    selectors.KBest().apply(
        classified,
        k = 2,
        score_func = 'sklearn.feature_selection.mutual_info_classif')
    tool = classified.fitted['k_best']
    assert tool.score_func.__name__ == 'mutual_info_classif'
