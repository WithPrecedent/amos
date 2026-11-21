"""Tests the model that wraps GPBoost, beyond what every model does."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos
from amos import models


@pytest.fixture(autouse = True)
def _gpboost() -> None:
    """Skips the tests if gpboost is not installed."""
    requires('gpboost')


def _grouped(task: str = 'regress', rows: int = 300) -> amos.Dataset:
    """Returns split data whose label depends on a feature and a group."""
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'x0': rng.normal(size = rows),
        'x1': rng.normal(size = rows),
        'court': rng.choice(list('abcdefgh'), rows),
        'latitude': rng.uniform(0, 1, rows),
        'longitude': rng.uniform(0, 1, rows)})
    effects = dict(zip('abcdefgh', rng.normal(scale = 2, size = 8)))
    score = 2 * data['x0'] + np.sin(data['x1']) + data['court'].map(effects)
    data['target'] = score + rng.normal(size = rows)
    if task == 'classify':
        data['target'] = np.where(data['target'] > score.median(), 'yes', 'no')
    dataset = amos.Dataset(
        data.drop(columns = ['latitude', 'longitude']) if task == 'classify'
        else data,
        label = 'target',
        groups = ['court'],
        seed = SEED)
    splitter = amos.splitters.Stratified if task == 'classify' else (
        amos.splitters.TrainTest)
    return splitter().apply(dataset)


def test_random_effects_are_stored_by_group() -> None:
    dataset = _grouped()
    models.Gpboost().apply(
        dataset, n_estimators = 20, coordinates = None,
        groups = 'court')
    table = dataset.tables['gpboost_random_effects']
    assert list(table.index) == ['Error_var', 'court']
    assert (table['estimate'] > 0).all()
    assert dataset.history[-1]['parameters']['likelihood'] == 'gaussian'


def test_the_first_group_is_the_default() -> None:
    dataset = _grouped()
    dataset.data = dataset.data.drop(columns = ['latitude', 'longitude'])
    models.Gpboost().apply(dataset, n_estimators = 10)
    assert dataset.model.groups == 'court'
    assert list(dataset.model.feature_names_in_) == ['x0', 'x1']


def test_features_alone_are_predicted_by_the_trees() -> None:
    dataset = _grouped()
    dataset.data = dataset.data.drop(columns = ['latitude', 'longitude'])
    models.Gpboost().apply(dataset, n_estimators = 10)
    model = dataset.model
    x = dataset.data.loc[dataset.test, ['x0', 'x1', 'court']]
    unseen = x.assign(court = 'a court that it has not seen')
    np.testing.assert_allclose(
        model.predict(x[['x0', 'x1']]), model.predict(unseen))
    assert not np.allclose(model.predict(x), model.predict(unseen))


def test_classifies_two_classes_of_any_kind() -> None:
    dataset = _grouped('classify')
    models.Gpboost().apply(dataset, n_estimators = 10)
    assert set(dataset.predictions) <= {'no', 'yes'}
    assert list(dataset.probabilities.columns) == ['no', 'yes']
    assert np.allclose(dataset.probabilities.sum(axis = 1), 1)
    assert dataset.history[-1]['parameters']['likelihood'] == (
        'bernoulli_logit')
    tree_only = dataset.model.predict_proba(
        dataset.data.loc[dataset.test, ['x0', 'x1']])
    assert np.allclose(tree_only.sum(axis = 1), 1)


def test_a_gaussian_process_needs_no_groups() -> None:
    dataset = _grouped(rows = 200)
    models.Gpboost().apply(
        dataset, n_estimators = 10, coordinates = ['latitude', 'longitude'],
        cov_function = 'exponential')
    assert dataset.model.groups is None
    assert list(dataset.model.feature_names_in_) == ['x0', 'x1']
    table = dataset.tables['gpboost_random_effects']
    assert 'GP_var' in table.index


def test_a_search_keeps_the_random_effects() -> None:
    dataset = _grouped()
    dataset.data = dataset.data.drop(columns = ['latitude', 'longitude'])
    models.Gpboost().apply(
        dataset, search = 'grid', cv = 2, n_estimators = [5, 10])
    assert len(dataset.tables['gpboost_search']) == 2
    assert dataset.model.groups == 'court'


def test_the_adapter_can_be_cloned() -> None:
    sklearn_base = importlib.import_module('sklearn.base')
    model = models.GaussianProcessBoosting(
        groups = 'court', likelihood = 'bernoulli_probit', n_estimators = 7)
    copy = sklearn_base.clone(model)
    assert copy.get_params()['n_estimators'] == 7
    assert copy.get_params()['likelihood'] == 'bernoulli_probit'
    assert sklearn_base.is_classifier(copy)
    assert copy.set_params(n_estimators = 3).parameters['n_estimators'] == 3


def test_a_likelihood_must_suit_the_task() -> None:
    dataset = _grouped()
    with pytest.raises(ValueError, match = 'does not suit a label'):
        models.Gpboost().apply(dataset, likelihood = 'bernoulli_logit')


def test_random_effects_need_groups_or_coordinates() -> None:
    model = models.GaussianProcessBoosting()
    x = pd.DataFrame({'x0': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match = 'needs "groups" or "coordinates"'):
        model.fit(x, pd.Series([1.0, 2.0, 3.0]))


def test_a_classifier_needs_two_classes() -> None:
    model = models.GaussianProcessBoosting(
        groups = 'court', likelihood = 'bernoulli_logit')
    x = pd.DataFrame({'x0': [1.0, 2.0, 3.0], 'court': ['a', 'b', 'a']})
    with pytest.raises(ValueError, match = 'needs two classes'):
        model.fit(x, pd.Series(['a', 'b', 'c']))
