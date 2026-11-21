"""Tests the Bayesian models (Bambi and PyMC), beyond what every model does."""

from __future__ import annotations

import copy
import importlib
import pickle

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos
from amos import models

# Few draws keep the tests quick.
QUICK = {'draws': 100, 'tune': 100}


def _grouped(task: str = 'regress', rows: int = 160) -> amos.Dataset:
    """Returns data split by group, whose label depends on a group."""
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'x0': rng.normal(size = rows),
        'x1': rng.normal(size = rows),
        'court': rng.choice(list('abcdefgh'), rows)})
    effects = dict(zip('abcdefgh', rng.normal(scale = 2, size = 8)))
    score = 2 * data['x0'] + np.sin(data['x1']) + data['court'].map(effects)
    data['target'] = score + rng.normal(size = rows)
    if task == 'classify':
        data['target'] = np.where(data['target'] > score.median(), 'yes', 'no')
    dataset = amos.Dataset(
        data, label = 'target', groups = ['court'], seed = SEED)
    return amos.splitters.GroupSplit().apply(dataset)


""" Bambi """


def test_bambi_stores_its_posteriors() -> None:
    requires('bambi')
    dataset = _grouped()
    models.Bambi().apply(dataset, **QUICK)
    table = dataset.tables['bambi_coefficients']
    assert list(table.index) == ['const', 'x0', 'x1']
    assert table['p_value'].isna().all()
    assert (table['ci_lower'] < table['coefficient']).all()
    assert table.loc['x0', 'ci_lower'] > 1
    effects = dataset.tables['bambi_random_effects']
    assert list(effects.index) == ['1|court']
    assert dataset.model.groups == 'court'
    assert dataset.history[-1]['parameters']['family'] == 'gaussian'


def test_bambi_takes_a_formula() -> None:
    requires('bambi')
    dataset = _grouped()
    models.Bambi().apply(
        dataset, formula = 'target ~ x0 + (1|court)', **QUICK)
    table = dataset.tables['bambi_coefficients']
    assert list(table.index) == ['const', 'x0']
    assert list(dataset.tables['bambi_random_effects'].index) == ['1|court']


def test_bambi_classifies_two_classes_of_any_kind() -> None:
    requires('bambi')
    dataset = _grouped('classify')
    models.Bambi().apply(dataset, **QUICK)
    assert set(dataset.predictions) <= {'no', 'yes'}
    assert list(dataset.probabilities.columns) == ['no', 'yes']
    assert np.allclose(dataset.probabilities.sum(axis = 1), 1)
    assert dataset.history[-1]['parameters']['family'] == 'bernoulli'


def test_bambi_predicts_rows_without_their_groups() -> None:
    requires('bambi')
    dataset = _grouped()
    models.Bambi().apply(dataset, **QUICK)
    predicted = dataset.model.predict(dataset.x_test)
    assert predicted.shape == (len(dataset.test),)
    assert np.isfinite(predicted).all()


def test_bambi_pickles_without_its_model() -> None:
    requires('bambi')
    dataset = _grouped()
    models.Bambi().apply(dataset, **QUICK)
    x = dataset.data.loc[dataset.test, ['x0', 'x1', 'court']]
    restored = pickle.loads(pickle.dumps(dataset.model))
    np.testing.assert_allclose(restored.predict(x), dataset.model.predict(x))
    np.testing.assert_allclose(
        copy.deepcopy(dataset.model).predict(x), dataset.model.predict(x))


def test_bambi_family_must_suit_the_task() -> None:
    requires('bambi')
    dataset = _grouped()
    with pytest.raises(ValueError, match = 'does not suit a label'):
        models.Bambi().apply(dataset, family = 'bernoulli', **QUICK)


""" PyMC """


def test_pymc_fits_trees_without_groups() -> None:
    requires('pymc_bart')
    dataset = _grouped()
    models.Pymc().apply(dataset, m = 20, **QUICK)
    model = dataset.model
    assert list(model.feature_names_in_) == ['x0', 'x1']
    assert np.isclose(model.feature_importances_.sum(), 1)
    assert model.feature_importances_[0] > model.feature_importances_[1]
    assert dataset.history[-1]['parameters']['likelihood'] == 'normal'
    assert np.corrcoef(dataset.predictions, dataset.y_test)[0, 1] > 0.5


def test_pymc_classifies_two_classes_of_any_kind() -> None:
    requires('pymc_bart')
    dataset = _grouped('classify')
    models.Pymc().apply(dataset, m = 20, **QUICK)
    assert set(dataset.predictions) <= {'no', 'yes'}
    assert list(dataset.probabilities.columns) == ['no', 'yes']
    assert np.allclose(dataset.probabilities.sum(axis = 1), 1)
    assert dataset.history[-1]['parameters']['likelihood'] == 'bernoulli'


def test_pymc_pickles_its_trees() -> None:
    requires('pymc_bart')
    dataset = _grouped()
    models.Pymc().apply(dataset, m = 20, **QUICK)
    restored = pickle.loads(pickle.dumps(dataset.model))
    np.testing.assert_allclose(
        restored.predict(dataset.x_test), dataset.predictions.to_numpy())


def test_pymc_needs_a_known_likelihood() -> None:
    requires('pymc_bart')
    model = models.BayesianAdditiveTrees(likelihood = 'gamma')
    with pytest.raises(ValueError, match = 'must be one of'):
        model.fit(pd.DataFrame({'x0': [1.0, 2.0]}), pd.Series([1.0, 2.0]))


def test_the_adapters_are_scikit_learn_classifiers_when_they_classify() -> None:
    sklearn_base = importlib.import_module('sklearn.base')
    assert sklearn_base.is_classifier(
        models.BayesianMixedModel(family = 'bernoulli'))
    assert sklearn_base.is_regressor(models.BayesianAdditiveTrees())
    copied = sklearn_base.clone(models.BayesianAdditiveTrees(m = 7))
    assert copied.get_params()['m'] == 7
