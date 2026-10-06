"""Tests the models module."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_mixed, package_of, requires, techniques

import amos
from amos import models

MODELS = techniques('model')


def _check(item: amos.Dataset, task: str) -> None:
    """Checks the predictions that a model stored in `item`."""
    assert item.model is not None
    assert list(item.predictions.index) == list(item.test)
    assert item.predictions.name == 'target'
    if task == 'classify':
        assert set(item.predictions) <= set(item.classes)
    record = item.history[-1]
    assert record['task'] == task
    assert record['rows'] == len(item.train)


@pytest.mark.parametrize(('name', 'kind'), [
    (n, k) for n, k in MODELS if 'classify' in k.tools])
def test_classifiers(
    name: str,
    kind: type[amos.Model],
    classified: amos.Dataset) -> None:
    requires(package_of(kind))
    kind().apply(classified)
    _check(classified, 'classify')
    if classified.probabilities is not None:
        assert list(classified.probabilities.columns) == [0, 1]
        sums = classified.probabilities.sum(axis = 1)
        assert np.allclose(sums, 1)


@pytest.mark.parametrize(('name', 'kind'), [
    (n, k) for n, k in MODELS if 'regress' in k.tools])
def test_regressors(
    name: str,
    kind: type[amos.Model],
    regressed: amos.Dataset) -> None:
    requires(package_of(kind))
    kind().apply(regressed)
    _check(regressed, 'regress')
    assert regressed.probabilities is None


def test_a_model_for_another_task_raises(regressed: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'can classify'):
        models.Logit().apply(regressed)


def test_a_model_needs_a_label() -> None:
    with pytest.raises(ValueError, match = 'needs a dataset with a label'):
        models.Logit().apply(pd.DataFrame({'a': [1, 2]}))


def _encoded(split: bool = True) -> amos.Dataset:
    """Returns the mixed dataset, ready for any model."""
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined'])
    if split:
        amos.splitters.Stratified().apply(dataset)
    amos.transformers.MedianImpute().apply(dataset)
    return amos.transformers.OneHot().apply(dataset)


def test_models_without_a_split_predict_every_row() -> None:
    dataset = _encoded(split = False)
    models.Logit().apply(dataset)
    assert len(dataset.predictions) == 200


def test_models_explain_that_they_need_numbers() -> None:
    dataset = amos.Dataset(make_mixed(), label = 'outcome')
    with pytest.raises(ValueError, match = r"\['region', 'joined'\]"):
        models.Logit().apply(dataset)


def test_models_accept_categories() -> None:
    requires('lightgbm')
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined'])
    amos.cleaners.AutoCategorize().apply(dataset, columns = ['region'])
    models.Lightgbm().apply(dataset, n_estimators = 5)
    assert len(dataset.predictions) == 200


def test_the_seed_makes_models_reproducible(classified: amos.Dataset) -> None:
    first = models.RandomForest().apply(classified, n_estimators = 10)
    probabilities = first.probabilities.copy()
    models.RandomForest().apply(classified, n_estimators = 10)
    pd.testing.assert_frame_equal(classified.probabilities, probabilities)
    assert classified.model.random_state == SEED


def test_contents_overrides_the_tools(classified: amos.Dataset) -> None:
    technique = amos.Model(
        name = 'tree', contents = 'sklearn.tree.ExtraTreeClassifier')
    technique.apply(classified)
    assert type(classified.model).__name__ == 'ExtraTreeClassifier'


def test_grid_search(classified: amos.Dataset) -> None:
    models.DecisionTree().apply(
        classified, search = 'grid', max_depth = [1, 3], cv = 3)
    table = classified.tables['decision_tree_search']
    assert len(table) == 2
    assert list(table['rank_test_score']) == [1, 2]
    assert classified.model.max_depth in {1, 3}
    assert classified.history[-1]['search'] == 'grid'


def test_random_search_draws_from_ranges(classified: amos.Dataset) -> None:
    models.RandomForest().apply(
        classified,
        search = 'random',
        n_estimators = [5, 20],
        max_features = [0.3, 0.9],
        n_iter = 4,
        cv = 3,
        scoring = 'roc_auc')
    table = classified.tables['random_forest_search']
    assert len(table) == 4
    assert table['param_n_estimators'].between(5, 20).all()


def test_lists_are_passed_as_they_are_without_a_search(
    regressed: amos.Dataset) -> None:
    models.NeuralNetwork().apply(
        regressed, hidden_layer_sizes = [8, 4], max_iter = 50)
    assert regressed.model.hidden_layer_sizes == [8, 4]


def test_xgboost_classifies_text_labels() -> None:
    requires('xgboost')
    dataset = _encoded()
    models.Xgboost().apply(dataset, n_estimators = 10)
    assert isinstance(dataset.model, models.LabelCoded)
    assert set(dataset.predictions) <= {'no', 'yes'}
    assert list(dataset.probabilities.columns) == ['no', 'yes']
    assert len(dataset.model.feature_importances_) == len(dataset.features)


def test_label_coded_works_with_a_search() -> None:
    requires('xgboost')
    dataset = _encoded()
    models.Xgboost().apply(
        dataset, search = 'grid', max_depth = [2, 3], n_estimators = 5, cv = 3)
    assert dataset.model.get_params()['estimator__max_depth'] in {2, 3}
    assert 0 <= dataset.model.score(dataset.x_test, dataset.y_test) <= 1


def test_label_coded_sets_parameters() -> None:
    wrapper = models.LabelCoded(estimator = models.Statsmodel())
    wrapper.set_params(estimator__family = 'poisson')
    assert wrapper.estimator.family == 'poisson'
    wrapper.set_params(estimator = models.Statsmodel(kind = 'glm'))
    assert wrapper.kind == 'glm'
    with pytest.raises(AttributeError):
        _ = wrapper._private


def test_ols_reports_coefficients(regressed: amos.Dataset) -> None:
    requires('statsmodels')
    models.OLS().apply(regressed)
    table = regressed.tables['ols_coefficients']
    assert list(table.index) == ['const', *regressed.features]
    assert list(table.columns) == [
        'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
        'ci_upper']
    assert (table['ci_lower'] <= table['coefficient']).all()
    assert regressed.model.get_params()['kind'] == 'ols'


def test_glm_is_logistic_for_classification(
    classified: amos.Dataset) -> None:
    requires('statsmodels')
    models.GLM().apply(classified)
    assert classified.model.family == 'binomial'
    assert 'glm_coefficients' in classified.tables
    assert list(classified.probabilities.columns) == [0, 1]


def test_glm_family_can_be_set(regressed: amos.Dataset) -> None:
    requires('statsmodels')
    models.GLM().apply(regressed, family = 'gamma')
    assert regressed.model.family == 'gamma'
    with pytest.raises(AttributeError, match = 'binomial'):
        regressed.model.predict_proba(regressed.x_test)


def test_binomial_needs_two_classes(multiclass: amos.Dataset) -> None:
    requires('statsmodels')
    with pytest.raises(ValueError, match = 'two classes'):
        models.GLM().apply(multiclass)


def test_statsmodel_without_a_constant(regressed: amos.Dataset) -> None:
    requires('statsmodels')
    models.OLS().apply(regressed, constant = False)
    assert 'const' not in regressed.tables['ols_coefficients'].index
    assert len(regressed.model.coef_) == len(regressed.features)


def _courts(rows: int = 300) -> amos.Dataset:
    """Returns a regression with judge and court groups (fixed effects)."""
    rng = np.random.default_rng(9)
    judge = rng.integers(0, 6, rows)
    x = rng.normal(size = rows)
    data = pd.DataFrame({
        'x': x,
        'judge': [f'j{j}' for j in judge],
        'court': [f'c{j % 3}' for j in judge],
        'target': 2.0 * x + judge + rng.normal(0, 1, rows)})
    dataset = amos.Dataset(
        data, label = 'target', seed = SEED, groups = ['judge', 'court'])
    return amos.splitters.TrainTest().apply(dataset)


def test_fixest_with_fixed_effects_and_clusters() -> None:
    requires('pyfixest')
    dataset = _courts()
    models.Fixest().apply(dataset, fixed_effects = 'judge', cluster = 'court')
    table = dataset.tables['fixest_coefficients']
    # The judge effects are absorbed, so only "x" is a coefficient.
    assert list(table.index) == ['x']
    assert table.loc['x', 'ci_lower'] < 2.0 < table.loc['x', 'ci_upper']
    assert dataset.model.fixed_effects == 'judge'
    assert dataset.predictions.notna().all()


def test_fixest_matches_pyfixest() -> None:
    requires('pyfixest')
    pyfixest = importlib.import_module('pyfixest')
    dataset = _courts()
    models.Fixest().apply(dataset, fixed_effects = 'judge', cluster = 'court')
    data = dataset.data.loc[dataset.train]
    expected = pyfixest.feols(
        'target ~ x | judge', data = data, vcov = {'CRV1': 'court'}).tidy()
    table = dataset.tables['fixest_coefficients']
    assert table.loc['x', 'coefficient'] == pytest.approx(
        expected.loc['x', 'Estimate'])
    assert table.loc['x', 'standard_error'] == pytest.approx(
        expected.loc['x', 'Std. Error'])


def test_fixest_logit_for_classification(classified: amos.Dataset) -> None:
    requires('pyfixest')
    models.Fixest().apply(classified)
    assert classified.model.family == 'logit'
    assert list(classified.probabilities.columns) == [0, 1]


def test_column_parameters_must_name_columns(regressed: amos.Dataset) -> None:
    requires('pyfixest')
    with pytest.raises(KeyError, match = 'fixed_effects column'):
        models.Fixest().apply(regressed, fixed_effects = 'missing')


def test_catboost_uses_categories_directly() -> None:
    requires('catboost')
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined'])
    amos.cleaners.AutoCategorize().apply(dataset, columns = ['region'])
    amos.splitters.Stratified().apply(dataset)
    models.Catboost().apply(dataset, iterations = 20)
    assert dataset.model.get_params()['cat_features'] == ['region']
    assert set(dataset.predictions) <= {'no', 'yes'}


def test_optuna_search(classified: amos.Dataset) -> None:
    requires('optuna_integration')
    models.RandomForest().apply(
        classified,
        search = 'optuna',
        n_estimators = [5, 20],
        max_features = [0.3, 0.9],
        criterion = ['gini', 'entropy'],
        n_iter = 4,
        cv = 3)
    table = classified.tables['random_forest_search']
    assert len(table) == 4
    assert {'param_n_estimators', 'param_criterion'} <= set(table.columns)
    assert table['param_n_estimators'].between(5, 20).all()
    assert list(table['rank_test_score'])[0] == 1
    assert classified.history[-1]['search'] == 'optuna'
