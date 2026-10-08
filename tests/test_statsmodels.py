"""Tests the models that wrap statsmodels, beyond what every model does."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_regression, requires

import amos
from amos import models

# The kinds of `Statsmodel`, by the models that wrap them.
STATSMODELS = sorted(
    (name, kind) for name, kind in amos.library.get_genre('model').items()
    if models.Statsmodel in kind.tools.values())
COLUMNS = [
    'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
    'ci_upper']


@pytest.fixture(autouse = True)
def _statsmodels() -> None:
    """Skips the tests if statsmodels is not installed."""
    requires('statsmodels')


def _data(kind: type[amos.Model]) -> tuple[amos.Dataset, dict[str, str]]:
    """Returns a split dataset that suits a model, and its parameters."""
    rng = np.random.default_rng(SEED)
    name = getattr(kind, 'kind', None)
    if 'regress' not in kind.tools:
        # Unlike `make_numeric`, no feature is a combination of the others,
        # so every coefficient has a standard error.
        data = pd.DataFrame(
            rng.normal(size = (300, 4)), columns = ['x0', 'x1', 'x2', 'x3'])
        score = data['x0'] - data['x1'] + rng.logistic(size = 300)
        cuts = [-1, 1] if name in {'mnlogit', 'ordinal_regression'} else [0]
        data['target'] = np.digitize(score, cuts)
    else:
        data = make_regression(rows = 300)
        if kind.counts:
            data['target'] = rng.poisson(np.exp(0.3 * data['x0'].clip(-2, 2)))
    data['court'] = np.resize(list('abcdef'), len(data))
    data['weight'] = rng.uniform(0.5, 2, len(data))
    task = 'regress' if 'regress' in kind.tools else 'classify'
    dataset = amos.Dataset(
        data,
        label = 'target',
        task = task,
        seed = SEED,
        groups = ['court', 'weight'])
    amos.splitters.TrainTest().apply(dataset)
    parameters = {'weights': 'weight'} if name == 'wls' else {}
    return dataset, parameters


@pytest.mark.parametrize(('name', 'kind'), STATSMODELS)
def test_every_statsmodel_reports_its_coefficients(
    name: str,
    kind: type[amos.Model]) -> None:
    dataset, parameters = _data(kind)
    kind().apply(dataset, **parameters)
    table = dataset.tables[f'{name}_coefficients']
    assert list(table.columns) == COLUMNS
    assert np.isfinite(table['coefficient']).all()
    features = list(dataset.model.feature_names_in_)
    assert features == dataset.features
    coefficients = np.atleast_2d(dataset.model.coef_)
    assert coefficients.shape[1] == len(features)
    # Every model reports the importance of the features from its
    # coefficients.
    amos.evaluators.FeatureImportance().apply(dataset)
    assert len(dataset.tables['feature_importance']) == len(features)


def test_classifiers_of_several_classes(multiclass: amos.Dataset) -> None:
    models.Mnlogit().apply(multiclass)
    probabilities = multiclass.probabilities
    assert list(probabilities.columns) == [0, 1, 2]
    assert np.allclose(probabilities.sum(axis = 1), 1)
    table = multiclass.tables['mnlogit_coefficients']
    assert {'const (1)', 'x0 (1)', 'x0 (2)'} <= set(table.index)
    assert multiclass.model.coef_.shape == (2, len(multiclass.features))
    amos.plots.CoefficientPlot().apply(multiclass)
    models.OrdinalRegression().apply(multiclass, distribution = 'probit')
    assert multiclass.model.distribution == 'probit'
    assert 'const' not in multiclass.tables['ordinal_regression_coefficients']


def test_ordinal_regression_keeps_the_order_of_its_classes() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'x0': rng.normal(size = 400), 'x1': rng.normal(size = 400)})
    # The classes are ranges of a hidden score, as ordered classes are.
    score = 1.5 * data['x0'] - data['x1'] + rng.logistic(size = 400)
    order = ['lenient', 'typical', 'harsh']
    data['target'] = pd.Categorical(
        np.asarray(order, dtype = object)[np.digitize(score, [-1, 1])],
        categories = order,
        ordered = True)
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    models.OrdinalRegression().apply(dataset)
    assert list(dataset.model.levels_) == order
    assert list(dataset.model.classes_) == sorted(order)
    # The probabilities are in the order of the sorted classes, as in
    # scikit-learn.
    assert list(dataset.probabilities.columns) == sorted(order)
    assert set(dataset.predictions) <= set(order)
    amos.metrics.RocAuc().apply(dataset)
    assert dataset.metrics['roc_auc'] > 0.7


def test_models_of_counts_need_counts(regressed: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'needs counts'):
        models.Poisson().apply(regressed)
    data = make_regression()
    data['target'] = np.random.default_rng(SEED).poisson(1, len(data))
    # Whole numbers with few values are classified unless "task" is set.
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    assert dataset.task == 'classify'
    with pytest.raises(ValueError, match = 'set "task" to "regress"'):
        models.Poisson().apply(dataset)


def test_a_glm_family_must_suit_the_task(classified: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'binomial'):
        models.GLM().apply(classified, family = 'poisson')
    with pytest.raises(ValueError, match = 'binomial'):
        models.GEE().apply(classified, groups = 'x0', family = 'gaussian')


def test_models_of_groups_and_weights_need_their_columns(
    regressed: amos.Dataset) -> None:
    for model in (models.Mixedlm(), models.GEE()):
        with pytest.raises(ValueError, match = '"groups" column'):
            model.apply(regressed)
    with pytest.raises(ValueError, match = '"weights" column'):
        models.WLS().apply(regressed)
    regressed.data['court'] = np.resize(list('abcd'), len(regressed.data))
    models.GEE().apply(regressed, groups = 'court', covariance = 'exchangeable')
    assert regressed.model.groups == 'court'
    # The column of groups is not a feature.
    assert 'court' not in regressed.tables['gee_coefficients'].index


def test_the_options_of_statsmodels(regressed: amos.Dataset) -> None:
    models.QuantileRegression().apply(regressed, quantile = 0.9)
    high = regressed.predictions.mean()
    models.QuantileRegression().apply(regressed)
    assert high > regressed.predictions.mean()
    models.RobustRegression().apply(regressed, norm = 'tukey_biweight')
    assert regressed.model.norm == 'tukey_biweight'
    with pytest.raises(ValueError, match = 'not in'):
        models.RobustRegression().apply(regressed, norm = 'missing')
    with pytest.raises(ValueError, match = 'not a kind'):
        models.Statsmodel(kind = 'missing').fit(
            regressed.x_train, regressed.y_train)


def test_influence_plot_draws_a_glm(regressed: amos.Dataset) -> None:
    requires('matplotlib')
    models.GLM().apply(regressed, family = 'gamma')
    amos.plots.InfluencePlot().apply(regressed)
    assert 'influence_plot' in regressed.figures
    models.QuantileRegression().apply(regressed)
    with pytest.raises(ValueError, match = '"ols" or "glm"'):
        amos.plots.InfluencePlot().apply(regressed)


def test_validators_refit_models_with_their_columns() -> None:
    dataset, _ = _data(models.Mixedlm)
    models.Mixedlm().apply(dataset)
    amos.validators.KFold().apply(dataset, n_splits = 3)
    assert 'cv_r2' in dataset.metrics
    models.WLS().apply(dataset, weights = 'weight')
    amos.validators.KFold().apply(dataset, n_splits = 3)
    assert 'cv_r2' in dataset.metrics


def test_the_same_results_as_statsmodels() -> None:
    api = importlib.import_module('statsmodels.api')
    dataset, _ = _data(models.Probit)
    models.Probit().apply(dataset)
    x_train = api.add_constant(dataset.x_train.astype(float))
    results = api.Probit(dataset.y_train.astype(float), x_train).fit(disp = 0)
    np.testing.assert_allclose(
        dataset.tables['probit_coefficients']['coefficient'], results.params)
    np.testing.assert_allclose(
        dataset.probabilities[1],
        results.predict(api.add_constant(dataset.x_test.astype(float))))
    dataset, _ = _data(models.Poisson)
    models.Poisson().apply(dataset)
    x_train = api.add_constant(dataset.x_train.astype(float))
    results = api.Poisson(dataset.y_train, x_train).fit(disp = 0)
    np.testing.assert_allclose(
        dataset.tables['poisson_coefficients']['standard_error'], results.bse)
    dataset, _ = _data(models.Mixedlm)
    models.Mixedlm().apply(dataset)
    x_train = api.add_constant(dataset.x_train.astype(float))
    results = api.MixedLM(
        dataset.y_train,
        x_train,
        groups = dataset.data.loc[dataset.train, 'court']).fit()
    np.testing.assert_allclose(
        dataset.tables['mixedlm_coefficients']['coefficient'],
        results.params)


def test_logit_is_scikit_learn_and_logit_sm_is_statsmodels() -> None:
    api = importlib.import_module('statsmodels.api')
    dataset, _ = _data(models.LogitSM)
    models.LogitSM().apply(dataset)
    results = api.Logit(
        dataset.y_train.astype(float),
        api.add_constant(dataset.x_train.astype(float))).fit(disp = 0)
    np.testing.assert_allclose(
        dataset.tables['logit_sm_coefficients']['p_value'], results.pvalues)
    models.Logit().apply(dataset)
    assert type(dataset.model).__name__ == 'LogisticRegression'
    # Effects find the same models by name, as the right kind.
    learner = amos.effects._learner('logit_sm', 'classify', SEED)
    assert (learner.kind, learner.output) == ('logit', 'binary')


def test_binomial_bayes_mixedglm_reports_its_posteriors() -> None:
    dataset, _ = _data(models.BinomialBayesMixedglm)
    models.BinomialBayesMixedglm().apply(dataset)
    assert dataset.model.groups == 'court'
    table = dataset.tables['binomial_bayes_mixedglm_coefficients']
    assert list(table.index) == ['const', *dataset.features]
    assert table['p_value'].isna().all()
    np.testing.assert_allclose(
        table['ci_upper'] - table['coefficient'],
        1.96 * table['standard_error'])
    assert set(dataset.predictions) <= {0, 1}
    assert np.allclose(dataset.probabilities.sum(axis = 1), 1)


def test_collinear_features_are_left_out(
    classified: amos.Dataset,
    multiclass: amos.Dataset) -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame(rng.normal(size = (300, 2)), columns = ['x0', 'x1'])
    data['copy'] = data['x0']
    data['target'] = (data['x0'] - data['x1'] + rng.logistic(size = 300) > 0)
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    models.LogitSM().apply(dataset)
    assert dataset.model.dropped_ == ['copy']
    table = dataset.tables['logit_sm_coefficients']
    assert table.loc['copy'].isna().all()
    estimated = table.drop(index = 'copy')
    assert np.isfinite(estimated[['coefficient', 'standard_error']]).all().all()
    assert np.isnan(dataset.model.coef_[-1])
    # A generalized linear model finds coefficients for collinear features
    # itself.
    models.GLM().apply(classified)
    assert classified.model.dropped_ == []
    # Two features of `classified` and `multiclass` are combinations of the
    # others, which these models leave out the same way on every computer.
    classified.data['court'] = np.resize(list('abcdef'), len(classified.data))
    models.GEE().apply(classified, groups = 'court')
    assert len(classified.model.dropped_) == 2
    assert np.allclose(classified.probabilities.sum(axis = 1), 1)
    models.Mnlogit().apply(multiclass)
    dropped = multiclass.model.dropped_
    assert len(dropped) == 2
    table = multiclass.tables['mnlogit_coefficients']
    assert table.loc[f'{dropped[0]} (2)'].isna().all()
