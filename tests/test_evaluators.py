"""Tests the evaluators module."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    SEED, make_fixest_courts, make_mixed, make_numeric, requires)

import amos
from amos import evaluators


def test_classification_report(fitted: amos.Dataset) -> None:
    evaluators.ClassificationReport().apply(fitted)
    table = fitted.tables['classification_report']
    assert {'0', '1', 'accuracy', 'macro avg'} <= set(table.index)
    assert 'f1-score' in table.columns


def test_confusion(fitted: amos.Dataset) -> None:
    evaluators.Confusion().apply(fitted)
    table = fitted.tables['confusion']
    assert table.index.name == 'actual'
    assert table.columns.name == 'predicted'
    assert table.to_numpy().sum() == len(fitted.test)


def test_feature_importance_from_coefficients(fitted: amos.Dataset) -> None:
    evaluators.FeatureImportance().apply(fitted)
    table = fitted.tables['feature_importance']
    assert set(table.index) == set(fitted.features)
    assert table.index.name == 'feature'
    assert table['importance'].is_monotonic_decreasing


def test_feature_importance_from_trees(classified: amos.Dataset) -> None:
    amos.models.RandomForest().apply(classified, n_estimators = 10)
    evaluators.FeatureImportance().apply(classified)
    table = classified.tables['feature_importance']
    assert table['importance'].sum() == pytest.approx(1.0)


def test_feature_importance_needs_a_model_that_reports_it(
    classified: amos.Dataset) -> None:
    amos.models.KNN().apply(classified)
    with pytest.raises(ValueError, match = 'permutation_importance'):
        evaluators.FeatureImportance().apply(classified)


def test_permutation_importance(fitted: amos.Dataset) -> None:
    evaluators.PermutationImportance().apply(fitted, n_repeats = 3)
    table = fitted.tables['permutation_importance']
    assert list(table.columns) == ['importance', 'std']
    assert len(table) == len(fitted.features)


def test_permutation_importance_for_regression(
    fitted_regression: amos.Dataset) -> None:
    evaluators.PermutationImportance().apply(fitted_regression, n_repeats = 2)
    assert 'permutation_importance' in fitted_regression.tables


@pytest.mark.parametrize('model', ['logit', 'random_forest', 'knn'])
def test_shap_importance(model: str, classified: amos.Dataset) -> None:
    requires('shap')
    amos.library.borrow(model)().apply(classified)
    evaluators.ShapImportance().apply(classified, rows = 20, background = 20)
    table = classified.tables['shap_importance']
    assert len(table) == len(classified.features)
    assert (table['importance'] >= 0).all()
    assert 'shap_importance' in classified.fitted


def test_evaluators_need_a_model(classified: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'apply a model first'):
        evaluators.Confusion().apply(classified)
    with pytest.raises(ValueError, match = 'apply a model first'):
        evaluators.FeatureImportance().apply(classified)


def test_conformal_sets_for_classification(fitted: amos.Dataset) -> None:
    requires('mapie')
    evaluators.Conformal().apply(fitted, confidence = 0.9, cv = 3)
    table = fitted.tables['conformal']
    assert list(table.columns) == [
        'actual', 'prediction', 'set', 'size', 'covered']
    assert len(table) == len(fitted.test)
    assert fitted.metrics['coverage'] == pytest.approx(table['covered'].mean())
    assert fitted.metrics['coverage'] >= 0.8
    assert fitted.metrics['set_size'] >= 1


def test_conformal_intervals_for_regression(
    fitted_regression: amos.Dataset) -> None:
    requires('mapie')
    evaluators.Conformal().apply(fitted_regression, confidence = 0.8, cv = 3)
    table = fitted_regression.tables['conformal']
    assert (table['lower'] <= table['upper']).all()
    assert fitted_regression.metrics['interval_width'] > 0
    assert fitted_regression.metrics['coverage'] >= 0.6


def test_bias_variance(fitted: amos.Dataset) -> None:
    requires('mlxtend')
    evaluators.BiasVariance().apply(fitted, rounds = 10)
    table = fitted.tables['bias_variance']
    assert list(table.columns) == ['loss', 'expected_loss', 'bias', 'variance']
    row = table.iloc[0]
    assert row['loss'] == '0-1_loss'
    assert 0 <= row['bias'] <= 1
    assert 0 <= row['variance'] <= 1
    assert fitted.history[-1] == {
        'technique': 'bias_variance', 'table': 'bias_variance'}


def test_bias_variance_of_a_regression(
    fitted_regression: amos.Dataset) -> None:
    requires('mlxtend')
    evaluators.BiasVariance().apply(fitted_regression, rounds = 10)
    row = fitted_regression.tables['bias_variance'].iloc[0]
    assert row['loss'] == 'mse'
    # The mean squared error is the squared bias plus the variance.
    assert row['expected_loss'] == pytest.approx(row['bias'] + row['variance'])


def test_bias_variance_of_coded_text_classes() -> None:
    requires('mlxtend')
    requires('xgboost')
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined', 'region'])
    amos.splitters.Stratified().apply(dataset)
    amos.models.Xgboost().apply(dataset, n_estimators = 5)
    evaluators.BiasVariance().apply(dataset, rounds = 5)
    assert 0 <= dataset.tables['bias_variance'].iloc[0]['bias'] <= 1


def test_bias_variance_needs_test_rows_and_scikit_learn(
    classified: amos.Dataset) -> None:
    requires('mlxtend')
    unsplit = amos.Dataset(classified.data, label = 'target', seed = SEED)
    amos.models.Logit().apply(unsplit)
    with pytest.raises(ValueError, match = 'split the data first'):
        evaluators.BiasVariance().apply(unsplit)
    requires('statsmodels')
    amos.models.LogitSM().apply(classified)
    with pytest.raises(TypeError, match = 'needs a scikit-learn model'):
        evaluators.BiasVariance().apply(classified)


def test_explain_weights(fitted: amos.Dataset) -> None:
    requires('eli5')
    evaluators.ExplainWeights().apply(fitted, top = 4)
    table = fitted.tables['explain_weights']
    assert {'feature', 'weight'} <= set(table.columns)
    assert len(table) == 4


def test_explain_weights_of_trees(classified: amos.Dataset) -> None:
    requires('eli5')
    amos.models.RandomForest().apply(classified, n_estimators = 10)
    evaluators.ExplainWeights().apply(classified)
    assert 'std' in classified.tables['explain_weights'].columns


def test_feature_importance_of_explainable_boosting(
    classified: amos.Dataset) -> None:
    requires('interpret')
    amos.models.ExplainableBoosting().apply(classified, interactions = 0)
    evaluators.FeatureImportance().apply(classified)
    table = classified.tables['feature_importance']
    assert set(table.index) == set(classified.features)


def test_pca_describes_the_features(
    classified: amos.Dataset) -> None:
    requires('statsmodels')
    evaluators.PCA().apply(classified)
    table = classified.tables['pca']
    features = len(classified.features)
    assert list(table.columns) == ['eigenvalue', 'share', 'cumulative']
    assert len(table) == features
    # The eigenvalues of the correlations of n columns add up to n.
    assert table['eigenvalue'].sum() == pytest.approx(features)
    assert table['share'].sum() == pytest.approx(1)
    # Two of the features are combinations of the others, so the last
    # components explain none of the variance (give or take rounding).
    assert (np.diff(table['cumulative']) > -1e-9).all()
    eigenvalues = np.linalg.eigvalsh(classified.x_train.corr().to_numpy())
    np.testing.assert_allclose(
        table['eigenvalue'], sorted(eigenvalues, reverse = True), atol = 1e-8)
    loadings = classified.tables['pca_loadings']
    assert loadings.shape == (features, features)
    evaluators.PCA().apply(
        classified, components = 2, standardize = False)
    assert len(classified.tables['pca']) == 2


def test_factor_analysis_describes_the_features(
    classified: amos.Dataset) -> None:
    requires('statsmodels')
    evaluators.FactorAnalysis().apply(classified)
    table = classified.tables['factor_analysis']
    factors = [c for c in table.columns if c.startswith('factor_')]
    # Kaiser's rule keeps a factor for each eigenvalue above 1.
    eigenvalues = np.linalg.eigvalsh(classified.x_train.corr().to_numpy())
    assert len(factors) == (eigenvalues > 1).sum()
    assert list(table.index) == classified.features
    np.testing.assert_allclose(table['communality'] + table['uniqueness'], 1)
    evaluators.FactorAnalysis().apply(
        classified, factors = 2, method = 'ml', rotation = 'none')
    table = classified.tables['factor_analysis']
    assert list(table.columns) == [
        'factor_1', 'factor_2', 'communality', 'uniqueness']


def test_analyses_of_the_features_use_the_real_training_rows(
    classified: amos.Dataset) -> None:
    requires('statsmodels')
    classified.data['constant'] = 1.0
    classified.synthetic = classified.train[:10]
    classified.data.loc[classified.synthetic, 'x0'] = 1e6
    evaluators.PCA().apply(classified)
    loadings = classified.tables['pca_loadings']
    # A column that does not vary is left out, and so are made-up rows,
    # whose extreme values would otherwise make "x0" a component alone.
    assert 'constant' not in loadings.index
    assert classified.tables['pca']['share'].iloc[0] < 0.9
    with pytest.raises(ValueError, match = 'two numeric columns'):
        evaluators.FactorAnalysis().apply(classified, columns = ['x0'])


def test_evaluators_name_the_columns_that_lightgbm_renamed() -> None:
    requires('lightgbm')
    data = make_numeric(rows = 200)
    data.columns = [
        str(c).replace('x', 'feature ') if c != 'target' else c
        for c in data.columns]
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    amos.models.Lightgbm().apply(dataset, n_estimators = 10)
    evaluators.FeatureImportance().apply(dataset)
    table = dataset.tables['feature_importance']
    assert set(table.index) == set(dataset.features)
    effects = evaluators.RegressionTable.create(dataset, simulations = 3)
    assert list(effects.estimates.loc['lightgbm'].index) == dataset.features


def test_shap_importance_explains_fixest_given_its_fixed_effects() -> None:
    requires('shap')
    dataset = make_fixest_courts()
    table = evaluators.ShapImportance().evaluate(
        dataset, rows = 12, background = 20)
    # Each row is explained with its own court's fixed effect, so every
    # feature (and only the features) has an importance.
    assert set(table.index) == set(dataset.features)
    explanation = dataset.fitted['shap_importance']
    assert np.asarray(explanation.values).shape[0] == 12
    assert np.isfinite(np.asarray(explanation.values)).all()


def test_permutation_importance_of_fixest_keeps_its_fixed_effects() -> None:
    dataset = make_fixest_courts()
    table = evaluators.PermutationImportance().evaluate(
        dataset, n_repeats = 3)
    # Only the features are shuffled: each row keeps its own court.
    assert set(table.index) == set(dataset.features)
    assert np.isfinite(table['importance']).all()
    assert table['importance'].max() > 0
    table = evaluators.PermutationImportance().evaluate(
        dataset, n_repeats = 2, scoring = 'roc_auc')
    assert np.isfinite(table['importance']).all()


def test_feature_importance_of_fixest_leaves_out_dropped_coefficients() -> None:
    dataset = make_fixest_courts()
    kept = list(dataset.model.coefficients().dropna(subset = ['coefficient']).index)
    table = evaluators.FeatureImportance().evaluate(dataset)
    # pyfixest drops collinear features, which then have no importance.
    assert len(kept) < len(dataset.features)
    assert sorted(table.index) == sorted(kept)
