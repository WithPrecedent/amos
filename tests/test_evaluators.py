"""Tests the evaluators module."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires

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


@pytest.mark.parametrize('model', ['sk_logit', 'random_forest', 'knn'])
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
