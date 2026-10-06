"""Tests the evaluators module."""

from __future__ import annotations

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
