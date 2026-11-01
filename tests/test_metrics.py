"""Tests the metrics module."""

from __future__ import annotations

import pytest
import sklearn.metrics
from conftest import SEED, make_mixed, make_numeric, techniques

import amos
from amos import metrics

# Group metrics need groups, so they are tested in test_fairness.py.
METRICS = [
    (n, k) for n, k in techniques('metric')
    if not issubclass(k, amos.GroupMetric)]


@pytest.mark.parametrize(('name', 'kind'), [
    (n, k) for n, k in METRICS if 'classify' in k.tasks])
def test_classification_metrics(
    name: str,
    kind: type[amos.Metric],
    fitted: amos.Dataset) -> None:
    metric = kind()
    assert metric.name == name
    assert isinstance(metric.measure(fitted), float)


@pytest.mark.parametrize(('name', 'kind'), [
    (n, k) for n, k in METRICS if 'regress' in k.tasks])
def test_regression_metrics(
    name: str,
    kind: type[amos.Metric],
    fitted_regression: amos.Dataset) -> None:
    assert isinstance(kind().measure(fitted_regression), float)


@pytest.mark.parametrize(('name', 'kind'), [
    (n, k) for n, k in METRICS if 'classify' in k.tasks])
def test_classification_metrics_with_three_classes(
    name: str,
    kind: type[amos.Metric],
    multiclass: amos.Dataset) -> None:
    if name == 'average_precision':
        pytest.skip('average precision is for binary labels')
    amos.models.Logit().apply(multiclass)
    assert isinstance(kind().measure(multiclass), float)


def test_metrics_match_scikit_learn(fitted: amos.Dataset) -> None:
    y_true = fitted.y_test
    positive = fitted.probabilities[1]
    expected = {
        'accuracy': sklearn.metrics.accuracy_score(y_true, fitted.predictions),
        'f1': sklearn.metrics.f1_score(y_true, fitted.predictions),
        'roc_auc': sklearn.metrics.roc_auc_score(y_true, positive),
        'log_loss': sklearn.metrics.log_loss(y_true, positive)}
    for name, value in expected.items():
        metric = amos.library.borrow(name)()
        assert metric.measure(fitted) == pytest.approx(value)


def test_text_classes_use_the_last_class_as_positive() -> None:
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined'])
    amos.splitters.Stratified().apply(dataset)
    amos.transformers.MedianImpute().apply(dataset)
    amos.transformers.OneHot().apply(dataset)
    amos.models.Logit().apply(dataset)
    expected = sklearn.metrics.precision_score(
        dataset.y_test, dataset.predictions, pos_label = 'yes')
    assert metrics.Precision().measure(dataset) == pytest.approx(expected)
    assert 0 < metrics.RocAuc().measure(dataset) <= 1


def test_parameters_change_the_measure(multiclass: amos.Dataset) -> None:
    amos.models.Logit().apply(multiclass)
    macro = metrics.F1().measure(multiclass)
    weighted = metrics.F1().measure(multiclass, average = 'weighted')
    expected = sklearn.metrics.f1_score(
        multiclass.y_test, multiclass.predictions, average = 'weighted')
    assert weighted == pytest.approx(expected)
    assert macro != weighted


def test_scores_are_higher_when_better(fitted: amos.Dataset) -> None:
    assert metrics.Accuracy().score(fitted) == metrics.Accuracy().measure(
        fitted)
    assert metrics.LogLoss().score(fitted) == -metrics.LogLoss().measure(
        fitted)
    assert metrics.Accuracy().test(fitted)


def test_metrics_need_predictions(classified: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'needs predictions'):
        metrics.Accuracy().measure(classified)


def test_metrics_need_probabilities(fitted: amos.Dataset) -> None:
    fitted.probabilities = None
    with pytest.raises(ValueError, match = 'probabilities'):
        metrics.RocAuc().measure(fitted)


def test_metrics_are_for_a_task(fitted: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'scores regress tasks'):
        metrics.RMSE().measure(fitted)


def test_a_metric_needs_a_scoring_function(fitted: amos.Dataset) -> None:
    class Empty(amos.Metric):
        pass

    with pytest.raises(NotImplementedError, match = 'no scoring function'):
        Empty().measure(fitted)


def test_metrics_use_their_parameters(multiclass: amos.Dataset) -> None:
    amos.models.Logit().apply(multiclass)
    weighted = metrics.F1(parameters = {'average': 'weighted'})
    assert weighted.measure(multiclass) == metrics.F1().measure(
        multiclass, average = 'weighted')
    # Parameters passed to `measure` take precedence.
    assert weighted.measure(multiclass, average = 'macro') == (
        metrics.F1().measure(multiclass))


def test_metrics_are_not_steps() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'analyst, critic'},
        'analyst': {'techniques': 'stratified, logit'},
        'critic': {'techniques': 'roc_auc'}}
    with pytest.raises(TypeError, match = 'is a metric, not a technique'):
        amos.Project.create(settings, item = make_numeric())
