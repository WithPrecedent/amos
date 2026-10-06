"""Tests the plots module."""

from __future__ import annotations

import pathlib

import pytest
from conftest import requires

import amos
from amos import plots


@pytest.fixture(autouse = True)
def _plotting() -> None:
    """Skips the tests if the plotting packages cannot be imported."""
    requires('matplotlib')
    requires('seaborn')


def _check(item: amos.Dataset, name: str, tmp_path: pathlib.Path) -> None:
    """Checks that the figure `name` was made and can be saved."""
    figure = item.figures[name]
    assert figure.axes
    path = tmp_path / f'{name}.png'
    figure.savefig(path)
    assert path.stat().st_size > 0
    assert item.history[-1] == {'technique': name, 'figure': name}


@pytest.mark.parametrize('name', [
    'confusion_heatmap', 'roc_curve', 'precision_recall_curve',
    'importance_plot'])
def test_classification_plots(
    name: str,
    fitted: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    amos.library.borrow(name)().apply(fitted)
    _check(fitted, name, tmp_path)


@pytest.mark.parametrize('name', ['roc_curve', 'precision_recall_curve'])
def test_curves_for_each_of_three_classes(
    name: str,
    multiclass: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    amos.models.Logit().apply(multiclass)
    amos.library.borrow(name)().apply(multiclass)
    _check(multiclass, name, tmp_path)
    assert len(multiclass.figures[name].axes[0].get_lines()) >= 3


@pytest.mark.parametrize('name', ['actual_vs_predicted', 'residual_plot'])
def test_regression_plots(
    name: str,
    fitted_regression: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    amos.library.borrow(name)().apply(fitted_regression)
    _check(fitted_regression, name, tmp_path)


def test_data_plots(mixed: amos.Dataset, tmp_path: pathlib.Path) -> None:
    plots.Histograms().apply(mixed, bins = 5)
    _check(mixed, 'histograms', tmp_path)
    plots.CorrelationHeatmap().apply(mixed, title = 'Correlations')
    _check(mixed, 'correlation_heatmap', tmp_path)
    figure = mixed.figures['correlation_heatmap']
    assert figure.get_suptitle() == 'Correlations'


def test_histograms_hide_unused_axes(mixed: amos.Dataset) -> None:
    plots.Histograms().apply(mixed, columns = ['age', 'income', 'member'])
    axes = mixed.figures['histograms'].axes
    assert [a.get_visible() for a in axes] == [True, True, True]
    plots.Histograms().apply(mixed, columns = ['age'], limit = 1)
    assert len(mixed.figures['histograms'].axes) == 1


def test_importance_plot_uses_an_earlier_table(fitted: amos.Dataset) -> None:
    amos.evaluators.PermutationImportance().apply(fitted, n_repeats = 2)
    plots.ImportancePlot().apply(fitted, limit = 3)
    axes = fitted.figures['importance_plot'].axes[0]
    assert len(axes.patches) == 3
    assert axes.get_xlabel() == 'drop in score when the feature is shuffled'


def test_size_of_a_figure(fitted: amos.Dataset) -> None:
    plots.RocCurve().apply(fitted, width = 3, height = 2)
    assert list(fitted.figures['roc_curve'].get_size_inches()) == [3, 2]


def test_curves_need_probabilities(fitted: amos.Dataset) -> None:
    fitted.probabilities = None
    with pytest.raises(ValueError, match = 'probabilities'):
        plots.RocCurve().apply(fitted)


def test_regression_plots_need_predictions(regressed: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'apply a model first'):
        plots.ResidualPlot().apply(regressed)
