"""Tests the plots module."""

from __future__ import annotations

import importlib
import pathlib

import numpy as np
import pytest
from conftest import SEED, make_numeric, requires

import amos
from amos import plots


SHAP_PLOTS = [
    'shap_bar', 'shap_beeswarm', 'shap_decision', 'shap_embedding',
    'shap_heatmap', 'shap_partial_dependence', 'shap_scatter', 'shap_violin',
    'shap_waterfall']


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
    'importance_plot', 'calibration_curve', 'det_curve', 'partial_dependence',
    'learning_curve', 'label_plot', 'box_plots', 'pair_plot',
    'prediction_intervals'])
def test_classification_plots(
    name: str,
    fitted: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    amos.library.borrow(name)().apply(fitted)
    _check(fitted, name, tmp_path)


@pytest.mark.parametrize('name', [
    'roc_curve', 'precision_recall_curve', 'calibration_curve', 'det_curve'])
def test_curves_for_each_of_three_classes(
    name: str,
    multiclass: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    amos.models.Logit().apply(multiclass)
    amos.library.borrow(name)().apply(multiclass)
    _check(multiclass, name, tmp_path)
    assert len(multiclass.figures[name].axes[0].get_lines()) >= 3


@pytest.mark.parametrize('name', [
    'actual_vs_predicted', 'residual_plot', 'qq_plot', 'prediction_intervals',
    'label_plot', 'learning_curve', 'partial_dependence'])
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


@pytest.mark.parametrize('name', SHAP_PLOTS)
def test_shap_plots(
    name: str,
    fitted: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    requires('shap')
    matplotlib = importlib.import_module('matplotlib')
    pyplot = importlib.import_module('matplotlib.pyplot')
    before = pyplot.get_fignums()
    amos.library.borrow(name)().apply(fitted, rows = 30, background = 30)
    _check(fitted, name, tmp_path)
    figure = fitted.figures[name]
    assert list(figure.get_size_inches()) == [6.4, 4.8]
    # pyplot forgets the figure (and any others that shap made).
    assert pyplot.get_fignums() == before
    assert figure.canvas.manager is None
    assert not matplotlib.is_interactive()


@pytest.mark.parametrize('name', SHAP_PLOTS)
def test_shap_plots_for_regression(
    name: str,
    fitted_regression: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    requires('shap')
    amos.library.borrow(name)().apply(fitted_regression, rows = 30)
    _check(fitted_regression, name, tmp_path)


def test_shap_bar_draws_the_shap_importance(fitted: amos.Dataset) -> None:
    requires('shap')
    amos.evaluators.ShapImportance().apply(fitted, rows = 30)
    plots.ShapBar().apply(fitted, limit = 4)
    widths = [p.get_width() for p in fitted.figures['shap_bar'].axes[0].patches]
    # The last of the four bars adds up the other features.
    for importance in fitted.tables['shap_importance']['importance'].head(3):
        assert np.isclose(widths, importance).any()


def test_shap_plots_explain_a_chosen_class(multiclass: amos.Dataset) -> None:
    requires('shap')
    amos.models.Logit().apply(multiclass)
    amos.evaluators.ShapImportance().apply(multiclass, rows = 30)
    values = np.abs(multiclass.fitted['shap_importance'].values)
    for position, category in enumerate(multiclass.classes):
        plots.ShapBar().apply(multiclass, category = category, limit = 6)
        axes = multiclass.figures['shap_bar'].axes[0]
        widths = [p.get_width() for p in axes.patches]
        largest = values[:, :, position].mean(axis = 0).max()
        assert np.isclose(widths, largest).any()
    with pytest.raises(ValueError, match = 'not one of the classes'):
        plots.ShapBar().apply(multiclass, category = 'none')


def test_shap_plots_use_the_rows_that_were_explained(
    fitted: amos.Dataset) -> None:
    requires('shap')
    amos.evaluators.ShapImportance().apply(fitted, rows = 10)
    plots.ShapHeatmap().apply(fitted)
    assert fitted.fitted['shap_importance'].shape[0] == 10
    plots.ShapWaterfall().apply(fitted, row = fitted.x_test.index[9])
    with pytest.raises(KeyError, match = 'first 10 rows'):
        plots.ShapWaterfall().apply(fitted, row = fitted.x_test.index[10])


def test_shap_waterfall_keeps_its_size_and_interactive_mode(
    fitted: amos.Dataset) -> None:
    requires('shap')
    matplotlib = importlib.import_module('matplotlib')
    # shap's waterfall turns off interactive mode and resizes the figure.
    matplotlib.interactive(True)
    try:
        plots.ShapWaterfall().apply(fitted, rows = 20, width = 5, height = 4)
        assert matplotlib.is_interactive()
    finally:
        matplotlib.interactive(False)
    assert list(fitted.figures['shap_waterfall'].get_size_inches()) == [5, 4]


def test_shap_scatter_chooses_features(fitted: amos.Dataset) -> None:
    requires('shap')
    amos.evaluators.ShapImportance().apply(fitted, rows = 30)
    plots.ShapScatter().apply(fitted)
    most = fitted.tables['shap_importance'].index[0]
    assert fitted.figures['shap_scatter'].axes[0].get_xlabel() == most
    plots.ShapScatter().apply(fitted, feature = 'x1', interaction = 'x2')
    assert fitted.figures['shap_scatter'].axes[0].get_xlabel() == 'x1'
    with pytest.raises(KeyError, match = 'not one of the explained features'):
        plots.ShapScatter().apply(fitted, feature = 'height')


def test_shap_force_draws_on_its_own_figure(fitted: amos.Dataset) -> None:
    requires('shap')
    pyplot = importlib.import_module('matplotlib.pyplot')
    row = fitted.x_test.index[2]
    plots.ShapForce().apply(fitted, row = row, rows = 20)
    figure = fitted.figures['shap_force']
    assert list(figure.get_size_inches()) == [14, 4]
    assert figure.axes
    assert pyplot.get_fignums() == []
    assert figure.canvas.manager is None


def test_shap_group_difference(fitted: amos.Dataset) -> None:
    requires('shap')
    with pytest.raises(ValueError, match = 'no groups'):
        plots.ShapGroupDifference().apply(fitted, rows = 20)
    fitted.data['court'] = ['high' if i % 3 else 'low' for i in range(200)]
    fitted.groups = ['court']
    plots.ShapGroupDifference().apply(fitted, limit = 4)
    assert len(fitted.figures['shap_group_difference'].axes[0].patches) == 4
    fitted.data['court'] = 'high'
    with pytest.raises(ValueError, match = 'both in and out'):
        plots.ShapGroupDifference().apply(fitted)


def test_shap_plots_are_reproducible(fitted: amos.Dataset) -> None:
    requires('shap')
    state = np.random.get_state()[1].copy()
    offsets = []
    for _ in range(2):
        plots.ShapBeeswarm().apply(fitted, rows = 30)
        axes = fitted.figures['shap_beeswarm'].axes[0]
        offsets.append(np.vstack([c.get_offsets() for c in axes.collections]))
    np.testing.assert_array_equal(offsets[0], offsets[1])
    # The global random numbers are put back as they were.
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_statsmodels_plots(
    regressed: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    requires('statsmodels')
    amos.models.OLS().apply(regressed)
    for name in ('influence_plot', 'coefficient_plot', 'qq_plot'):
        amos.library.borrow(name)().apply(regressed)
        _check(regressed, name, tmp_path)
    axes = regressed.figures['coefficient_plot'].axes[0]
    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert labels == ['x4', 'x3', 'x2', 'x1', 'x0']
    plots.CoefficientPlot().apply(regressed, intercept = True)
    axes = regressed.figures['coefficient_plot'].axes[0]
    assert 'const' in [t.get_text() for t in axes.get_yticklabels()]
    with pytest.raises(TypeError, match = 'needs a scikit-learn model'):
        plots.LearningCurve().apply(regressed)


def test_plots_that_need_a_kind_of_model(fitted: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'no table of coefficients'):
        plots.CoefficientPlot().apply(fitted)
    with pytest.raises(ValueError, match = '"ols" or "glm"'):
        plots.InfluencePlot().apply(fitted)
    with pytest.raises(ValueError, match = 'regression model'):
        plots.QqPlot().apply(fitted)
    with pytest.raises(ValueError, match = 'decision tree'):
        plots.TreePlot().apply(fitted)
    with pytest.raises(ValueError, match = 'explainable_boosting'):
        plots.ShapeFunctions().apply(fitted)
    with pytest.raises(ValueError, match = 'no search table'):
        plots.SearchPlot().apply(fitted)
    with pytest.raises(ValueError, match = '"parameter" and its "values"'):
        plots.ValidationCurve().apply(fitted)


def test_tree_plot(classified: amos.Dataset, tmp_path: pathlib.Path) -> None:
    amos.models.DecisionTree().apply(classified, max_depth = 3)
    plots.TreePlot().apply(classified, depth = 2)
    _check(classified, 'tree_plot', tmp_path)
    assert classified.figures['tree_plot'].canvas.manager is None
    amos.models.RandomForest().apply(classified, n_estimators = 5)
    plots.TreePlot().apply(classified, tree = 4)
    _check(classified, 'tree_plot', tmp_path)


def test_validation_curve(fitted: amos.Dataset, tmp_path: pathlib.Path) -> None:
    plots.ValidationCurve().apply(
        fitted, parameter = 'C', values = [0.01, 1.0], cv = 3)
    _check(fitted, 'validation_curve', tmp_path)


def test_curves_refit_models_with_coded_classes(
    multiclass: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    requires('xgboost')
    amos.models.Xgboost().apply(multiclass, n_estimators = 10)
    plots.LearningCurve().apply(multiclass, cv = 3)
    _check(multiclass, 'learning_curve', tmp_path)
    plots.ValidationCurve().apply(
        multiclass, parameter = 'max_depth', values = [1, 3], cv = 3)
    _check(multiclass, 'validation_curve', tmp_path)
    plots.PartialDependence().apply(multiclass, category = 2)
    _check(multiclass, 'partial_dependence', tmp_path)


def test_data_plots_split_by_class(
    mixed: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    for name in ('count_plots', 'missing_heatmap', 'box_plots', 'pair_plot'):
        amos.library.borrow(name)().apply(mixed)
        _check(mixed, name, tmp_path)
    for name in ('count_plots', 'pair_plot'):
        figure = mixed.figures[name]
        # The legend is beside the plots, with the classes in order.
        assert [t.get_text() for t in figure.legends[0].get_texts()] == [
            'no', 'yes']
        assert all(a.get_legend() is None for a in figure.axes)
    axes = mixed.figures['missing_heatmap'].axes[0]
    assert [t.get_text() for t in axes.get_xticklabels()] == ['age']
    assert len(mixed.figures['pair_plot'].axes) == 4


def test_missing_heatmap_without_missing_values(
    classified: amos.Dataset) -> None:
    plots.MissingHeatmap().apply(classified)
    axes = classified.figures['missing_heatmap'].axes[0]
    assert axes.texts[0].get_text() == 'no missing values'


def test_fairness_plot(fitted: amos.Dataset, tmp_path: pathlib.Path) -> None:
    requires('fairlearn')
    fitted.data['court'] = ['high' if i % 3 else 'low' for i in range(200)]
    fitted.groups = ['court']
    plots.FairnessPlot().apply(fitted)
    _check(fitted, 'fairness_plot', tmp_path)
    axes = fitted.figures['fairness_plot'].axes
    assert [a.get_title() for a in axes] == [
        'selection rate', 'accuracy', 'true positive rate',
        'false positive rate']
    assert [t.get_text() for t in axes[0].get_xticklabels()] == [
        'high', 'low']


def test_prediction_intervals_draw_the_conformal_table(
    fitted_regression: amos.Dataset) -> None:
    requires('mapie')
    amos.evaluators.Conformal().apply(fitted_regression, cv = 3)
    table = fitted_regression.tables['conformal']
    for rows, most in ((20, 20), (1000, len(table))):
        plots.PredictionIntervals().apply(fitted_regression, rows = rows)
        axes = fitted_regression.figures['prediction_intervals'].axes[0]
        # The first collection is the intervals, and the others are dots.
        points = sum(len(c.get_offsets()) for c in axes.collections[1:])
        assert most / 2 < points <= most


def test_search_plot(classified: amos.Dataset, tmp_path: pathlib.Path) -> None:
    amos.models.RandomForest().apply(
        classified, search = 'random', n_estimators = [5, 15],
        max_features = ['sqrt', None], n_iter = 4, cv = 3)
    plots.SearchPlot().apply(classified)
    _check(classified, 'search_plot', tmp_path)
    axes = classified.figures['search_plot'].axes
    assert [a.get_xlabel() for a in axes] == ['max_features', 'n_estimators']


def test_shape_functions(
    classified: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    requires('interpret')
    amos.models.ExplainableBoosting().apply(classified, interactions = 0)
    plots.ShapeFunctions().apply(classified, limit = 4)
    _check(classified, 'shape_functions', tmp_path)
    titles = [a.get_title() for a in classified.figures['shape_functions'].axes]
    model = classified.model
    order = np.argsort(model.term_importances())[::-1][:4]
    assert titles == [model.term_names_[i] for i in order]


def test_plots_have_their_own_sizes(fitted: amos.Dataset) -> None:
    plots.PairPlot().apply(fitted, limit = 2)
    assert list(fitted.figures['pair_plot'].get_size_inches()) == [8, 8]
    plots.PairPlot().apply(fitted, limit = 2, width = 4)
    assert list(fitted.figures['pair_plot'].get_size_inches()) == [4, 8]


def test_label_plot_counts_each_class() -> None:
    dataset = amos.Dataset(
        make_numeric(weights = [0.8, 0.2]), label = 'target', seed = SEED)
    plots.LabelPlot().apply(dataset)
    axes = dataset.figures['label_plot'].axes[0]
    heights = [p.get_height() for p in axes.patches]
    assert heights == list(dataset.y.value_counts().sort_index())
