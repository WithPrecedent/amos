"""Plots of a model: how well it predicts, what it learned, and whether its assumptions hold.

Contents:
    ActualVsPredicted: the label against the predictions (regression).
    BiasVariancePlot: the model's expected loss, bias, and variance.
    CalibrationCurve: the share in each class against its probability.
    CausalGraph: the causal links that tigramite found, as a graph.
    ClassificationReportHeatmap: the precision, recall, and f1 of each class.
    CoefficientPlot: coefficients with their confidence intervals.
    ConfusionHeatmap: the confusion matrix as a heatmap (classification).
    DecisionRegions: the class predicted across two features (classification).
    DetCurve: the detection error tradeoff curve (classification).
    EstimatesPlot: every model's estimates in a regression table.
    FairnessPlot: the model's fairness metrics for each group.
    ImportancePlot: the most important features, as bars.
    InfluencePlot: the influence of each row on a statsmodels regression.
    LearningCurve: the model's score as it learns from more rows.
    LoadingsHeatmap: how strongly each feature loads on each factor.
    PartialDependence: the average prediction as features change.
    PrecisionRecallCurve: precision against recall (classification).
    PredictionIntervals: conformal prediction intervals (or set sizes).
    QqPlot: the quantiles of the residuals against a normal distribution's.
    ResidualPlot: the errors against the predictions (regression).
    RocCurve: the ROC curve (classification).
    ScorecardPlot: each branch's score on each metric of a scorecard.
    ScreePlot: the share of the variance of each principal component.
    SearchPlot: the score of each try in a hyperparameter search.
    ShapeFunctions: an explainable boosting model's shape functions.
    ValidationCurve: the model's score as one of its parameters changes.
    WeightsPlot: the weights of the features that eli5 explains.

"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Hashable, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, evaluators, models, utilities
from ._base import (
    _IMPORTANCE_LABELS,
    Plot,
    _classes,
    _figure_legend,
    _grid,
    _importances,
    _lent,
    _model,
    _table,
    _top_features,
    _unwrapped,
)

# Names that statsmodels, pyfixest, and others give the intercept.
_INTERCEPTS: frozenset[str] = frozenset({
    'const', 'Intercept', '(Intercept)', 'intercept'})


# The kinds of `models.Statsmodel` that an `influence_plot` draws.
_INFLUENCE: frozenset[str] = frozenset({'glm', 'ols'})


# Columns of a table of coefficients that a `coefficient_plot` draws.
_INTERVAL_COLUMNS: tuple[str, ...] = ('coefficient', 'ci_lower', 'ci_upper')


# Columns of a table of causal links that a `causal_graph` draws.
_LINK_COLUMNS: tuple[str, ...] = ('cause', 'effect', 'lag', 'link', 'strength')


# Most cells of a heatmap that are labeled with their numbers.
_ANNOTATED_CELLS: int = 100


# Swaps the ends of a link of tigramite's (such as "o->"), which is read
# backward from its other end (as "<-o").
_REVERSED_ENDS: dict[int, int] = str.maketrans('<>', '><')


@dataclasses.dataclass
class ActualVsPredicted(Plot):
    """The label against the model's predictions (regression).

    Wraps:
        [`sklearn.metrics.PredictionErrorDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html#sklearn.metrics.PredictionErrorDisplay.from_predictions)
        from scikit-learn.

    """

    label_axes: ClassVar[str] = 'xy'

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the actual labels against the predictions.

        Args:
            item: the dataset with a fitted regression model.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        _prediction_error(item, figure, 'actual_vs_predicted')


@dataclasses.dataclass
class BiasVariancePlot(Plot):
    """The model's expected loss, beside its bias and variance, as bars.

    Draws the table made by `bias_variance` (which is applied if it has not
    been, or the table named by "source"). For the mean squared error of a
    regression, the expected loss is the bias (squared) plus the variance;
    for the share of a classifier's errors, it is not their sum, but the bars
    still show which of the two is larger.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the expected loss, the bias, and the variance.

        Args:
            item: the dataset with a fitted scikit-learn model and test rows.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "bias_variance", or makes it.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'bias_variance', evaluators.BiasVariance().evaluate)
        row = table.iloc[0]
        axes = figure.subplots()
        axes.bar(
            ['expected loss', 'bias', 'variance'],
            [row['expected_loss'], row['bias'], row['variance']],
            color = ['C0', 'C1', 'C2'])
        axes.set_ylabel(f'loss ({row["loss"]})')


@dataclasses.dataclass
class CalibrationCurve(Plot):
    """The share of rows in each class against its predicted probability.

    A well calibrated classifier's probabilities are as often right as they
    say: of the rows given a probability of 0.8, 80% are in the class. The
    test rows are put in `bins` by their probability, and each bin's share in
    the class is drawn against its mean probability (with scikit-learn's
    calibration display). For more than two classes, each class is compared
    with the rest.

    Wraps:
        [`sklearn.calibration.CalibrationDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html#sklearn.calibration.CalibrationDisplay.from_predictions)
        from scikit-learn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        bins: int = 10,
        strategy: str = 'uniform',
        **kwargs: Any) -> None:
        """Draws the calibration curve (for each class if more than two).

        Args:
            item: the dataset with a fitted classifier that predicts
                probabilities.
            figure: the figure to draw on.
            bins: number of bins of probabilities. Defaults to 10.
            strategy: "uniform" (bins of the same width) or "quantile" (bins
                with the same number of rows). Defaults to "uniform".
            **kwargs: not used.

        """
        calibration = importlib.import_module('sklearn.calibration')
        axes = figure.subplots()
        for number, (name, y_true, y_score, positive) in enumerate(
            _curves(item)):
            calibration.CalibrationDisplay.from_predictions(
                y_true,
                y_score,
                n_bins = bins,
                strategy = strategy,
                pos_label = positive,
                name = name,
                ref_line = number == 0,
                ax = axes)


@dataclasses.dataclass
class CausalGraph(Plot):
    """The causal links that tigramite found, as a graph of the variables.

    Draws the table of links made by `pcmci`, `pcmci_plus`, or `lpcmci` (the
    table named by "source", or the last table of links made) with
    tigramite's `plot_graph`: a node for each variable (the features and the
    label), colored by how strongly it depends on its own past, and an arrow
    for each link between two variables, colored by its strength and
    labeled with its lags. A link whose direction is not known is drawn
    without an arrowhead.

    Wraps:
        [`tigramite.plotting.plot_graph`](https://jakobrunge.github.io/tigramite/#tigramite.plotting.plot_graph)
        from tigramite.

    """

    starts_at_zero: ClassVar[str] = ''

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the graph of the causal links.

        Args:
            item: the dataset with a table of causal links.
            figure: the figure to draw on.
            source: name of the table of links. Defaults to `None`, which
                uses the last table of links made.
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if no table of links was made and
                `source` does not name one.

        """
        plotting = utilities.import_tool('tigramite.plotting')
        matplotlib = importlib.import_module('matplotlib')
        table = _links(item, source)
        names = [
            str(c) for c in item.data.columns
            if c in item.features or c == item.label]
        found = pd.unique(table[['cause', 'effect']].to_numpy().ravel())
        names.extend(str(n) for n in found if str(n) not in names)
        positions = {n: i for i, n in enumerate(names)}
        lags = int(table['lag'].max()) if len(table) else 0
        graph = np.full((len(names), len(names), lags + 1), '', dtype = '<U3')
        strengths = np.zeros(graph.shape)
        for cause, effect, lag, link, strength in table[
            ['cause', 'effect', 'lag', 'link', 'strength']].itertuples(
                index = False):
            i, j = positions[str(cause)], positions[str(effect)]
            graph[i, j, lag], strengths[i, j, lag] = link, strength
            if lag == 0:
                # tigramite lists a link at the same time from both ends.
                graph[j, i, 0] = link[::-1].translate(_REVERSED_ENDS)
                strengths[j, i, 0] = strength
        limit = max(1.0, float(np.abs(strengths).max(initial = 0)))
        # tigramite sizes its text in points, so it is given the style's.
        text = matplotlib.rcParams['font.size']
        ticks = matplotlib.rcParams['xtick.labelsize']
        axes = figure.subplots()
        # tigramite draws its colorbars with `pyplot.colorbar`, which would
        # open a figure of pyplot's own if `figure` were not lent to it.
        with _lent(figure):
            plotting.plot_graph(
                graph,
                val_matrix = strengths,
                var_names = names,
                fig_ax = (figure, axes),
                vmin_edges = -limit,
                vmax_edges = limit,
                vmin_nodes = -limit,
                vmax_nodes = limit,
                label_fontsize = text,
                node_label_size = text,
                link_label_fontsize = text,
                tick_label_size = ticks,
                link_colorbar_label = 'strength of the link',
                node_colorbar_label = 'strength of its own past')


@dataclasses.dataclass
class ClassificationReportHeatmap(Plot):
    """The precision, recall, and f1 of each class, as a heatmap.

    Draws the table made by `classification_report` (which is applied if it
    has not been, or the table named by "source"): a row for each class and
    for the macro and weighted averages, and a column for each score, from
    0 (pale) to 1 (dark).

    Wraps:
        [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        from seaborn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the scores of each class.

        Args:
            item: the dataset with a fitted classifier.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "classification_report", or makes it.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        table = _table(
            item, source, 'classification_report',
            evaluators.ClassificationReport().evaluate)
        scores = table.drop(index = 'accuracy', errors = 'ignore')[
            ['precision', 'recall', 'f1-score']].astype(float)
        axes = figure.subplots()
        seaborn.heatmap(
            scores, annot = True, fmt = '.2f', cmap = 'Blues', vmin = 0,
            vmax = 1, ax = axes)
        # Minor ticks (which some styles show) mean nothing between classes.
        axes.minorticks_off()


@dataclasses.dataclass
class CoefficientPlot(Plot):
    """Coefficients with their confidence intervals (a forest plot).

    Draws a table of coefficients made by `ols`, `glm`, `fixest`, or `cox`
    ("{model}_coefficients"), or of the effects that an inferer (such as
    `partially_linear`) estimated: a dot for each coefficient and a line for
    its confidence interval, with a dashed line at zero. The table is
    `source`, or the last such table made. The intercept is left out unless
    `intercept` is `True`.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        intercept: bool = False,
        limit: int = 30,
        **kwargs: Any) -> None:
        """Draws the coefficients and their confidence intervals.

        Args:
            item: the dataset with a table of coefficients.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the last table of coefficients made.
            intercept: whether to draw the intercept. Defaults to `False`.
            limit: most coefficients to draw. Defaults to 30.
            **kwargs: not used.

        """
        table = _coefficients(item, source)
        if not intercept:
            # A multinomial logit names its intercepts "const ({class})".
            table = table.loc[[
                str(i).split(' (')[0] not in _INTERCEPTS for i in table.index]]
        table = table.head(limit).iloc[::-1]
        coefficients = table['coefficient'].to_numpy(dtype = float)
        errors = [
            coefficients - table['ci_lower'].to_numpy(dtype = float),
            table['ci_upper'].to_numpy(dtype = float) - coefficients]
        axes = figure.subplots()
        axes.errorbar(
            coefficients,
            [str(i) for i in table.index],
            xerr = errors,
            fmt = 'o',
            capsize = 3)
        axes.axvline(0, color = 'gray', linestyle = '--', linewidth = 1)
        axes.set_xlabel('coefficient, with its confidence interval')


@dataclasses.dataclass
class ConfusionHeatmap(Plot):
    """The confusion matrix as a heatmap (classification).

    Wraps:
        [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        from seaborn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the confusion matrix.

        Args:
            item: the dataset with a fitted classifier.
            figure: the figure to draw on.
            source: name of a table made by `confusion` to draw. Defaults to
                `None`, which finds the confusion matrix again.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        matrix = (
            evaluators.Confusion().evaluate(item) if source is None
            else _table(item, source))
        axes = figure.subplots()
        seaborn.heatmap(
            matrix, annot = True, fmt = 'd', cmap = 'Blues', ax = axes)
        # Minor ticks (which some styles show) mean nothing between classes.
        axes.minorticks_off()


@dataclasses.dataclass
class DecisionRegions(Plot):
    """The class that the model predicts across two features (classification).

    mlxtend's plot of the decision regions of a classifier: the plane of two
    `features` (by default, the two most important) is shaded by the class
    that the model predicts at each point, with the test rows (or every row,
    if the data is not split) drawn over it as points of their classes. The
    other features are held at their medians in the training rows, so the
    regions are a slice of what the model does. Columns that the model needs
    besides its features (such as the fixed effects and cluster of a
    `fixest` model) are held at their most common values in the real
    training rows.

    Wraps:
        [`mlxtend.plotting.plot_decision_regions`](https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/)
        from mlxtend.

    """

    starts_at_zero: ClassVar[str] = ''

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        features: Sequence[str] | None = None,
        **kwargs: Any) -> None:
        """Draws the decision regions of the model across two features.

        Args:
            item: the dataset with a fitted classifier.
            figure: the figure to draw on.
            features: the two features to draw. Defaults to `None`, which
                uses the two most important (see `importance_plot`).
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if the task is not classification.
            ValueError: if there are not two features, or a feature is not a
                number.

        """
        if item.task != 'classify':
            message = 'decision_regions needs a classifier'
            raise base.IncompatibleTechniqueError(message)
        plotting = utilities.import_tool('mlxtend.plotting')
        matplotlib = importlib.import_module('matplotlib')
        model = _model(item)
        names = list(
            utilities.feature_names(model, item.data.columns) or item.features)
        chosen = _top_features(item, names, features, self.name)
        data = item.data[names]
        if not all(pd.api.types.is_numeric_dtype(data[n]) or
                   pd.api.types.is_bool_dtype(data[n]) for n in names):
            message = 'decision_regions needs features that are all numbers'
            raise ValueError(message)
        x = (item.x_test if item.is_split else item.x)[names]
        classes = _classes(item)
        positions = {c: i for i, c in enumerate(classes)}
        y = (item.y_test if item.is_split else item.y).map(positions)
        held = {
            names.index(n): float(m)
            for n, m in item.x_train[names].astype(float).median().items()
            if n not in chosen}
        others = evaluators._other_columns(model, item)
        real = item.real_train_rows
        given = {
            c: values.iloc[0]
            for c, values in item.data.loc[real, others].mode().items()}
        cycle = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        axes = figure.subplots()
        # mlxtend sizes its grid of points by pyplot's current figure.
        with _lent(figure):
            plotting.plot_decision_regions(
                x.to_numpy(dtype = float),
                y.to_numpy(dtype = int),
                clf = _ClassPositions(model, names, classes, given),
                feature_index = [names.index(n) for n in chosen],
                filler_feature_values = held or None,
                filler_feature_ranges = dict.fromkeys(held, np.inf) or None,
                colors = ','.join(
                    matplotlib.colors.to_hex(c) for c in cycle[:len(classes)]),
                legend = 0,
                ax = axes)
        handles = [
            h for h in axes.collections if h.get_label().lstrip('-').isdigit()]
        axes.legend(
            handles, [str(classes[int(h.get_label())]) for h in handles],
            title = str(item.label))
        axes.set_xlabel(str(chosen[0]))
        axes.set_ylabel(str(chosen[1]))


@dataclasses.dataclass
class DetCurve(Plot):
    """The detection error tradeoff (DET) curve (classification).

    scikit-learn's DET curve: the false negative rate against the false
    positive rate at every threshold, on the scales of a normal
    distribution, which makes the curves of good classifiers nearly straight
    and easy to compare. For more than two classes, each class is compared
    with the rest.

    Wraps:
        [`sklearn.metrics.DetCurveDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DetCurveDisplay.html#sklearn.metrics.DetCurveDisplay.from_predictions)
        from scikit-learn.

    """

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the DET curve (one for each class if there are more than two).

        Args:
            item: the dataset with a fitted classifier that predicts
                probabilities.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        metrics = importlib.import_module('sklearn.metrics')
        axes = figure.subplots()
        for name, y_true, y_score, positive in _curves(item):
            metrics.DetCurveDisplay.from_predictions(
                y_true, y_score, pos_label = positive, name = name, ax = axes)


@dataclasses.dataclass
class EstimatesPlot(Plot):
    """Every model's estimates with their confidence intervals.

    Draws the table made by `regression_table` (which is applied if it has
    not been, or the table named by "source") as a forest plot: a row for
    each term, with a dot for each model's estimate and a line for its 95%
    confidence interval, in a color for each model, and a dashed line at
    zero. The intercept is left out unless `intercept` is `True`.

    """

    starts_at_zero: ClassVar[str] = 'x'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        intercept: bool = False,
        limit: int = 30,
        **kwargs: Any) -> None:
        """Draws the estimates of each model.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "regression_table", or makes it.
            intercept: whether to draw the intercept. Defaults to `False`.
            limit: most terms to draw. Defaults to 30.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'regression_table',
            evaluators.RegressionTable().evaluate)
        models = list(dict.fromkeys(table.index.get_level_values('model')))
        terms = [
            t for t in dict.fromkeys(table.index.get_level_values('term'))
            if intercept or str(t).split(' (')[0] not in _INTERCEPTS][:limit]
        rows = np.arange(len(terms))[::-1]
        # The models' dots for a term are spread across a band around it.
        offsets = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else (
            np.zeros(1))
        axes = figure.subplots()
        for model, offset in zip(models, offsets, strict = True):
            estimates = table.xs(model, level = 'model').reindex(terms)
            shown = estimates['estimate'].notna().to_numpy()
            values = estimates['estimate'].to_numpy(dtype = float)[shown]
            axes.errorbar(
                values,
                (rows + offset)[shown],
                xerr = [
                    values - estimates['ci_lower'].to_numpy(
                        dtype = float)[shown],
                    estimates['ci_upper'].to_numpy(dtype = float)[shown]
                    - values],
                fmt = 'o',
                markersize = 3,
                capsize = 2,
                label = str(model))
        axes.set_yticks(rows, [str(t) for t in terms])
        axes.axvline(0, color = 'gray', linestyle = '--', linewidth = 1)
        axes.set_xlabel('estimate, with its 95% confidence interval')
        if len(models) > 1:
            axes.legend()
            _figure_legend(figure, axes, 'model')


@dataclasses.dataclass
class FairnessPlot(Plot):
    """The model's fairness metrics for each group, as bars.

    Draws the table made with fairlearn by the `fairness` evaluator (which is
    applied for `group` if it has not been): a panel for each metric (the
    selection rate, the accuracy, and the true and false positive rates),
    with a bar for each group. A model treats groups alike when their bars
    are about the same.

    """

    size: ClassVar[tuple[float, float]] = (9.6, 3.6)
    shared_axes: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        group: str | None = None,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws each fairness metric for each group.

        Args:
            item: the dataset with a fitted classifier and groups.
            figure: the figure to draw on.
            group: column of groups. Defaults to `None`, which uses the table
                that `fairness` made or the dataset's first group.
            source: name of a table made by `fairness` to draw. Defaults to
                `None`, which uses the table named "fairness" (if it is for
                `group`), or makes it.
            **kwargs: not used.

        """
        found = item.tables.get('fairness')
        if source is not None:
            table = _table(item, source)
        elif found is None or group not in {None, found.index.name}:
            table = evaluators.Fairness().evaluate(item, group = group)
        else:
            table = found
        groups = table.drop(index = ['difference', 'ratio'], errors = 'ignore')
        columns = [c for c in groups.columns if c != 'count']
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            axes.bar([str(g) for g in groups.index], groups[column])
            axes.set_ylim(0, 1)
            axes.set_title(column.replace('_', ' '), fontsize = 'small')


@dataclasses.dataclass
class ImportancePlot(Plot):
    """The most important features, as horizontal bars.

    The importances come from a table made earlier by `shap_importance`,
    `permutation_importance`, or `feature_importance` (in that order), or are
    found with `feature_importance` if there is no such table.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        source: str | None = None,
        limit: int = 15,
        **kwargs: Any) -> None:
        """Draws the importance of the most important features.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            source: name of the table of importances to draw. Defaults to
                `None`, which uses the first one found.
            limit: most features to draw. Defaults to 15.
            **kwargs: not used.

        """
        found, table = _importances(item, source)
        top = table['importance'].head(limit).iloc[::-1]
        axes = figure.subplots()
        axes.barh([str(i) for i in top.index], top.to_numpy())
        axes.set_xlabel(_IMPORTANCE_LABELS.get(found, 'importance'))


@dataclasses.dataclass
class InfluencePlot(Plot):
    """The influence of each training row on a statsmodels regression.

    statsmodels' influence plot for an `ols` or `glm` model: each training
    row's studentized residual against its leverage, with the size of its
    dot showing its influence on the coefficients (by Cook's distance, or
    "dffits"), so that rows that sway the results stand out. The most
    influential rows are labeled.

    Wraps:
        [`statsmodels.graphics.regressionplots.influence_plot`](https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.influence_plot.html)
        from statsmodels.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        criterion: str = 'cooks',
        **kwargs: Any) -> None:
        """Draws the influence of each row.

        Args:
            item: the dataset with a fitted `ols` or `glm` model.
            figure: the figure to draw on.
            criterion: "cooks" or "dffits". Defaults to "cooks".
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if the model is not one of those.

        """
        model = _unwrapped(item)
        kind = getattr(model, 'kind', None)
        if not isinstance(model, models.Statsmodel) or kind not in _INFLUENCE:
            message = 'influence_plot needs an "ols" or "glm" model'
            raise base.IncompatibleTechniqueError(message)
        regression = utilities.import_tool(
            'statsmodels.graphics.regressionplots')
        # statsmodels finds the externally studentized residuals of an ols
        # model, but only the internally studentized residuals of a glm.
        regression.influence_plot(
            model.results,
            external = kind != 'glm',
            criterion = criterion,
            ax = figure.subplots())


@dataclasses.dataclass
class LearningCurve(Plot):
    """The model's score as it learns from more of the training rows.

    scikit-learn's learning curve: a copy of the model is fitted to growing
    shares (`sizes`) of the training rows in `cv`-fold cross-validation, and
    its scores on the rows it learned from and on the rows held out are
    drawn, which shows whether more data would help. The score is `scoring`,
    or accuracy for classification and R² for regression.

    Wraps:
        [`sklearn.model_selection.LearningCurveDisplay.from_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LearningCurveDisplay.html#sklearn.model_selection.LearningCurveDisplay.from_estimator)
        from scikit-learn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        sizes: Sequence[float] = (0.1, 0.325, 0.55, 0.775, 1.0),
        cv: int = 5,
        scoring: str | None = None,
        **kwargs: Any) -> None:
        """Draws the learning curve.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            sizes: shares (or numbers) of the training rows to learn from.
                Defaults to five shares from 0.1 to 1.
            cv: number of folds of cross-validation. Defaults to 5.
            scoring: name of a scikit-learn scorer. Defaults to `None`, which
                uses accuracy or R².
            **kwargs: not used.

        """
        model_selection = importlib.import_module('sklearn.model_selection')
        estimator, x, y = _refittable(item, self.name)
        model_selection.LearningCurveDisplay.from_estimator(
            estimator,
            x,
            y,
            train_sizes = list(sizes),
            cv = cv,
            scoring = scoring or _scoring(item),
            score_name = scoring or _scoring(item),
            shuffle = True,
            random_state = item.seed,
            ax = figure.subplots())


@dataclasses.dataclass
class LoadingsHeatmap(Plot):
    """How strongly each feature loads on each factor, as a heatmap.

    Draws the loadings in the table made by `factor_analysis` (which is
    applied if it has not been, or the table named by "source"): a row for
    each feature and a column for each factor, from -1 (blue) to 1 (red).
    Features that load strongly on the same factor move together.

    Wraps:
        [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        from seaborn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the loadings of each feature.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "factor_analysis", or makes it.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        table = _table(
            item, source, 'factor_analysis',
            evaluators.FactorAnalysis().evaluate)
        loadings = table[
            [c for c in table.columns if str(c).startswith('factor_')]]
        axes = figure.subplots()
        seaborn.heatmap(
            loadings.astype(float),
            cmap = 'vlag',
            center = 0,
            vmin = -1,
            vmax = 1,
            # Numbers in more cells than this would not fit.
            annot = loadings.size <= _ANNOTATED_CELLS,
            fmt = '.2f',
            ax = axes)
        # Minor ticks (which some styles show) mean nothing between features.
        axes.minorticks_off()


@dataclasses.dataclass
class PartialDependence(Plot):
    """The model's average prediction as features change (partial dependence).

    scikit-learn's partial dependence plot of `features` (by default, the
    three most important): for each value of a feature, the model's average
    prediction on the training rows when every row is given that value. Set
    `kind` to "individual" to draw a line for each row instead (ICE lines)
    or to "both". A pair of features, such as ["age", "income"], draws their
    joint dependence as contours (of the average). For a classifier, the
    prediction is the probability of `category` (by default, the last
    class).

    Wraps:
        [`sklearn.inspection.PartialDependenceDisplay.from_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay.from_estimator)
        from scikit-learn.

    """

    size: ClassVar[tuple[float, float]] = (9.6, 3.6)
    starts_at_zero: ClassVar[str] = 'y'
    label_axes: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        features: Sequence[str | Sequence[str]] | None = None,
        kind: str = 'average',
        category: Hashable | None = None,
        limit: int = 3,
        **kwargs: Any) -> None:
        """Draws the partial dependence of the prediction on each feature.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            features: features (or pairs of features) to draw. Defaults to
                `None`, which draws the most important.
            kind: "average", "individual", or "both". Defaults to "average".
            category: class of the label whose probability is drawn.
                Defaults to `None`, which uses the last class.
            limit: most features to draw if `features` is `None`. Defaults to
                3.
            **kwargs: not used.

        """
        inspection = importlib.import_module('sklearn.inspection')
        if features is None:
            _, table = _importances(item, None)
            features = [str(f) for f in table.index[:limit]]
        model, target = _model(item), None
        if item.task == 'classify':
            classes = _classes(item)
            target = classes[-1] if category is None else category
            if isinstance(model, models.LabelCoded):
                # The wrapped model knows the classes by their positions.
                model, target = model.estimator, classes.index(target)
        _require_sklearn(model, self.name)
        x = item.x_train if item.is_split else item.x
        features = list(features)
        # scikit-learn only draws the average for a pair of features.
        kinds = [kind if isinstance(f, str) else 'average' for f in features]
        # With "both", scikit-learn draws the rows and the average in its own
        # blue and orange, rather than in the colors of the style.
        colors = {}
        if 'both' in kinds:
            colors = {
                'ice_lines_kw': {'color': 'C0'},
                'pd_line_kw': {'color': 'C1'}}
        inspection.PartialDependenceDisplay.from_estimator(
            model,
            x,
            features,
            kind = kinds if len(set(kinds)) > 1 else kind,
            target = target,
            random_state = item.seed,
            ax = figure.subplots(),
            **colors)


@dataclasses.dataclass
class PrecisionRecallCurve(Plot):
    """Precision against recall at every threshold (classification).

    Wraps:
        [`sklearn.metrics.PrecisionRecallDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay.from_predictions)
        from scikit-learn.

    """

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the precision-recall curve (for each class if more than two).

        Args:
            item: the dataset with a fitted classifier that predicts
                probabilities.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        metrics = importlib.import_module('sklearn.metrics')
        axes = figure.subplots()
        for name, y_true, y_score, positive in _curves(item):
            metrics.PrecisionRecallDisplay.from_predictions(
                y_true, y_score, pos_label = positive, name = name, ax = axes)


@dataclasses.dataclass
class PredictionIntervals(Plot):
    """Conformal prediction intervals of the test rows (or the set sizes).

    Draws the table made with MAPIE by the `conformal` evaluator (which is
    applied if it has not been). For regression, each test row (up to
    `rows`, in order of their predictions) is a line from the lower to the
    upper end of its interval, with a dot for its label, which is red when
    it is outside the interval. For classification, the bars are the number
    of rows with prediction sets of each size, split by whether the set
    holds the true class.

    """

    label_axes: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        rows: int = 100,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the prediction intervals or the sizes of the sets.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            rows: most rows to draw (evenly spaced) for regression. Defaults
                to 100.
            source: name of the table made by `conformal` to draw. Defaults
                to `None`, which uses the table named "conformal", or makes
                it.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'conformal', evaluators.Conformal().evaluate)
        axes = figure.subplots()
        if 'lower' not in table:
            counts = pd.crosstab(table['size'], table['covered'])
            sizes = [str(s) for s in counts.index]
            held = counts.get(True, pd.Series(0, index = counts.index))
            missed = counts.get(False, pd.Series(0, index = counts.index))
            axes.bar(sizes, held, label = 'holds the true class')
            axes.bar(
                sizes, missed, bottom = held, color = 'C3',
                label = 'misses it')
            axes.set_xlabel('size of the prediction set')
            axes.set_ylabel('test rows')
            axes.legend()
            return
        shown = table.sort_values('prediction')
        shown = shown.iloc[::max(1, math.ceil(len(shown) / rows))]
        positions = np.arange(len(shown))
        covered = shown['covered'].to_numpy(dtype = bool)
        axes.vlines(
            positions, shown['lower'], shown['upper'], color = 'C0',
            alpha = 0.4, label = 'interval')
        axes.plot(
            positions, shown['prediction'], color = 'C0', linewidth = 1,
            label = 'prediction')
        actual = shown['actual'].to_numpy(dtype = float)
        axes.scatter(
            positions[covered], actual[covered], s = 10, color = 'black',
            label = 'label (inside)')
        axes.scatter(
            positions[~covered], actual[~covered], s = 10, color = 'C3',
            label = 'label (outside)')
        axes.set_xlabel('test rows, in order of their predictions')
        axes.set_ylabel(str(item.label))
        axes.legend(loc = 'upper left')


@dataclasses.dataclass
class QqPlot(Plot):
    """The quantiles of the residuals against a normal distribution's.

    statsmodels' Q-Q plot of the model's errors on the test rows (the label
    minus the prediction), standardized. If the errors are normal, as the
    standard errors of a linear regression assume, the dots fall on the
    line.

    Wraps:
        [`statsmodels.graphics.gofplots.qqplot`](https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html)
        from statsmodels.

    """

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the quantiles of the residuals.

        Args:
            item: the dataset with a fitted regression model.
            figure: the figure to draw on.
            **kwargs: not used.

        Raises:
            ValueError: if there are no predictions.
            IncompatibleTechniqueError: if the label is classified.

        """
        if item.predictions is None:
            message = 'there are no predictions: apply a model first'
            raise ValueError(message)
        if item.task == 'classify':
            message = 'qq_plot needs a regression model'
            raise base.IncompatibleTechniqueError(message)
        gofplots = utilities.import_tool('statsmodels.graphics.gofplots')
        y = item.y.loc[item.predictions.index]
        residuals = (y - item.predictions).astype(float)
        axes = figure.subplots()
        # statsmodels draws the points in blue and the line in red, rather
        # than in the colors of the style.
        gofplots.qqplot(
            residuals,
            line = '45',
            fit = True,
            ax = axes,
            markerfacecolor = 'C0',
            markeredgecolor = 'C0')
        axes.get_lines()[-1].set_color('C1')


@dataclasses.dataclass
class ResidualPlot(Plot):
    """The model's errors against its predictions (regression).

    Wraps:
        [`sklearn.metrics.PredictionErrorDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html#sklearn.metrics.PredictionErrorDisplay.from_predictions)
        from scikit-learn.

    """

    label_axes: ClassVar[str] = 'x'

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the residuals against the predictions.

        Args:
            item: the dataset with a fitted regression model.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        _prediction_error(item, figure, 'residual_vs_predicted')


@dataclasses.dataclass
class RocCurve(Plot):
    """The receiver operating characteristic (ROC) curve (classification).

    Wraps:
        [`sklearn.metrics.RocCurveDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay.from_predictions)
        from scikit-learn.

    """

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the ROC curve (one for each class if there are more than two).

        Args:
            item: the dataset with a fitted classifier that predicts
                probabilities.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        metrics = importlib.import_module('sklearn.metrics')
        axes = figure.subplots()
        for name, y_true, y_score, positive in _curves(item):
            metrics.RocCurveDisplay.from_predictions(
                y_true, y_score, pos_label = positive, name = name, ax = axes)
        axes.plot([0, 1], [0, 1], linestyle = '--', color = 'gray')


@dataclasses.dataclass
class ScorecardPlot(Plot):
    """Each branch's score on each metric of a scorecard, as dots.

    Draws the table made by `scorecard` (which is applied if it has not
    been, or the table named by "source"): a panel for each metric, with a
    dot for each branch (named by its techniques), best first. Branches
    whose dots are close did about as well.

    """

    size: ClassVar[tuple[float, float]] = (9.6, 3.6)

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the score of each branch on each metric.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "scorecard", or makes it.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'scorecard', evaluators.Scorecard().evaluate)
        metrics = [
            c for c in table.columns
            if c != 'rank' and pd.api.types.is_numeric_dtype(table[c])]
        steps = [c for c in table.columns if c != 'rank' and c not in metrics]
        if steps:
            names = [
                ' / '.join(str(v) for v in row)
                for row in table[steps].itertuples(index = False)]
        else:
            names = [f'rank {r}' for r in table['rank']]
        # The best branch is at the top.
        rows = np.arange(len(table))[::-1]
        for axes, metric in zip(
            _grid(figure, len(metrics)), metrics, strict = True):
            axes.scatter(table[metric].to_numpy(dtype = float), rows, s = 12)
            # Only the panels at the left name the branches.
            first = axes.get_subplotspec().is_first_col()
            axes.set_yticks(rows, names if first else [])
            axes.set_title(str(metric).replace('_', ' '), fontsize = 'small')


@dataclasses.dataclass
class ScreePlot(Plot):
    """The share of the variance that each principal component explains.

    Draws the table made by `pca` (which is applied if it has not been, or
    the table named by "source"): a bar for each component's share of the
    variance of the features, and a line for the share of the components up
    to it. Where the bars level off (the "elbow") suggests how many
    components to keep.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the share of the variance of each component.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "pca", or makes it.
            **kwargs: not used.

        """
        ticker = importlib.import_module('matplotlib.ticker')
        table = _table(item, source, 'pca', evaluators.PCA().evaluate)
        components = np.arange(1, len(table) + 1)
        axes = figure.subplots()
        axes.bar(components, table['share'], label = 'component')
        axes.plot(
            components, table['cumulative'], marker = 'o', markersize = 3,
            color = 'C1', label = 'cumulative')
        axes.xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
        axes.set_xlabel('principal component')
        axes.set_ylabel('share of the variance')
        axes.legend()


@dataclasses.dataclass
class SearchPlot(Plot):
    """The cross-validated score of each try in a hyperparameter search.

    Draws the table that a model's search (grid, random, or Optuna) made,
    "{model}_search": a panel for each searched parameter, with the mean
    score of each try against the value of the parameter, and the best try
    marked with a star. The table is `source`, or the last search made.

    """

    shared_axes: ClassVar[str] = 'y'
    # The parameters of the model have no natural zero.
    starts_at_zero: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the score of each try against each parameter.

        Args:
            item: the dataset with a search table.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the last search table made.
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if no search table was made and
                `source` does not name one.

        """
        names = [
            n for n, t in item.tables.items()
            if n.endswith('_search') and 'mean_test_score' in t]
        if source is None and not names:
            message = (
                'there is no search table: apply a model with "search" first')
            raise base.IncompatibleTechniqueError(message)
        table = item.tables[source or names[-1]]
        parameters = [c for c in table.columns if c.startswith('param_')]
        best = table['rank_test_score'].to_numpy() == 1
        scores = table['mean_test_score'].to_numpy(dtype = float)
        for number, (axes, column) in enumerate(zip(
            _grid(figure, len(parameters)), parameters, strict = True)):
            values = pd.to_numeric(table[column], errors = 'coerce')
            if values.isna().any():
                # Parameters that are not all numbers are drawn as text (and
                # a missing value is a parameter of `None`).
                values = pd.Series([
                    'None' if pd.isna(v) else str(v) for v in table[column]])
            points = values.to_numpy()
            axes.scatter(points, scores, s = 20)
            axes.scatter(
                points[best], scores[best], marker = '*', s = 150,
                color = 'C3', label = 'best')
            axes.set_xlabel(column.removeprefix('param_'))
            axes.set_ylabel('mean score' if number == 0 else '')


@dataclasses.dataclass
class ShapeFunctions(Plot):
    """An explainable boosting model's contribution from each feature.

    InterpretML's explanation of an `explainable_boosting` model: for each of
    its most important features (up to `limit`), the feature's contribution
    to the model's output (on the scale of the log odds for a classifier)
    across the feature's values, with its confidence band. These shape
    functions are the whole model: its output for a row is the sum of the
    contributions, plus the intercept (and any pairs of features, which are
    not drawn).

    Wraps:
        - [`interpret.glassbox.ExplainableBoostingClassifier.explain_global`](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
          from InterpretML, for classes.
        - [`interpret.glassbox.ExplainableBoostingRegressor.explain_global`](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)
          from InterpretML, for values.

    """

    size: ClassVar[tuple[float, float]] = (9.6, 6.4)
    starts_at_zero: ClassVar[str] = 'y'
    shared_axes: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 6,
        category: Hashable | None = None,
        **kwargs: Any) -> None:
        """Draws the shape function of each of the most important features.

        Args:
            item: the dataset with a fitted `explainable_boosting` model.
            figure: the figure to draw on.
            limit: most features to draw. Defaults to 6.
            category: class of the label to draw, if there are more than two.
                Defaults to `None`, which uses the last class.
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if the model is not an explainable
                boosting model.

        """
        model = _unwrapped(item)
        if not hasattr(model, 'term_importances'):
            message = 'shape_functions needs an "explainable_boosting" model'
            raise base.IncompatibleTechniqueError(message)
        explanation = model.explain_global()
        order = np.argsort(model.term_importances())[::-1]
        terms = [
            int(t) for t in order
            if explanation.data(int(t))['type'] == 'univariate'][:limit]
        for axes, term in zip(_grid(figure, len(terms)), terms, strict = True):
            data = explanation.data(term)
            scores, lower, upper = (
                np.asarray(data[k], dtype = float)
                for k in ('scores', 'lower_bounds', 'upper_bounds'))
            if scores.ndim == 2:  # noqa: PLR2004
                classes = _classes(item)
                column = -1 if category is None else classes.index(category)
                scores, lower, upper = (
                    a[:, column] for a in (scores, lower, upper))
            names = list(data['names'])
            if len(names) == len(scores) + 1:
                # The names of a numeric feature are the edges of its bins.
                edges = np.asarray(names, dtype = float)
                axes.stairs(scores, edges, baseline = None)
                axes.fill_between(
                    edges, np.append(lower, lower[-1]),
                    np.append(upper, upper[-1]), step = 'post', alpha = 0.3)
            else:
                axes.bar(
                    [str(n) for n in names], scores,
                    yerr = [scores - lower, upper - scores], capsize = 2)
            axes.axhline(0, color = 'gray', linewidth = 0.5)
            axes.set_title(str(model.term_names_[term]), fontsize = 'small')


@dataclasses.dataclass
class ValidationCurve(Plot):
    """The model's score as one of its parameters changes.

    scikit-learn's validation curve: a copy of the model is fitted with each
    of `values` for `parameter` (in `cv`-fold cross-validation on the
    training rows), and its scores on the rows it learned from and on the
    rows held out are drawn, which shows where it underfits or overfits. The
    score is `scoring`, or accuracy for classification and R² for
    regression.

    Wraps:
        [`sklearn.model_selection.ValidationCurveDisplay.from_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ValidationCurveDisplay.html#sklearn.model_selection.ValidationCurveDisplay.from_estimator)
        from scikit-learn.

    """

    # The parameters of the model have no natural zero.
    starts_at_zero: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        parameter: str | None = None,
        values: Sequence[Any] | None = None,
        cv: int = 5,
        scoring: str | None = None,
        **kwargs: Any) -> None:
        """Draws the validation curve.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            parameter: name of the model's parameter to change, such as
                "max_depth". Required.
            values: values of the parameter to try. Required.
            cv: number of folds of cross-validation. Defaults to 5.
            scoring: name of a scikit-learn scorer. Defaults to `None`, which
                uses accuracy or R².
            **kwargs: not used.

        Raises:
            ValueError: if `parameter` or `values` is missing.

        """
        if parameter is None or values is None:
            message = 'validation_curve needs a "parameter" and its "values"'
            raise ValueError(message)
        model_selection = importlib.import_module('sklearn.model_selection')
        estimator, x, y = _refittable(item, self.name)
        model_selection.ValidationCurveDisplay.from_estimator(
            estimator,
            x,
            y,
            param_name = parameter,
            param_range = list(values),
            cv = cv,
            scoring = scoring or _scoring(item),
            score_name = scoring or _scoring(item),
            ax = figure.subplots())


@dataclasses.dataclass
class WeightsPlot(Plot):
    """The weights of the features that eli5 explains, as bars.

    Draws the table made by `explain_weights` (which is applied if it has
    not been, or the table named by "source"): a bar for each of the `limit`
    features with the largest weights, with its standard deviation if eli5
    found one (for the trees of a forest), and a panel for each class if the
    weights differ by class.

    """

    shared_axes: ClassVar[str] = 'x'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        limit: int = 15,
        **kwargs: Any) -> None:
        """Draws the weight of each feature.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "explain_weights", or makes it.
            limit: most features to draw in each panel. Defaults to 15.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'explain_weights',
            evaluators.ExplainWeights().evaluate)
        targets = (
            list(dict.fromkeys(table['target'])) if 'target' in table
            else [None])
        for axes, target in zip(
            _grid(figure, len(targets)), targets, strict = True):
            weights = table if target is None else table.loc[
                table['target'] == target]
            weights = weights.head(limit).iloc[::-1]
            axes.barh(
                weights['feature'].astype(str),
                weights['weight'],
                xerr = weights.get('std'))
            if len(targets) > 1:
                axes.set_title(str(target), fontsize = 'small')
        figure.supxlabel('weight', fontsize = 'small')


""" Private Classes """


@dataclasses.dataclass
class _ClassPositions:
    """Predicts the position of each class from an array of features.

    mlxtend's `plot_decision_regions` passes the model an array (without the
    names of the features) and needs classes numbered 0 to n - 1. This gives
    the model its features by name, and any other columns it needs, and
    numbers its predictions.

    Args:
        model: the fitted classifier.
        columns: the names of the model's features, in its order.
        classes: the classes, in the order of their numbers.
        given: the value of each other column that the model needs (such as
            the fixed effects of a `fixest` model), given to every row.
            Defaults to an empty `dict`.

    """

    model: Any
    columns: list[str]
    classes: list[Any]
    given: dict[Any, Any] = dataclasses.field(default_factory = dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns the number of the class predicted for each row of `x`.

        Args:
            x: the features, in the order of `columns`.

        Returns:
            The position in `classes` of each prediction.

        """
        frame = pd.DataFrame(x, columns = self.columns)
        for column, value in self.given.items():
            frame[column] = value
        predictions = pd.Series(np.asarray(self.model.predict(frame)))
        positions = {c: i for i, c in enumerate(self.classes)}
        return predictions.map(positions).to_numpy(dtype = int)


""" Private Functions """


def _coefficients(item: base.Dataset, source: str | None) -> pd.DataFrame:
    """Returns a table of coefficients with confidence intervals.

    Args:
        item: the dataset with a table of coefficients.
        source: name of the table, or `None` for the last one made.

    Raises:
        IncompatibleTechniqueError: if no such table was made and `source`
            does not name one.

    Returns:
        The table, with "coefficient", "ci_lower", and "ci_upper" columns.

    """
    names = [
        n for n, t in item.tables.items()
        if set(_INTERVAL_COLUMNS) <= set(t.columns)]
    if source is None and not names:
        message = (
            'there is no table of coefficients: apply "ols", "glm", "fixest", '
            '"cox", or an inferer first')
        raise base.IncompatibleTechniqueError(message)
    return item.tables[source or names[-1]]


def _curves(item: base.Dataset) -> list[tuple[str, Any, pd.Series, Any]]:
    """Returns what is needed to draw a curve for each class.

    Args:
        item: the dataset with a fitted classifier.

    Raises:
        IncompatibleTechniqueError: if the model does not predict
            probabilities.

    Returns:
        For a binary label, the positive class only: its name, the true
            labels, its predicted probability, and the class itself (the
            positive label). For more labels, each class against the rest:
            its name, whether each row is in the class, its predicted
            probability, and `True`.

    """
    if item.probabilities is None:
        message = 'the model does not predict probabilities'
        raise base.IncompatibleTechniqueError(message)
    probabilities = item.probabilities
    y_true = item.y.loc[probabilities.index].to_numpy()
    classes = list(probabilities.columns)
    if len(classes) == 2:  # noqa: PLR2004
        positive = classes[-1]
        return [(str(positive), y_true, probabilities[positive], positive)]
    return [
        (str(c), y_true == c, probabilities[c], True) for c in classes]


def _links(item: base.Dataset, source: str | None) -> pd.DataFrame:
    """Returns a table of the causal links that tigramite found.

    Args:
        item: the dataset with a table of links.
        source: name of the table, or `None` for the last one made.

    Raises:
        IncompatibleTechniqueError: if no such table was made and `source`
            does not name one.

    Returns:
        The table, with "cause", "effect", "lag", "link", and "strength"
            columns.

    """
    names = [
        n for n, t in item.tables.items()
        if set(_LINK_COLUMNS) <= set(t.columns)]
    if source is None and not names:
        message = (
            'there is no table of causal links: apply "pcmci", "pcmci_plus", '
            'or "lpcmci" first')
        raise base.IncompatibleTechniqueError(message)
    return _table(item, source or names[-1])


def _prediction_error(item: base.Dataset, figure: Any, kind: str) -> None:
    """Draws scikit-learn's prediction error display.

    Wraps:
        [`sklearn.metrics.PredictionErrorDisplay.from_predictions`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html#sklearn.metrics.PredictionErrorDisplay.from_predictions)
        from scikit-learn.

    Args:
        item: the dataset with a fitted regression model.
        figure: the figure to draw on.
        kind: "actual_vs_predicted" or "residual_vs_predicted".

    Raises:
        ValueError: if there are no predictions.

    """
    if item.predictions is None:
        message = 'there are no predictions: apply a model first'
        raise ValueError(message)
    metrics = importlib.import_module('sklearn.metrics')
    axes = figure.subplots()
    # scikit-learn draws the points in its own blue, rather than in the first
    # color of the style.
    metrics.PredictionErrorDisplay.from_predictions(
        item.y.loc[item.predictions.index],
        item.predictions,
        kind = kind,
        scatter_kwargs = {'color': 'C0'},
        ax = axes)


def _refittable(
    item: base.Dataset,
    use: str) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Returns an unfitted copy of the dataset's model, and rows to fit it to.

    Args:
        item: the dataset with a fitted model.
        use: what fits the model again (for the message of an error).

    Returns:
        An unfitted copy of the model, and the features and labels of the
            training rows (or of every row, if the data is not split). For a
            model whose classes are coded as numbers, the copy is of the
            model that it wraps, and the labels are the classes' positions.

    """
    sklearn = importlib.import_module('sklearn.base')
    model = _model(item)
    x, y = (item.x_train, item.y_train) if item.is_split else (item.x, item.y)
    if isinstance(model, models.LabelCoded):
        model = model.estimator
        y = y.map({c: i for i, c in enumerate(_classes(item))})
    _require_sklearn(model, use)
    return sklearn.clone(model), x, y


def _require_sklearn(model: Any, use: str) -> None:
    """Raises an error if `model` is not a scikit-learn estimator.

    Args:
        model: a fitted model.
        use: what needs a scikit-learn estimator (for the message).

    Raises:
        IncompatibleTechniqueError: if `model` is not a scikit-learn estimator
            (as the models of statsmodels and pyfixest are not).

    """
    sklearn = importlib.import_module('sklearn.base')
    if not isinstance(model, sklearn.BaseEstimator):
        message = (
            f'{use} needs a scikit-learn model, not a {type(model).__name__}')
        raise base.IncompatibleTechniqueError(message)


def _scoring(item: base.Dataset) -> str:
    """Returns the name of the default scikit-learn scorer for `item`.

    Args:
        item: the dataset.

    Returns:
        "accuracy" for classification, and "r2" otherwise.

    """
    return 'accuracy' if item.task == 'classify' else 'r2'
