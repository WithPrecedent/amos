"""Techniques that draw figures of the data and the model.

These are the techniques of the "artist" stage. Each one draws a `matplotlib`
figure and adds it to the dataset's `figures` (under the technique's name).
Figures are made without `matplotlib.pyplot`, so they do not open windows or
depend on a display. (shap draws its plots with `matplotlib.pyplot`, so the
SHAP plots lend it their figure while shap draws, and take it back after.)
`Project.export` saves them as image files, and in a notebook, a figure is
shown by making it the last line of a cell.

Every plot accepts "width" and "height" (in inches) and "title" parameters,
and "style", "colors", and "latex" parameters that set how it looks (see
`Plot.implement`). By default, figures have SciencePlots' style for
scientific figures, with the size and fonts of a figure in Nature, and a
cycle of colors that people with color blindness can tell apart.

Contents:
    Plot: base class for techniques that draw a figure.
    ActualVsPredicted: the label against the predictions (regression).
    BoxPlots: each numeric feature as a box, split by class.
    CalibrationCurve: the share in each class against its probability.
    CoefficientPlot: coefficients with their confidence intervals.
    ConfusionHeatmap: the confusion matrix as a heatmap (classification).
    CorrelationHeatmap: the correlations between numeric columns.
    CountPlots: the count of each value of each categorical feature.
    DetCurve: the detection error tradeoff curve (classification).
    FairnessPlot: the model's fairness metrics for each group.
    Histograms: the distribution of each numeric feature.
    ImportancePlot: the most important features, as bars.
    InfluencePlot: the influence of each row on a statsmodels regression.
    LabelPlot: the distribution of the label.
    LearningCurve: the model's score as it learns from more rows.
    MissingHeatmap: where values are missing.
    PairPlot: each pair of numeric features against each other.
    PartialDependence: the average prediction as features change.
    PrecisionRecallCurve: precision against recall (classification).
    PredictionIntervals: conformal prediction intervals (or set sizes).
    QqPlot: the quantiles of the residuals against a normal distribution's.
    ResidualPlot: the errors against the predictions (regression).
    RocCurve: the ROC curve (classification).
    SearchPlot: the score of each try in a hyperparameter search.
    ShapBar: the mean absolute SHAP value of the most important features.
    ShapBeeswarm: the SHAP value of each row for the most important features.
    ShapDecision: how the features move each row's prediction, as lines.
    ShapEmbedding: the rows placed by their SHAP values.
    ShapForce: how each feature pushes one row's prediction up or down.
    ShapGroupDifference: how the SHAP values differ between two groups.
    ShapHeatmap: the SHAP values of every explained row, as a heatmap.
    ShapPartialDependence: the average prediction as a feature changes.
    ShapScatter: a feature's SHAP values against its values.
    ShapViolin: the distribution of the SHAP values of each feature.
    ShapWaterfall: how each feature moves one row's prediction.
    ShapeFunctions: an explainable boosting model's shape functions.
    SurvivalCurves: the share of rows without an event over time.
    TreePlot: the splits of a decision tree.
    ValidationCurve: the model's score as one of its parameters changes.

"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import importlib
import math
import re
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from . import base, describers, evaluators, models, options, utilities

# Labels of the axis of an `importance_plot`, by the table it draws, in the
# order that the tables are looked for.
_IMPORTANCE_LABELS: dict[str, str] = {
    'shap_importance': 'mean absolute SHAP value',
    'permutation_importance': 'drop in score when the feature is shuffled',
    'feature_importance': "the model's importance"}
# Names that statsmodels, pyfixest, and others give the intercept.
_INTERCEPTS: frozenset[str] = frozenset({
    'const', 'Intercept', '(Intercept)', 'intercept'})
# A color of the cycle of the style, such as "C0" (the first).
_CYCLE_COLOR: re.Pattern[str] = re.compile(r'C\d+')
# Settings of `matplotlib` for the sizes of text.
_FONT_SIZES: tuple[str, ...] = (
    'font.size', 'axes.labelsize', 'axes.titlesize', 'figure.labelsize',
    'figure.titlesize', 'legend.fontsize', 'legend.title_fontsize',
    'xtick.labelsize', 'ytick.labelsize')
# The kinds of `models.Statsmodel` that an `influence_plot` draws.
_INFLUENCE: frozenset[str] = frozenset({'glm', 'ols'})
# Columns of a table of coefficients that a `coefficient_plot` draws.
_INTERVAL_COLUMNS: tuple[str, ...] = ('coefficient', 'ci_lower', 'ci_upper')


@dataclasses.dataclass
class Plot(base.Operation, abc.ABC):
    """Base class for techniques that draw a figure.

    A subclass writes a `draw` method, which draws on a `matplotlib` figure.
    The figure is stored in the dataset's `figures` under the technique's
    name.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most plots. Defaults to `None`.
        parameters: keyword arguments for `draw`. Defaults to an empty `dict`.

    """

    # The usual width and height of the figure, in inches, for `matplotlib`'s
    # default style. They are scaled to the width of the style that is used.
    size: ClassVar[tuple[float, float]] = (6.4, 4.8)
    # Whether `size` is scaled to the width of the style. The plots of tools
    # that fix the sizes of their text (such as shap) keep their usual sizes,
    # so that the text fits, and the style's sizes of text are enlarged to
    # match.
    scaled: ClassVar[bool] = True

    """ Required Methods """

    @abc.abstractmethod
    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> Any:
        """Draws on `figure`.

        Args:
            item: the dataset to draw.
            figure: a `matplotlib.figure.Figure` to draw on.
            **kwargs: parameters for the drawing.

        Returns:
            `None`, or a figure to store instead of `figure` (for a tool that
                only draws on a figure of its own).

        """

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        width: float | None = None,
        height: float | None = None,
        title: str | None = None,
        *,
        style: str | Sequence[str] | None = options._PLOT_STYLE,
        colors: str | None = options._PLOT_COLORS,
        latex: bool = False,
        **kwargs: Any) -> base.Dataset:
        """Draws a figure in a style and adds it to the figures of `item`.

        Args:
            item: the dataset to draw.
            width: width of the figure in inches. Defaults to `None`, which
                uses the plot's usual width for the style (3.3 for most plots
                in the default style, the width of a column in Nature).
            height: height of the figure in inches. Defaults to `None`, which
                uses the plot's usual height for the style (2.5 for most
                plots in the default style).
            title: title of the figure. Defaults to `None`.
            style: name of a `matplotlib` or SciencePlots style (such as
                "science", "nature", "ieee", "ggplot", or "default"), a list
                of them (applied in order), or "xkcd", which makes figures
                look drawn by hand (on its own or after other styles). `None`
                (or "none") uses the current settings of `matplotlib`.
                Defaults to `options._PLOT_STYLE`.
            colors: name of a color cycle (a style that only sets colors,
                such as SciencePlots' "bright", "vibrant", "muted", or
                "high-contrast"), applied after `style`, or `None` (or
                "none") for the colors of `style`. Defaults to
                `options._PLOT_COLORS`.
            latex: whether to set the text with LaTeX, which must be
                installed. Defaults to `False`, in which case `matplotlib`
                sets the text (and math), whatever `style` says.
            **kwargs: parameters for `draw`.

        Returns:
            The dataset, with the new figure.

        """
        figures = utilities.import_tool('matplotlib.figure')
        with contextlib.ExitStack() as stack:
            matplotlib = stack.enter_context(
                _styled(style, colors, latex = latex))
            # How much smaller (or larger) figures are in the style than in
            # `matplotlib`'s default style.
            ratio = (
                matplotlib.rcParams['figure.figsize'][0]
                / matplotlib.rcParamsDefault['figure.figsize'][0])
            scale = ratio if self.scaled else 1.0
            if not self.scaled:
                # A plot that keeps its usual size has the text of the
                # style, enlarged as much as the plot was not shrunk.
                stack.enter_context(matplotlib.rc_context(
                    _font_sizes(matplotlib.rcParams, 1 / ratio)))
            figure = figures.Figure(
                figsize = (
                    self.size[0] * scale if width is None else width,
                    self.size[1] * scale if height is None else height),
                layout = 'constrained')
            drawn = self.draw(item, figure, **kwargs)
            if drawn is not None:
                figure = drawn
            if title is not None:
                figure.suptitle(title)
            # `matplotlib` makes some parts of a figure (such as most of its
            # ticks) only when the figure is drawn, and finds the colors of
            # lines in the cycle (such as "C0") then too, so both are done
            # here, in the style, rather than when it is shown or saved.
            _fix_colors(figure)
            _hide_minor_ticks(figure)
            figure.draw_without_rendering()
        item.figures[self.name] = figure
        item.record(self.name, figure = self.name)
        return item


@dataclasses.dataclass
class ActualVsPredicted(Plot):
    """The label against the model's predictions (regression)."""

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the actual labels against the predictions.

        Args:
            item: the dataset with a fitted regression model.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        _prediction_error(item, figure, 'actual_vs_predicted')


@dataclasses.dataclass
class BoxPlots(Plot):
    """Each numeric feature as a box, split by class.

    Draws seaborn's box plot of each numeric feature (or of `columns`), in a
    grid. The boxes are split by the column `by`, or by the classes of the
    label (for classification), or by the dataset's first group.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        limit: int = 16,
        **kwargs: Any) -> None:
        """Draws a box plot of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Defaults to `None`, which draws the
                numeric features.
            by: column that splits the boxes. Defaults to `None`, which uses
                the label's classes or the dataset's first group (if either).
            limit: most columns to draw. Defaults to 16.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        columns = list(item.numerics if columns is None else columns)[:limit]
        data, by, order = _split(item, by)
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            seaborn.boxplot(
                data = data, x = by, y = column, order = order, ax = axes)
            axes.set_title(str(column), fontsize = 'small')
            axes.set_ylabel('')


@dataclasses.dataclass
class CalibrationCurve(Plot):
    """The share of rows in each class against its predicted probability.

    A well calibrated classifier's probabilities are as often right as they
    say: of the rows given a probability of 0.8, 80% are in the class. The
    test rows are put in `bins` by their probability, and each bin's share in
    the class is drawn against its mean probability (with scikit-learn's
    calibration display). For more than two classes, each class is compared
    with the rest.

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
class CoefficientPlot(Plot):
    """Coefficients with their confidence intervals (a forest plot).

    Draws a table of coefficients made by `ols`, `glm`, `fixest`, or `cox`
    ("{model}_coefficients"), or of effects estimated by `partially_linear`
    or `interactive_regression`: a dot for each coefficient and a line for
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
    """The confusion matrix as a heatmap (classification)."""

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the confusion matrix.

        Args:
            item: the dataset with a fitted classifier.
            figure: the figure to draw on.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        matrix = evaluators.Confusion().evaluate(item)
        axes = figure.subplots()
        seaborn.heatmap(
            matrix, annot = True, fmt = 'd', cmap = 'Blues', ax = axes)
        # Minor ticks (which some styles show) mean nothing between classes.
        axes.minorticks_off()


@dataclasses.dataclass
class CorrelationHeatmap(Plot):
    """The correlations between the numeric columns as a heatmap."""

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        method: describers.CorrelationMethod = 'pearson',
        **kwargs: Any) -> None:
        """Draws the correlations.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            method: "pearson", "spearman", or "kendall". Defaults to
                "pearson".
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        table = describers.Correlations().describe(item, method = method)
        axes = figure.subplots()
        seaborn.heatmap(
            table,
            cmap = 'vlag',
            center = 0,
            vmin = -1,
            vmax = 1,
            square = True,
            ax = axes)
        # Minor ticks (which some styles show) mean nothing between columns.
        axes.minorticks_off()


@dataclasses.dataclass
class CountPlots(Plot):
    """The count of each value of each categorical feature, split by class.

    Draws seaborn's count plot of each categorical and boolean feature (or of
    `columns`), in a grid, with the `top` most common values of each. The
    bars are split by the column `by`, or by the classes of the label (for
    classification), or by the dataset's first group.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        limit: int = 16,
        top: int = 10,
        **kwargs: Any) -> None:
        """Draws a count plot of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Defaults to `None`, which draws the
                categorical and boolean features.
            by: column that splits the bars. Defaults to `None`, which uses
                the label's classes or the dataset's first group (if either).
            limit: most columns to draw. Defaults to 16.
            top: most values of each column to draw. Defaults to 10.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        if columns is None:
            columns = [*item.categoricals, *item.booleans]
        columns = list(columns)[:limit]
        data, by, order = _split(item, by)
        grid = _grid(figure, len(columns))
        for number, (axes, column) in enumerate(
            zip(grid, columns, strict = True)):
            values = data[column].astype(str)
            seaborn.countplot(
                data = data.assign(**{column: values}),
                y = column,
                hue = by,
                order = list(values.value_counts().index[:top]),
                hue_order = order,
                legend = 'auto' if number == 0 else False,
                ax = axes)
            axes.set_title(str(column), fontsize = 'small')
            axes.set_ylabel('')
        if grid:
            _figure_legend(figure, grid[0], by)


@dataclasses.dataclass
class DetCurve(Plot):
    """The detection error tradeoff (DET) curve (classification).

    scikit-learn's DET curve: the false negative rate against the false
    positive rate at every threshold, on the scales of a normal
    distribution, which makes the curves of good classifiers nearly straight
    and easy to compare. For more than two classes, each class is compared
    with the rest.

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
class FairnessPlot(Plot):
    """The model's fairness metrics for each group, as bars.

    Draws the table made with fairlearn by the `fairness` evaluator (which is
    applied for `group` if it has not been): a panel for each metric (the
    selection rate, the accuracy, and the true and false positive rates),
    with a bar for each group. A model treats groups alike when their bars
    are about the same.

    """

    size: ClassVar[tuple[float, float]] = (9.6, 3.6)

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        group: str | None = None,
        **kwargs: Any) -> None:
        """Draws each fairness metric for each group.

        Args:
            item: the dataset with a fitted classifier and groups.
            figure: the figure to draw on.
            group: column of groups. Defaults to `None`, which uses the table
                that `fairness` made or the dataset's first group.
            **kwargs: not used.

        """
        table = item.tables.get('fairness')
        if table is None or group not in {None, table.index.name}:
            table = evaluators.Fairness().evaluate(item, group = group)
        groups = table.drop(index = ['difference', 'ratio'], errors = 'ignore')
        columns = [c for c in groups.columns if c != 'count']
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            axes.bar([str(g) for g in groups.index], groups[column])
            axes.set_ylim(0, 1)
            axes.set_title(column.replace('_', ' '), fontsize = 'small')


@dataclasses.dataclass
class Histograms(Plot):
    """The distribution of each numeric feature, in a grid."""

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        columns: Sequence[str] | None = None,
        bins: int = 20,
        limit: int = 16,
        **kwargs: Any) -> None:
        """Draws a histogram of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Booleans are drawn as 0 and 1. Defaults
                to `None`, which draws the numeric features.
            bins: number of bins in each histogram. Defaults to 20.
            limit: most columns to draw. Defaults to 16.
            **kwargs: not used.

        """
        columns = list(item.numerics if columns is None else columns)[:limit]
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            values = item.data[column].dropna()
            if pd.api.types.is_bool_dtype(values.dtype):
                # `matplotlib` cannot put booleans in bins.
                values = values.astype(int)
            axes.hist(values, bins = bins)
            axes.set_title(str(column), fontsize = 'small')


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
            ValueError: if the model is not one of those.

        """
        model = _unwrapped(item)
        kind = getattr(model, 'kind', None)
        if not isinstance(model, models.Statsmodel) or kind not in _INFLUENCE:
            message = 'influence_plot needs an "ols" or "glm" model'
            raise ValueError(message)
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
class LabelPlot(Plot):
    """The distribution of the label: a bar for each class, or a histogram."""

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        bins: int = 20,
        **kwargs: Any) -> None:
        """Draws the number of rows in each class, or a histogram of the label.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            bins: number of bins in the histogram of a label that is not
                classified. Defaults to 20.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        axes = figure.subplots()
        y = item.y.dropna()
        if item.task == 'classify':
            seaborn.countplot(
                x = y.astype(str),
                order = [str(c) for c in item.classes],
                ax = axes)
        else:
            seaborn.histplot(y, bins = bins, kde = True, ax = axes)
        axes.set_xlabel(str(item.label))


@dataclasses.dataclass
class LearningCurve(Plot):
    """The model's score as it learns from more of the training rows.

    scikit-learn's learning curve: a copy of the model is fitted to growing
    shares (`sizes`) of the training rows in `cv`-fold cross-validation, and
    its scores on the rows it learned from and on the rows held out are
    drawn, which shows whether more data would help. The score is `scoring`,
    or accuracy for classification and R² for regression.

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
class MissingHeatmap(Plot):
    """Where values are missing, as a heatmap of the rows and columns.

    Each column of the heatmap is a column of the data and each line a row
    (up to `rows` of them, evenly spaced, in order), dark where the value is
    missing, so that patterns (such as columns that are missing together)
    stand out. Columns without missing values are left out unless `columns`
    names them.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        rows: int = 500,
        **kwargs: Any) -> None:
        """Draws where values are missing.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Defaults to `None`, which draws the
                columns with missing values.
            rows: most rows to draw. Defaults to 500.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        data = item.data if columns is None else item.data[list(columns)]
        missing = data.isna()
        if columns is None:
            missing = missing.loc[:, missing.any()]
        axes = figure.subplots()
        if missing.empty:
            axes.text(
                0.5, 0.5, 'no missing values', ha = 'center',
                transform = axes.transAxes)
            axes.set_axis_off()
            return
        step = max(1, math.ceil(len(missing) / rows))
        seaborn.heatmap(
            missing.iloc[::step].astype(int),
            cmap = 'Greys',
            vmin = 0,
            vmax = 1,
            cbar = False,
            yticklabels = False,
            ax = axes)
        # Minor ticks (which some styles show) mean nothing between columns.
        axes.minorticks_off()
        axes.set_ylabel('rows')


@dataclasses.dataclass
class PairPlot(Plot):
    """Each pair of numeric features against each other, colored by class.

    Like seaborn's pair plot: the distribution of each numeric feature (or of
    each of `columns`) is on the diagonal, and every other panel is a
    scatter plot of two features. The colors are the values of the column
    `by`, or the classes of the label (for classification), or the dataset's
    first group. Only the first `limit` features and `rows` rows are drawn,
    since the number of panels grows with the square of the features.

    """

    size: ClassVar[tuple[float, float]] = (8.0, 8.0)

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        limit: int = 4,
        rows: int = 1000,
        **kwargs: Any) -> None:
        """Draws each pair of features.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Defaults to `None`, which draws the
                numeric features.
            by: column that colors the dots. Defaults to `None`, which uses
                the label's classes or the dataset's first group (if either).
            limit: most columns to draw. Defaults to 4.
            rows: most rows to draw (chosen at random). Defaults to 1000.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        columns = list(item.numerics if columns is None else columns)[:limit]
        data, by, order = _split(item, by)
        data = data.sample(min(rows, len(data)), random_state = item.seed)
        count = len(columns)
        grid = figure.subplots(count, count, squeeze = False)
        for i, row in enumerate(columns):
            for j, column in enumerate(columns):
                axes = grid[i, j]
                # Only the panel in the first row and second column has a
                # legend, which is moved to the side of the figure.
                labeled = 'auto' if (i, j) == (0, 1) else False
                if i == j:
                    seaborn.histplot(
                        data = data, x = column, hue = by, hue_order = order,
                        element = 'step', stat = 'density',
                        common_norm = False, legend = False, ax = axes)
                else:
                    seaborn.scatterplot(
                        data = data, x = column, y = row, hue = by,
                        hue_order = order, s = 10, legend = labeled,
                        ax = axes)
                axes.set_xlabel(str(column) if i == count - 1 else '')
                axes.set_ylabel(str(row) if j == 0 else '')
        if count > 1:
            _figure_legend(figure, grid[0, 1], by)


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

    """

    size: ClassVar[tuple[float, float]] = (9.6, 3.6)

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
    """Precision against recall at every threshold (classification)."""

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

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        rows: int = 100,
        **kwargs: Any) -> None:
        """Draws the prediction intervals or the sizes of the sets.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            rows: most rows to draw (evenly spaced) for regression. Defaults
                to 100.
            **kwargs: not used.

        """
        table = item.tables.get('conformal')
        if table is None:
            table = evaluators.Conformal().evaluate(item)
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

    """

    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws the quantiles of the residuals.

        Args:
            item: the dataset with a fitted regression model.
            figure: the figure to draw on.
            **kwargs: not used.

        Raises:
            ValueError: if there are no predictions, or the label is
                classified.

        """
        if item.predictions is None:
            message = 'there are no predictions: apply a model first'
            raise ValueError(message)
        if item.task == 'classify':
            message = 'qq_plot needs a regression model'
            raise ValueError(message)
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
    """The model's errors against its predictions (regression)."""

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
    """The receiver operating characteristic (ROC) curve (classification)."""

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
class SearchPlot(Plot):
    """The cross-validated score of each try in a hyperparameter search.

    Draws the table that a model's search (grid, random, or Optuna) made,
    "{model}_search": a panel for each searched parameter, with the mean
    score of each try against the value of the parameter, and the best try
    marked with a star. The table is `source`, or the last search made.

    """

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
            ValueError: if there is no search table.

        """
        names = [
            n for n, t in item.tables.items()
            if n.endswith('_search') and 'mean_test_score' in t]
        if source is None and not names:
            message = (
                'there is no search table: apply a model with "search" first')
            raise ValueError(message)
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
class ShapBar(Plot):
    """The mean absolute SHAP value of the most important features.

    This is shap's bar plot. The SHAP values are the ones that
    `shap_importance` found, if it was applied, or are found as it finds them.
    The features past the `limit` are added together in the last bar.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the mean absolute SHAP value of each feature.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            limit: most bars to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.bar(
                explanation,
                max_display = limit,
                ax = figure.subplots(),
                show = False)


@dataclasses.dataclass
class ShapBeeswarm(Plot):
    """The SHAP value of each row for the most important features.

    This is shap's beeswarm plot (once called its summary plot). Each dot is
    a row, placed by its SHAP value and colored by the value of the feature,
    so it shows both how much a feature matters and in which direction. The
    SHAP values are the ones that `shap_importance` found, if it was applied,
    or are found as it finds them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the SHAP values of the most important features.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            limit: most features to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.beeswarm(
                explanation,
                max_display = limit,
                ax = figure.subplots(),
                plot_size = None,
                show = False)


@dataclasses.dataclass
class ShapDecision(Plot):
    """How the features move each row's prediction, as lines.

    This is shap's decision plot. Each line is a row: it starts at the
    model's average output at the bottom and moves by the SHAP value of each
    feature (the most important at the top) to reach the row's output at
    the top. Rows that the model treats alike follow similar paths. The SHAP
    values are the ones that `shap_importance` found, if it was applied, or
    are found as it finds them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 20,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws a line for each explained row.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            limit: most features to draw. Defaults to 20.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        names = list(explanation.feature_names)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.decision(
                float(np.asarray(explanation.base_values).flat[0]),
                np.asarray(explanation.values),
                features = pd.DataFrame(
                    np.asarray(explanation.data), columns = names),
                feature_display_range = slice(None, -limit - 1, -1),
                auto_size_plot = False,
                ignore_warnings = True,
                show = False)


@dataclasses.dataclass
class ShapEmbedding(Plot):
    """The rows placed by their SHAP values, colored by one feature's.

    This is shap's embedding plot. The SHAP values of all the features are
    reduced to two dimensions (with principal components), so that rows that
    are explained alike are close together, and each row is colored by the
    SHAP value of `feature` (by default, the most important). The SHAP values
    are the ones that `shap_importance` found, if it was applied, or are
    found as it finds them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        feature: str | None = None,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the explained rows by their SHAP values.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            feature: feature whose SHAP values color the rows. Defaults to
                `None`, which uses the feature with the largest mean absolute
                SHAP value.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        feature = _feature(explanation, feature)
        # shap resizes the figure, but every plot keeps the size that it was
        # given.
        figure.set_size_inches = _keep_size
        try:
            with _pyplot(figure, item.seed) as shap:
                shap.plots.embedding(
                    feature,
                    np.asarray(explanation.values),
                    feature_names = list(explanation.feature_names),
                    show = False)
        finally:
            del figure.set_size_inches


@dataclasses.dataclass
class ShapForce(Plot):
    """How each feature pushes one row's prediction up or down.

    This is shap's force plot (drawn with matplotlib). From the model's
    average output, the features that raise the row's output push from the
    left (in red) and the features that lower it push from the right (in
    blue), and they meet at the row's output. The SHAP values are the ones
    that `shap_importance` found, if it was applied, or are found as it
    finds them, for the first rows of the test data.

    """

    size: ClassVar[tuple[float, float]] = (14.0, 4.0)
    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        row: Hashable | None = None,
        rotation: float = 30,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> Any:
        """Draws how each feature pushes the prediction for `row`.

        Args:
            item: the dataset with a fitted model.
            figure: the figure whose size to use.
            row: index label of the row to explain. Defaults to `None`, which
                explains the first explained row.
            rotation: angle of the features' labels, in degrees, so that they
                do not overlap. Defaults to 30.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        Returns:
            The figure that shap drew on, since its force plot makes its own.

        """
        explanation = _explanation(item, category, rows, background)
        one = explanation[_explained_position(item, explanation, row)]
        with _pyplot(figure, item.seed) as shap:
            drawn = shap.plots.force(
                one,
                matplotlib = True,
                figsize = tuple(figure.get_size_inches()),
                text_rotation = rotation,
                show = False)
        # The layout keeps the labels above and below the bar in the figure.
        drawn.set_layout_engine('constrained')
        return drawn


@dataclasses.dataclass
class ShapGroupDifference(Plot):
    """How the SHAP values of the features differ between two groups.

    This is shap's group difference plot. For each feature, the bar is the
    mean SHAP value of the explained rows in a group minus that of the other
    rows, with a 95% confidence interval (from bootstrapping), so it shows
    which features explain why the model's outputs differ between them. The
    group is the rows whose `group` column is `value`. The SHAP values are
    the ones that `shap_importance` found, if it was applied, or are found
    as it finds them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        group: str | None = None,
        value: Any = None,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the difference in each feature's SHAP values.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            group: column of groups. Defaults to `None`, which uses the
                dataset's first group.
            value: the group to compare with the other rows. Defaults to
                `None`, which uses the first value of `group` (in sorted
                order).
            limit: most features to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        Raises:
            ValueError: if there is no `group`, or the explained rows are not
                both in and out of the group.

        """
        explanation = _explanation(item, category, rows, background)
        if group is None:
            if not item.groups:
                message = 'the dataset has no groups: name a "group" column'
                raise ValueError(message)
            group = item.groups[0]
        groups = item.data.loc[_explained_rows(item, explanation), group]
        if value is None:
            value = sorted(groups.dropna().unique(), key = str)[0]
        mask = (groups == value).to_numpy()
        if mask.all() or not mask.any():
            message = (
                f'the explained rows must be both in and out of the group '
                f'{value!r} of {group!r}')
            raise ValueError(message)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.group_difference(
                np.asarray(explanation.values),
                mask,
                feature_names = list(explanation.feature_names),
                xlabel = f'SHAP value of {group} = {value} minus the rest',
                max_display = limit,
                ax = figure.subplots(),
                show = False)


@dataclasses.dataclass
class ShapHeatmap(Plot):
    """The SHAP values of every explained row, as a heatmap.

    This is shap's heatmap plot. The rows are put in order so that rows that
    are explained alike are next to each other, and the line above the
    heatmap is the model's output for each row. The SHAP values are the ones
    that `shap_importance` found, if it was applied, or are found as it finds
    them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the SHAP values of each row for the most important features.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            limit: most features to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.heatmap(
                explanation,
                max_display = limit,
                ax = figure.subplots(),
                show = False)


@dataclasses.dataclass
class ShapPartialDependence(Plot):
    """The model's prediction as one feature changes, with each row's.

    This is shap's partial dependence plot. The thick line is the model's
    average output over the explained rows when each of them is given each
    value of `feature` (by default, the most important), the thin lines are
    the outputs of single rows, and the histogram is the distribution of the
    feature. The dashed lines are the average output and the feature's
    average. For a classifier, the output is the probability of `category`.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        feature: str | None = None,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the model's output as `feature` changes.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            feature: feature to draw. Defaults to `None`, which draws the
                feature with the largest mean absolute SHAP value.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        feature = _feature(explanation, feature)
        x = item.data.loc[_explained_rows(item, explanation), item.features]
        with _pyplot(figure, item.seed) as shap:
            shap.plots.partial_dependence(
                feature,
                _predictor(item, category),
                x,
                ice = True,
                model_expected_value = True,
                feature_expected_value = True,
                ax = figure.subplots(),
                show = False)


@dataclasses.dataclass
class ShapScatter(Plot):
    """A feature's SHAP values against its values (a dependence plot).

    This is shap's scatter plot. Each dot is a row. The dots are colored by
    the feature that seems to interact with `feature` the most, or by
    `interaction`. The SHAP values are the ones that `shap_importance` found,
    if it was applied, or are found as it finds them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        feature: str | None = None,
        interaction: str | None = None,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the SHAP values of a feature against its values.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            feature: feature to draw. Defaults to `None`, which draws the
                feature with the largest mean absolute SHAP value.
            interaction: feature that colors the dots. Defaults to `None`,
                which lets shap choose the one that interacts the most.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        Raises:
            KeyError: if `feature` or `interaction` is not a feature.

        """
        explanation = _explanation(item, category, rows, background)
        feature = _feature(explanation, feature)
        color = (
            explanation if interaction is None
            else explanation[:, _feature(explanation, interaction)])
        with _pyplot(figure, item.seed) as shap:
            shap.plots.scatter(
                explanation[:, feature],
                color = color,
                ax = figure.subplots(),
                show = False)


@dataclasses.dataclass
class ShapViolin(Plot):
    """The distribution of the SHAP values of the most important features.

    This is shap's violin plot: like the beeswarm plot, but each feature's
    SHAP values are drawn as a violin (their density), with dots colored by
    the value of the feature. The SHAP values are the ones that
    `shap_importance` found, if it was applied, or are found as it finds
    them.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws the distribution of each feature's SHAP values.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            limit: most features to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        """
        explanation = _explanation(item, category, rows, background)
        with _pyplot(figure, item.seed) as shap:
            shap.plots.violin(
                explanation,
                max_display = limit,
                plot_size = None,
                show = False)


@dataclasses.dataclass
class ShapWaterfall(Plot):
    """How each feature moves one row's prediction (a waterfall plot).

    This is shap's waterfall plot. It starts from the model's average output
    on the background data and adds the SHAP value of each feature, largest
    first, to reach the model's output for the row. The SHAP values are the
    ones that `shap_importance` found, if it was applied, or are found as it
    finds them, for the first rows of the test data.

    """

    scaled: ClassVar[bool] = False

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        row: Hashable | None = None,
        limit: int = 10,
        category: Hashable | None = None,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> None:
        """Draws how each feature moves the prediction for `row`.

        Args:
            item: the dataset with a fitted model.
            figure: the figure to draw on.
            row: index label of the row to explain. Defaults to `None`, which
                explains the first explained row.
            limit: most features to draw. Defaults to 10.
            category: class of the label to explain. Defaults to `None`,
                which explains the last class (the positive class of a binary
                label).
            rows: most rows to explain, if the SHAP values have not been
                found yet. Defaults to 200.
            background: most training rows to use as the background data, if
                the SHAP values have not been found yet. Defaults to 100.
            **kwargs: not used.

        Raises:
            KeyError: if `row` is not one of the explained rows.

        """
        explanation = _explanation(item, category, rows, background)
        position = _explained_position(item, explanation, row)
        # shap resizes the figure before it fits the labels to the arrows, but
        # every plot keeps the size that it was given.
        figure.set_size_inches = _keep_size
        try:
            with _pyplot(figure, item.seed) as shap:
                shap.plots.waterfall(
                    explanation[position], max_display = limit, show = False)
        finally:
            del figure.set_size_inches


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

    """

    size: ClassVar[tuple[float, float]] = (9.6, 6.4)

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
            ValueError: if the model is not an explainable boosting model.

        """
        model = _unwrapped(item)
        if not hasattr(model, 'term_importances'):
            message = 'shape_functions needs an "explainable_boosting" model'
            raise ValueError(message)
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
class SurvivalCurves(Plot):
    """The share of rows without an event over time (Kaplan-Meier curves).

    The label is the time until the event. Set "event" to the column that is
    1 if the event happened (by default, every event was observed) and
    "group" to a column to draw a curve for each group. The shading is the
    95% confidence interval. Set "at_risk" to add a table of the number of
    rows at risk (whose time is at least each tick's) below the curves, as
    papers in medicine usually do.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        event: str | None = None,
        group: str | None = None,
        *,
        at_risk: bool = False,
        **kwargs: Any) -> None:
        """Draws the survival curves.

        Args:
            item: the dataset to draw. Its label is the time.
            figure: the figure to draw on.
            event: name of the column that is 1 if the event happened.
                Defaults to `None`, which means every event was observed.
            group: name of a column of groups. Defaults to `None`, which draws
                one curve.
            at_risk: whether to add the table of the number of rows at risk.
                Defaults to `False`.
            **kwargs: not used.

        """
        curves = describers.survival_curves(item, event, group)
        if not at_risk:
            axes = figure.subplots()
        else:
            axes, table = figure.subplots(
                2, 1, sharex = True,
                height_ratios = [4, 0.6 + 0.3 * len(curves)])
        for name, curve in curves.items():
            steps = describers.survival_steps(curve)
            [line] = axes.step(
                steps['time'], steps['survival'], where = 'post',
                label = name)
            axes.fill_between(
                steps['time'], steps['ci_lower'], steps['ci_upper'],
                step = 'post', alpha = 0.25, color = line.get_color(),
                linewidth = 0)
        axes.legend()
        axes.set_ylabel('share without the event')
        if at_risk:
            _at_risk(table, axes, curves)
            axes = table
        axes.set_xlabel(str(item.label))


@dataclasses.dataclass
class TreePlot(Plot):
    """The splits of a decision tree (or of one tree of a forest).

    scikit-learn's tree plot of a `decision_tree`, or of the tree numbered
    `tree` of a `random_forest`, `extra_trees`, `gradient_boosting`, or
    `adaboost` model, down to `depth` levels (deep trees are hard to read).
    Each box is a split, with the share of rows in each class (or the mean
    label) shaded.

    """

    size: ClassVar[tuple[float, float]] = (12.0, 6.0)

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        tree: int = 0,
        depth: int = 3,
        **kwargs: Any) -> None:
        """Draws the tree.

        Args:
            item: the dataset with a fitted tree model from scikit-learn.
            figure: the figure to draw on.
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            depth: most levels to draw. Defaults to 3.
            **kwargs: not used.

        Raises:
            ValueError: if the model is not a scikit-learn tree (or trees).

        """
        tree_module = importlib.import_module('sklearn.tree')
        model = _unwrapped(item)
        names = getattr(model, 'feature_names_in_', None)
        chosen = model
        if hasattr(model, 'estimators_'):
            chosen = list(np.ravel(np.asarray(
                model.estimators_, dtype = object)))[tree]
        if not hasattr(chosen, 'tree_'):
            message = (
                'tree_plot needs a decision tree, or a forest of them, from '
                'scikit-learn')
            raise ValueError(message)
        classes = getattr(chosen, 'classes_', None)
        # scikit-learn sizes the text of the boxes with a canvas that draws.
        agg = importlib.import_module('matplotlib.backends.backend_agg')
        bases = importlib.import_module('matplotlib.backend_bases')
        agg.FigureCanvasAgg(figure)
        try:
            tree_module.plot_tree(
                chosen,
                max_depth = depth,
                feature_names = None if names is None else list(names),
                class_names = None if classes is None else [
                    str(c) for c in _classes(item)],
                filled = True,
                ax = figure.subplots())
        finally:
            bases.FigureCanvasBase(figure)


@dataclasses.dataclass
class ValidationCurve(Plot):
    """The model's score as one of its parameters changes.

    scikit-learn's validation curve: a copy of the model is fitted with each
    of `values` for `parameter` (in `cv`-fold cross-validation on the
    training rows), and its scores on the rows it learned from and on the
    rows held out are drawn, which shows where it underfits or overfits. The
    score is `scoring`, or accuracy for classification and R² for
    regression.

    """

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


""" Private Functions """


def _at_risk(table: Any, axes: Any, curves: dict[str, Any]) -> None:
    """Writes the number of rows at risk at each tick of `axes` in `table`.

    Args:
        table: the axes to write the numbers in.
        axes: the axes of the survival curves.
        curves: the fitted Kaplan-Meier estimators, by name.

    """
    low, high = axes.get_xlim()
    ticks = [t for t in axes.get_xticks() if low <= t <= high]
    for row, curve in enumerate(curves.values()):
        durations = np.asarray(curve.time)
        for tick in ticks:
            table.text(
                tick, row, str(int((durations >= tick).sum())),
                ha = 'center', va = 'center', fontsize = 'small')
    table.set_yticks(range(len(curves)), [str(n) for n in curves])
    table.set_ylim(len(curves) - 0.5, -0.5)
    table.set_title('number at risk', fontsize = 'small', loc = 'left')
    table.tick_params(left = False)
    for spine in table.spines.values():
        spine.set_visible(False)


def _classes(item: base.Dataset) -> list[Any]:
    """Returns the classes in the order of the model's probabilities.

    Args:
        item: the dataset with a fitted classifier.

    Returns:
        The columns of the dataset's `probabilities`, or its `classes` if the
            model does not predict probabilities.

    """
    if item.probabilities is not None:
        return list(item.probabilities.columns)
    return item.classes


def _coefficients(item: base.Dataset, source: str | None) -> pd.DataFrame:
    """Returns a table of coefficients with confidence intervals.

    Args:
        item: the dataset with a table of coefficients.
        source: name of the table, or `None` for the last one made.

    Raises:
        ValueError: if there is no such table.

    Returns:
        The table, with "coefficient", "ci_lower", and "ci_upper" columns.

    """
    names = [
        n for n, t in item.tables.items()
        if set(_INTERVAL_COLUMNS) <= set(t.columns)]
    if source is None and not names:
        message = (
            'there is no table of coefficients: apply "ols", "glm", "fixest", '
            '"cox", or an effect first')
        raise ValueError(message)
    return item.tables[source or names[-1]]


def _curves(item: base.Dataset) -> list[tuple[str, Any, pd.Series, Any]]:
    """Returns what is needed to draw a curve for each class.

    Args:
        item: the dataset with a fitted classifier.

    Raises:
        ValueError: if the model does not predict probabilities.

    Returns:
        For a binary label, the positive class only: its name, the true
            labels, its predicted probability, and the class itself (the
            positive label). For more labels, each class against the rest:
            its name, whether each row is in the class, its predicted
            probability, and `True`.

    """
    if item.probabilities is None:
        message = 'the model does not predict probabilities'
        raise ValueError(message)
    probabilities = item.probabilities
    y_true = item.y.loc[probabilities.index].to_numpy()
    classes = list(probabilities.columns)
    if len(classes) == 2:  # noqa: PLR2004
        positive = classes[-1]
        return [(str(positive), y_true, probabilities[positive], positive)]
    return [
        (str(c), y_true == c, probabilities[c], True) for c in classes]


def _explained_position(
    item: base.Dataset,
    explanation: Any,
    row: Hashable | None) -> int:
    """Returns the position of `row` among the rows that were explained.

    Args:
        item: the dataset with SHAP values.
        explanation: the SHAP values.
        row: index label of a row, or `None` for the first explained row.

    Raises:
        KeyError: if `row` was not explained.

    Returns:
        The position of the row in `explanation`.

    """
    explained = list(_explained_rows(item, explanation))
    if row is None:
        return 0
    if row not in explained:
        message = (
            f'row {row!r} was not explained: SHAP values were found for the '
            f'first {len(explained)} rows of the test data')
        raise KeyError(message)
    return explained.index(row)


def _explained_rows(item: base.Dataset, explanation: Any) -> pd.Index:
    """Returns the index labels of the rows that were explained.

    `shap_importance` explains the first rows of the test data (or of all
    the data, if it is not split).

    Args:
        item: the dataset with SHAP values.
        explanation: the SHAP values.

    Returns:
        The index labels of the explained rows, in order.

    """
    x = item.x_test if item.is_split else item.x
    return x.index[:explanation.shape[0]]


def _explanation(
    item: base.Dataset,
    category: Hashable | None,
    rows: int,
    background: int) -> Any:
    """Returns the SHAP values of the model's output for one class.

    The SHAP values are the ones that `shap_importance` stored in the
    dataset's `fitted`, if it was applied. Otherwise, they are found (and
    stored) as `shap_importance` finds them.

    Args:
        item: the dataset with a fitted model.
        category: class of the label to explain, or `None` for the last
            class. Models whose SHAP values are for one output only (such as
            regressions) do not use it.
        rows: most rows to explain, if the SHAP values have not been found.
        background: most training rows to use as the background data, if the
            SHAP values have not been found.

    Raises:
        ValueError: if `category` is not a class of the label.

    Returns:
        A `shap.Explanation` with a SHAP value for each row and feature.

    """
    explanation = item.fitted.get('shap_importance')
    if explanation is None:
        evaluators.ShapImportance().evaluate(
            item, rows = rows, background = background)
        explanation = item.fitted['shap_importance']
    if len(explanation.shape) == 2:  # noqa: PLR2004
        return explanation
    classes = _classes(item)
    if category is None:
        return explanation[:, :, -1]
    if category not in classes:
        message = f'{category!r} is not one of the classes {classes}'
        raise ValueError(message)
    return explanation[:, :, classes.index(category)]


def _feature(explanation: Any, name: str | None) -> str:
    """Returns `name`, or the feature that matters most.

    Args:
        explanation: the SHAP values.
        name: name of a feature, or `None`.

    Raises:
        KeyError: if `name` is not one of the explained features.

    Returns:
        `name`, or the feature with the largest mean absolute SHAP value.

    """
    names = list(explanation.feature_names)
    if name is None:
        values = np.abs(np.asarray(explanation.values))
        return str(names[int(values.mean(axis = 0).argmax())])
    if name not in names:
        message = f'{name!r} is not one of the explained features'
        raise KeyError(message)
    return name


def _figure_legend(figure: Any, axes: Any, title: str | None) -> None:
    """Moves the legend of `axes` to the side of `figure`.

    Args:
        figure: the figure.
        axes: axes with a legend (or without one, which does nothing).
        title: title of the legend.

    """
    legend = axes.get_legend()
    if legend is None:
        return
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    legend.remove()
    figure.legend(handles, labels, title = title, loc = 'outside right upper')


def _fix_colors(figure: Any) -> None:
    """Replaces the colors of lines that name colors of the cycle.

    A line keeps a color such as "C0" (the first color of the cycle) as it
    was given, and finds it in the cycle of whatever style is used when the
    line is drawn. The color is found now instead, in the current style.

    Args:
        figure: a figure. Its lines are changed in place.

    """
    colors = importlib.import_module('matplotlib.colors')
    lines = importlib.import_module('matplotlib.lines')
    for line in figure.findobj(lines.Line2D):
        for part in ('color', 'markerfacecolor', 'markeredgecolor'):
            value = getattr(line, f'get_{part}')()
            if isinstance(value, str) and _CYCLE_COLOR.fullmatch(value):
                getattr(line, f'set_{part}')(colors.to_rgba(value))


def _font_sizes(settings: Any, factor: float) -> dict[str, float]:
    """Returns the sizes of text in `settings`, multiplied by `factor`.

    Sizes given as words (such as "large") are relative to "font.size", so
    they change with it.

    Args:
        settings: the settings of `matplotlib` (its `rcParams`).
        factor: how much larger to make the text.

    Returns:
        The new sizes, by the names of the settings.

    """
    return {
        name: settings[name] * factor for name in _FONT_SIZES
        if isinstance(settings[name], int | float)}


def _grid(figure: Any, count: int) -> list[Any]:
    """Returns axes for `count` small plots, in rows of up to four.

    Args:
        figure: the figure to draw on.
        count: number of plots.

    Returns:
        The axes of the plots, in order. Axes in the grid that are not needed
            are hidden.

    """
    width = min(max(count, 1), 4)
    grid = figure.subplots(
        math.ceil(max(count, 1) / width), width, squeeze = False)
    for axes in list(grid.flat)[count:]:
        axes.set_visible(False)
    return list(grid.flat)[:count]


def _hide_minor_ticks(figure: Any) -> None:
    """Hides the minor ticks of axes of categories, such as features' names.

    Some styles (such as "science") show minor ticks, which mean nothing
    between categories.

    Args:
        figure: a figure. Its axes are changed in place.

    """
    category = importlib.import_module('matplotlib.category')
    ticker = importlib.import_module('matplotlib.ticker')
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            if isinstance(axis.units, category.UnitData):
                axis.set_minor_locator(ticker.NullLocator())


def _importances(
    item: base.Dataset,
    source: str | None) -> tuple[str, pd.DataFrame]:
    """Returns a table of the importance of each feature.

    Args:
        item: the dataset with a fitted model.
        source: name of the table, or `None` for the first one found of
            "shap_importance", "permutation_importance", and
            "feature_importance".

    Returns:
        The name of the table and the table (largest importance first). If
            there is no such table, the importances are found with
            `feature_importance`.

    """
    names = [source] if source else list(_IMPORTANCE_LABELS)
    found = next((n for n in names if n in item.tables), None)
    if found is None:
        return 'feature_importance', evaluators.FeatureImportance().evaluate(
            item)
    return found, item.tables[found]


def _keep_size(*args: Any, **kwargs: Any) -> None:
    """Does nothing, in place of a figure's `set_size_inches`.

    Args:
        *args: not used.
        **kwargs: not used.

    """


def _model(item: base.Dataset) -> Any:
    """Returns the dataset's fitted model.

    Args:
        item: the dataset.

    Raises:
        ValueError: if there is no fitted model.

    Returns:
        The fitted model.

    """
    if item.model is None:
        message = 'there is no fitted model: apply a model first'
        raise ValueError(message)
    return item.model


def _predictor(
    item: base.Dataset,
    category: Hashable | None) -> Callable[[pd.DataFrame], np.ndarray]:
    """Returns a function of the features that returns the model's output.

    Args:
        item: the dataset with a fitted model.
        category: class of the label whose probability is returned (for a
            classifier), or `None` for the last class.

    Returns:
        A function that takes features and returns the model's predictions
            (for regression) or the probability of `category`.

    """
    model = _model(item)
    if item.task != 'classify' or not hasattr(model, 'predict_proba'):
        return lambda x: np.asarray(model.predict(x), dtype = float)
    classes = _classes(item)
    position = -1 if category is None else classes.index(category)
    return lambda x: np.asarray(model.predict_proba(x))[:, position]


def _prediction_error(item: base.Dataset, figure: Any, kind: str) -> None:
    """Draws scikit-learn's prediction error display.

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


@contextlib.contextmanager
def _pyplot(figure: Any, seed: int | None) -> Iterator[Any]:
    """Lends `figure` to `matplotlib.pyplot` while shap draws on it.

    shap draws with `matplotlib.pyplot`, which only draws on figures that it
    manages. In this context, `figure` is pyplot's current figure, with a
    canvas that draws images but never opens a window. Afterwards, pyplot
    forgets `figure` and any other figure made in the context (which can
    still be used), and its interactive mode (which some shap plots turn
    off) is restored.

    Some shap plots also use `numpy`'s global random numbers (to jitter dots
    or to bootstrap), so they are seeded with `seed` in the context, and put
    back as they were afterwards. And shap's bar and waterfall plots are
    colored with the colors of the style (see `_shap_colors`), and shap's own
    colors are put back afterwards.

    Args:
        figure: the figure to lend.
        seed: seed for `numpy`'s global random numbers.

    Yields:
        Any: the `shap` module.

    """
    shap = utilities.import_tool('shap')
    matplotlib = importlib.import_module('matplotlib')
    pyplot = importlib.import_module('matplotlib.pyplot')
    # pyplot keeps its figures in `Gcf`. Newer versions of matplotlib can
    # adopt a figure with `pyplot.figure(figure)`, but that would also give
    # it a window if the backend has them.
    figures = importlib.import_module('matplotlib._pylab_helpers').Gcf
    agg = importlib.import_module('matplotlib.backends.backend_agg')
    bases = importlib.import_module('matplotlib.backend_bases')
    before = set(pyplot.get_fignums())
    interactive = matplotlib.is_interactive()
    state = np.random.get_state()  # noqa: NPY002
    manager = agg.FigureCanvasAgg.new_manager(
        figure, max(before, default = 0) + 1)
    figures.set_active(manager)
    np.random.seed(seed)  # noqa: NPY002
    # shap places the ticks of some plots (such as its bar and waterfall
    # plots) so that `matplotlib` would make billions of minor ticks between
    # them, so styles that show minor ticks (such as "science") cannot.
    minor = {'xtick.minor.visible': False, 'ytick.minor.visible': False}
    # shap's colors can only be set in versions that have this (experimental)
    # module.
    try:
        styles = importlib.import_module('shap.plots._style')
    except ImportError:
        styles = None
    colors = None if styles is None else styles.get_style()
    try:
        if styles is not None:
            styles.set_style(**_shap_colors(matplotlib))
        with matplotlib.rc_context(minor):
            yield shap
    finally:
        if styles is not None:
            styles.set_style(colors)
        np.random.set_state(state)  # noqa: NPY002
        for number in set(pyplot.get_fignums()) - before:
            made = figures.get_fig_manager(number).canvas.figure
            pyplot.close(number)
            # Without pyplot's canvas, the figure cannot open a window.
            bases.FigureCanvasBase(made)
        matplotlib.interactive(interactive)


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
        TypeError: if `model` is not a scikit-learn estimator (as the models
            of statsmodels and pyfixest are not).

    """
    sklearn = importlib.import_module('sklearn.base')
    if not isinstance(model, sklearn.BaseEstimator):
        message = (
            f'{use} needs a scikit-learn model, not a {type(model).__name__}')
        raise TypeError(message)


def _scoring(item: base.Dataset) -> str:
    """Returns the name of the default scikit-learn scorer for `item`.

    Args:
        item: the dataset.

    Returns:
        "accuracy" for classification, and "r2" otherwise.

    """
    return 'accuracy' if item.task == 'classify' else 'r2'


def _shap_colors(matplotlib: Any) -> dict[str, tuple[float, ...]]:
    """Returns the colors of the style for shap's bar and waterfall plots.

    shap colors features that raise a prediction red and those that lower it
    blue (with lighter versions of both for parts of its waterfall plot),
    rather than with the colors of the style. These are the first two colors
    of the style's cycle instead: in "bright", blue (the first) for features
    that lower a prediction and red (the second) for those that raise it.
    The lighter versions are halfway to white.

    Args:
        matplotlib: the `matplotlib` module, in the style.

    Returns:
        shap's style options for the colors, by their names.

    """
    colors = importlib.import_module('matplotlib.colors')
    cycle = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color')
    cycle = list(cycle or ['C0', 'C1'])
    lower, higher = (np.asarray(colors.to_rgb(c)) for c in (cycle * 2)[:2])
    return {
        'primary_color_positive': tuple(higher),
        'primary_color_negative': tuple(lower),
        'secondary_color_positive': tuple((higher + 1) / 2),
        'secondary_color_negative': tuple((lower + 1) / 2)}


def _split(
    item: base.Dataset,
    by: str | None) -> tuple[pd.DataFrame, str | None, list[str] | None]:
    """Returns the data and the column that splits (or colors) a plot.

    Args:
        item: the dataset.
        by: name of a column, or `None` for the label (for classification),
            or the dataset's first group (if it has groups).

    Returns:
        The data, with `by` as text (unless it holds decimal numbers), so
            that seaborn gives each value its own color; the name of the
            column (or `None`); and the order of its values (the classes,
            for the label), or `None` if it is not text.

    """
    if by is None:
        if item.task == 'classify':
            by = item.label
        elif item.groups:
            by = item.groups[0]
    data = item.data
    if by is None or pd.api.types.is_float_dtype(data[by]):
        return data, by, None
    values = data[by].map(lambda v: None if pd.isna(v) else str(v))
    if by == item.label:
        order = [str(c) for c in item.classes]
    else:
        order = sorted(values.dropna().unique())
    return data.assign(**{by: values}), by, order


@contextlib.contextmanager
def _styled(
    style: str | Sequence[str] | None,
    colors: str | None,
    *,
    latex: bool) -> Iterator[Any]:
    """Applies a style to the figures made and drawn in the context.

    The styles are applied over the current settings of `matplotlib`, which
    are put back afterwards. SciencePlots, which adds its styles to those of
    `matplotlib`, is only imported if a style is not one of `matplotlib`'s.

    Args:
        style: names of `matplotlib` or SciencePlots styles, applied in order,
            and "xkcd", which is applied after them. `None` (or "none")
            applies none.
        colors: name of a color cycle, applied after the styles (but before
            "xkcd", which keeps the colors), or `None` (or "none") for the
            colors of the styles.
        latex: whether to set the text with LaTeX.

    Raises:
        ValueError: if a style is not one of `matplotlib` or SciencePlots, or
            `colors` is not a color cycle.

    Yields:
        Any: the `matplotlib` module.

    """
    matplotlib = utilities.import_tool('matplotlib')
    styles = importlib.import_module('matplotlib.style')
    # Settings files may list several styles in one text, as "science, nature".
    names = [] if style is None else style.split(',') if isinstance(
        style, str) else list(style)
    names = [str(n).strip() for n in names]
    names = [n for n in names if n and n.lower() != 'none']
    cycle = None if colors is None or str(colors).lower() == 'none' else (
        str(colors))
    sheets = [n for n in names if n != 'xkcd'] + ([cycle] if cycle else [])
    # "default" (the settings `matplotlib` starts with) is not in its library.
    if any(s not in styles.library and s != 'default' for s in sheets):
        utilities.import_tool('scienceplots')
    unknown = [
        n for n in names
        if n not in styles.library and n not in {'default', 'xkcd'}]
    if unknown:
        message = (
            f'{unknown} are not styles of matplotlib or SciencePlots (see '
            f'matplotlib.style.available)'
        )
        raise ValueError(message)
    # A color cycle is a style that sets only the colors of the cycle.
    if cycle and set(styles.library.get(cycle, ())) != {'axes.prop_cycle'}:
        message = (
            f'colors must be a color cycle (such as "bright", "vibrant", '
            f'"muted", or "high-contrast"), not {cycle!r}'
        )
        raise ValueError(message)
    with contextlib.ExitStack() as stack:
        stack.enter_context(matplotlib.rc_context())
        if sheets:
            styles.use(sheets)
        matplotlib.rcParams['text.usetex'] = bool(latex)
        if 'xkcd' in names:
            pyplot = importlib.import_module('matplotlib.pyplot')
            stack.enter_context(pyplot.xkcd())
            # `matplotlib` warns each time that it looks for a font of xkcd's
            # that is not installed, so only those that are installed are
            # used (or the usual font, if none are).
            fonts = importlib.import_module('matplotlib.font_manager')
            installed = {f.name for f in fonts.fontManager.ttflist}
            matplotlib.rcParams['font.family'] = [
                f for f in matplotlib.rcParams['font.family']
                if f in installed] or ['sans-serif']
        yield matplotlib


def _unwrapped(item: base.Dataset) -> Any:
    """Returns the dataset's fitted model, without an `amos` wrapper.

    Args:
        item: the dataset with a fitted model.

    Returns:
        The fitted model (or the model that a `LabelCoded` wraps).

    """
    model = _model(item)
    if isinstance(model, models.LabelCoded):
        return model.estimator
    return model
