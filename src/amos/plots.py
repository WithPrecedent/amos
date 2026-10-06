"""Techniques that draw figures of the data and the model.

These are the techniques of the "artist" stage. Each one draws a `matplotlib`
figure and adds it to the dataset's `figures` (under the technique's name).
Figures are made without `matplotlib.pyplot`, so they do not open windows or
depend on a display. `Project.export` saves them as image files, and in a
notebook, a figure is shown by making it the last line of a cell.

Every plot accepts "width" and "height" (in inches) and "title" parameters.

Contents:
    Plot: base class for techniques that draw a figure.
    ActualVsPredicted: the label against the predictions (regression).
    ConfusionHeatmap: the confusion matrix as a heatmap (classification).
    CorrelationHeatmap: the correlations between numeric columns.
    Histograms: the distribution of each numeric feature.
    ImportancePlot: the most important features, as bars.
    PrecisionRecallCurve: precision against recall (classification).
    ResidualPlot: the errors against the predictions (regression).
    RocCurve: the ROC curve (classification).
    SurvivalCurves: the share of rows without an event over time.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from . import base, describers, evaluators, utilities

# Labels of the axis of an `importance_plot`, by the table it draws, in the
# order that the tables are looked for.
_IMPORTANCE_LABELS: dict[str, str] = {
    'shap_importance': 'mean absolute SHAP value',
    'permutation_importance': 'drop in score when the feature is shuffled',
    'feature_importance': "the model's importance"}


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

    """ Required Methods """

    @abc.abstractmethod
    def draw(self, item: base.Dataset, figure: Any, **kwargs: Any) -> None:
        """Draws on `figure`.

        Args:
            item: the dataset to draw.
            figure: a `matplotlib.figure.Figure` to draw on.
            **kwargs: parameters for the drawing.

        """

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        width: float = 6.4,
        height: float = 4.8,
        title: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Draws a figure and adds it to the figures of `item`.

        Args:
            item: the dataset to draw.
            width: width of the figure in inches. Defaults to 6.4.
            height: height of the figure in inches. Defaults to 4.8.
            title: title of the figure. Defaults to `None`.
            **kwargs: parameters for `draw`.

        Returns:
            The dataset, with the new figure.

        """
        figures = utilities.import_tool('matplotlib.figure')
        figure = figures.Figure(
            figsize = (width, height), layout = 'constrained')
        self.draw(item, figure, **kwargs)
        if title is not None:
            figure.suptitle(title)
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
        count = max(len(columns), 1)
        width = min(count, 4)
        grid = figure.subplots(
            math.ceil(count / width), width, squeeze = False)
        for axes, column in zip(grid.flat, columns, strict = False):
            values = item.data[column].dropna()
            if pd.api.types.is_bool_dtype(values.dtype):
                # `matplotlib` cannot put booleans in bins.
                values = values.astype(int)
            axes.hist(values, bins = bins)
            axes.set_title(str(column), fontsize = 'small')
        for axes in list(grid.flat)[len(columns):]:
            axes.set_visible(False)


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
        names = [source] if source else list(_IMPORTANCE_LABELS)
        found = next((n for n in names if n in item.tables), None)
        if found is None:
            found = 'feature_importance'
            table = evaluators.FeatureImportance().evaluate(item)
        else:
            table = item.tables[found]
        top = table['importance'].head(limit).iloc[::-1]
        axes = figure.subplots()
        axes.barh([str(i) for i in top.index], top.to_numpy())
        axes.set_xlabel(_IMPORTANCE_LABELS.get(found, 'importance'))


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
class SurvivalCurves(Plot):
    """The share of rows without an event over time (Kaplan-Meier curves).

    The label is the time until the event. Set "event" to the column that is
    1 if the event happened (by default, every event was observed) and
    "group" to a column to draw a curve for each group. The shading is the
    95% confidence interval.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        event: str | None = None,
        group: str | None = None,
        **kwargs: Any) -> None:
        """Draws the survival curves.

        Args:
            item: the dataset to draw. Its label is the time.
            figure: the figure to draw on.
            event: name of the column that is 1 if the event happened.
                Defaults to `None`, which means every event was observed.
            group: name of a column of groups. Defaults to `None`, which draws
                one curve.
            **kwargs: not used.

        """
        axes = figure.subplots()
        for fitter in describers.survival_curves(item, event, group).values():
            fitter.plot_survival_function(ax = axes)
        axes.set_xlabel(str(item.label))
        axes.set_ylabel('share without the event')


""" Private Functions """


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
    metrics.PredictionErrorDisplay.from_predictions(
        item.y.loc[item.predictions.index],
        item.predictions,
        kind = kind,
        ax = axes)
