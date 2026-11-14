"""Plots of the data, which explorers draw.

Contents:
    AssociationRulesPlot: the support and confidence of each association rule,
        colored by lift.
    BoxPlots: each numeric feature as a box, split by class.
    CorrelationHeatmap: the correlations between numeric columns.
    CountPlots: the count of each value of each categorical feature.
    DistributionPlots: the distribution of every column.
    FrequencyPlots: the share of the rows with each value of each column.
    Histograms: the distribution of each numeric feature.
    LabelPlot: the distribution of the label.
    MissingBars: the share of each column's values that are missing.
    MissingHeatmap: where values are missing.
    PairPlot: each pair of numeric features against each other.
    SurvivalCurves: the share of rows without an event over time.

"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, describers, utilities
from ._base import Plot, _figure_legend, _grid, _match_ranges, _table


@dataclasses.dataclass
class AssociationRulesPlot(Plot):
    """The support and confidence of each association rule, colored by lift.

    Draws the table made by `association_rules` (which is applied if it has
    not been, or the table named by "source"): a dot for each rule, at the
    share of the rows that have all of its items (its support) and the share
    of the rows with its antecedents that also have its consequents (its
    confidence), colored by its lift. The strongest rules are at the top
    right, in the brightest colors.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws a dot for each rule.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            source: name of the table of rules. Defaults to `None`, which
                uses the table named "association_rules", or makes it.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'association_rules',
            describers.AssociationRules().describe)
        axes = figure.subplots()
        if len(table):
            points = axes.scatter(
                table['support'], table['confidence'], c = table['lift'],
                cmap = 'viridis', s = 12)
            figure.colorbar(points, ax = axes, label = 'lift')
        axes.set_xlabel('support')
        axes.set_ylabel('confidence')


@dataclasses.dataclass
class BoxPlots(Plot):
    """Each numeric feature as a box, split by class.

    Draws seaborn's box plot of each numeric feature (or of `columns`), in a
    grid. The boxes are split by the column `by`, or by the classes of the
    label (for classification), or by the dataset's first group, unless
    `split` is `False`.

    Wraps:
        [`seaborn.boxplot`](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
        from seaborn.

    """

    starts_at_zero: ClassVar[str] = ''

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        by: str | None = None,
        split: bool = True,
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
            split: whether to split the boxes. Defaults to `True`. If it is
                `False`, each column is one box, and `by` is not used.
            limit: most columns to draw. Defaults to 16.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        columns = list(item.numerics if columns is None else columns)[:limit]
        if split:
            data, by, order = _split(item, by)
        else:
            data, by, order = item.data, None, None
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            seaborn.boxplot(
                data = data, x = by, y = column, order = order, ax = axes)
            axes.set_title(str(column), fontsize = 'small')
            axes.set_ylabel('')


@dataclasses.dataclass
class CorrelationHeatmap(Plot):
    """The correlations between the numeric columns as a heatmap.

    Wraps:
        [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        from seaborn.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        method: describers.CorrelationMethod = 'pearson',
        *,
        source: str | None = None,
        **kwargs: Any) -> None:
        """Draws the correlations.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            method: "pearson", "spearman", or "kendall". Defaults to
                "pearson".
            source: name of a table made by `correlations` to draw, in place
                of finding the correlations with `method`. Defaults to
                `None`.
            **kwargs: not used.

        """
        seaborn = utilities.import_tool('seaborn')
        table = (
            describers.Correlations().describe(item, method = method)
            if source is None else _table(item, source))
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

    Wraps:
        [`seaborn.countplot`](https://seaborn.pydata.org/generated/seaborn.countplot.html)
        from seaborn.

    """

    shared_axes: ClassVar[str] = 'x'

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
class DistributionPlots(Plot):
    """The distribution of every column, in a grid.

    Each column (or each of `columns`) is a panel: a histogram of a column of
    numbers or dates, and bars of the `top` most common values of any other
    column (with missing values left out). Unlike `histograms` and
    `count_plots`, it draws the label and the groups too, and it does not
    split the rows by class.

    Wraps:
        [`matplotlib.axes.Axes.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)
        from matplotlib.

    """

    # Wider than most plots, so that the numbers of each panel fit.
    size: ClassVar[tuple[float, float]] = (9.6, 6.4)
    starts_at_zero: ClassVar[str] = 'y'
    shared_axes: ClassVar[str] = 'y'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        columns: Sequence[str] | None = None,
        bins: int = 20,
        top: int = 10,
        limit: int = 16,
        **kwargs: Any) -> None:
        """Draws the distribution of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            columns: columns to draw. Defaults to `None`, which draws every
                column of the data.
            bins: number of bins in each histogram. Defaults to 20.
            top: most values of each other column to draw. Defaults to 10.
            limit: most columns to draw. Defaults to 16.
            **kwargs: not used.

        """
        dates = importlib.import_module('matplotlib.dates')
        ticker = importlib.import_module('matplotlib.ticker')
        columns = list(item.data.columns if columns is None else columns)
        columns = columns[:limit]
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            values = item.data[column].dropna()
            kind = base._kind_of(values)
            if kind == 'dates':
                axes.hist(values, bins = bins)
                # A small panel has room for only a few dates.
                locator = dates.AutoDateLocator(minticks = 2, maxticks = 3)
                axes.xaxis.set_major_locator(locator)
                axes.xaxis.set_major_formatter(
                    dates.ConciseDateFormatter(locator))
            elif kind == 'numerics':
                axes.hist(values, bins = bins)
                axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins = 4))
                # Large numbers would run into each other side by side.
                axes.tick_params(axis = 'x', labelrotation = 90)
            else:
                counts = values.astype(str).value_counts().head(top)
                axes.bar(list(counts.index), counts.to_numpy())
                axes.tick_params(axis = 'x', labelrotation = 90)
            axes.set_title(str(column), fontsize = 'small')


@dataclasses.dataclass
class FrequencyPlots(Plot):
    """The share of the rows with each value of each column, in a grid.

    Draws the table made by `frequencies` (which is applied if it has not
    been, or the table named by "source"): a panel for each column, with a
    bar for each of its `top` most common values (missing values included).

    """

    shared_axes: ClassVar[str] = 'x'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        top: int = 10,
        limit: int = 16,
        **kwargs: Any) -> None:
        """Draws the share of each value of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "frequencies", or makes it.
            top: most values of each column to draw. Defaults to 10.
            limit: most columns to draw. Defaults to 16.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'frequencies', describers.Frequencies().describe)
        columns = list(dict.fromkeys(table['column']))[:limit]
        for axes, column in zip(
            _grid(figure, len(columns)), columns, strict = True):
            shares = table.loc[table['column'] == column].nlargest(
                top, 'share').iloc[::-1]
            axes.barh(shares['value'].astype(str), shares['share'])
            axes.set_title(str(column), fontsize = 'small')
        if columns:
            figure.supxlabel('share of the rows', fontsize = 'small')


@dataclasses.dataclass
class Histograms(Plot):
    """The distribution of each numeric feature, in a grid.

    Wraps:
        [`matplotlib.axes.Axes.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)
        from matplotlib.

    """

    starts_at_zero: ClassVar[str] = 'y'
    shared_axes: ClassVar[str] = 'y'

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
class LabelPlot(Plot):
    """The distribution of the label: a bar for each class, or a histogram.

    Wraps:
        - [`seaborn.countplot`](https://seaborn.pydata.org/generated/seaborn.countplot.html)
          from seaborn, for classes.
        - [`seaborn.histplot`](https://seaborn.pydata.org/generated/seaborn.histplot.html)
          from seaborn, for values.

    """

    label_axes: ClassVar[str] = 'x'

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
class MissingBars(Plot):
    """The share of each column's values that are missing, as bars.

    Draws the table made by `missing_values` (which is applied if it has not
    been, or the table named by "source"): a bar for each column with
    missing values (up to `limit`, most missing first). If no value is
    missing, the figure says so.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        source: str | None = None,
        limit: int = 30,
        **kwargs: Any) -> None:
        """Draws the share of missing values of each column.

        Args:
            item: the dataset to draw.
            figure: the figure to draw on.
            source: name of the table to draw. Defaults to `None`, which uses
                the table named "missing_values", or makes it.
            limit: most columns to draw. Defaults to 30.
            **kwargs: not used.

        """
        table = _table(
            item, source, 'missing_values',
            describers.MissingValues().describe)
        shares = table['share'].loc[table['share'] > 0]
        shares = shares.sort_values(
            ascending = False, kind = 'stable').head(limit).iloc[::-1]
        axes = figure.subplots()
        axes.barh([str(c) for c in shares.index], shares.to_numpy())
        axes.set_xlabel('share of the rows that are missing')
        if shares.empty:
            axes.text(
                0.5, 0.5, 'no values are missing', ha = 'center',
                va = 'center', transform = axes.transAxes)


@dataclasses.dataclass
class MissingHeatmap(Plot):
    """Where values are missing, as a heatmap of the rows and columns.

    Each column of the heatmap is a column of the data and each line a row
    (up to `rows` of them, evenly spaced, in order), dark where the value is
    missing, so that patterns (such as columns that are missing together)
    stand out. Columns without missing values are left out unless `columns`
    names them.

    Wraps:
        [`seaborn.heatmap`](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
        from seaborn.

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

    Wraps:
        - [`seaborn.scatterplot`](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)
          from seaborn, off the diagonal.
        - [`seaborn.histplot`](https://seaborn.pydata.org/generated/seaborn.histplot.html)
          from seaborn, on the diagonal.

    """

    size: ClassVar[tuple[float, float]] = (8.0, 8.0)
    starts_at_zero: ClassVar[str] = ''

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
        # Each feature has one range: on the x axes of its column, and on the
        # y axes of its row (but for the histogram, which shows densities).
        for k in range(count):
            _match_ranges([
                *((grid[i, k], 'x') for i in range(count)),
                *((grid[k, j], 'y') for j in range(count) if j != k)])
        if count > 1:
            _figure_legend(figure, grid[0, 1], by)


@dataclasses.dataclass
class SurvivalCurves(Plot):
    """The share of rows without an event over time (Kaplan-Meier curves).

    The label is the time until the event. Set "event" to the column that is
    1 if the event happened (by default, every event was observed) and
    "group" to a column to draw a curve for each group. The shading is the
    95% confidence interval. Set "at_risk" to add a table of the number of
    rows at risk (whose time is at least each tick's) below the curves, as
    papers in medicine usually do.

    Wraps:
        [`statsmodels.duration.survfunc.SurvfuncRight`](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.SurvfuncRight.html)
        from statsmodels.

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
