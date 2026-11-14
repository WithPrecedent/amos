"""The base class of plots, and what plots share.

Contents:
    Plot: base class for techniques that draw a figure.

"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import enum
import importlib
import math
import re
from collections.abc import Callable, Generator, Sequence
from typing import Any, ClassVar

import pandas as pd

from .. import base, evaluators, models, options, utilities

# Labels of the axis of an `importance_plot`, by the table it draws, in the
# order that the tables are looked for.
_IMPORTANCE_LABELS: dict[str, str] = {
    'shap_importance': 'mean absolute SHAP value',
    'permutation_importance': 'drop in score when the feature is shuffled',
    'feature_importance': "the model's importance"}


# A color of the cycle of the style, such as "C0" (the first).
_CYCLE_COLOR: re.Pattern[str] = re.compile(r'C\d+')


# Settings of `matplotlib` for the sizes of text.
_FONT_SIZES: tuple[str, ...] = (
    'font.size', 'axes.labelsize', 'axes.titlesize', 'figure.labelsize',
    'figure.titlesize', 'legend.fontsize', 'legend.title_fontsize',
    'xtick.labelsize', 'ytick.labelsize')


class _FromOptions(enum.Enum):
    """Marks an argument whose default is read from `options` when it is used.

    `None` is a value of its own for the style and colors of a figure (no
    style), so a missing argument needs another marker.

    """

    VALUE = 'from options'


# The default `style` and `colors` of a figure, which are those of
# `options._PLOT_STYLE` and `options._PLOT_COLORS` when it is drawn.
_FROM_OPTIONS: _FromOptions = _FromOptions.VALUE


@dataclasses.dataclass
class Plot(base.Operation, abc.ABC):
    """Base class for techniques that draw a figure.

    A subclass writes a `draw` method, which draws on a `matplotlib` figure.
    The figure is stored in the dataset's `figures` under the technique's
    name.

    Wraps:
        - [`matplotlib.figure.Figure`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html)
          from matplotlib, to draw on.
        - [`matplotlib.style.use`](https://matplotlib.org/stable/api/style_api.html#matplotlib.style.use)
          from matplotlib, for the style of the figure.
        - [SciencePlots](https://github.com/garrettj403/SciencePlots),
          for its styles (such as "science", "nature", and "bright").
        - [`matplotlib.pyplot.xkcd`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xkcd.html)
          from matplotlib, for the "xkcd" style.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
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
    # The axes ("x", "y", or both) whose ranges start at 0 (or end at 0, if
    # their values are all negative), unless the plot is given
    # "start_at_zero". A plot leaves out an axis whose values have no natural
    # zero, such as the values of a feature.
    starts_at_zero: ClassVar[str] = 'xy'
    # The axes ("x", "y", or both) that measure the same thing in every
    # panel of the plot, so that the panels share their ranges.
    shared_axes: ClassVar[str] = ''
    # The axes ("x", "y", or both) that show the label (or the model's
    # predictions of it). A regression's label has no natural zero, so for a
    # regression these axes do not start at 0, unless the plot is given
    # "start_at_zero".
    label_axes: ClassVar[str] = ''
    model_specific_setting: ClassVar[str] = 'plots'

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

    def apply(
        self,
        item: base.Dataset,
        width: float | None = None,
        height: float | None = None,
        title: str | None = None,
        *,
        style: str | Sequence[str] | _FromOptions | None = _FROM_OPTIONS,
        colors: str | _FromOptions | None = _FROM_OPTIONS,
        latex: bool = False,
        start_at_zero: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Draws a figure in a style and adds it to the figures of `item`.

        The figure is drawn by `make_figure`, which describes the parameters.

        Args:
            item: the dataset to draw.
            width: width of the figure in inches. Defaults to `None`, which
                uses the plot's usual width for the style.
            height: height of the figure in inches. Defaults to `None`, which
                uses the plot's usual height for the style.
            title: title of the figure. Defaults to `None`.
            style: names of `matplotlib` or SciencePlots styles, or "xkcd".
                Defaults to `options._PLOT_STYLE` (read when the figure is
                drawn).
            colors: name of a color cycle. Defaults to
                `options._PLOT_COLORS` (read when the figure is drawn).
            latex: whether to set the text with LaTeX. Defaults to `False`.
            start_at_zero: the axes whose ranges start at 0. Defaults to
                `None`, which uses the plot's `starts_at_zero`.
            **kwargs: parameters for `draw`.

        Returns:
            The dataset, with the new figure.

        Raises:
            ValueError: if `start_at_zero` names an axis that is not "x" or
                "y".

        """
        item.figures[self.name] = self.make_figure(
            item,
            width,
            height,
            title,
            style = style,
            colors = colors,
            latex = latex,
            start_at_zero = start_at_zero,
            **kwargs)
        item.record(self.name, figure = self.name)
        return item

    def make_figure(
        self,
        item: base.Dataset,
        width: float | None = None,
        height: float | None = None,
        title: str | None = None,
        *,
        style: str | Sequence[str] | _FromOptions | None = _FROM_OPTIONS,
        colors: str | _FromOptions | None = _FROM_OPTIONS,
        latex: bool = False,
        start_at_zero: str | None = None,
        **kwargs: Any) -> Any:
        """Returns a figure drawn in a style, without adding it to `item`.

        `apply` stores the figure in the dataset's `figures`, and explorers
        and critics use this to draw the figures of their tables (see
        `base.Visualizer`). The ranges of the axes in `starts_at_zero` are
        widened to start at 0, and the panels of a plot share the ranges of
        its `shared_axes`. An axis that is not a linear scale of numbers
        (such as one of names, dates, or a logarithmic scale), and an axis of
        an image (such as a heatmap), are left as they are.

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
                Defaults to `options._PLOT_STYLE` (read when the figure is
                drawn).
            colors: name of a color cycle (a style that only sets colors,
                such as SciencePlots' "bright", "vibrant", "muted", or
                "high-contrast"), applied after `style`, or `None` (or
                "none") for the colors of `style`. Defaults to
                `options._PLOT_COLORS` (read when the figure is drawn).
            latex: whether to set the text with LaTeX, which must be
                installed. Defaults to `False`, in which case `matplotlib`
                sets the text (and math), whatever `style` says.
            start_at_zero: the axes ("x", "y", "xy", or "" for neither) whose
                ranges start at 0. Defaults to `None`, which uses the plot's
                `starts_at_zero`, without its `label_axes` for a regression.
            **kwargs: parameters for `draw`.

        Returns:
            The `matplotlib.figure.Figure`.

        Raises:
            ValueError: if `start_at_zero` names an axis that is not "x" or
                "y".

        """
        if start_at_zero is not None and set(start_at_zero) - set('xy'):
            message = (
                f'start_at_zero must be "x", "y", "xy", or "", not '
                f'"{start_at_zero}"')
            raise ValueError(message)
        if isinstance(style, _FromOptions):
            style = options._PLOT_STYLE
        if isinstance(colors, _FromOptions):
            colors = options._PLOT_COLORS
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
            if start_at_zero is None:
                start_at_zero = self.starts_at_zero
                if item.task == 'regress':
                    start_at_zero = ''.join(
                        a for a in start_at_zero if a not in self.label_axes)
            _start_at_zero(figure, start_at_zero)
            _share_ranges(figure, self.shared_axes)
            # `matplotlib` makes some parts of a figure (such as most of its
            # ticks) only when the figure is drawn, and finds the colors of
            # lines in the cycle (such as "C0") then too, so both are done
            # here, in the style, rather than when it is shown or saved.
            _fix_colors(figure)
            _hide_minor_ticks(figure)
            figure.draw_without_rendering()
        return figure


""" Private Functions """


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


def _do_nothing(*args: Any, **kwargs: Any) -> None:
    """Does nothing, in place of a method that should not run.

    Such as a figure's `set_size_inches` or `matplotlib.pyplot.show`.

    Args:
        *args: not used.
        **kwargs: not used.

    """


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


@contextlib.contextmanager
def _lent(figure: Any) -> Generator[Any, None, None]:
    """Lends `figure` to `matplotlib.pyplot` while a tool draws on it.

    Some tools (such as shap, mlxtend, and dtreeviz) draw with
    `matplotlib.pyplot`, which only draws on figures that it manages. In this
    context, `figure` is pyplot's current figure, with a canvas that draws
    images but never opens a window, and `pyplot.show` (which dtreeviz calls
    after it draws) does nothing. Afterwards, pyplot forgets `figure` and any
    other figure made in the context (which can still be used), and its
    interactive mode (which some shap plots turn off) and `show` are
    restored.

    Args:
        figure: the figure to lend.

    Yields:
        Any: the `matplotlib.pyplot` module.

    """
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
    show = pyplot.show
    manager = agg.FigureCanvasAgg.new_manager(
        figure, max(before, default = 0) + 1)
    figures.set_active(manager)
    setattr(pyplot, 'show', _do_nothing)  # noqa: B010
    try:
        yield pyplot
    finally:
        setattr(pyplot, 'show', show)  # noqa: B010
        for number in set(pyplot.get_fignums()) - before:
            made = figures.get_fig_manager(number).canvas.figure
            pyplot.close(number)
            # Without pyplot's canvas, the figure cannot open a window.
            bases.FigureCanvasBase(made)
        matplotlib.interactive(interactive)


def _linear_axes(figure: Any, name: str) -> list[Any]:
    """Returns the axes of `figure` whose axis `name` is a range of numbers.

    Args:
        figure: a drawn figure.
        name: "x" or "y".

    Returns:
        The visible axes of the figure whose axis `name` is a linear scale of
            numbers. Colorbars, axes with an image or contours (such as a
            heatmap or a map of the regions of a classifier), and axes whose
            axis has units (such as names or dates) or ticks at fixed
            positions (such as the names of the features in a SHAP plot)
            are left out.

    """
    contour = importlib.import_module('matplotlib.contour')
    ticker = importlib.import_module('matplotlib.ticker')
    found = []
    for axes in figure.axes:
        axis = getattr(axes, f'{name}axis')
        if (
            not axes.get_visible()
            or not axes.axison
            or axes.images
            or any(
                isinstance(c, contour.ContourSet) for c in axes.collections)
            or getattr(axes, '_colorbar', None) is not None
            or axis.get_scale() != 'linear'
            or axis.get_converter() is not None
            or isinstance(axis.get_major_locator(), ticker.FixedLocator)):
            continue
        found.append(axes)
    return found


def _match_ranges(panels: Sequence[tuple[Any, str]]) -> None:
    """Gives `panels` the range that holds all of their ranges.

    Args:
        panels: pairs of axes and the name of one of their axes ("x" or
            "y"), which all measure the same thing. They are changed in
            place.

    """
    if len(panels) < 2:  # noqa: PLR2004
        return
    limits = [getattr(a, f'get_{name}lim')() for a, name in panels]
    lower = min(min(pair) for pair in limits)
    upper = max(max(pair) for pair in limits)
    for (axes, name), (start, end) in zip(panels, limits, strict = True):
        # An inverted axis stays inverted.
        shared = (upper, lower) if start > end else (lower, upper)
        getattr(axes, f'set_{name}lim')(shared)


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


def _share_ranges(figure: Any, names: str) -> None:
    """Gives the panels of `figure` the same ranges on the axes in `names`.

    Args:
        figure: a drawn figure. It is changed in place.
        names: the axes ("x", "y", or both) that measure the same thing in
            every panel.

    """
    for name in names:
        _match_ranges([(a, name) for a in _linear_axes(figure, name)])


def _start_at_zero(figure: Any, names: str) -> None:
    """Widens the ranges of the axes in `names` to reach 0.

    A range of positive numbers is widened to start at 0, and a range of
    negative numbers to end at 0. A range that holds 0 is left as it is.

    Args:
        figure: a drawn figure. It is changed in place.
        names: the axes ("x", "y", or both) to widen.

    """
    for name in names:
        for axes in _linear_axes(figure, name):
            start, end = getattr(axes, f'get_{name}lim')()
            lower, upper = min(start, end), max(start, end)
            if lower <= 0 <= upper:
                continue
            lower, upper = min(lower, 0.0), max(upper, 0.0)
            # An inverted axis stays inverted.
            widened = (upper, lower) if start > end else (lower, upper)
            getattr(axes, f'set_{name}lim')(widened)


@contextlib.contextmanager
def _styled(
    style: str | Sequence[str] | None,
    colors: str | None,
    *,
    latex: bool) -> Generator[Any, None, None]:
    """Applies a style to the figures made and drawn in the context.

    The styles are applied over the current settings of `matplotlib`, which
    are put back afterwards. SciencePlots, which adds its styles to those of
    `matplotlib`, is only imported if a style is not one of `matplotlib`'s.

    Wraps:
        - [`matplotlib.style.use`](https://matplotlib.org/stable/api/style_api.html#matplotlib.style.use)
          from matplotlib.
        - [SciencePlots](https://github.com/garrettj403/SciencePlots),
          for its styles (such as "science", "nature", and "bright").
        - [`matplotlib.pyplot.xkcd`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xkcd.html)
          from matplotlib, for the "xkcd" style.

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


def _table(
    item: base.Dataset,
    source: str | None,
    default: str | None = None,
    make: Callable[[base.Dataset], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
    """Returns the table that a plot draws.

    Args:
        item: the dataset.
        source: name of the table in the dataset's `tables`, or `None` for
            the table named `default`.
        default: the name that the technique that makes the table gives it.
            Defaults to `None`.
        make: makes the table (with that technique), if `source` is `None`
            and there is no table named `default`. Defaults to `None`.

    Raises:
        KeyError: if `source` is not the name of a table, or if there is no
            table to draw.

    Returns:
        The table.

    """
    if source is not None:
        if source not in item.tables:
            message = f'there is no table named {source!r} to draw'
            raise KeyError(message)
        return item.tables[source]
    if default is not None and default in item.tables:
        return item.tables[default]
    if make is None:
        message = 'there is no table to draw: name one as "source"'
        raise KeyError(message)
    return make(item)


def _top_features(
    item: base.Dataset,
    names: list[str],
    features: Sequence[str] | None,
    use: str,
    counts: tuple[int, ...] = (2,)) -> list[str]:
    """Returns the features to draw: `features`, or the most important.

    Args:
        item: the dataset with a fitted model.
        names: the model's features.
        features: the features that were asked for, or `None`.
        use: the plot that draws them (for the message of an error).
        counts: the numbers of features that the plot can draw. The most
            important are the largest number. Defaults to (2,).

    Raises:
        ValueError: if the model has too few features, or `features` are not
            the model's or are not a number that the plot can draw.

    Returns:
        The names of the features.

    """
    if features is None:
        try:
            _, table = _importances(item, None)
            ranked = [n for n in table.index if n in names]
        except ValueError:
            ranked = []
        ranked += [n for n in names if n not in ranked]
        features = ranked[:max(counts)]
    chosen = list(features)
    unknown = [f for f in chosen if f not in names]
    if unknown:
        message = f'{use} cannot draw {unknown}: they are not features'
        raise ValueError(message)
    if len(chosen) not in counts:
        wanted = ' or '.join(str(c) for c in counts)
        message = f'{use} needs {wanted} features, not {len(chosen)}'
        raise ValueError(message)
    return chosen


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
