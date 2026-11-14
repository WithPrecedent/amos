"""Plots of the SHAP values of a model, drawn with shap.

Contents:
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

"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from collections.abc import Callable, Generator, Hashable
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, evaluators, options, utilities
from ._base import Plot, _classes, _do_nothing, _lent, _model


@dataclasses.dataclass
class ShapBar(Plot):
    """The mean absolute SHAP value of the most important features.

    This is shap's bar plot. The SHAP values are the ones that
    `shap_importance` found, if it was applied, or are found as it finds them.
    The features past the `limit` are added together in the last bar.

    Wraps:
        [`shap.plots.bar`](https://shap.readthedocs.io/en/latest/generated/shap.plots.bar.html)
        from shap.

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

    Wraps:
        [`shap.plots.beeswarm`](https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html)
        from shap.

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

    Wraps:
        [`shap.plots.decision`](https://shap.readthedocs.io/en/latest/generated/shap.plots.decision.html)
        from shap.

    """

    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = ''

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

    Wraps:
        [`shap.plots.embedding`](https://shap.readthedocs.io/en/latest/generated/shap.plots.embedding.html)
        from shap.

    """

    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = ''

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
        figure.set_size_inches = _do_nothing
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

    Wraps:
        [`shap.plots.force`](https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html)
        from shap.

    """

    size: ClassVar[tuple[float, float]] = (14.0, 4.0)
    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = ''

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

    Wraps:
        [`shap.plots.group_difference`](https://shap.readthedocs.io/en/latest/generated/shap.plots.group_difference.html)
        from shap.

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

    Wraps:
        [`shap.plots.heatmap`](https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html)
        from shap.

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
    Columns that the model needs besides its features (such as the fixed
    effects and cluster of a `fixest` model) are held at each row's own
    values.

    Wraps:
        [`shap.plots.partial_dependence`](https://shap.readthedocs.io/en/latest/generated/shap.plots.partial_dependence.html)
        from shap.

    """

    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = 'y'
    label_axes: ClassVar[str] = 'y'

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
        explained = _explained_rows(item, explanation)
        x = item.data.loc[explained, item.features]
        others = evaluators._other_columns(_model(item), item)
        given = item.data.loc[explained, others] if others else None
        with _pyplot(figure, item.seed) as shap:
            shap.plots.partial_dependence(
                feature,
                _predictor(item, category, given = given),
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

    Wraps:
        [`shap.plots.scatter`](https://shap.readthedocs.io/en/latest/generated/shap.plots.scatter.html)
        from shap.

    """

    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = 'y'

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

    Wraps:
        [`shap.plots.violin`](https://shap.readthedocs.io/en/latest/generated/shap.plots.violin.html)
        from shap.

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

    Wraps:
        [`shap.plots.waterfall`](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)
        from shap.

    """

    scaled: ClassVar[bool] = False
    starts_at_zero: ClassVar[str] = ''

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
        figure.set_size_inches = _do_nothing
        try:
            with _pyplot(figure, item.seed) as shap:
                shap.plots.waterfall(
                    explanation[position], max_display = limit, show = False)
                _labels_beside_arrows(figure)
        finally:
            del figure.set_size_inches


""" Private Functions """


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


def _labels_beside_arrows(figure: Any) -> None:
    """Puts the labels of the arrows of shap's waterfall plot beside them.

    shap writes the SHAP value of each feature inside its arrow, in white, if
    it fits there when shap draws it, and beside the arrow otherwise. But the
    figure's layout then makes room for the names of the features, which
    narrows the arrows, so a label can spill out of its arrow onto the white
    background. Every label is put just past the point of its arrow instead,
    in the arrow's color, and the x axis is widened if it needs to be to hold
    them (with shap's twin axes for the expected value and the prediction,
    which must match it).

    Args:
        figure: a figure with shap's waterfall plot, which pyplot manages.
            It is changed in place.

    """
    patches = importlib.import_module('matplotlib.patches')
    transforms = importlib.import_module('matplotlib.transforms')
    axes = figure.axes[0]
    arrows = {}
    for arrow in axes.patches:
        if isinstance(arrow, patches.FancyArrow):
            ys = arrow.get_xy()[:, 1]
            arrows[round((ys.min() + ys.max()) / 2, 6)] = arrow
    for label in axes.texts:
        arrow = arrows.get(round(label.get_position()[1], 6))
        if arrow is None:
            continue
        xs = arrow.get_xy()[:, 0]
        # shap writes the values that raise the prediction with a plus sign.
        higher = label.get_text().startswith('+')
        label.set_x(xs.max() if higher else xs.min())
        label.set_horizontalalignment('left' if higher else 'right')
        label.set_color(arrow.get_facecolor())
        label.set_transform(transforms.offset_copy(
            axes.transData, fig = figure, x = 5 if higher else -5,
            units = 'points'))
    # Widening the axis makes the arrows shorter, so it is widened until the
    # labels stop moving past its ends.
    for _ in range(5):
        figure.draw_without_rendering()
        renderer = figure.canvas.get_renderer()
        frame = axes.get_window_extent(renderer)
        labels = transforms.Bbox.union(
            [t.get_window_extent(renderer) for t in axes.texts] or [frame])
        lower, upper = axes.get_xlim()
        per_pixel = (upper - lower) / frame.width
        widened = (
            lower - max(0.0, frame.x0 - labels.x0) * per_pixel,
            upper + max(0.0, labels.x1 - frame.x1) * per_pixel)
        if np.allclose(widened, (lower, upper)):
            break
        for twin in axes.get_shared_y_axes().get_siblings(axes):
            twin.set_xlim(widened)


def _predictor(
    item: base.Dataset,
    category: Hashable | None,
    *,
    given: pd.DataFrame | None = None) -> Callable[[pd.DataFrame], np.ndarray]:
    """Returns a function of the features that returns the model's output.

    Args:
        item: the dataset with a fitted model.
        category: class of the label whose probability is returned (for a
            classifier), or `None` for the last class.
        given: other columns that the model needs (such as the fixed effects
            of a `fixest` model) of the rows that the function is given, in
            the same order. Defaults to `None`.

    Returns:
        A function that takes features and returns the model's predictions
            (for regression) or the probability of `category`.

    """
    model = _model(item)

    def joined(x: pd.DataFrame) -> pd.DataFrame:
        """Returns `x` with the columns of `given`, if there are any.

        Args:
            x: features of the rows of `given`, in the same order.

        Returns:
            `x`, with the columns of `given`.

        """
        return x if given is None else evaluators._with_given(x, given)

    if item.task != 'classify' or not hasattr(model, 'predict_proba'):
        return lambda x: np.asarray(model.predict(joined(x)), dtype = float)
    classes = _classes(item)
    position = -1 if category is None else classes.index(category)
    return lambda x: np.asarray(model.predict_proba(joined(x)))[:, position]


@contextlib.contextmanager
def _pyplot(figure: Any, seed: int | None) -> Generator[Any, None, None]:
    """Lends `figure` to `matplotlib.pyplot` while shap draws on it.

    shap draws with `matplotlib.pyplot`, so `figure` is lent to it (see
    `_lent`). Some shap plots also use `numpy`'s global random numbers (to
    jitter dots or to bootstrap), so they are seeded with `seed` in the
    context, and put back as they were afterwards. And shap's bar and
    waterfall plots are colored with the colors of the style (see
    `_shap_colors`), and shap's own colors are put back afterwards. The
    numbers on the axes are written out in full, rather than in scientific
    notation or beside a power of 10 or a number to add to them, unless they
    are too large (see `options._SCIENTIFIC_LIMIT`).

    Args:
        figure: the figure to lend.
        seed: seed for `numpy`'s global random numbers.

    Yields:
        Any: the `shap` module.

    """
    shap = utilities.import_tool('shap')
    matplotlib = importlib.import_module('matplotlib')
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    # shap places the ticks of some plots (such as its bar and waterfall
    # plots) so that `matplotlib` would make billions of minor ticks between
    # them, so styles that show minor ticks (such as "science") cannot.
    minor = {'xtick.minor.visible': False, 'ytick.minor.visible': False}
    plain = {
        'axes.formatter.useoffset': False,
        # No number is this small, so small numbers are never scientific.
        'axes.formatter.limits': [-1000, options._SCIENTIFIC_LIMIT]}
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
        with _lent(figure), matplotlib.rc_context({**minor, **plain}):
            yield shap
    finally:
        if styles is not None:
            styles.set_style(colors)
        np.random.set_state(state)  # noqa: NPY002


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
