"""Plots of a decision tree, drawn with dtreeviz.

Contents:
    TreeFeatureSpace: how a decision tree divides one or two features.
    TreeLeafDistributions: the label of the training rows in each leaf.
    TreeLeafPurity: the impurity of each leaf of a decision tree.
    TreeLeafSizes: the number of training rows in each leaf.
    TreePlot: the splits of a decision tree.
    TreePredictionPath: how much each feature decides one row's path down a
        decision tree.

"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any, ClassVar

import numpy as np

from .. import base, utilities
from ._base import Plot, _classes, _lent, _model, _top_features, _unwrapped

# The warning of scikit-learn when a model that learned from named features
# is given an array.
_UNNAMED_WARNING: str = 'X does not have valid feature names'


@dataclasses.dataclass
class TreeFeatureSpace(Plot):
    """How a decision tree divides one or two features.

    dtreeviz's feature space of a `decision_tree`, or of one tree (numbered
    `tree`) of a `random_forest`, `extra_trees`, or `adaboost` model: the
    training rows across one or two `features` (by default,
    the two most important), with the tree's splits on them as lines. For a
    classifier, the rows are colored by class; for a regression, by the
    label, with the mean of the label in each region.

    Wraps:
        - [`dtreeviz.trees.DTreeVizAPI.ctree_feature_space`](https://github.com/parrt/dtreeviz)
          from dtreeviz, for classes.
        - [`dtreeviz.trees.DTreeVizAPI.rtree_feature_space`](https://github.com/parrt/dtreeviz)
          from dtreeviz, for values.

    """

    starts_at_zero: ClassVar[str] = ''

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        features: Sequence[str] | None = None,
        tree: int = 0,
        **kwargs: Any) -> None:
        """Draws the training rows and the tree's splits across `features`.

        Args:
            item: the dataset with a fitted tree model.
            figure: the figure to draw on.
            features: one or two features to draw. Defaults to `None`, which
                uses the two most important (see `importance_plot`).
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            **kwargs: not used.

        """
        viz, names = _tree_viz(item, tree, self.name)
        chosen = _top_features(
            item, names, features, self.name, counts = (1, 2))
        method = (
            viz.ctree_feature_space if item.task == 'classify'
            else viz.rtree_feature_space)
        _draw_tree(figure, method, features = [str(c) for c in chosen])


@dataclasses.dataclass
class TreeLeafDistributions(Plot):
    """The label of the training rows in each leaf of a decision tree.

    dtreeviz's distributions of the leaves of a tree (see `tree_leaf_sizes`
    for the trees it draws): for a classifier, the number of training rows of
    each class in each leaf, as stacked bars; for a regression, the label of
    each training row in each leaf, as points, with the leaf's prediction as
    a line.

    Wraps:
        - [`dtreeviz.trees.DTreeVizAPI.ctree_leaf_distributions`](https://github.com/parrt/dtreeviz)
          from dtreeviz, for classes.
        - [`dtreeviz.trees.DTreeVizAPI.rtree_leaf_distributions`](https://github.com/parrt/dtreeviz)
          from dtreeviz, for values.

    """

    label_axes: ClassVar[str] = 'x'

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        tree: int = 0,
        **kwargs: Any) -> None:
        """Draws the label of the training rows in each leaf.

        Args:
            item: the dataset with a fitted tree model.
            figure: the figure to draw on.
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            **kwargs: not used.

        """
        viz, _ = _tree_viz(item, tree, self.name, boosters = True)
        method = (
            viz.ctree_leaf_distributions if item.task == 'classify'
            else viz.rtree_leaf_distributions)
        _draw_tree(figure, method)


@dataclasses.dataclass
class TreeLeafPurity(Plot):
    """The impurity of each leaf of a decision tree.

    dtreeviz's purity of the leaves of a `decision_tree`, or of one tree
    (numbered `tree`) of a `random_forest`, `extra_trees`, or `adaboost`
    model: the value of the tree's criterion in each leaf (such as
    its Gini impurity, for a classifier, or the squared error of its
    prediction, for a regression). A leaf of rows that are all alike has an
    impurity of 0.

    Wraps:
        [`dtreeviz.trees.DTreeVizAPI.leaf_purity`](https://github.com/parrt/dtreeviz)
        from dtreeviz.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        tree: int = 0,
        **kwargs: Any) -> None:
        """Draws the impurity of each leaf.

        Args:
            item: the dataset with a fitted tree model.
            figure: the figure to draw on.
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            **kwargs: not used.

        """
        viz, _ = _tree_viz(item, tree, self.name)
        _draw_tree(figure, viz.leaf_purity)


@dataclasses.dataclass
class TreeLeafSizes(Plot):
    """The number of training rows in each leaf of a decision tree.

    dtreeviz's sizes of the leaves of a `decision_tree`, of one tree (numbered
    `tree`) of a `random_forest`, `extra_trees`, or `adaboost` model, or of
    one tree of an `xgboost` or `lightgbm` model. A tree with many tiny
    leaves has learned rows by heart (it overfits).

    Wraps:
        [`dtreeviz.trees.DTreeVizAPI.leaf_sizes`](https://github.com/parrt/dtreeviz)
        from dtreeviz.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        tree: int = 0,
        **kwargs: Any) -> None:
        """Draws the number of training rows in each leaf.

        Args:
            item: the dataset with a fitted tree model.
            figure: the figure to draw on.
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            **kwargs: not used.

        """
        viz, _ = _tree_viz(item, tree, self.name, boosters = True)
        _draw_tree(figure, viz.leaf_sizes)


@dataclasses.dataclass
class TreePlot(Plot):
    """The splits of a decision tree (or of one tree of a forest).

    scikit-learn's tree plot of a `decision_tree`, or of the tree numbered
    `tree` of a `random_forest`, `extra_trees`, `gradient_boosting`, or
    `adaboost` model, down to `depth` levels (deep trees are hard to read).
    Each box is a split, with the share of rows in each class (or the mean
    label) shaded.

    Wraps:
        [`sklearn.tree.plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
        from scikit-learn.

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
            IncompatibleTechniqueError: if the model is not a scikit-learn
                tree (or trees).

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
            raise base.IncompatibleTechniqueError(message)
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
class TreePredictionPath(Plot):
    """How much each feature decides one row's path down a decision tree.

    dtreeviz's importance of the features for one row, in a `decision_tree`
    or in one tree (numbered `tree`) of a `random_forest`, `extra_trees`, or
    `adaboost` model: the row goes down the tree from its root to a
    leaf, and each feature's importance is how much the splits on it along
    that path reduce the impurity, as the tree's `feature_importances_` are
    for every row. The row is `row`, or the first row of the test data (or
    of all the data, if it is not split).

    Wraps:
        [`dtreeviz.trees.DTreeVizAPI.instance_feature_importance`](https://github.com/parrt/dtreeviz)
        from dtreeviz.

    """

    def draw(
        self,
        item: base.Dataset,
        figure: Any,
        *,
        row: Hashable | None = None,
        tree: int = 0,
        **kwargs: Any) -> None:
        """Draws the importance of each feature on the path of `row`.

        Args:
            item: the dataset with a fitted tree model.
            figure: the figure to draw on.
            row: index label of the row. Defaults to `None`, which uses the
                first row of the test data (or of all the data).
            tree: number of the tree to draw, for a model of many trees.
                Defaults to 0.
            **kwargs: not used.

        """
        viz, names = _tree_viz(item, tree, self.name)
        if row is None:
            row = (item.x_test if item.is_split else item.x).index[0]
        _draw_tree(
            figure,
            viz.instance_feature_importance,
            item.data.loc[[row], names].to_numpy(dtype = float)[0])


""" Private Functions """


def _draw_tree(
    figure: Any,
    method: Callable[..., Any],
    *args: Any,
    **kwargs: Any) -> None:
    """Draws a plot of dtreeviz on new axes of `figure`, in the style.

    The figure is lent to pyplot while dtreeviz draws (see `_lent`), and the
    plot is given the font and sizes of text of the style (see
    `_tree_viz_text`).

    Args:
        figure: the figure to draw on.
        method: the method of dtreeviz's adapter that draws the plot.
        *args: positional arguments for `method`.
        **kwargs: keyword arguments for `method`.

    """
    with _lent(figure), warnings.catch_warnings():
        # dtreeviz passes the rows to the tree as arrays, without the names
        # of the features that it learned from.
        warnings.filterwarnings('ignore', _UNNAMED_WARNING, UserWarning)
        method(
            *args, ax = figure.subplots(), **_tree_viz_text(method), **kwargs)


def _tree_viz(
    item: base.Dataset,
    tree: int,
    use: str,
    *,
    boosters: bool = False) -> tuple[Any, list[str]]:
    """Returns dtreeviz's adapter of the dataset's tree, with the training rows.

    The tree is a scikit-learn decision tree, the tree numbered `tree` of a
    scikit-learn forest (or of `adaboost`), or (if `boosters`) the tree
    numbered `tree` of an xgboost or lightgbm model. The adapter is given the
    training rows, with the classes numbered in the order of the model's
    classes.

    Wraps:
        [`dtreeviz.model`](https://github.com/parrt/dtreeviz)
        from dtreeviz.

    Args:
        item: the dataset with a fitted tree model.
        tree: number of the tree, for a model of many trees.
        use: the plot that draws the tree (for the message of an error).
        boosters: whether the plot can draw a tree of xgboost or lightgbm
            (dtreeviz draws only some plots of them). Defaults to `False`.

    Raises:
        IncompatibleTechniqueError: if the model is not a tree, or trees,
            that dtreeviz draws for the plot.

    Returns:
        The adapter, and the names of the model's features.

    """
    dtreeviz = utilities.import_tool('dtreeviz')
    fitted = _model(item)
    model = _unwrapped(item)
    names = list(
        utilities.feature_names(model, item.data.columns) or item.features)
    chosen, index = model, None
    booster = hasattr(model, 'booster_') or hasattr(model, 'get_booster')
    if booster and not boosters:
        message = (
            f'{use} needs a decision tree or a forest of them from '
            f'scikit-learn: dtreeviz cannot draw it for xgboost or lightgbm')
        raise base.IncompatibleTechniqueError(message)
    if hasattr(model, 'booster_'):
        # lightgbm's models are drawn from their boosters.
        chosen, index = model.booster_, tree
    elif hasattr(model, 'get_booster'):
        index = tree
    elif hasattr(model, 'estimators_') and not hasattr(model, 'train_score_'):
        # The trees of scikit-learn's gradient boosting fit the errors of the
        # trees before them, not the label, so they are not drawn.
        chosen = list(np.ravel(np.asarray(
            model.estimators_, dtype = object)))[tree]
    trees = index is not None or (
        hasattr(chosen, 'tree_')
        and (item.task != 'classify' or hasattr(chosen, 'classes_')))
    if not trees:
        message = (
            f'{use} needs a decision tree, a forest of them, or an xgboost or '
            f'lightgbm model')
        raise base.IncompatibleTechniqueError(message)
    y = item.y_train
    classes = None
    if item.task == 'classify':
        classes = list(getattr(fitted, 'classes_', _classes(item)))
        y = y.map({c: i for i, c in enumerate(classes)})
    viz = dtreeviz.model(
        chosen,
        item.x_train[names],
        y.to_numpy(),
        tree_index = index,
        feature_names = [str(n) for n in names],
        target_name = str(item.label),
        class_names = None if classes is None else [str(c) for c in classes])
    return viz, names


def _tree_viz_text(method: Callable[..., Any]) -> dict[str, Any]:
    """Returns the font and sizes of text of the style, for a dtreeviz plot.

    dtreeviz sets the font (Arial) and sizes of its text itself, so the
    style's are passed to each of its plots, as the parameters it accepts.

    Args:
        method: the method of dtreeviz's adapter that draws the plot.

    Returns:
        The parameters for the font and sizes of text that `method` accepts.

    """
    matplotlib = importlib.import_module('matplotlib')
    fonts = importlib.import_module('matplotlib.font_manager')
    settings = matplotlib.rcParams
    size = fonts.FontProperties(size = settings['font.size']).get_size()
    ticks = fonts.FontProperties(
        size = settings['xtick.labelsize']).get_size()
    text = {
        'fontname': fonts.FontProperties(
            family = settings['font.family']).get_name(),
        'fontsize': size,
        'label_fontsize': size,
        'ticks_fontsize': ticks}
    return utilities.accepted_parameters(method, text)
