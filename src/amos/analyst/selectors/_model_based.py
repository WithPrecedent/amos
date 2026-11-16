"""Selectors that choose features with a model.

Contents:
    ExhaustiveSelection: keeps the best of every set of 1 to 3 features.
    RecursiveElimination: removes the least important feature of a model, again
        and again.
    RecursiveEliminationCV: removes features as `recursive_elimination` does and
        keeps the number that scores best.
    SelectFromModel: keeps the features that a model finds most important.
    SequentialSelection: adds (or removes) features with mlxtend while the score
        improves.
    SequentialSelectionSK: adds (or removes) features with scikit-learn while
        the score improves.

"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from typing import Any, cast

import chrisjen
import numpy as np
import pandas as pd

from ...core import base, utilities
from .. import models
from ._base import Selector


class _ModelBased:
    """Makes a selector one that chooses features with a model.

    The model is "estimator": the name of a model in the library (such as
    "random_forest"), which is built for the task with its default
    parameters, or an estimator, its class, or its import path. By default,
    it is a logistic regression (to classify) or a linear regression, so
    scale the features first. The model must report the importance of each
    feature (as `coef_` or `feature_importances_`) unless the selector
    scores sets of features instead.

    It is a plain class rather than a `Selector`, so that the library does
    not store it as a technique.

    """

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the model.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with "estimator".

        """
        return {
            **parameters,
            'estimator': _build_estimator(item, parameters.get('estimator'))}


class _ModelSearch(_ModelBased):
    """Makes a selector one that scores sets of features by cross-validation.

    The selector fits a model (see `_ModelBased`) to sets of the features, in
    cross-validation on the training rows, and keeps the set with the best
    score. The score is "scoring" (the name of a scikit-learn scorer), or by
    default accuracy or R². The folds ("cv", 5 by default) keep the rows of
    each of the dataset's first groups together, as a model's search does
    (see `models.search_folds`).

    """

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the model, scorer, and folds of the search.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with "estimator", "scoring", and "cv".

        """
        scoring = parameters.get('scoring') or (
            'accuracy' if item.task == 'classify' else 'r2')
        cv = models.search_folds(
            item, item.x_train, item.y_train, parameters.get('cv', 5))
        return {
            **super()._prepare(item, parameters),
            'scoring': scoring,
            'cv': cv}


class _MlxtendSearch(_ModelSearch):
    """Makes a selector one of mlxtend's searches for the best features.

    mlxtend's selectors return an array rather than a `DataFrame`, and name
    the features they keep in their own attributes.

    """

    """ Private Methods """

    def _to_frame(
        self,
        values: Any,
        tool: Any,
        columns: list[str],  # noqa: ARG002
        index: pd.Index) -> pd.DataFrame:
        """Returns the selected features as a `DataFrame`.

        Args:
            values: the output of the selector's `transform` method.
            tool: the fitted selector.
            columns: not used.
            index: index of the data.

        Returns:
            The selected features, with `index` and their names.

        """
        names = getattr(tool, 'k_feature_names_', None)
        if names is None:
            names = tool.best_feature_names_
        # mlxtend returns an array, which is of objects if the features were
        # of several types, so each column is given back its type.
        return pd.DataFrame(
            np.asarray(values), index = index,
            columns = [str(n) for n in names]).infer_objects()


@dataclasses.dataclass
class ExhaustiveSelection(_MlxtendSearch, Selector):
    """Keeps the best of every set of 1 to 3 features (by cross-validation).

    mlxtend's exhaustive search tries every set of `min_features` to
    `max_features` features (1 to 3 by default), so it is slow for many
    features. The model ("estimator"), score ("scoring"), and folds ("cv") are
    described in the introduction of the module. The score of each set is a
    table in the dataset's `tables` as "{name}_search".

    Wraps:
        [`mlxtend.feature_selection.ExhaustiveFeatureSelector`](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/)
        from mlxtend.

    """

    contents: str = 'mlxtend.feature_selection.ExhaustiveFeatureSelector'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'min_features': 1, 'max_features': 3, 'print_progress': False})

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the best set of features, and adds the score of each set.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        _add_search(item, self.name)
        return item

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Tries no more features than there are.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        most = min(tool.max_features, x.shape[1])
        tool.set_params(
            max_features = most, min_features = min(tool.min_features, most))
        tool.fit(x, y)


@dataclasses.dataclass
class RecursiveElimination(_ModelBased, Selector):
    """Removes the least important feature of a model, again and again.

    It fits the model to the features, removes the "step" least important (1 by
    default), and fits it again, until `n_features_to_select` are left (half of
    the features by default). The model ("estimator") is described in the
    introduction of the module.

    Wraps:
        [`sklearn.feature_selection.RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.RFE'

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than it would keep.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        wanted = tool.n_features_to_select
        if isinstance(wanted, int) and wanted > x.shape[1]:
            tool.set_params(n_features_to_select = x.shape[1])
        super()._fit(tool, x, y)


@dataclasses.dataclass
class RecursiveEliminationCV(_ModelSearch, Selector):
    """Removes features as `recursive_elimination` does, keeping the best.

    It removes the least important feature of the model again and again, in
    cross-validation on the training rows, and keeps the number of features that
    scored best (at least `min_features_to_select`, 1 by default). The model
    ("estimator"), score ("scoring"), and folds ("cv") are described in the
    introduction of the module. The score of each number of features is a table
    in the dataset's `tables` as "{name}_search", with a row for each number
    ("size") and the mean and standard deviation of its scores ("score" and
    "std"), best first.

    Wraps:
        [`sklearn.feature_selection.RFECV`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.RFECV'

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the best number of features, and adds the score of each.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        tool = item.fitted.get(self.name)
        if tool is not None:
            results = tool.cv_results_
            item.tables[f'{self.name}_search'] = pd.DataFrame({
                'size': np.asarray(results['n_features'], dtype = int),
                'score': np.asarray(results['mean_test_score'], dtype = float),
                'std': np.asarray(results['std_test_score'], dtype = float),
                }).sort_values(
                    'score', ascending = False, kind = 'stable',
                    ignore_index = True)
        return item


@dataclasses.dataclass
class SelectFromModel(_ModelBased, Selector):
    """Keeps the features that a model finds most important.

    It fits the model and keeps the features whose importance (the size of their
    coefficients, or `feature_importances_`) is at least "threshold" (by
    default, the mean importance). The model ("estimator") is described in the
    introduction of the module.

    Wraps:
        [`sklearn.feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFromModel'


@dataclasses.dataclass
class SequentialSelection(_MlxtendSearch, Selector):
    """Adds (or removes) features one at a time while the score improves.

    mlxtend's sequential search starts with no features and adds the one that
    most improves the score of the model, again and again (or, if "forward" is
    false, starts with every feature and removes them). If "floating" is true,
    it also tries to remove (or add back) a feature after each step. It keeps
    `k_features` features, or by default ("best") the number with the best
    score. The model ("estimator"), score ("scoring"), and folds ("cv") are
    described in the introduction of the module. The score of the set of each
    size is a table in the dataset's `tables` as "{name}_search".
    `sequential_selection_sk` does much the same without mlxtend.

    Wraps:
        [`mlxtend.feature_selection.SequentialFeatureSelector`](https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/)
        from mlxtend.

    """

    contents: str = 'mlxtend.feature_selection.SequentialFeatureSelector'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'k_features': 'best'})

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the features that were chosen, and adds the score of each set.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        _add_search(item, self.name)
        return item

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than `k_features`.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        if isinstance(tool.k_features, int) and tool.k_features > x.shape[1]:
            tool.set_params(k_features = x.shape[1])
        tool.fit(x, y)


@dataclasses.dataclass
class SequentialSelectionSK(_ModelSearch, Selector):
    """Adds (or removes) features one at a time until it keeps enough.

    scikit-learn's sequential search adds the feature that most improves the
    score of the model, again and again (or, if "direction" is "backward",
    removes the feature whose removal least lowers it). It keeps
    `n_features_to_select` features, or by default ("auto") half of them, or if
    "tol" is set, stops when the score improves by less than "tol". The model
    ("estimator"), score ("scoring"), and folds ("cv") are described in the
    introduction of the module. Unlike `sequential_selection`, it needs no
    optional package, but it does not report the score of each set.

    Wraps:
        [`sklearn.feature_selection.SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SequentialFeatureSelector'


""" Private Functions """


def _add_search(item: base.Dataset, name: str) -> None:
    """Adds the score of each set of features that an mlxtend search tried.

    The table is "{name}_search" in the dataset's `tables`, with a row for
    each set, best first: its features (joined by commas), their number
    ("size"), and the mean and standard deviation of its scores in
    cross-validation ("score" and "std").

    Args:
        item: the dataset with the fitted selector.
        name: name of the selector.

    """
    tool = item.fitted.get(name)
    if tool is None:
        return
    rows = []
    for subset in tool.get_metric_dict().values():
        names = [str(n) for n in subset['feature_names']]
        rows.append({
            'features': ', '.join(names),
            'size': len(names),
            'score': float(subset['avg_score']),
            'std': float(subset['std_dev'])})
    item.tables[f'{name}_search'] = pd.DataFrame(rows).sort_values(
        'score', ascending = False, kind = 'stable', ignore_index = True)


def _build_estimator(item: base.Dataset, estimator: Any) -> Any:
    """Returns the model that a selector chooses features with.

    Args:
        item: the dataset to transform.
        estimator: the name of a model in the library, an estimator, its
            class, its import path, or `None` for a logistic regression (to
            classify) or a linear regression.

    Raises:
        ValueError: if `estimator` names a model in the library that cannot
            do the dataset's task.

    Returns:
        The built (and unfitted) estimator, with the dataset's seed as its
            `random_state` if it has one that is not set.

    """
    if estimator is None:
        linear_model = importlib.import_module('sklearn.linear_model')
        estimator = (
            linear_model.LogisticRegression(max_iter = 1000)
            if item.task == 'classify'
            else linear_model.LinearRegression())
    elif isinstance(estimator, str) and _is_model(estimator):
        kind = cast(
            'type[models.Model]',
            chrisjen.library.borrow(estimator, genre = 'model'))
        technique = kind()
        task = technique._check_task(item)
        estimator = technique._make_tool(
            item, technique._keywords(), tool = technique._choose_tool(task))
    else:
        estimator = utilities.import_tool(estimator)
        if isinstance(estimator, type):
            estimator = estimator()
    if item.seed is not None and hasattr(estimator, 'get_params') and (
        estimator.get_params().get('random_state', False) is None):
        estimator.set_params(random_state = item.seed)
    return estimator


def _is_model(name: str) -> bool:
    """Returns whether `name` is the name of a model in the library.

    Args:
        name: a name, or an import path.

    Returns:
        Whether a class in the "model" genre of the library has the name.

    """
    try:
        chrisjen.library.borrow(name, genre = 'model')
    except KeyError:
        return False
    return True
