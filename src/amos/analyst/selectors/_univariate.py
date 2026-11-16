"""Selectors that test each feature on its own.

Contents:
    GenericUnivariate: keeps the features most related to the label, by a mode
        that the parameters choose.
    KBest: keeps the k features most related to the label.
    SelectFdr: keeps the features related to the label, controlling the false
        discovery rate.
    SelectFpr: keeps the features related to the label, controlling the false
        positive rate.
    SelectFwe: keeps the features related to the label, controlling the family-
        wise error rate.
    SelectPercentile: keeps a share of the features most related to the label.
    VarianceThreshold: removes features whose variance is at or below a
        threshold.

"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from ...core import base, utilities
from ._base import Selector


@dataclasses.dataclass
class GenericUnivariate(Selector):
    """Keeps the features most related to the label, by a mode it is given.

    "mode" is "percentile" (the default), "k_best", "fpr", "fdr", or "fwe",
    and "param" is the setting of that mode (50 by default, so it keeps half
    of the features). This lets an experiment search the mode as a parameter.
    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.GenericUnivariateSelect'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'mode': 'percentile', 'param': 50})

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class KBest(Selector):
    """Keeps the k features (10 by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectKBest'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'k': 10})

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than k.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels, or `None` if there is no label.

        """
        if isinstance(tool.k, int) and tool.k > x.shape[1]:
            tool.set_params(k = 'all')
        super()._fit(tool, x, y)

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectFdr(Selector):
    """Keeps the features related to the label, controlling false discovery.

    It keeps the features whose p-values pass the Benjamini-Hochberg
    procedure for a false discovery rate of "alpha" (0.05 by default). The
    default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFdr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFdr'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectFpr(Selector):
    """Keeps the features related to the label, controlling false positives.

    It keeps the features whose p-values are below "alpha" (0.05 by
    default). The default test is an F test suited to the task
    (`f_classif` or `f_regression`). Set "score_func" to another function or
    its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFpr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFpr'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectFwe(Selector):
    """Keeps the features related to the label, controlling family-wise error.

    It keeps the features whose p-values pass a Bonferroni correction for a
    family-wise error rate of "alpha" (0.05 by default). The default test is
    an F test suited to the task (`f_classif` or `f_regression`). Set
    "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFwe`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFwe'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectPercentile(Selector):
    """Keeps the features (50% by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectPercentile'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'percentile': 50})

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class VarianceThreshold(Selector):
    """Removes features whose variance is at or below a threshold (0).

    Wraps:
        [`sklearn.feature_selection.VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.VarianceThreshold'


""" Private Functions """


def _with_score_function(
    item: base.Dataset,
    parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns `parameters` with a scoring function for feature selection.

    Wraps:
        - [`sklearn.feature_selection.f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
          from scikit-learn, by default, to classify.
        - [`sklearn.feature_selection.f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
          from scikit-learn, by default, to regress.

    Args:
        item: the dataset.
        parameters: parameters for a feature selector.

    Returns:
        The parameters, with "score_func" imported if it was an import path,
            or chosen for the task if it was not given.

    """
    score = parameters.get('score_func')
    if score is None:
        name = 'f_classif' if item.task == 'classify' else 'f_regression'
        score = f'sklearn.feature_selection.{name}'
    return {**parameters, 'score_func': utilities.import_tool(score)}
