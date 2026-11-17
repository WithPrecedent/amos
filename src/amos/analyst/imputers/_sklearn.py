"""Imputers from scikit-learn.

Contents:
    IterativeImpute: imputers.
    KnnImpute: imputers.
    MeanImpute: imputers.
    MedianImpute: imputers.
    ModeImpute: imputers.

"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

from ...core import base
from ._base import Imputer

""" Imputers """


@dataclasses.dataclass
class IterativeImpute(Imputer):
    """Fills missing values by modeling each feature from the others.

    Wraps:
        [`sklearn.impute.IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
        from scikit-learn.

    """

    assumptions: ClassVar[tuple[str, ...]] = ('missing_at_random',)

    contents: str = 'sklearn.impute.IterativeImputer'

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Enables scikit-learn's experimental imputer and then applies it.

        Args:
            item: the dataset to transform.
            columns: names of the columns to transform. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the transformer.

        Returns:
            The transformed dataset.

        """
        importlib.import_module('sklearn.experimental.enable_iterative_imputer')
        return super().apply(item, columns = columns, **kwargs)


@dataclasses.dataclass
class KnnImpute(Imputer):
    """Fills missing values with the average of the most similar rows.

    Wraps:
        [`sklearn.impute.KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
        from scikit-learn.

    """

    assumptions: ClassVar[tuple[str, ...]] = ('missing_at_random',)

    contents: str = 'sklearn.impute.KNNImputer'


@dataclasses.dataclass
class MeanImpute(Imputer):
    """Fills missing values with the mean of the training rows.

    Wraps:
        [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
        from scikit-learn.

    """

    assumptions: ClassVar[tuple[str, ...]] = ('missing_completely_at_random',)

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'mean'})


@dataclasses.dataclass
class MedianImpute(Imputer):
    """Fills missing values with the median of the training rows.

    Wraps:
        [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
        from scikit-learn.

    """

    assumptions: ClassVar[tuple[str, ...]] = ('missing_completely_at_random',)

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'median'})


@dataclasses.dataclass
class ModeImpute(Imputer):
    """Fills missing values with the most common value of the training rows.

    Unlike the other imputers, this fills features of every kind.

    Wraps:
        [`sklearn.impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
        from scikit-learn.

    """

    assumptions: ClassVar[tuple[str, ...]] = ('missing_completely_at_random',)

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'categoricals', 'booleans')

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'most_frequent'})
