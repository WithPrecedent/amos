"""Scalers from scikit-learn.

Contents:
    Bins: scalers.
    Binarize: scalers.
    Gauss: scalers.
    MaxAbs: scalers.
    MinMax: scalers.
    Normalize: scalers.
    Quantile: scalers.
    Robust: scalers.
    Standard: scalers.

"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from ...core import base
from ._base import Scaler

""" Scalers """


@dataclasses.dataclass
class Bins(Scaler):
    """Sorts each feature into bins with about the same number of rows.

    Wraps:
        [`sklearn.preprocessing.KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.KBinsDiscretizer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'n_bins': 5,
            'encode': 'ordinal',
            'strategy': 'quantile',
            # Only passed to versions of scikit-learn that accept it.
            'quantile_method': 'averaged_inverted_cdf'})


@dataclasses.dataclass
class Binarize(Scaler):
    """Makes each feature 1 if it is above a threshold (0 by default).

    Wraps:
        [`sklearn.preprocessing.Binarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.Binarizer'


@dataclasses.dataclass
class Gauss(Scaler):
    """Makes each feature more like a normal distribution (Yeo-Johnson).

    Wraps:
        [`sklearn.preprocessing.PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.PowerTransformer'


@dataclasses.dataclass
class MaxAbs(Scaler):
    """Divides each feature by its largest absolute value.

    Wraps:
        [`sklearn.preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.MaxAbsScaler'


@dataclasses.dataclass
class MinMax(Scaler):
    """Rescales each feature to the range from 0 to 1.

    Wraps:
        [`sklearn.preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.MinMaxScaler'


@dataclasses.dataclass
class Normalize(Scaler):
    """Rescales each row to a length of 1.

    Wraps:
        [`sklearn.preprocessing.Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.Normalizer'


@dataclasses.dataclass
class Quantile(Scaler):
    """Replaces each value with its quantile in the training rows.

    Wraps:
        [`sklearn.preprocessing.QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.QuantileTransformer'

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Uses no more quantiles than there are training rows.

        Args:
            tool: the built transformer.
            x: the training rows of the columns to transform.
            y: the training labels, or `None` if there is no label.

        """
        if tool.n_quantiles > len(x):
            tool.set_params(n_quantiles = len(x))
        super()._fit(tool, x, y)


@dataclasses.dataclass
class Robust(Scaler):
    """Centers on the median and scales by the interquartile range.

    Wraps:
        [`sklearn.preprocessing.RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.RobustScaler'


@dataclasses.dataclass
class Standard(Scaler):
    """Centers each feature on 0 with a standard deviation of 1.

    Wraps:
        [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.StandardScaler'
