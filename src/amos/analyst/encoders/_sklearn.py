"""Encoders from scikit-learn.

Contents:
    OneHot: encoders.
    Ordinal: encoders.

"""

from __future__ import annotations

import dataclasses

from ...core import base
from ._base import Encoder

""" Encoders """


@dataclasses.dataclass
class OneHot(Encoder):
    """Makes a column of 0s and 1s for each category (dummy variables).

    Categories that are not in the training rows are all 0s.

    Wraps:
        [`sklearn.preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.OneHotEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'handle_unknown': 'ignore',
            'sparse_output': False})


@dataclasses.dataclass
class Ordinal(Encoder):
    """Numbers the categories (0, 1, 2, ...) in sorted order.

    Categories that are not in the training rows are -1.

    Wraps:
        [`sklearn.preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.OrdinalEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'handle_unknown': 'use_encoded_value',
            'unknown_value': -1})
