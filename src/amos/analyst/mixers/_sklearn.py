"""Mixers from scikit-learn.

Contents:
    Polynomial: mixers.
    Splines: mixers.

"""

from __future__ import annotations

import dataclasses

from ...core import base
from ._base import Mixer

""" Mixers """


@dataclasses.dataclass
class Polynomial(Mixer):
    """Adds the squares and products of the features (degree 2 by default).

    Wraps:
        [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.PolynomialFeatures'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'include_bias': False})


@dataclasses.dataclass
class Splines(Mixer):
    """Replaces each feature with a set of spline curves.

    Wraps:
        [`sklearn.preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.preprocessing.SplineTransformer'
