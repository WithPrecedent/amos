"""Reducers from scikit-learn.

Contents:
    PCAReduce: reducers.

"""

from __future__ import annotations

import dataclasses

from ...core import base
from ._base import Reducer

""" Reducers """


@dataclasses.dataclass
class PCAReduce(Reducer):
    """Replaces the features with their principal components.

    By default, it keeps enough components to explain 95% of the variance.
    Scale the features first.

    Wraps:
        [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.decomposition.PCA'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'n_components': 0.95})
