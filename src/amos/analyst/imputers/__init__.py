"""Techniques that fill missing values.

These are preprocessing techniques of the "analyst" stage. Each one wraps an
imputer from scikit-learn, which is fitted to the training rows only and then
fills the missing values of every row, so nothing is learned from the test
rows. By default, an imputer changes the features with missing values. Pass
`columns` as a parameter to choose others.

Contents:
    Imputer: base class for techniques that fill missing values.
    IterativeImpute: fills missing values by modeling each feature from the
        others.
    KnnImpute: fills missing values with the average of the most similar rows.
    MeanImpute: fills missing values with the mean of the training rows.
    MedianImpute: fills missing values with the median of the training rows.
    ModeImpute: fills missing values with the most common value of the training
        rows.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Imputer',
    'IterativeImpute',
    'KnnImpute',
    'MeanImpute',
    'MedianImpute',
    'ModeImpute']

from ._base import Imputer
from ._sklearn import (
    IterativeImpute,
    KnnImpute,
    MeanImpute,
    MedianImpute,
    ModeImpute,
)

# The public names say that they are in `amos.imputers` (rather than in the
# private module that defines them, or in `amos.analyst.imputers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.imputers.Name`. `amos` makes `amos.imputers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.imputers'
