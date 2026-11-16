"""Techniques that rescale numbers.

These are preprocessing techniques of the "analyst" stage. Each one wraps a
scaler from scikit-learn, which is fitted to the training rows only and then
rescales every row, so nothing is learned from the test rows. By default, a
scaler changes the numeric features. Pass `columns` as a parameter to choose
others.

Contents:
    Scaler: base class for techniques that rescale the numeric features.
    Bins: sorts each feature into bins with about the same number of rows.
    Binarize: makes each feature 1 if it is above a threshold (0 by default).
    Gauss: makes each feature more like a normal distribution (Yeo-Johnson).
    MaxAbs: divides each feature by its largest absolute value.
    MinMax: rescales each feature to the range from 0 to 1.
    Normalize: rescales each row to a length of 1.
    Quantile: replaces each value with its quantile in the training rows.
    Robust: centers on the median and scales by the interquartile range.
    Standard: centers each feature on 0 with a standard deviation of 1.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Binarize',
    'Bins',
    'Gauss',
    'MaxAbs',
    'MinMax',
    'Normalize',
    'Quantile',
    'Robust',
    'Scaler',
    'Standard']

from ._base import Scaler
from ._sklearn import (
    Binarize,
    Bins,
    Gauss,
    MaxAbs,
    MinMax,
    Normalize,
    Quantile,
    Robust,
    Standard,
)

# The public names say that they are in `amos.scalers` (rather than in the
# private module that defines them, or in `amos.analyst.scalers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.scalers.Name`. `amos` makes `amos.scalers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.scalers'
