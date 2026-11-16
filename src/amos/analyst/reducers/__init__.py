"""Techniques that reduce the number of features.

These are preprocessing techniques of the "analyst" stage. Each one is fitted
to the training rows only and then changes every row, so nothing is learned
from the test rows. By default, a reducer uses the numeric and boolean
features. Pass `columns` as a parameter to choose others.

Contents:
    Reducer: base class for techniques that reduce the number of features.
    PCAReduce: replaces the features with their principal components.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'PCAReduce',
    'Reducer']

from ._base import Reducer
from ._sklearn import PCAReduce

# The public names say that they are in `amos.reducers` (rather than in the
# private module that defines them, or in `amos.analyst.reducers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.reducers.Name`. `amos` makes `amos.reducers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.reducers'
