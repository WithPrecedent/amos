"""Techniques that make new features from combinations of others.

These are preprocessing techniques of the "analyst" stage. By default, a
mixer combines the numeric features. Pass `columns` as a parameter to choose
others.

Contents:
    Mixer: base class for techniques that make new features from numeric ones.
    Interactions: adds the product of each pair of features, or of the terms of
        formulas.
    Polynomial: adds the squares and products of the features (degree 2 by
        default).
    Splines: replaces each feature with a set of spline curves.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Interactions',
    'Mixer',
    'Polynomial',
    'Splines']

from ._base import Mixer
from ._interactions import Interactions
from ._sklearn import Polynomial, Splines

# The public names say that they are in `amos.mixers` (rather than in the
# private module that defines them, or in `amos.analyst.mixers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.mixers.Name`. `amos` makes `amos.mixers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.mixers'
