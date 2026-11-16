r"""Techniques that change what is in columns, usually before data is studied.

These are techniques of the "wrangler" stage of a project. Cleaners (see the
`cleaners` module) change which rows and columns there are. Mungers change
what is in the columns, or make new columns from them, without adding or
removing rows. Each one changes a whole column at once with the vectorized
methods of `pandas` (such as those of `Series.str`), so they are fast even
with a lot of data. Like cleaners, mungers learn nothing from the data that
could leak from the test rows into the training rows, so they are safe to use
before the data is split.

Parsers (see the `parsers` module) search text for patterns and make
columns of what they find.

Contents:
    Munger: base class for techniques that change the values in columns.
    NormalizeText: collapses spaces in text, and optionally changes its case and
        removes accents and punctuation.
    StripText: trims spaces from (and optionally lowercases) text.
    AutoCategorize: makes columns with few unique values categorical.
    ConvertTypes: changes the data types of columns.
    SetKinds: sets the kind of each column (dummy, categorical, or numerical),
        with the smallest type that holds it.
    Coalesce: takes the first value that is not missing from several columns.
    CombineFlags: combines flags into one.
    DeriveColumns: makes columns from expressions of other columns, such as "a /
        b".
    MapValues: replaces values with others, such as "N/A" with a missing value.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'AutoCategorize',
    'Coalesce',
    'CombineFlags',
    'ConvertTypes',
    'DeriveColumns',
    'MapValues',
    'Munger',
    'NormalizeText',
    'SetKinds',
    'StripText']

from ._base import Munger
from ._text import NormalizeText, StripText
from ._types import AutoCategorize, ConvertTypes, SetKinds
from ._values import Coalesce, CombineFlags, DeriveColumns, MapValues

# The public names say that they are in `amos.mungers` (rather than in the
# private module that defines them, or in `amos.wrangler.mungers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.mungers.Name`. `amos` makes `amos.mungers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.mungers'
