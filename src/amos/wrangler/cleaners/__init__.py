"""Techniques that clean data, usually before it is studied or analyzed.

These are techniques of the "wrangler" stage of a project. Each one removes
(or keeps) rows or columns, renames columns, or cleans the values of columns
into their proper types (numbers, dates, or booleans written as text).
Techniques that otherwise change what is in the columns are mungers (see the
`mungers` module), those that add the columns of another table are mergers
(see the `mergers` module), and those that rearrange the rows and columns are
shapers (see the `shapers` module).
Cleaners learn nothing from the data that could leak from the test rows into
the training rows (they do not, for example, fill missing values with an
average), so they are safe to use before the data is split. To fill missing
values, use an imputer (see the `imputers` module) after splitting.

Contents:
    Cleaner: base class for techniques that clean data.
    DropColumns: removes columns.
    DropConstant: removes columns that have only one value.
    KeepColumns: keeps only some columns (and the label).
    RenameColumns: renames columns, including the label and groups of the
        dataset.
    DropDuplicates: removes rows that duplicate an earlier row.
    DropMissing: removes rows with missing values.
    FilterRows: keeps the rows that match a query, or a condition for each
        column.
    CleanBooleans: turns text such as "yes" and "no" into booleans.
    CleanDates: turns text into dates.
    CleanNumbers: turns text such as "$1,234.50" into numbers.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'CleanBooleans',
    'CleanDates',
    'CleanNumbers',
    'Cleaner',
    'DropColumns',
    'DropConstant',
    'DropDuplicates',
    'DropMissing',
    'FilterRows',
    'KeepColumns',
    'RenameColumns']

from ._base import Cleaner
from ._columns import DropColumns, DropConstant, KeepColumns, RenameColumns
from ._rows import DropDuplicates, DropMissing, FilterRows
from ._values import CleanBooleans, CleanDates, CleanNumbers

# The public names say that they are in `amos.cleaners` (rather than in the
# private module that defines them, or in `amos.wrangler.cleaners`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.cleaners.Name`. `amos` makes `amos.cleaners` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.cleaners'
