r"""Techniques that reshape data, usually before it is studied or analyzed.

These are techniques of the "wrangler" stage of a project. Cleaners (see the
`cleaners` module) remove rows and columns, and mungers (see the `mungers`
module) change what is in the columns. Shapers change what a row is: they
rearrange the same data so that each row stands for something else. A table
with a row for each order becomes one with a row for each item of each order
(`lists_to_rows`), and back again (`group_rows`). A table with a column for
each year becomes one with a row for each year (`wide_to_long`), and back
again (`long_to_wide`).

The rows of reshaped data are not the rows that it began with, so:

* The rows get new index labels (0, 1, 2, and so on). When rows are repeated
  (by `lists_to_rows` and `wide_to_long`), an index with a name is kept as a
  column of that name, so each new row still says which row it came from.
* A shaper must come before the data is split into training and test rows.
* The label and groups of the dataset must still be columns afterward, or
  the shaper's "label" and "groups" parameters must name those of the
  reshaped data.

Like cleaners and mungers, shapers learn nothing from the data that could
leak from the test rows into the training rows.

Contents:
    Shaper: base class for techniques that reshape data.
    GroupRows: makes one row for each group of rows, with summaries of its
        columns.
    ListsToRows: makes a row for each item of a list.
    LongToWide: spreads the rows of each group into columns, such as one for
        each year.
    WideToLong: stacks columns into rows, such as a row for each year.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'GroupRows',
    'ListsToRows',
    'LongToWide',
    'Shaper',
    'WideToLong']

from ._base import Shaper
from ._groups import GroupRows, ListsToRows
from ._pivots import LongToWide, WideToLong

# The public names say that they are in `amos.shapers` (rather than in the
# private module that defines them, or in `amos.wrangler.shapers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.shapers.Name`. `amos` makes `amos.shapers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.shapers'
