r"""Techniques that add the columns of another table to the data.

These are techniques of the "wrangler" stage of a project. Research often
combines tables that are organized differently: one table has a row for each
patient visit, another a row for each hospital, and a third a row for each
period that a doctor worked at a hospital. A merger matches each row of the
data to a row of another table and adds the columns of that row. The mergers
differ in how they match rows:

* `merge_keys` matches the row with the same keys (such as the same
  hospital).
* `merge_nearest` matches the row with the nearest number or date (such as
  the last price of a stock before a trade).
* `merge_ranges` matches the row whose range has a number or date in it (such
  as the job that an employee held on the date of a survey).
* `merge_summary` matches every row with the same keys, and adds summaries of
  them (such as the number of students at a school and their mean age).

A merger never adds, removes, or reorders the rows of the data, so the data
keeps the unit that it studies, and its label, groups, and index. A row
without a match gets missing values. Start from the table whose rows are what
you study, and merge the others into it. To change what a row is first (to
make a row for each item of each order, for example), use a shaper (see the
`shapers` module).

The other table is the "source" parameter of every merger: the path of a data
file, which is loaded with the project's clerk as loaders load theirs, or a
table (in Python). Mergers learn nothing from the data that could leak from
the test rows into the training rows.

Contents:
    Merger: base class for techniques that add columns from another table.
    MergeKeys: adds columns from another table's row that has the same keys.
    MergeSummary: adds summaries of the rows of another table that have the same
        keys.
    MergeNearest: adds columns from another table's row with the nearest date or
        number.
    MergeRanges: adds columns from another table's row whose range has a date or
        number in it.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'MergeKeys',
    'MergeNearest',
    'MergeRanges',
    'MergeSummary',
    'Merger']

from ._base import Merger
from ._keys import MergeKeys, MergeSummary
from ._nearest import MergeNearest
from ._ranges import MergeRanges

# The public names say that they are in `amos.mergers` (rather than in the
# private module that defines them, or in `amos.wrangler.mergers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.mergers.Name`. `amos` makes `amos.mergers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.mergers'
