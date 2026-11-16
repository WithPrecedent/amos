"""Techniques that split data into training and test sets.

A splitter is usually the first step of the "analyst" stage. Every technique
after it learns from the training rows only, and models are evaluated on the
test rows. Splitting with the dataset's `seed` makes the split reproducible:
in an `experiment` or `contest`, every combination of techniques gets the same
split.

Contents:
    Splitter: base class for techniques that split data.
    GroupSplit: keeps all the rows of each group in the same set.
    Stratified: splits at random, keeping the share of each class in both sets.
    TimeSplit: uses the latest rows as the test set.
    TrainTest: splits the rows at random.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'GroupSplit',
    'Splitter',
    'Stratified',
    'TimeSplit',
    'TrainTest']

from ._base import Splitter
from ._splits import GroupSplit, Stratified, TimeSplit, TrainTest

# The public names say that they are in `amos.splitters` (rather than in the
# private module that defines them, or in `amos.analyst.splitters`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.splitters.Name`. `amos` makes `amos.splitters` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.splitters'
