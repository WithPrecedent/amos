"""Techniques that resample the training rows to balance the classes.

These are techniques of the "analyst" stage for classification tasks in which
one class is rare. Each wraps a sampler from imbalanced-learn, which adds rows
of the rare classes (over-sampling), removes rows of the common classes
(under-sampling), or both. Only the training rows are resampled, so models are
still evaluated on the real (and imbalanced) test rows. The features must be
numbers, so put a sampler after any encoders.

Contents:
    Sampler: base class for techniques that resample the training rows.
    Adasyn: adds synthetic rows where the rare class is hardest to learn.
    BorderlineSmote: adds synthetic rows near the border between the classes.
    NearMiss: removes rows of the common classes that are far from the rare
        rows.
    RandomOver: duplicates random rows of the rare classes.
    RandomUnder: removes random rows of the common classes.
    Smote: adds synthetic rows between rare rows and their nearest neighbors.
    SmoteEnn: `smote`, then removes rows that their neighbors misclassify.
    SmoteTomek: `smote`, then removes pairs of rows from different classes that
        are each other's nearest neighbors.
    TomekLinks: removes rows of the common class that are paired with rare rows.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Adasyn',
    'BorderlineSmote',
    'NearMiss',
    'RandomOver',
    'RandomUnder',
    'Sampler',
    'Smote',
    'SmoteEnn',
    'SmoteTomek',
    'TomekLinks']

from ._base import Sampler
from ._base import _leading_copies as _leading_copies
from ._imblearn import (
    Adasyn,
    BorderlineSmote,
    NearMiss,
    RandomOver,
    RandomUnder,
    Smote,
    SmoteEnn,
    SmoteTomek,
    TomekLinks,
)

# The public names say that they are in `amos.samplers` (rather than in the
# private module that defines them, or in `amos.analyst.samplers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.samplers.Name`. `amos` makes `amos.samplers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.samplers'
