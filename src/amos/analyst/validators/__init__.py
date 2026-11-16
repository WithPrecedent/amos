"""Techniques that cross-validate a model, as the last step of the "analyst".

A validator measures how well the dataset's model does on rows that it did
not learn from, without using the test rows. It divides the training rows into
folds with one of scikit-learn's cross-validation splitters, fits a new copy
of the model to the rest of the training rows for each fold, and scores the
copy on the fold with its "metrics" (by default, those of a scorecard for
the task). The scores of each fold are a table, stored in the dataset's
`tables` under the validator's name, and their means are stored in its
`metrics` as "cv_{metric}" (such as "cv_roc_auc"), so that a scorecard or
an experiment's comparison shows them beside the scores on the test rows.

Rows that a sampler made up (the dataset's `synthetic` rows) are never scored.
Instead, each sampler that was applied to the training rows is applied again
to the rows that each copy of the model learns from. Transformers are not
fitted again: a validator scores the model, given the features that the
transformers made after learning from all of the training rows.

Contents:
    Validator: base class for techniques that cross-validate a model.
    GroupKFold: validators that wrap the cross-validation splitter of scikit-
        learn with the same name.
    GroupShuffleSplit: validators that wrap the cross-validation splitter of
        scikit-learn with the same name.
    KFold: validators that wrap the cross-validation splitter of scikit-learn
        with the same name.
    LeaveOneGroupOut: validators that wrap the cross-validation splitter of
        scikit-learn with the same name.
    LeaveOneRowOut: a validator that wraps scikit-learn's `LeaveOneOut` (the
        name `leave_one_out` is an encoder).
    RepeatedKFold: validators that wrap the cross-validation splitter of scikit-
        learn with the same name.
    RepeatedStratifiedKFold: validators that wrap the cross-validation splitter
        of scikit-learn with the same name.
    ShuffleSplit: validators that wrap the cross-validation splitter of scikit-
        learn with the same name.
    StratifiedGroupKFold: validators that wrap the cross-validation splitter of
        scikit-learn with the same name.
    StratifiedKFold: validators that wrap the cross-validation splitter of
        scikit-learn with the same name.
    StratifiedShuffleSplit: validators that wrap the cross-validation splitter
        of scikit-learn with the same name.
    TimeSeriesSplit: validators that wrap the cross-validation splitter of
        scikit-learn with the same name.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'GroupKFold',
    'GroupShuffleSplit',
    'KFold',
    'LeaveOneGroupOut',
    'LeaveOneRowOut',
    'RepeatedKFold',
    'RepeatedStratifiedKFold',
    'ShuffleSplit',
    'StratifiedGroupKFold',
    'StratifiedKFold',
    'StratifiedShuffleSplit',
    'TimeSeriesSplit',
    'Validator']

from ._base import Validator
from ._base import _columns as _columns
from ._base import _features as _features
from ._sklearn import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneRowOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

# The public names say that they are in `amos.validators` (rather than in the
# private module that defines them, or in `amos.analyst.validators`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.validators.Name`. `amos` makes `amos.validators` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.validators'
