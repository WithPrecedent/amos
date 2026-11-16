"""Validators with the folds of scikit-learn.

Contents:
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

import dataclasses
from typing import ClassVar

from ._base import Validator


@dataclasses.dataclass
class GroupKFold(Validator):
    """Folds that keep all of the rows of each group in the same fold.

    Wraps:
        [`sklearn.model_selection.GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.GroupKFold'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class GroupShuffleSplit(Validator):
    """Repeated random splits of the groups (not the rows) into two sets.

    Wraps:
        [`sklearn.model_selection.GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.GroupShuffleSplit'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class KFold(Validator):
    """Divides the rows into folds ("n_splits", 5 by default) at random.

    Wraps:
        [`sklearn.model_selection.KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.KFold'


@dataclasses.dataclass
class LeaveOneGroupOut(Validator):
    """Scores each group with a copy of the model fitted to the other groups.

    Wraps:
        [`sklearn.model_selection.LeaveOneGroupOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.LeaveOneGroupOut'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class LeaveOneRowOut(Validator):
    """Predicts each row with a copy of the model fitted to every other row.

    A fold of one row cannot be scored alone, so the predictions of every
    fold are scored together. The table has the prediction of each row.

    Wraps:
        [`sklearn.model_selection.LeaveOneOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.LeaveOneOut'
    pools: ClassVar[bool] = True


@dataclasses.dataclass
class RepeatedKFold(Validator):
    """`k_fold` repeated with different random folds ("n_repeats" times).

    Wraps:
        [`sklearn.model_selection.RepeatedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.RepeatedKFold'


@dataclasses.dataclass
class RepeatedStratifiedKFold(Validator):
    """`stratified_k_fold` repeated with different random folds.

    Wraps:
        [`sklearn.model_selection.RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.RepeatedStratifiedKFold'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class ShuffleSplit(Validator):
    """Repeated random splits of the rows into two sets (Monte Carlo).

    Wraps:
        [`sklearn.model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.ShuffleSplit'


@dataclasses.dataclass
class StratifiedGroupKFold(Validator):
    """Folds that keep each group together and the classes in proportion.

    Wraps:
        [`sklearn.model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.StratifiedGroupKFold'
    uses_groups: ClassVar[bool] = True
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class StratifiedKFold(Validator):
    """Folds that keep the share of each class the same in every fold.

    Wraps:
        [`sklearn.model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.StratifiedKFold'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class StratifiedShuffleSplit(Validator):
    """Repeated random splits into two sets, with the classes in proportion.

    Wraps:
        [`sklearn.model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.StratifiedShuffleSplit'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class TimeSeriesSplit(Validator):
    """Scores later rows with copies of the model fitted to earlier rows.

    Use this when the rows are ordered in time (or set "order" to a column,
    such as a date), so that the model is never scored on rows that came
    before the rows it learned from.

    Wraps:
        [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.model_selection.TimeSeriesSplit'
