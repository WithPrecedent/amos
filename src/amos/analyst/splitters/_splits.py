"""Splitters that divide rows into training and test sets.

Contents:
    GroupSplit: keeps all the rows of each group in the same set.
    Stratified: keeps the share of each class the same in both sets.
    TimeSplit: uses the latest rows as the test set.
    TrainTest: splits the rows at random.

"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import numpy as np
import pandas as pd

from ...core import base
from ._base import Splitter


@dataclasses.dataclass
class GroupSplit(Splitter):
    """Keeps all of the rows of each group in the same set.

    Use this when rows are not independent (for example, several students
    in the same school, or several observations of one person), so
    that a model is tested on groups it has not seen.

    Wraps:
        [`sklearn.model_selection.GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
        from scikit-learn.

    """

    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        groups: str | None = None,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Splits the groups (not the rows) at random.

        Args:
            item: the dataset to split.
            test_size: share of the groups to put in the test set.
            groups: name of the column that identifies each row's group.
                Defaults to `None`, in which case the dataset's first group
                is used.
            **kwargs: not used.

        Raises:
            ValueError: if `groups` is not given and the dataset has no
                groups.

        Returns:
            The training rows and the test rows.

        """
        if groups is None and item.groups:
            groups = item.groups[0]
        if groups is None:
            message = f'{self.name!r} needs the name of a "groups" column'
            raise ValueError(message)
        model_selection = importlib.import_module('sklearn.model_selection')
        splitter = model_selection.GroupShuffleSplit(
            n_splits = 1, test_size = test_size, random_state = item.seed)
        train, test = next(splitter.split(
            item.data, groups = item.data[groups]))
        return item.data.index[train], item.data.index[test]


@dataclasses.dataclass
class Stratified(Splitter):
    """Splits at random, keeping the share of each class in both sets.

    This is the usual choice for a classification task, especially when one
    class is rare.

    Wraps:
        [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        from scikit-learn.

    """

    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Splits the rows at random, stratified by the label.

        Args:
            item: the dataset to split. It must have a label.
            test_size: share of the rows to put in the test set.
            **kwargs: not used.

        Returns:
            The training rows and the test rows.

        """
        model_selection = importlib.import_module('sklearn.model_selection')
        # The positions of the rows are split, rather than their index labels,
        # because scikit-learn cannot select from an index of text that is
        # stored by pyarrow.
        positions = np.arange(len(item.data))
        train, test = model_selection.train_test_split(
            positions,
            test_size = test_size,
            random_state = item.seed,
            stratify = item.y)
        return item.data.index[train], item.data.index[test]


@dataclasses.dataclass
class TimeSplit(Splitter):
    """Uses the latest rows as the test set.

    Use this when the data is ordered in time, so that a model is tested on
    observations that come after the ones it learned from.

    """

    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        order: str | None = None,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Puts the last `test_size` share of the rows in the test set.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set.
            order: name of the column to order the rows by (such as a date).
                Defaults to `None`, which uses the current order of the rows.
            **kwargs: not used.

        Returns:
            The training rows and the test rows.

        """
        data = item.data
        if order is not None:
            data = data.sort_values(order, kind = 'stable')
        cut = len(data) - round(len(data) * test_size)
        return data.index[:cut], data.index[cut:]


@dataclasses.dataclass
class TrainTest(Splitter):
    """Splits the rows at random.

    Wraps:
        [`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        from scikit-learn.

    """

    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        *,
        shuffle: bool = True,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Splits the rows at random.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set.
            shuffle: whether to shuffle the rows before splitting. If `False`,
                the last rows are the test set. Defaults to `True`.
            **kwargs: not used.

        Returns:
            The training rows and the test rows.

        """
        model_selection = importlib.import_module('sklearn.model_selection')
        # The positions of the rows are split, as in `Stratified`.
        positions = np.arange(len(item.data))
        train, test = model_selection.train_test_split(
            positions,
            test_size = test_size,
            random_state = item.seed,
            shuffle = shuffle)
        return item.data.index[train], item.data.index[test]
