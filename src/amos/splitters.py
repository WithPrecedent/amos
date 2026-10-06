"""Techniques that split data into training and test sets.

A splitter is usually the first step of the "analyst" stage. Every technique
after it learns from the training rows only, and models are evaluated on the
test rows. Splitting with the dataset's `seed` makes the split reproducible:
in an `experiment` or `contest`, every combination of techniques gets the same
split.

Contents:
    Splitter: base class for techniques that split data.
    GroupSplit: keeps all the rows of each group in the same set.
    Stratified: keeps the share of each class the same in both sets.
    TimeSplit: uses the latest rows as the test set.
    TrainTest: splits the rows at random.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from typing import Any

import pandas as pd

from . import base, options


@dataclasses.dataclass
class Splitter(base.Operation, abc.ABC):
    """Base class for techniques that split data into training and test sets.

    A subclass writes a `divide` method, which returns the index labels of the
    training and test rows.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most splitters. Defaults to `None`.
        parameters: keyword arguments for `divide`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Returns the index labels of the training and test rows.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set.
            **kwargs: parameters for the split.

        Returns:
            The training rows and the test rows.

        """

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        test_size: float = options._DEFAULT_TEST_SIZE,
        **kwargs: Any) -> base.Dataset:
        """Splits `item` into training and test sets.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set. Defaults to
                `options._DEFAULT_TEST_SIZE`.
            **kwargs: parameters for `divide`.

        Raises:
            ValueError: if `test_size` is not between 0 and 1.

        Returns:
            The split dataset.

        """
        if not 0 < test_size < 1:
            message = f'test_size must be between 0 and 1, not {test_size}'
            raise ValueError(message)
        train, test = self.divide(item, test_size, **kwargs)
        item.split(train, test)
        item.record(
            self.name,
            test_size = test_size,
            train = len(train),
            test = len(test),
            seed = item.seed)
        return item


@dataclasses.dataclass
class GroupSplit(Splitter):
    """Keeps all of the rows of each group in the same set.

    Use this when rows are not independent (for example, several cases
    decided by the same judge, or several observations of one person), so
    that a model is tested on groups it has not seen.

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
        train, test = model_selection.train_test_split(
            item.data.index,
            test_size = test_size,
            random_state = item.seed,
            stratify = item.y)
        return pd.Index(train), pd.Index(test)


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
    """Splits the rows at random."""

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
        train, test = model_selection.train_test_split(
            item.data.index,
            test_size = test_size,
            random_state = item.seed,
            shuffle = shuffle)
        return pd.Index(train), pd.Index(test)
