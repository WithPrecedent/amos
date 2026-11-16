"""The base class of shapers, and what they share.

Contents:
    Shaper: base class for techniques that reshape data.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Any

import camina
import pandas as pd

from ...core import base


@dataclasses.dataclass
class Shaper(base.Operation, abc.ABC):
    """Base class for techniques that reshape data.

    A subclass writes a `shape` method, which takes the data and returns it in
    its new shape. The dataset's `history` records how many rows and columns
    there were before and after.

    Every shaper takes "label", "task", and "groups" parameters, which name
    the label and groups of the reshaped data. Without them, the dataset
    keeps the label and groups that it has.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most shapers. Defaults to `None`.
        parameters: keyword arguments for `shape`, and "label", "task", and
            "groups". Defaults to an empty `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def shape(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns `data` in its new shape.

        Args:
            data: the data to reshape. It may be changed in place.
            **kwargs: parameters for reshaping.

        Returns:
            The reshaped data.

        """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        *,
        label: str | None = None,
        task: str | None = None,
        groups: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Reshapes the data of `item`.

        Args:
            item: the dataset to reshape.
            label: name of the label of the reshaped data. Defaults to `None`,
                which keeps the label of `item`.
            task: "classify" or "regress". Defaults to `None`, which keeps
                the task of `item`, unless `label` names a new label, whose
                task is inferred (see `Dataset.infer_task`).
            groups: names of the groups of the reshaped data. Defaults to
                `None`, which keeps the groups of `item`.
            **kwargs: parameters for `shape`.

        Raises:
            KeyError: if the label or a group is not a column of the reshaped
                data.
            ValueError: if the data has been split, or `task` is not a task.

        Returns:
            The reshaped dataset.

        """
        if item.is_split:
            message = (
                f'{self.name!r} changes the rows of the data, so it must come '
                f'before the data is split'
            )
            raise ValueError(message)
        if task is not None and task not in base._TASKS:
            message = f'task must be one of {base._TASKS}, not {task!r}'
            raise ValueError(message)
        label = label or item.label
        named = item.groups if groups is None else camina.listify(groups)
        before = item.data.shape
        shaped = self.shape(item.data.copy(), **kwargs)
        lost = [
            c for c in [label, *named]
            if c is not None and c not in shaped.columns]
        if lost:
            message = (
                f'the reshaped data has no columns {lost}, which are the '
                f'label or groups: set "label" or "groups" in the parameters '
                f'of {self.name!r} to name those of the reshaped data'
            )
            raise KeyError(message)
        named = base._validate_groups(named, shaped, label)
        relabeled = label != item.label
        item.label, item.groups = label, named
        item.replace(shaped)
        if task is not None:
            item.task = task
        elif relabeled:
            item.task = item.infer_task()
        item.record(
            self.name,
            rows = [before[0], shaped.shape[0]],
            columns = [before[1], shaped.shape[1]])
        return item


""" Private Functions """


def _claim(name: str, taken: Sequence[str], technique: str) -> None:
    """Checks that a new column can have a name.

    Args:
        name: the name of a column to make.
        taken: the names of the other columns of the reshaped data.
        technique: name of the technique that makes the column, for messages.

    Raises:
        ValueError: if `name` is one of `taken`.

    """
    if name in taken:
        message = (
            f'{technique!r} would make two columns named {name!r}: rename '
            f'one of them first (with rename_columns), or give the new '
            f'column another name'
        )
        raise ValueError(message)


def _keep_index(data: pd.DataFrame) -> pd.DataFrame:
    """Returns `data` with an index that has a name as a column.

    Rows that are repeated cannot keep their index labels, which must differ.
    An index with a name (such as "case_id") says something about each row,
    so it becomes a column, which is repeated with the rows.

    Args:
        data: the data to reshape.

    Returns:
        The data with its index as columns, if every level of the index has
            a name that no column has, and otherwise as it is.

    """
    names = list(data.index.names)
    if any(n is None or n in data.columns for n in names):
        return data
    return data.reset_index()
