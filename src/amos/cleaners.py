"""Techniques that clean data, usually before it is studied or analyzed.

These are techniques of the "wrangler" stage of a project. Each one removes
(or keeps) rows or columns, or renames columns. Techniques that change what is
in the columns are mungers (see the `mungers` module). Cleaners learn nothing
from the data that could leak from the test rows into the training rows (they
do not, for example, fill missing values with an average), so they are safe
to use before the data is split. To fill missing values, use an imputer from
the `transformers` module after splitting.

Contents:
    Cleaner: base class for techniques that clean data.
    DropColumns: removes columns.
    DropConstant: removes columns that have only one value.
    DropDuplicates: removes duplicate rows.
    DropMissing: removes rows with missing values.
    FilterRows: keeps the rows that match a query.
    KeepColumns: keeps only some columns (and the label).
    RenameColumns: renames columns.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from . import base, utilities


@dataclasses.dataclass
class Cleaner(base.Operation, abc.ABC):
    """Base class for techniques that clean data.

    A subclass writes a `clean` method, which takes the data and returns the
    cleaned data. The dataset's `history` records how many rows and columns
    there were before and after.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most cleaners. Defaults to `None`.
        parameters: keyword arguments for `clean`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def clean(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns the cleaned `data`.

        Args:
            data: the data to clean. It may be changed in place.
            **kwargs: parameters for cleaning.

        Returns:
            The cleaned data.

        """

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Cleans the data of `item`.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        before = item.data.shape
        item.replace(self.clean(item.data.copy(), **kwargs))
        item.record(
            self.name,
            rows = [before[0], item.data.shape[0]],
            columns = [before[1], item.data.shape[1]])
        return item


@dataclasses.dataclass
class DropColumns(Cleaner):
    """Removes columns.

    Wraps:
        [`pandas.DataFrame.drop`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Removes `columns`.

        Args:
            data: the data to clean.
            columns: names of the columns to remove. Defaults to `None`, which
                removes nothing.
            **kwargs: not used.

        Returns:
            The data, without `columns`.

        """
        dropped = utilities.select_columns(data, columns or [])
        return data.drop(columns = dropped)


@dataclasses.dataclass
class DropConstant(Cleaner):
    """Removes columns that have only one value (and so tell you nothing)."""

    def clean(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Removes columns with one unique value (counting missing values).

        Args:
            data: the data to clean.
            **kwargs: not used.

        Returns:
            The data, without constant columns.

        """
        constant = [
            c for c in data.columns
            if data[c].nunique(dropna = False) <= 1]  # noqa: PD101
        return data.drop(columns = constant)


@dataclasses.dataclass
class DropDuplicates(Cleaner):
    """Removes rows that duplicate an earlier row.

    Wraps:
        [`pandas.DataFrame.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Removes duplicate rows, keeping the first.

        Args:
            data: the data to clean.
            columns: columns to compare. Defaults to `None`, which compares
                every column.
            **kwargs: not used.

        Returns:
            The data, without duplicate rows.

        """
        subset = (
            None if columns is None
            else utilities.select_columns(data, columns))
        return data.drop_duplicates(subset = subset)


@dataclasses.dataclass
class DropMissing(Cleaner):
    """Removes rows with missing values.

    Wraps:
        [`pandas.DataFrame.dropna`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Removes rows with a missing value in any of `columns`.

        Args:
            data: the data to clean.
            columns: columns to check. Defaults to `None`, which checks every
                column.
            **kwargs: not used.

        Returns:
            The data, without rows that have missing values.

        """
        subset = (
            None if columns is None
            else utilities.select_columns(data, columns))
        return data.dropna(subset = subset)


@dataclasses.dataclass
class FilterRows(Cleaner):
    """Keeps the rows that match a query.

    Wraps:
        [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        query: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Keeps the rows that match `query`.

        Args:
            data: the data to clean.
            query: a `pandas` query, such as "age >= 18". Column names with
                spaces go inside backticks. Defaults to `None`, which keeps
                every row.
            **kwargs: not used.

        Returns:
            The rows that match `query`.

        """
        return data if query is None else data.query(query)


@dataclasses.dataclass
class KeepColumns(Cleaner):
    """Keeps only some columns. The label and groups are always kept."""

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        label: str | None = None,
        groups: Sequence[str] = (),
        **kwargs: Any) -> pd.DataFrame:
        """Keeps `columns` (and `label` and `groups`).

        Args:
            data: the data to clean.
            columns: names of the columns to keep. Defaults to `None`, which
                keeps every column.
            label: name of the label, which is always kept. Defaults to `None`.
            groups: names of the group columns, which are always kept.
                Defaults to an empty tuple.
            **kwargs: not used.

        Returns:
            The data, with only `columns`, the label, and the groups.

        """
        if columns is None:
            return data
        kept = utilities.select_columns(data, columns)
        for name in [label, *groups]:
            if name is not None and name not in kept:
                kept.append(name)
        return data[kept]

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Keeps the chosen columns of `item`, its label, and its groups.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        return super().implement(
            item, **{'label': item.label, 'groups': item.groups, **kwargs})


@dataclasses.dataclass
class RenameColumns(Cleaner):
    """Renames columns, including the label and groups of the dataset.

    Wraps:
        [`pandas.DataFrame.rename`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        names: Mapping[str, str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Renames columns.

        Args:
            data: the data to clean.
            names: `dict` mapping current names to new names. Defaults to
                `None`, which renames nothing.
            **kwargs: not used.

        Returns:
            The data, with new column names.

        """
        return data.rename(columns = dict(names or {}))

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Renames the columns of `item`, including its label.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        names = dict(kwargs.get('names') or {})
        # The label and groups are renamed before the data so that `replace`
        # finds them.
        special = [c for c in [item.label, *item.groups] if c in names]
        if special:
            item.data = item.data.rename(
                columns = {c: names[c] for c in special})
            item.groups = [names.get(g, g) for g in item.groups]
            if item.label in names:
                item.label = names[item.label]
        return super().implement(item, **kwargs)
