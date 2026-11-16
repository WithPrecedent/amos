"""Cleaners that remove, keep, or rename columns.

Contents:
    DropColumns: removes columns.
    DropConstant: removes columns that have only one value.
    KeepColumns: keeps only some columns (and the label).
    RenameColumns: renames columns.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ...core import base, utilities
from ._base import Cleaner


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

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Keeps the chosen columns of `item`, its label, and its groups.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        return super().apply(
            item, **{'label': item.label, 'groups': item.groups, **kwargs})


@dataclasses.dataclass
class RenameColumns(Cleaner):
    """Renames columns, including the label and groups of the dataset.

    The names are "names", a `dict` of current names and new names, and any
    other parameter, whose name is a current name and whose value is the new
    name (as in "state = state_abbreviation").

    Wraps:
        [`pandas.DataFrame.rename`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        names: Mapping[str, str] | None = None,
        **renames: str) -> pd.DataFrame:
        """Renames columns.

        Args:
            data: the data to clean.
            names: `dict` mapping current names to new names. Defaults to
                `None`.
            **renames: new names, by the current names.

        Returns:
            The data, with new column names.

        """
        return data.rename(columns = {**dict(names or {}), **renames})

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Renames the columns of `item`, including its label and groups.

        The kinds of the renamed columns (see `Dataset.kinds`) are kept.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        names = {
            **dict(kwargs.pop('names', None) or {}),
            **{str(k): str(v) for k, v in kwargs.items()}}
        # The label and groups are renamed before the data so that `replace`
        # finds them.
        special = [c for c in [item.label, *item.groups] if c in names]
        if special:
            item.data = item.data.rename(
                columns = {c: names[c] for c in special})
            item.groups = [names.get(g, g) for g in item.groups]
            if item.label in names:
                item.label = names[item.label]
        item.kinds = {names.get(c, c): k for c, k in item.kinds.items()}
        return super().apply(item, names = names)
