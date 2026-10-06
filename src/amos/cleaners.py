"""Techniques that clean data, usually before it is studied or analyzed.

These are the techniques of the "wrangler" stage of a project. Each one changes
the rows or columns of the data. Cleaners learn nothing from the data that
could leak from the test rows into the training rows (they do not, for
example, fill missing values with an average), so they are safe to use before
the data is split. To fill missing values, use an imputer from the
`transformers` module after splitting.

Contents:
    Cleaner: base class for techniques that clean data.
    AutoCategorize: makes columns with few unique values categorical.
    ConvertTypes: changes the data types of columns.
    DropColumns: removes columns.
    DropConstant: removes columns that have only one value.
    DropDuplicates: removes duplicate rows.
    DropMissing: removes rows with missing values.
    FilterRows: keeps the rows that match a query.
    KeepColumns: keeps only some columns (and the label).
    RenameColumns: renames columns.
    StripText: trims spaces from (and optionally lowercases) text.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from . import base, options


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
class AutoCategorize(Cleaner):
    """Makes columns with few unique values categorical.

    Text and number columns with `threshold` unique values or fewer become the
    `pandas` "category" type, so they are encoded (rather than scaled) by the
    analyst. Boolean columns are not changed.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        threshold: int = options._CATEGORY_THRESHOLD,
        **kwargs: Any) -> pd.DataFrame:
        """Makes columns with `threshold` unique values or fewer categorical.

        Args:
            data: the data to clean.
            columns: columns to check. Defaults to `None`, which checks every
                column.
            threshold: most unique values that a categorical column can have.
                Defaults to `options._CATEGORY_THRESHOLD`.
            **kwargs: not used.

        Returns:
            The data, with categorical columns.

        """
        for column in _columns(data, columns):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if values.nunique(dropna = True) <= threshold:
                data[column] = values.astype('category')
        return data


@dataclasses.dataclass
class ConvertTypes(Cleaner):
    """Changes the data types of columns."""

    def clean(
        self,
        data: pd.DataFrame,
        types: Mapping[str, Any] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Changes the type of each column in `types`.

        Args:
            data: the data to clean.
            types: `dict` mapping column names to `pandas` data types (such as
                "category", "float", or "boolean"). Defaults to `None`, which
                changes nothing.
            **kwargs: not used.

        Returns:
            The data, with the new types.

        """
        return data.astype(dict(types or {}))


@dataclasses.dataclass
class DropColumns(Cleaner):
    """Removes columns."""

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
        return data.drop(columns = _columns(data, columns or []))


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
    """Removes rows that duplicate an earlier row."""

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
        subset = None if columns is None else _columns(data, columns)
        return data.drop_duplicates(subset = subset)


@dataclasses.dataclass
class DropMissing(Cleaner):
    """Removes rows with missing values."""

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
        subset = None if columns is None else _columns(data, columns)
        return data.dropna(subset = subset)


@dataclasses.dataclass
class FilterRows(Cleaner):
    """Keeps the rows that match a query."""

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
    """Keeps only some columns. The label is always kept."""

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        label: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Keeps `columns` (and `label`).

        Args:
            data: the data to clean.
            columns: names of the columns to keep. Defaults to `None`, which
                keeps every column.
            label: name of the label, which is always kept. Defaults to `None`.
            **kwargs: not used.

        Returns:
            The data, with only `columns` and the label.

        """
        if columns is None:
            return data
        kept = _columns(data, columns)
        if label is not None and label not in kept:
            kept.append(label)
        return data[kept]

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Keeps the chosen columns of `item` and its label.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        return super().implement(item, **{'label': item.label, **kwargs})


@dataclasses.dataclass
class RenameColumns(Cleaner):
    """Renames columns. Renaming the label also renames it in the dataset."""

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
        if item.label in names:
            # The label is renamed before the data so that `replace` finds it.
            label = names[item.label]
            item.data = item.data.rename(columns = {item.label: label})
            item.label = label
        return super().implement(item, **kwargs)


@dataclasses.dataclass
class StripText(Cleaner):
    """Trims spaces from text, and optionally makes it lowercase."""

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        lowercase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Trims spaces from the text in `columns`.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes every
                text column.
            lowercase: whether to make the text lowercase too. Defaults to
                `False`.
            **kwargs: not used.

        Returns:
            The data, with trimmed text. Empty text becomes a missing value.

        """
        if columns is None:
            columns = [
                c for c in data.columns
                if pd.api.types.is_string_dtype(data[c].dtype)
                and not isinstance(data[c].dtype, pd.CategoricalDtype)]
        for column in _columns(data, columns):
            text = data[column].str.strip()
            if lowercase:
                text = text.str.lower()
            data[column] = text.mask(text == '')
        return data


""" Private Functions """


def _columns(
    data: pd.DataFrame,
    columns: Sequence[str] | str | None) -> list[str]:
    """Returns `columns` as a `list`, checking that they are in `data`.

    Args:
        data: the data.
        columns: a column name, a sequence of names, or `None` for every
            column.

    Raises:
        KeyError: if a column is not in `data`.

    Returns:
        The names of the columns.

    """
    if columns is None:
        return list(data.columns)
    names = [columns] if isinstance(columns, str) else list(columns)
    missing = [c for c in names if c not in data.columns]
    if missing:
        message = f'the columns {missing} are not in the data'
        raise KeyError(message)
    return names
