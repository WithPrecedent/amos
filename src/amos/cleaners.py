"""Techniques that clean data, usually before it is studied or analyzed.

These are techniques of the "wrangler" stage of a project. Each one removes
(or keeps) rows or columns, renames columns, or cleans the values of columns
into their proper types (numbers, dates, or booleans written as text).
Techniques that otherwise change what is in the columns are mungers (see the
`mungers` module), those that add the columns of another table are mergers
(see the `mergers` module), and those that rearrange the rows and columns are
shapers (see the `shapers` module).
Cleaners learn nothing from the data that could leak from the test rows into
the training rows (they do not, for example, fill missing values with an
average), so they are safe to use before the data is split. To fill missing
values, use an imputer from the `transformers` module after splitting.

Contents:
    Cleaner: base class for techniques that clean data.
    CleanBooleans: turns text such as "yes" and "no" into booleans.
    CleanDates: turns text into dates.
    CleanNumbers: turns text such as "$1,234.50" into numbers.
    DropColumns: removes columns.
    DropConstant: removes columns that have only one value.
    DropDuplicates: removes duplicate rows.
    DropMissing: removes rows with missing values.
    FilterRows: keeps the rows that match a query, or a condition for each
        column.
    KeepColumns: keeps only some columns (and the label).
    RenameColumns: renames columns.

"""

from __future__ import annotations

import abc
import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from . import base, utilities

# The start of a condition of `filter_rows` that compares a column to a value:
# an operator, or "in" or "not in" as words.
_COMPARISON: re.Pattern[str] = re.compile(r'(==|!=|<=|>=|<|>|in\b|not\s+in\b)')
# Words (in lower case) that `CleanBooleans` reads as false and as true.
_FALSE: tuple[str, ...] = ('false', 'f', 'no', 'n', '0')
_TRUE: tuple[str, ...] = ('true', 't', 'yes', 'y', '1')


@dataclasses.dataclass
class Cleaner(base.Operation, abc.ABC):
    """Base class for techniques that clean data.

    A subclass writes a `clean` method, which takes the data and returns the
    cleaned data. The dataset's `history` records how many rows and columns
    there were before and after ("rows" and "columns") and, if the rows are
    the same and the values or type of any kept column changed, those columns
    ("changed").

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
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

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Cleans the data of `item`.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        before = item.data
        item.replace(self.clean(before.copy(), **kwargs))
        after = item.data
        details: dict[str, Any] = {
            'rows': [before.shape[0], after.shape[0]],
            'columns': [before.shape[1], after.shape[1]]}
        if after.index.equals(before.index):
            changed = [
                c for c in before.columns
                if c in after.columns and not after[c].equals(before[c])]
            if changed:
                details['changed'] = changed
        item.record(self.name, **details)
        return item


@dataclasses.dataclass
class CleanBooleans(Cleaner):
    """Turns text such as "yes" and "no" into booleans.

    Text is compared without regard to case or the spaces around it. Numbers
    are true if they are 1 and false if they are 0. Anything else is missing,
    so the columns have the `pandas` "boolean" type, which allows missing
    values.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        true_values: Sequence[Any] = _TRUE,
        false_values: Sequence[Any] = _FALSE,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the values in `columns` into booleans.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            true_values: words that mean true. Defaults to "true", "t", "yes",
                "y", and "1".
            false_values: words that mean false. Defaults to "false", "f",
                "no", "n", and "0".
            **kwargs: not used.

        Returns:
            The data, with booleans. Columns that are already boolean are not
                changed.

        """
        trues = _words(true_values)
        falses = _words(false_values)
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if pd.api.types.is_numeric_dtype(values.dtype):
                true = (values == 1).to_numpy(dtype = bool)
                false = (values == 0).to_numpy(dtype = bool)
            else:
                words = values.astype('str').str.strip().str.lower()
                true = words.isin(trues).to_numpy(dtype = bool)
                false = words.isin(falses).to_numpy(dtype = bool)
            flags = pd.array(true, dtype = 'boolean')
            flags[~(true | false)] = pd.NA
            data[column] = pd.Series(flags, index = values.index)
        return data


@dataclasses.dataclass
class CleanDates(Cleaner):
    """Turns text into dates.

    Text that is not a date becomes a missing value, rather than an error (as
    it would with `convert_types`).

    Wraps:
        [`pandas.to_datetime`](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        date_format: str | None = None,
        dayfirst: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into dates.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            date_format: the format of the dates, with the codes of Python's
                `strftime` (such as "%m/%d/%Y"), or "mixed" to find the
                format of each date separately (which is slower). Defaults to
                `None`, in which case `pandas` finds the format from the first
                date and uses it for all of them.
            dayfirst: whether the day comes before the month in dates such as
                "01/02/2020". Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with dates.

        """
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype('str')
            data[column] = pd.to_datetime(
                values,
                errors = 'coerce',
                format = date_format,
                dayfirst = dayfirst)
        return data


@dataclasses.dataclass
class CleanNumbers(Cleaner):
    """Turns text such as "$1,234.50" into numbers.

    The first number in the text is kept, so "$1,234.50" becomes 1234.5,
    "12%" becomes 12, and "about 3 years" becomes 3. Text without a number
    becomes a missing value.

    Wraps:
        [`pandas.to_numeric`](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        thousands: str = ',',
        decimal: str = '.',
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into numbers.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            thousands: the character that separates thousands. Defaults to
                ",". Set it to "." (and `decimal` to ",") for numbers such as
                "1.234,50".
            decimal: the character before the decimals. Defaults to ".".
            **kwargs: not used.

        Raises:
            ValueError: if `thousands` and `decimal` are the same.

        Returns:
            The data, with numbers. Columns that are already numbers are not
                changed.

        """
        if thousands == decimal:
            message = 'the thousands and decimal separators must differ'
            raise ValueError(message)
        digits = r'\d+'
        if thousands:
            digits += rf'(?:{re.escape(thousands)}\d{{3}})*'
        point = re.escape(decimal)
        pattern = (
            rf'([-+]?(?:{digits}(?:{point}\d*)?|{point}\d+)'
            rf'(?:[eE][-+]?\d+)?)')
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_numeric_dtype(values.dtype):
                continue
            found = values.astype('str').str.extract(pattern, expand = False)
            if thousands:
                found = found.str.replace(thousands, '', regex = False)
            if decimal != '.':
                found = found.str.replace(decimal, '.', regex = False)
            data[column] = pd.to_numeric(found, errors = 'coerce')
        return data


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
    """Keeps the rows that match a query, or a condition for each column.

    Each parameter other than "query" is a column, and its value is a
    condition for that column, so "state = not in ['GU', 'PR']" keeps the
    rows whose state is not "GU" or "PR". A condition can start with a
    comparison ("==", "!=", "<", "<=", ">", ">=", "in", or "not in"); "not"
    and a value (as in "not 1") means "!="; a list means "in"; and a value
    alone means "==". Words that are not columns are values, so they need no
    quotes (see `utilities.quote_bare_values`). The rows kept match the query
    and every condition.

    Wraps:
        [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        query: str | None = None,
        **conditions: Any) -> pd.DataFrame:
        """Keeps the rows that match `query` and every condition.

        Args:
            data: the data to clean.
            query: a `pandas` query, such as "age >= 18". Column names with
                spaces go inside backticks. Defaults to `None`.
            **conditions: a condition for each column, by its name (see the
                class docstring).

        Raises:
            KeyError: if a condition is for a column that is not in the data.

        Returns:
            The rows that match `query` and every condition. If there are
                neither, every row.

        """
        missing = [c for c in conditions if c not in data.columns]
        if missing:
            message = (
                f'the columns {missing} that filter_rows has conditions for '
                f'are not in the data'
            )
            raise KeyError(message)
        parts = [] if query is None else [str(query)]
        parts.extend(_condition_query(c, v) for c, v in conditions.items())
        if not parts:
            return data
        joined = ' and '.join(f'({p})' for p in parts)
        return data.query(utilities.quote_bare_values(joined, data))


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


""" Private Functions """


def _condition_query(column: str, condition: Any) -> str:
    """Returns a `pandas` query of `condition` for `column`.

    Args:
        column: name of the column.
        condition: a comparison and a value (such as "> 5" or "not in [1,
            2]"), "not" and a value (which means "!="), a list (which means
            "in"), or a value alone (which means "==").

    Returns:
        The query, with the column's name in backticks.

    """
    name = f'`{column}`'
    if isinstance(condition, list | tuple | set):
        return f'{name} in {list(condition)!r}'
    if not isinstance(condition, str):
        return f'{name} == {condition!r}'
    text = condition.strip()
    if _COMPARISON.match(text):
        return f'{name} {text}'
    if text.startswith('not '):
        return f'{name} != {text[4:].strip()}'
    return f'{name} == {text}'


def _words(values: Sequence[Any] | Any) -> list[str]:
    """Returns words that mean true or false, in lower case.

    Args:
        values: a word or a sequence of them. Numbers and booleans (as a
            settings file may give them) are made into text.

    Returns:
        The words, trimmed and in lower case.

    """
    items = [values] if isinstance(values, str) or not isinstance(
        values, Iterable) else list(values)
    return [str(v).strip().lower() for v in items]
